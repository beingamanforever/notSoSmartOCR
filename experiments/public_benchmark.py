"""Run transcription benchmarks through the real OCR pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ocr_pipeline.pipeline import IMAGE_SUFFIXES, process_document
from ocr_pipeline.providers import TesseractReader

MAX_WORKERS = 32
NORMALIZATION = "Unicode NFKC, casefold, and collapsed whitespace"


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    subset: str
    image_path: Path
    reference: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the OCR pipeline on public transcription datasets"
    )
    parser.add_argument("dataset", choices=("clinocr", "funsd"))
    parser.add_argument("root", type=Path, help="Extracted dataset root")
    parser.add_argument("output", type=Path, help="JSON results path")
    parser.add_argument(
        "--workers",
        type=_worker_count,
        default=min(4, os.cpu_count() or 1),
        help=f"Concurrent OCR workers, from 1 to {MAX_WORKERS}",
    )
    parser.add_argument("--language", default="eng", help="Tesseract language")
    parser.add_argument("--tesseract", default="tesseract", help="Tesseract executable")
    args = parser.parse_args(argv)

    try:
        payload = run_benchmark(
            args.dataset,
            args.root,
            args.workers,
            TesseractReader(language=args.language, executable=args.tesseract),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset: str,
    root: Path,
    workers: int,
    reader: TesseractReader,
) -> dict[str, object]:
    cases = discover_cases(dataset, root)
    if not cases:
        raise ValueError(f"No evaluation cases found in {root}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(
            executor.map(lambda case: _evaluate_case(case, root, reader), cases)
        )

    subsets = {
        subset: _summarize([record for record in records if record["subset"] == subset])
        for subset in sorted({case.subset for case in cases})
    }
    return {
        "dataset": dataset,
        "dataset_root": str(root.resolve()),
        "reader": reader.name,
        "normalization": NORMALIZATION,
        "summary": _summarize(records),
        "subsets": subsets,
        "cases": records,
    }


def discover_cases(dataset: str, root: Path) -> list[BenchmarkCase]:
    if dataset == "clinocr":
        return _discover_clinocr(root)
    if dataset == "funsd":
        return _discover_funsd(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _discover_clinocr(root: Path) -> list[BenchmarkCase]:
    lookup_path = root / "oneshot_lookup.csv"
    if not lookup_path.is_file():
        raise ValueError(f"ClinOCR lookup not found: {lookup_path}")

    cases: list[BenchmarkCase] = []
    with lookup_path.open(encoding="utf-8-sig", newline="") as lookup_file:
        rows = csv.DictReader(lookup_file)
        required = {"subset", "template", "sample", "role"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError(f"ClinOCR lookup has invalid columns: {lookup_path}")

        for row in rows:
            if row["role"].strip().casefold() != "eval":
                continue
            subset = row["subset"].strip()
            template = row["template"].strip()
            sample = row["sample"].strip()
            stem = f"template_{template}_sample_{sample}_{subset}"
            reference_path = root / "ground_truth" / subset / f"{stem}.txt"
            if not reference_path.is_file():
                raise ValueError(f"ClinOCR reference not found: {reference_path}")
            image_path = _find_image(root / "scans" / subset, stem, ".jpg")
            cases.append(
                BenchmarkCase(
                    id=f"{subset}/{stem}",
                    subset=subset,
                    image_path=image_path,
                    reference=reference_path.read_text(encoding="utf-8-sig"),
                )
            )
    return sorted(cases, key=lambda case: case.id)


def _discover_funsd(root: Path) -> list[BenchmarkCase]:
    annotations_dir = root / "testing_data" / "annotations"
    images_dir = root / "testing_data" / "images"
    if not annotations_dir.is_dir():
        raise ValueError(f"FUNSD test annotations not found: {annotations_dir}")

    cases = []
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        cases.append(
            BenchmarkCase(
                id=f"test/{annotation_path.stem}",
                subset="test",
                image_path=_find_image(images_dir, annotation_path.stem, ".png"),
                reference=_funsd_reference(annotation, annotation_path),
            )
        )
    return cases


def _evaluate_case(
    case: BenchmarkCase,
    root: Path,
    reader: TesseractReader,
) -> dict[str, object]:
    started = time.perf_counter()
    result = process_document(case.image_path, reader)
    latency_ms = (time.perf_counter() - started) * 1000
    prediction = "\n\n".join(page.text.value for page in result.pages)
    metrics = _score(prediction, case.reference)
    return {
        "id": case.id,
        "subset": case.subset,
        "image": _relative_path(case.image_path, root),
        "prediction": prediction,
        "reference": case.reference,
        "status": result.status,
        "metrics": metrics,
        "failures": [failure.__dict__ for failure in result.failures],
        "latency_ms": round(latency_ms, 3),
    }


def _score(prediction: str, reference: str) -> dict[str, dict[str, float | int]]:
    normalized_prediction = _normalize(prediction)
    normalized_reference = _normalize(reference)
    prediction_words = normalized_prediction.split()
    reference_words = normalized_reference.split()
    character_edits = _edit_distance(normalized_prediction, normalized_reference)
    word_edits = _edit_distance(prediction_words, reference_words)
    return {
        "cer": _case_metric(character_edits, len(normalized_reference)),
        "wer": _case_metric(word_edits, len(reference_words)),
    }


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    case_count = len(records)
    covered_count = sum(
        bool(_normalize(str(record["prediction"]))) for record in records
    )
    status_counts = Counter(str(record["status"]) for record in records)
    failure_codes = Counter(
        str(failure["code"])
        for record in records
        for failure in record["failures"]  # type: ignore[union-attr]
    )
    latencies = [float(record["latency_ms"]) for record in records]

    return {
        "cases": case_count,
        "covered_cases": covered_count,
        "coverage": _ratio(covered_count, case_count),
        "failed_cases": case_count - status_counts.get("success", 0),
        "pipeline_failures": sum(failure_codes.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_codes": dict(sorted(failure_codes.items())),
        "cer": _aggregate_metric(records, "cer"),
        "wer": _aggregate_metric(records, "wer"),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
    }


def _aggregate_metric(
    records: list[dict[str, object]], name: str
) -> dict[str, float | int]:
    case_metrics = [record["metrics"][name] for record in records]  # type: ignore[index]
    edits = sum(int(metric["edits"]) for metric in case_metrics)
    reference_units = sum(int(metric["reference_units"]) for metric in case_metrics)
    return {
        "case_mean": round(
            sum(float(metric["rate"]) for metric in case_metrics) / len(case_metrics),
            6,
        ),
        "micro": round(_ratio(edits, reference_units), 6),
        "edits": edits,
        "reference_units": reference_units,
    }


def _case_metric(edits: int, reference_units: int) -> dict[str, float | int]:
    return {
        "edits": edits,
        "reference_units": reference_units,
        "rate": round(_ratio(edits, reference_units), 6),
    }


def _edit_distance(left: Sequence[object], right: Sequence[object]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _funsd_reference(annotation: object, path: Path) -> str:
    if not isinstance(annotation, dict) or not isinstance(annotation.get("form"), list):
        raise ValueError(f"Invalid FUNSD annotation: {path}")

    words: list[tuple[int, int, str]] = []
    for item in annotation["form"]:
        if not isinstance(item, dict) or not isinstance(item.get("words"), list):
            raise ValueError(f"Invalid FUNSD annotation item: {path}")
        for word in item["words"]:
            if not isinstance(word, dict) or not isinstance(word.get("box"), list):
                raise ValueError(f"Invalid FUNSD word: {path}")
            text = str(word.get("text", "")).strip()
            box = word["box"]
            if text and len(box) >= 2:
                words.append((int(box[1]), int(box[0]), text))
    words.sort()
    return " ".join(text for _, _, text in words)


def _find_image(directory: Path, stem: str, default_suffix: str) -> Path:
    matches = [
        path
        for path in directory.glob(f"{stem}.*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple images found for {stem}: {directory}")
    return matches[0] if matches else directory / f"{stem}{default_suffix}"


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else float(numerator)
    return numerator / denominator


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _worker_count(value: str) -> int:
    workers = int(value)
    if not 1 <= workers <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"workers must be between 1 and {MAX_WORKERS}")
    return workers


if __name__ == "__main__":
    raise SystemExit(main())
