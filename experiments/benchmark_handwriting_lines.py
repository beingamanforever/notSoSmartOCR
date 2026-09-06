"""Evaluate existing handwriting readers on pinned public text-line test sets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any
import unicodedata

from PIL import Image

from ocr_pipeline.verification import EditCounts, edit_counts


@dataclass(frozen=True)
class DatasetSpec:
    repository: str
    revision: str
    split: str
    expected_lines: int
    language: str


DATASETS = {
    "iam": DatasetSpec(
        repository="Teklia/IAM-line",
        revision="fbdad97500ce54635c0d1ba306bf535cb40656cf",
        split="test",
        expected_lines=2915,
        language="English",
    ),
    "rimes": DatasetSpec(
        repository="Teklia/RIMES-2011-line",
        revision="ba3e6b5573094208b30a134e32d9b65dab18e74e",
        split="test",
        expected_lines=778,
        language="French",
    ),
}
READERS = ("nemotron", "phi4", "falcon", "trocr")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            prepare_dataset(args.dataset, args.data_root, args.cache_dir)
        else:
            run_benchmark(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if getattr(args, "output", None) and not args.output.exists():
            _write_json(
                args.output,
                {
                    "status": "rejected_incomplete",
                    "failure": f"{type(error).__name__}: {error}",
                },
            )
        parser.error(str(error))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("dataset", choices=DATASETS)
    prepare.add_argument("data_root", type=Path)
    prepare.add_argument("--cache-dir", type=Path)

    run = actions.add_parser("run")
    run.add_argument("dataset", choices=DATASETS)
    run.add_argument("reader", choices=READERS)
    run.add_argument("data_root", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--model-path", type=Path)
    run.add_argument("--adapter-path", type=Path)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=_positive_int, default=1)
    run.add_argument("--max-new-tokens", type=_positive_int, default=128)
    run.add_argument("--limit", type=_positive_int)
    run.add_argument("--binarize", action="store_true")
    return parser


def prepare_dataset(
    dataset_name: str,
    data_root: Path,
    cache_dir: Path | None = None,
) -> Path:
    from datasets import load_dataset

    spec = DATASETS[dataset_name]
    destination = data_root / dataset_name / spec.revision / spec.split
    ground_truth = destination / "ground_truth.jsonl"
    dataset = load_dataset(
        spec.repository,
        revision=spec.revision,
        split=spec.split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if len(dataset) != spec.expected_lines:
        raise ValueError(
            f"{spec.repository} {spec.split} has {len(dataset)} lines, "
            f"expected {spec.expected_lines}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    temporary_ground_truth = destination / "ground_truth.jsonl.tmp"
    with temporary_ground_truth.open("w", encoding="utf-8") as stream:
        for index, item in enumerate(dataset):
            reference = item.get("text")
            source = item.get("image")
            if not isinstance(reference, str) or not reference:
                raise ValueError(f"line {index} has no ground-truth text")
            if not isinstance(source, Image.Image):
                raise ValueError(f"line {index} has no decoded image")
            image_name = f"{index:06d}.png"
            image_path = destination / image_name
            presented = source.convert("RGB")
            try:
                temporary_image = destination / f".{image_name}.tmp"
                presented.save(temporary_image, format="PNG")
                os.replace(temporary_image, image_path)
                record = {
                    "id": f"{dataset_name}-{spec.split}-{index:06d}",
                    "image": image_name,
                    "reference": reference,
                    "source_dimensions": list(source.size),
                    "source_mode": source.mode,
                    "presented_dimensions": list(presented.size),
                    "presented_mode": presented.mode,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            finally:
                presented.close()
    os.replace(temporary_ground_truth, ground_truth)
    _load_cases(dataset_name, data_root, None)
    return ground_truth


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[args.dataset]
    cases = _load_cases(args.dataset, args.data_root, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.output.with_suffix(".rows.jsonl")
    if args.output.exists() or rows_path.exists():
        raise ValueError("benchmark output already exists")

    started = time.perf_counter()
    reader = _load_reader(args)
    model_load_ms = round((time.perf_counter() - started) * 1000, 3)
    warmup_started = time.perf_counter()
    warmup = _predict(args.reader, reader, cases[:1])
    warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 3)
    warmup_result = {
        "sample_id": cases[0]["id"],
        "status": warmup[0]["status"],
        "failure": warmup[0]["failure"],
        "latency_ms": warmup_ms,
        "sample_is_scored_in_benchmark": True,
    }

    _reset_peak_memory()
    rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    with rows_path.open("x", encoding="utf-8") as stream:
        for start in range(0, len(cases), args.batch_size):
            batch = cases[start : start + args.batch_size]
            batch_started = time.perf_counter()
            predictions = _predict(args.reader, reader, batch)
            batch_ms = (time.perf_counter() - batch_started) * 1000
            for case, prediction in zip(batch, predictions, strict=True):
                row = _row(case, prediction, batch_ms / len(batch), len(batch))
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if len(rows) % max(args.batch_size * 25, 25) == 0:
                _write_json(
                    args.output,
                    _summary_payload(
                        args,
                        spec,
                        reader,
                        rows,
                        rows_path,
                        model_load_ms,
                        warmup_result,
                        time.perf_counter() - benchmark_started,
                        status="running",
                        total_cases=len(cases),
                    ),
                )

    payload = _summary_payload(
        args,
        spec,
        reader,
        rows,
        rows_path,
        model_load_ms,
        warmup_result,
        time.perf_counter() - benchmark_started,
        status="complete",
        total_cases=len(cases),
    )
    _write_json(args.output, payload)
    return payload


def _load_cases(
    dataset_name: str,
    data_root: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    spec = DATASETS[dataset_name]
    root = data_root / dataset_name / spec.revision / spec.split
    ground_truth = root / "ground_truth.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in ground_truth.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared dataset: {ground_truth}") from error
    if len(rows) != spec.expected_lines:
        raise ValueError(
            f"prepared {dataset_name} has {len(rows)} lines, "
            f"expected {spec.expected_lines}"
        )
    for index, row in enumerate(rows):
        image_path = root / str(row.get("image", ""))
        expected_id = f"{dataset_name}-{spec.split}-{index:06d}"
        expected_image = f"{index:06d}.png"
        if (
            row.get("id") != expected_id
            or row.get("image") != expected_image
            or not isinstance(row.get("reference"), str)
            or row.get("presented_mode") != "RGB"
            or not image_path.is_file()
        ):
            raise ValueError("prepared dataset has an invalid line")
        with Image.open(image_path) as image:
            if image.mode != "RGB" or list(image.size) != row.get(
                "presented_dimensions"
            ):
                raise ValueError("prepared dataset image does not match its record")
        row["image_path"] = image_path
    return rows[:limit] if limit else rows


def _load_reader(args: argparse.Namespace) -> object:
    if args.reader == "trocr":
        from ocr_pipeline.handwriting import TrOCRHandwritingReader

        overrides = {"model_name_or_path": args.model_path} if args.model_path else {}
        overrides["binarize"] = bool(getattr(args, "binarize", False))
        reader = TrOCRHandwritingReader(
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_batch_items=max(16, args.batch_size),
            batch_size=args.batch_size,
            **overrides,
        )
        reader.check_health()
        return reader
    if args.reader == "falcon":
        from ocr_pipeline.falcon import FalconOCRReader

        if args.model_path is None:
            raise ValueError("Falcon requires --model-path")
        reader = FalconOCRReader(
            local_model_path=args.model_path,
            category="text",
            device_map=args.device,
            max_new_tokens=args.max_new_tokens,
            max_dimension=1536,
        )
        reader.check_health()
        return reader
    if args.reader == "phi4":
        from ocr_pipeline.providers import Phi4HandwritingReader

        if args.adapter_path is None:
            raise ValueError("Phi-4 requires --adapter-path")
        reader = Phi4HandwritingReader(
            args.adapter_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_batch_items=max(16, args.batch_size),
            batch_size=args.batch_size,
        )
        reader._initialize_components()
        return reader

    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    from ocr_pipeline.providers import NemotronOCRV2Reader

    if args.model_path is None:
        raise ValueError("Nemotron requires --model-path")
    native = NemotronOCRV2(model_dir=str(args.model_path), lang="multi")
    return NemotronOCRV2Reader(
        language="multi",
        merge_level="paragraph",
        batch_size=args.batch_size,
        pipeline=native,
    )


def _predict(
    reader_name: str,
    reader: object,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        if reader_name == "nemotron":
            results = reader.read_batch(  # type: ignore[attr-defined]
                [case["image_path"] for case in cases],
                list(range(1, len(cases) + 1)),
            )
            output = []
            for case, result in zip(cases, results, strict=True):
                if isinstance(result, Exception):
                    output.append(_failure(result))
                    continue
                regions = sorted(
                    result,
                    key=lambda region: (
                        region.reading_order,
                        region.bounding_box.top,
                        region.bounding_box.left,
                    ),
                )
                output.append(
                    {
                        "status": "success",
                        "prediction": "\n".join(region.text for region in regions),
                        "provider_output": [
                            {
                                "text": region.text,
                                "kind": region.kind,
                                "confidence": region.confidence,
                                "bounding_box": asdict(region.bounding_box),
                            }
                            for region in regions
                        ],
                        "model_input_dimensions": {
                            "reader_input": case["presented_dimensions"],
                            "processor": "unexposed_by_nemotron_ocr_v2",
                        },
                        "failure": None,
                    }
                )
            return output

        images = []
        try:
            for case in cases:
                with Image.open(case["image_path"]) as source:
                    images.append(source.convert("RGB"))
            model_inputs = [
                _model_input_dimensions(reader_name, reader, image) for image in images
            ]
            if reader_name == "falcon":
                texts = reader.transcribe_crops(  # type: ignore[attr-defined]
                    images,
                    ["text"] * len(images),
                )
            else:
                texts = reader.transcribe_batch(images)  # type: ignore[attr-defined]
        finally:
            for image in images:
                image.close()
        return [
            {
                "status": "success",
                "prediction": text,
                "provider_output": text,
                "model_input_dimensions": model_input,
                "failure": None,
            }
            for text, model_input in zip(texts, model_inputs, strict=True)
        ]
    except Exception as error:
        return [_failure(error) for _ in cases]


def _failure(error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "prediction": "",
        "provider_output": None,
        "model_input_dimensions": None,
        "failure": {
            "type": type(error).__name__,
            "code": getattr(error, "code", None),
            "message": str(error),
        },
    }


def _row(
    case: dict[str, Any],
    prediction: dict[str, Any],
    latency_ms: float,
    batch_size: int,
) -> dict[str, Any]:
    raw_prediction = str(prediction["prediction"])
    reference = str(case["reference"])
    return {
        "id": case["id"],
        "image": case["image"],
        "source_dimensions": case["source_dimensions"],
        "presented_dimensions": case["presented_dimensions"],
        "reference": reference,
        "reader_prediction": raw_prediction,
        "output_boundary": "ocr_pipeline_reader_public_output",
        "status": prediction["status"],
        "failure": prediction["failure"],
        "provider_output": prediction["provider_output"],
        "model_input_dimensions": prediction["model_input_dimensions"],
        "latency_ms": round(latency_ms, 3),
        "batch_size": batch_size,
        "metrics": {"literal": _score(raw_prediction, reference, _normalize_line)},
    }


def _score(
    prediction: str,
    reference: str,
    normalize: Any,
) -> dict[str, Any]:
    scored_prediction = normalize(prediction)
    scored_reference = normalize(reference)
    character_counts = edit_counts(scored_prediction, scored_reference)
    word_counts = edit_counts(scored_prediction.split(), scored_reference.split())
    return {
        "prediction": scored_prediction,
        "reference": scored_reference,
        "exact": scored_prediction == scored_reference,
        "reference_characters": len(scored_reference),
        "reference_words": len(scored_reference.split()),
        "character_edits": asdict(character_counts),
        "word_edits": asdict(word_counts),
    }


def _summary_payload(
    args: argparse.Namespace,
    spec: DatasetSpec,
    reader: object,
    rows: list[dict[str, Any]],
    rows_path: Path,
    model_load_ms: float,
    warmup: dict[str, Any],
    wall_seconds: float,
    *,
    status: str,
    total_cases: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "dataset": {
            **asdict(spec),
            "name": args.dataset,
            "full_test_split": args.limit is None,
        },
        "reader": {
            "name": args.reader,
            "provenance": _reader_provenance(args, reader),
            "generation": getattr(reader, "generation", None),
            "model_path": str(args.model_path) if args.model_path else None,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        },
        "configuration": {
            "batch_size": args.batch_size,
            "task_mode": "direct_complete_text_line",
            "reader_contract": _reader_contract(args),
            "preprocessing": "dataset_decoded_pixels_converted_to_rgb_png",
            "model_input_dimensions": "recorded_per_row_or_explicitly_unexposed",
            "output_boundary": "ocr_pipeline_reader_public_output",
            "text_correction": False,
        },
        "progress": {"completed": len(rows), "expected": total_cases},
        "latency": {
            "model_load_ms": model_load_ms,
            "warmup": warmup,
            "benchmark_wall_ms": round(wall_seconds * 1000, 3),
            **_latency(rows),
        },
        "metrics": {
            "literal": _aggregate(rows, "literal"),
            "failed_lines": sum(row["status"] != "success" for row in rows),
            "empty_predictions": sum(not row["reader_prediction"] for row in rows),
            "failures_remain_in_denominators": True,
        },
        "peak_cuda_memory_mib": _peak_memory_mib(),
        "rows": rows_path.name,
    }


def _aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    character_counts = _sum_counts(rows, mode, "character_edits")
    word_counts = _sum_counts(rows, mode, "word_edits")
    reference_characters = sum(
        row["metrics"][mode]["reference_characters"] for row in rows
    )
    reference_words = sum(row["metrics"][mode]["reference_words"] for row in rows)
    exact = sum(row["metrics"][mode]["exact"] for row in rows)
    return {
        "cer": _ratio(character_counts.edits, reference_characters),
        "wer": _ratio(word_counts.edits, reference_words),
        "exact": exact,
        "exact_rate": _ratio(exact, len(rows)),
        "reference_characters": reference_characters,
        "reference_words": reference_words,
        "character_edits": asdict(character_counts),
        "word_edits": asdict(word_counts),
    }


def _sum_counts(rows: list[dict[str, Any]], mode: str, key: str) -> EditCounts:
    values = [row["metrics"][mode][key] for row in rows]
    return EditCounts(
        insertions=sum(value["insertions"] for value in values),
        deletions=sum(value["deletions"] for value in values),
        substitutions=sum(value["substitutions"] for value in values),
    )


def _latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row["latency_ms"]) for row in rows)
    if not values:
        return {"p50_ms": None, "p95_ms": None, "lines_per_second": None}
    return {
        "basis": "amortized_batch_wall_time_per_line",
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(values[math.ceil(len(values) * 0.95) - 1], 3),
        "lines_per_second": round(1000 / statistics.mean(values), 3),
    }


def _reader_provenance(args: argparse.Namespace, reader: object) -> Any:
    provenance = getattr(reader, "provenance", None)
    if args.reader == "falcon" and args.model_path is not None:
        return {
            **(provenance or {}),
            "revision": "unverified",
            "identity_verified": False,
            "identity_reason": "Explicit local model paths are not revision proof.",
        }
    if args.reader != "nemotron":
        return provenance
    repository = args.model_path.parent
    relative_model_path = args.model_path.relative_to(repository)
    revision = _git(repository, "rev-parse", "HEAD")
    remote = _git(repository, "config", "--get", "remote.origin.url")
    return {
        "id": "unresolved",
        "loaded_from": str(args.model_path),
        "repository_remote": remote or "unresolved",
        "repository_revision": revision or "unresolved",
        "weight_revision": "unresolved",
        "weights_tracked_and_clean": False,
        "weight_revision_reason": (
            "The code checkout revision does not identify every file under "
            f"{relative_model_path}."
        ),
        "identity_verified": False,
    }


def _git(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _model_input_dimensions(
    reader_name: str,
    reader: object,
    image: Image.Image,
) -> dict[str, Any]:
    dimensions: dict[str, Any] = {"reader_input": list(image.size)}
    if reader_name == "trocr":
        image_processor = getattr(
            getattr(reader, "_processor", None), "image_processor"
        )
        dimensions["processor_size"] = _mapping_or_value(image_processor.size)
        dimensions["processor_crop_size"] = _mapping_or_value(image_processor.crop_size)
    elif reader_name == "falcon":
        from ocr_pipeline.falcon import _prepare_falcon_crop

        prepared = _prepare_falcon_crop(
            image,
            reader.generation_config.max_dimension,  # type: ignore[attr-defined]
        )
        dimensions["prepared_crop"] = list(prepared.image.size)
        dimensions["processor"] = "unexposed_by_falcon_generate"
        if prepared.owned:
            prepared.image.close()
    else:
        dimensions["processor"] = "unexposed_by_phi4_processor"
    return dimensions


def _mapping_or_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    dimensions = {
        name: getattr(value, name)
        for name in (
            "height",
            "width",
            "shortest_edge",
            "longest_edge",
            "max_height",
            "max_width",
        )
        if getattr(value, name, None) is not None
    }
    return dimensions or str(value)


def _reader_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.reader == "phi4":
        from ocr_pipeline.providers import PHI4_HANDWRITING_PROMPT

        return {
            "prompt": PHI4_HANDWRITING_PROMPT,
            "dynamic_hd": 1,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
    if args.reader == "falcon":
        return {
            "category": "text",
            "max_dimension": 1536,
            "device_map": args.device,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
        }
    if args.reader == "nemotron":
        return {
            "language": "multi",
            "merge_level": "paragraph",
            "device": "library_default_unexposed",
            "decoding": "library_default_unexposed",
            "max_new_tokens": "not_configurable_by_reader",
        }
    return {
        "scope": "single_text_line",
        "processor": "TrOCRProcessor",
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }


def _normalize_line(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def _peak_memory_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    except (ImportError, RuntimeError):
        pass
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
