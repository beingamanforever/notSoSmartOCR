"""Export raw Heron detections for OmniDocBench's official layout evaluator."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from experiments.omnidocbench_layout_benchmark import select_panel
from ocr_pipeline.layout import HeronLayoutDetector, LayoutDetection, LayoutDetector

MODEL_NAME = "docling-project/docling-layout-heron"
MODEL_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
PREDICTION_NAME = "predictions.json"
REPORT_NAME = "run_report.json"
OMNIDOC_CATEGORIES = (
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
)
CATEGORY_IDS = {name: index for index, name in enumerate(OMNIDOC_CATEGORIES)}
HERON_TO_OMNIDOC = {
    "title": "title",
    "section_header": "title",
    "text": "plain text",
    "list_item": "plain text",
    "document_index": "plain text",
    "page_footer": "abandon",
    "page_header": "abandon",
    "picture": "figure",
    "caption": "figure_caption",
    "table": "table",
    "footnote": "table_footnote",
    "formula": "isolate_formula",
    "code": "figure",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export raw Heron layout detections for OmniDocBench"
    )
    parser.add_argument("annotations", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--base-per-language",
        type=_positive_int,
        help="Select the same balanced v1.5 panel as the layout evaluator",
    )
    args = parser.parse_args(argv)

    try:
        detector = HeronLayoutDetector(
            model_name=args.model_name,
            model_revision=args.model_revision,
            device=args.device,
            threshold=args.threshold,
        )
        export_predictions(
            args.annotations,
            args.image_root,
            args.output_dir,
            detector,
            dataset_revision=args.dataset_revision,
            limit=args.limit,
            base_per_language=args.base_per_language,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def export_predictions(
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    detector: LayoutDetector,
    *,
    dataset_revision: str,
    limit: int | None = None,
    base_per_language: int | None = None,
) -> dict[str, object]:
    if not dataset_revision.strip():
        raise ValueError("OmniDocBench dataset revision must not be empty")
    records = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("OmniDocBench annotations must be a JSON list")
    selected = select_panel(records, base_per_language)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No OmniDocBench pages were selected")

    case_ids = [_image_path(record, index) for index, record in enumerate(selected)]
    if len({Path(case_id).stem for case_id in case_ids}) != len(case_ids):
        raise ValueError("Selected pages have duplicate prediction names")
    output_dir.mkdir(parents=True)

    results: list[dict[str, object]] = []
    pages = []
    unsupported = Counter()
    for index, case_id in enumerate(case_ids):
        started = time.perf_counter()
        failure = None
        mapped = []
        raw_count = 0
        try:
            detections = detector.detect(image_root / case_id)
            raw_count = len(detections)
            for detection in detections:
                category = HERON_TO_OMNIDOC.get(detection.label)
                if category is None:
                    unsupported[detection.label] += 1
                    continue
                mapped.append(_prediction(case_id, detection, category))
        except Exception as error:
            failure = {
                "stage": "layout",
                "code": "layout_failed",
                "message": str(error),
            }
        results.extend(mapped)
        pages.append(
            {
                "case_id": case_id,
                "status": "failed"
                if failure
                else ("success" if mapped else "abstained"),
                "raw_detections": raw_count,
                "mapped_detections": len(mapped),
                "failure": failure,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        print(
            f"heron-layout {index + 1}/{len(case_ids)} {case_id} {pages[-1]['status']}",
            file=sys.stderr,
            flush=True,
        )

    predictions = {
        "results": results,
        "categories": {
            str(index): name for index, name in enumerate(OMNIDOC_CATEGORIES)
        },
    }
    (output_dir / PREDICTION_NAME).write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latencies = [float(page["latency_ms"]) for page in pages]
    states = Counter(str(page["status"]) for page in pages)
    report: dict[str, object] = {
        "dataset": "OmniDocBench",
        "dataset_revision": dataset_revision,
        "model": detector.name,
        "model_name": getattr(detector, "model_name", None),
        "model_revision": getattr(detector, "model_revision", None),
        "device": getattr(detector, "device", None),
        "threshold": getattr(detector, "threshold", None),
        "selection": {
            "base_per_language": base_per_language,
            "limit": limit,
        },
        "mapping": HERON_TO_OMNIDOC,
        "case_ids": case_ids,
        "attempted": len(pages),
        "covered": states["success"],
        "failed": states["failed"],
        "abstained": states["abstained"],
        "mapped_detections": len(results),
        "unsupported_detections": dict(sorted(unsupported.items())),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "predictions": PREDICTION_NAME,
        "pages": pages,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _prediction(
    case_id: str,
    detection: LayoutDetection,
    category: str,
) -> dict[str, object]:
    return {
        "image_name": Path(case_id).stem,
        "bbox": list(detection.box),
        "category_id": CATEGORY_IDS[category],
        "score": detection.confidence,
    }


def _image_path(record: object, index: int) -> str:
    if not isinstance(record, dict) or not isinstance(record.get("page_info"), dict):
        raise ValueError(f"Invalid page_info in annotation record {index}")
    image_path = record["page_info"].get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"Invalid image_path in annotation record {index}")
    return image_path


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
