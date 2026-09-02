"""Benchmark docTR FAST as a missing-text region proposal detector."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ARCHITECTURE = "fast_base"
TARGET_CASE_IDS = ("C14-D001-P001", "C14-D002-P001")
CURRENT_REGION_KINDS = {"text", "word"}
NEGATIVE_EXCLUDED_CATEGORIES = {"C08", "C13", "C14"}
DEFAULT_NEGATIVE_PAGES = 20
IOU_THRESHOLD = 0.5
COVERAGE_THRESHOLD = 0.5
NOVELTY_COVERAGE_THRESHOLD = 0.5
BIN_THRESHOLD = 0.1
BOX_THRESHOLD = 0.1
CONFIDENCE_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

Box = list[float]
Metric = Callable[[Box, Box], float]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run detector-only docTR inference")
    infer.add_argument("images", type=Path)
    infer.add_argument("output", type=Path)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--architecture", default=ARCHITECTURE)
    infer.add_argument("--case-id", action="append", dest="case_ids")

    evaluate = subparsers.add_parser(
        "evaluate", help="Score saved detections against the private challenge panel"
    )
    evaluate.add_argument("challenge_root", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument(
        "--additional-predictions", type=Path, action="append", default=[]
    )
    evaluate.add_argument("--overlays", type=Path)
    evaluate.add_argument(
        "--negative-pages", type=_positive_int, default=DEFAULT_NEGATIVE_PAGES
    )

    selected = subparsers.add_parser(
        "selected-images", help="Print the deterministic local panel image paths"
    )
    selected.add_argument("challenge_root", type=Path)
    selected.add_argument(
        "--negative-pages", type=_positive_int, default=DEFAULT_NEGATIVE_PAGES
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "infer":
            payload = run_inference(
                args.images,
                args.output,
                device=args.device,
                architecture=args.architecture,
                case_ids=args.case_ids,
            )
            print(json.dumps(_inference_console_summary(payload), indent=2))
        elif args.command == "evaluate":
            payload = evaluate_predictions(
                args.challenge_root,
                args.predictions,
                args.output,
                overlays=args.overlays,
                negative_pages=args.negative_pages,
                additional_predictions=args.additional_predictions,
            )
            print(json.dumps(_evaluation_console_summary(payload), indent=2))
        else:
            for case in select_panel(
                args.challenge_root, negative_pages=args.negative_pages
            ):
                print(case["image_path"])
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_inference(
    images_root: Path,
    output: Path,
    *,
    device: str,
    architecture: str = ARCHITECTURE,
    case_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    if not images_root.is_dir():
        raise ValueError(f"Image directory does not exist: {images_root}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    image_paths = sorted(
        (
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(images_root).as_posix(),
    )
    if not image_paths:
        raise ValueError(f"No images found below: {images_root}")
    if case_ids:
        by_case_id: dict[str, Path] = {}
        for path in image_paths:
            if path.stem in by_case_id:
                raise ValueError(f"Duplicate image case_id: {path.stem}")
            by_case_id[path.stem] = path
        missing = sorted(set(case_ids) - set(by_case_id))
        if missing:
            raise ValueError(f"Requested images are missing: {missing}")
        image_paths = [by_case_id[case_id] for case_id in sorted(set(case_ids))]

    started = time.perf_counter()
    predictor, torch_module, doctr_version = _load_predictor(architecture, device)
    model_load_ms = (time.perf_counter() - started) * 1000
    first_image, _, _, _ = _load_image(image_paths[0])
    predictor([first_image])
    _synchronize(torch_module, device)

    if device.startswith("cuda"):
        torch_module.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, object]] = []
    for image_path in image_paths:
        case_id = image_path.stem
        try:
            image, width, height, decode_ms = _load_image(image_path)
            _synchronize(torch_module, device)
            inference_started = time.perf_counter()
            raw_prediction = predictor([image])
            _synchronize(torch_module, device)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            detections = _normalize_doctr_prediction(raw_prediction, width, height)
            record = {
                "case_id": case_id,
                "status": "success",
                "width": width,
                "height": height,
                "detections": detections,
                "decode_ms": round(decode_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "failure": None,
            }
        except Exception as error:
            _synchronize(torch_module, device)
            record = {
                "case_id": case_id,
                "status": "failed",
                "width": None,
                "height": None,
                "detections": [],
                "decode_ms": None,
                "inference_ms": None,
                "failure": f"{type(error).__name__}",
            }
        records.append(record)

    latencies = [
        float(record["inference_ms"])
        for record in records
        if record["inference_ms"] is not None
    ]
    status_counts = Counter(str(record["status"]) for record in records)
    model = predictor.model
    postprocessor = model.postprocessor
    payload: dict[str, object] = {
        "benchmark": "docTR FAST missing-text proposals",
        "status": "complete",
        "model": {
            "library": "python-doctr",
            "library_version": doctr_version,
            "architecture": architecture,
            "pretrained": True,
            "checkpoint_url": model.cfg.get("url"),
            "assume_straight_pages": True,
            "preserve_aspect_ratio": True,
            "symmetric_pad": True,
            "bin_threshold": float(postprocessor.bin_thresh),
            "box_threshold": float(postprocessor.box_thresh),
            "device": device,
            "dtype": str(next(model.parameters()).dtype),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch_module.__version__,
            "cuda": torch_module.version.cuda,
            "gpu": (
                torch_module.cuda.get_device_name(device)
                if device.startswith("cuda")
                else None
            ),
        },
        "panel": {
            "images_root": str(images_root.resolve()),
            "attempted": len(records),
            "case_ids": [record["case_id"] for record in records],
        },
        "operations": {
            "warmup_pages": 1,
            "model_load_ms": round(model_load_ms, 3),
            "inference_ms": _latency_summary(latencies),
            "decode_ms": _latency_summary(
                [
                    float(record["decode_ms"])
                    for record in records
                    if record["decode_ms"] is not None
                ]
            ),
            "cuda_memory_mib": _cuda_memory(torch_module, device),
        },
        "coverage": {
            "attempted": len(records),
            "succeeded": status_counts["success"],
            "failed": status_counts["failed"],
            "failure_policy": "failed pages retain zero proposals",
        },
        "cases": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return payload


def evaluate_predictions(
    challenge_root: Path,
    predictions_path: Path,
    output: Path,
    *,
    overlays: Path | None = None,
    negative_pages: int = DEFAULT_NEGATIVE_PAGES,
    additional_predictions: Sequence[Path] = (),
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if overlays is not None and overlays.exists():
        raise FileExistsError(f"Overlay directory already exists: {overlays}")
    panel = select_panel(challenge_root, negative_pages=negative_pages)
    prediction_runs = [
        _load_json(path) for path in (predictions_path, *additional_predictions)
    ]
    prediction_records: dict[str, dict[str, Any]] = {}
    for predictions in prediction_runs:
        model = predictions.get("model")
        if not isinstance(model, dict) or model.get("architecture") != ARCHITECTURE:
            raise ValueError(f"Predictions must come from docTR {ARCHITECTURE}")
        for case_id, record in _prediction_records(predictions).items():
            if case_id in prediction_records:
                raise ValueError(f"Duplicate prediction across runs for {case_id}")
            prediction_records[case_id] = record
    expected_ids = {str(case["case_id"]) for case in panel}
    extras = sorted(set(prediction_records) - expected_ids)
    if extras:
        raise ValueError(f"Predictions contain cases outside the fixed panel: {extras}")

    case_records = []
    for case in panel:
        case_id = str(case["case_id"])
        annotation = _load_json(Path(case["annotation_path"]))
        current_output = _load_json(Path(case["current_output_path"]))
        current_boxes = _current_boxes(current_output)
        target_boxes = (
            _handwriting_boxes(annotation) if case["role"] == "target" else []
        )
        prediction = prediction_records.get(
            case_id,
            {
                "case_id": case_id,
                "status": "failed",
                "detections": [],
                "failure": "missing_prediction",
            },
        )
        detections = _detections(prediction)
        case_records.append(
            {
                "case_id": case_id,
                "category_id": case["category_id"],
                "role": case["role"],
                "image_path": case["image_path"],
                "status": prediction.get("status", "failed"),
                "failure": prediction.get("failure"),
                "target_boxes": target_boxes,
                "current_boxes": current_boxes,
                "detections": detections,
            }
        )

    default_score = _score_variant(case_records, BOX_THRESHOLD)
    sweeps = {
        _confidence_key(threshold): _score_variant(case_records, threshold)
        for threshold in CONFIDENCE_SWEEP
    }
    failures = Counter(
        str(case["failure"] or "unspecified")
        for case in case_records
        if case["status"] != "success"
    )
    payload: dict[str, object] = {
        "benchmark": "docTR FAST missing-text proposals",
        "status": "complete",
        "panel": {
            "challenge": "challenging-formats-20260902",
            "target_pages": len(TARGET_CASE_IDS),
            "negative_pages": negative_pages,
            "attempted_pages": len(case_records),
            "target_case_ids": list(TARGET_CASE_IDS),
            "negative_case_ids": [
                case["case_id"] for case in case_records if case["role"] == "negative"
            ],
            "negative_selection": (
                "deterministic evenly spaced pages across categories other than "
                "C08, C13, and C14; annotation handwriting must be empty"
            ),
        },
        "model": prediction_runs[0].get("model"),
        "runtime": prediction_runs[0].get("runtime"),
        "operations": prediction_runs[0].get("operations"),
        "prediction_runs": [
            {
                "model": predictions.get("model"),
                "runtime": predictions.get("runtime"),
                "operations": predictions.get("operations"),
                "panel": predictions.get("panel"),
            }
            for predictions in prediction_runs
        ],
        "coverage": {
            "attempted": len(case_records),
            "succeeded": sum(case["status"] == "success" for case in case_records),
            "failed": sum(case["status"] != "success" for case in case_records),
            "failure_types": dict(sorted(failures.items())),
            "failure_policy": "failed and missing pages retain zero proposals",
        },
        "scorer": {
            "current_region_kinds": sorted(CURRENT_REGION_KINDS),
            "includes_table_source_regions": True,
            "iou_threshold": IOU_THRESHOLD,
            "target_coverage_threshold": COVERAGE_THRESHOLD,
            "target_coverage_definition": "intersection divided by target area",
            "novel_proposal_definition": (
                "less than 50 percent of proposal area is covered by the union "
                "of current text and word regions"
            ),
            "matching": (
                "score-descending one-to-one proposals to the best unmatched "
                "baseline-missed target"
            ),
            "precision_denominator": (
                "all novel proposals on C14 targets and no-handwriting negatives"
            ),
        },
        "default_threshold": default_score,
        "confidence_sweep": sweeps,
        "case_summary": [
            _public_case_summary(case, BOX_THRESHOLD) for case in case_records
        ],
    }
    if overlays is not None:
        _write_overlays(case_records, overlays, BOX_THRESHOLD)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return payload


def select_panel(
    challenge_root: Path, *, negative_pages: int = DEFAULT_NEGATIVE_PAGES
) -> list[dict[str, str]]:
    annotations_root = challenge_root / "annotations" / "primary"
    sources_root = challenge_root / "sources"
    specialist_v8 = challenge_root / "runs" / "specialist-v8" / "model-output"
    specialist_v10 = challenge_root / "runs" / "specialist-v10" / "model-output"
    for required in (annotations_root, sources_root, specialist_v8, specialist_v10):
        if not required.is_dir():
            raise ValueError(f"Required challenge directory is missing: {required}")

    targets = [
        _case_paths(
            case_id,
            "target",
            sources_root,
            annotations_root,
            specialist_v10,
        )
        for case_id in TARGET_CASE_IDS
    ]
    by_category: dict[str, list[dict[str, str]]] = {}
    for category_dir in sorted(annotations_root.iterdir()):
        if (
            not category_dir.is_dir()
            or category_dir.name in NEGATIVE_EXCLUDED_CATEGORIES
        ):
            continue
        cases = []
        for annotation_path in sorted(category_dir.glob("*.json")):
            annotation = _load_json(annotation_path)
            if annotation.get("handwriting") != []:
                continue
            case_id = annotation.get("case_id")
            if not isinstance(case_id, str):
                raise ValueError(f"Annotation lacks case_id: {annotation_path}")
            try:
                cases.append(
                    _case_paths(
                        case_id,
                        "negative",
                        sources_root,
                        annotations_root,
                        specialist_v8,
                    )
                )
            except ValueError:
                continue
        if cases:
            by_category[category_dir.name] = cases
    if not by_category:
        raise ValueError("No eligible no-handwriting negative pages were found")

    categories = sorted(by_category)
    base, remainder = divmod(negative_pages, len(categories))
    negatives = []
    for index, category in enumerate(categories):
        count = base + (1 if index < remainder else 0)
        negatives.extend(_evenly_spaced(by_category[category], count))
    if len(negatives) != negative_pages:
        raise ValueError(
            f"Requested {negative_pages} negative pages but selected {len(negatives)}"
        )
    return targets + negatives


def _case_paths(
    case_id: str,
    role: str,
    sources_root: Path,
    annotations_root: Path,
    outputs_root: Path,
) -> dict[str, str]:
    image_path = _unique_path(sources_root, f"{case_id}.*", IMAGE_SUFFIXES)
    annotation_path = annotations_root / case_id.split("-", 1)[0] / f"{case_id}.json"
    if not annotation_path.is_file():
        raise ValueError(f"Missing annotation for {case_id}")
    output_path = _unique_path(outputs_root, f"{case_id}.json", {".json"})
    return {
        "case_id": case_id,
        "category_id": case_id.split("-", 1)[0],
        "role": role,
        "image_path": str(image_path.resolve()),
        "annotation_path": str(annotation_path.resolve()),
        "current_output_path": str(output_path.resolve()),
    }


def _score_variant(
    cases: list[dict[str, object]], confidence_threshold: float
) -> dict[str, object]:
    negatives = [case for case in cases if case["role"] == "negative"]
    raw_count = 0
    candidate_count = 0
    false_per_negative = []
    prepared: list[dict[str, object]] = []
    for case in cases:
        detections = [
            detection
            for detection in case["detections"]
            if float(detection["score"]) >= confidence_threshold
        ]
        candidates = [
            detection
            for detection in detections
            if _covered_fraction(detection["bbox"], case["current_boxes"])
            < NOVELTY_COVERAGE_THRESHOLD
        ]
        raw_count += len(detections)
        candidate_count += len(candidates)
        if case["role"] == "negative":
            false_per_negative.append(len(candidates))
        prepared.append({**case, "filtered_detections": candidates})

    metrics = {}
    for name, metric, threshold in (
        ("iou_0_50", _iou, IOU_THRESHOLD),
        ("coverage_0_50", _target_coverage, COVERAGE_THRESHOLD),
    ):
        baseline_total = baseline_matched = union_matched = 0
        proposal_counts = Counter()
        recovered_targets = []
        for case in prepared:
            if case["role"] != "target":
                proposal_counts["false_positives"] += len(case["filtered_detections"])
                continue
            targets = case["target_boxes"]
            current = case["current_boxes"]
            missed = [
                (index, box)
                for index, box in enumerate(targets)
                if max((metric(box, current_box) for current_box in current), default=0)
                < threshold
            ]
            matched = _match(
                [box for _, box in missed],
                case["filtered_detections"],
                metric,
                threshold,
            )
            baseline_total += len(targets)
            baseline_matched += len(targets) - len(missed)
            union_matched += len(targets) - len(missed) + matched["true_positives"]
            proposal_counts.update(
                {
                    "true_positives": matched["true_positives"],
                    "false_positives": matched["false_positives"],
                    "false_negatives": matched["false_negatives"],
                }
            )
            for local_index in matched["matched_reference_indices"]:
                recovered_targets.append(
                    {
                        "case_id": case["case_id"],
                        "target_index": missed[local_index][0],
                    }
                )
        baseline_recall = baseline_matched / baseline_total if baseline_total else 0.0
        union_recall = union_matched / baseline_total if baseline_total else 0.0
        metrics[name] = {
            "baseline": {
                "matched": baseline_matched,
                "targets": baseline_total,
                "recall": round(baseline_recall, 6),
            },
            "union": {
                "matched": union_matched,
                "targets": baseline_total,
                "recall": round(union_recall, 6),
                "absolute_recall_gain_points": round(
                    (union_recall - baseline_recall) * 100, 3
                ),
            },
            "novel_proposals": {
                **_metrics(proposal_counts),
                "recovered_targets": recovered_targets,
            },
        }

    return {
        "confidence_threshold": confidence_threshold,
        "raw_detections": raw_count,
        "novel_proposals": candidate_count,
        "redundant_detections": raw_count - candidate_count,
        "metrics": metrics,
        "negative_false_proposals_per_page": {
            "pages": len(negatives),
            "total": sum(false_per_negative),
            "mean": round(sum(false_per_negative) / len(negatives), 3)
            if negatives
            else None,
            "p50": _rounded_percentile(false_per_negative, 0.5),
            "p95": _rounded_percentile(false_per_negative, 0.95),
            "max": max(false_per_negative, default=None),
        },
    }


def _match(
    references: list[Box],
    detections: list[dict[str, object]],
    metric: Metric,
    threshold: float,
) -> dict[str, object]:
    remaining = set(range(len(references)))
    matched_indices = []
    true_positives = 0
    for detection in sorted(
        detections, key=lambda item: float(item["score"]), reverse=True
    ):
        best = max(
            remaining,
            key=lambda index: metric(references[index], detection["bbox"]),
            default=None,
        )
        if best is None or metric(references[best], detection["bbox"]) < threshold:
            continue
        remaining.remove(best)
        matched_indices.append(best)
        true_positives += 1
    return {
        "true_positives": true_positives,
        "false_positives": len(detections) - true_positives,
        "false_negatives": len(references) - true_positives,
        "matched_reference_indices": matched_indices,
    }


def _metrics(counts: Counter[str]) -> dict[str, float | int]:
    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _normalize_doctr_prediction(
    value: object, width: int, height: int
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("docTR returned an invalid page prediction")
    raw_boxes = value[0].get("words")
    if not isinstance(raw_boxes, np.ndarray) or raw_boxes.ndim != 2:
        raise ValueError("docTR prediction lacks a words array")
    detections = []
    for row in raw_boxes.tolist():
        if len(row) != 5:
            raise ValueError("docTR straight-page boxes must have five values")
        left, top, right, bottom, score = (float(item) for item in row)
        box = _box(
            [left * width, top * height, right * width, bottom * height],
            width=width,
            height=height,
        )
        if not 0 <= score <= 1:
            raise ValueError("docTR detection score is outside [0, 1]")
        detections.append({"bbox": box, "score": round(score, 6)})
    return detections


def _load_predictor(architecture: str, device: str):
    os.environ.setdefault("USE_TORCH", "1")
    try:
        import doctr
        import torch
        from doctr.models import detection_predictor
    except ImportError as error:
        raise RuntimeError(
            "python-doctr and a compatible PyTorch build are required"
        ) from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {device}")
    predictor = detection_predictor(
        architecture,
        pretrained=True,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        batch_size=1,
    ).to(device)
    predictor.model.postprocessor.bin_thresh = BIN_THRESHOLD
    predictor.model.postprocessor.box_thresh = BOX_THRESHOLD
    return predictor, torch, doctr.__version__


def _prediction_records(predictions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = predictions.get("cases")
    if not isinstance(records, list):
        raise ValueError("Prediction file lacks cases")
    result = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str):
            raise ValueError("Prediction case is invalid")
        case_id = record["case_id"]
        if case_id in result:
            raise ValueError(f"Duplicate prediction for {case_id}")
        result[case_id] = record
    return result


def _detections(prediction: dict[str, Any]) -> list[dict[str, object]]:
    if prediction.get("status") != "success":
        return []
    raw = prediction.get("detections")
    if not isinstance(raw, list):
        raise ValueError("Prediction detections must be a list")
    result = []
    for detection in raw:
        if not isinstance(detection, dict):
            raise ValueError("Detection must be an object")
        score = detection.get("score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("Detection score must be from 0 to 1")
        result.append({"bbox": _box(detection.get("bbox")), "score": float(score)})
    return result


def _current_boxes(output: dict[str, Any]) -> list[Box]:
    result = output.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("pages"), list):
        raise ValueError("Current OCR output lacks result pages")
    boxes = []
    for page in result["pages"]:
        if not isinstance(page, dict) or not isinstance(page.get("regions"), list):
            continue
        for region in page["regions"]:
            if (
                not isinstance(region, dict)
                or region.get("kind") not in CURRENT_REGION_KINDS
            ):
                continue
            bounding_box = region.get("bounding_box")
            if not isinstance(bounding_box, dict):
                continue
            boxes.append(
                _box(
                    [
                        bounding_box.get("left"),
                        bounding_box.get("top"),
                        bounding_box.get("right"),
                        bounding_box.get("bottom"),
                    ]
                )
            )
    return boxes


def _handwriting_boxes(annotation: dict[str, Any]) -> list[Box]:
    handwriting = annotation.get("handwriting")
    if not isinstance(handwriting, list):
        raise ValueError("Annotation handwriting must be a list")
    boxes = []
    for item in handwriting:
        if not isinstance(item, dict) or item.get("legibility") != "legible":
            continue
        boxes.append(_box(item.get("bbox")))
    return boxes


def _box(value: object, *, width: int | None = None, height: int | None = None) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Bounding box must contain four coordinates")
    if not all(
        isinstance(item, (int, float)) and math.isfinite(item) for item in value
    ):
        raise ValueError("Bounding box coordinates must be finite numbers")
    left, top, right, bottom = (float(item) for item in value)
    if width is not None:
        left, right = max(0.0, left), min(float(width), right)
    if height is not None:
        top, bottom = max(0.0, top), min(float(height), bottom)
    if left >= right or top >= bottom:
        raise ValueError("Bounding box must have positive area")
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def _intersection(left: Box, right: Box) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(target: Box, proposal: Box) -> float:
    intersection = _intersection(target, proposal)
    union = _area(target) + _area(proposal) - intersection
    return intersection / union if union else 0.0


def _target_coverage(target: Box, proposal: Box) -> float:
    return _intersection(target, proposal) / _area(target)


def _covered_fraction(proposal: Box, current_boxes: list[Box]) -> float:
    clipped = []
    for box in current_boxes:
        left = max(proposal[0], box[0])
        top = max(proposal[1], box[1])
        right = min(proposal[2], box[2])
        bottom = min(proposal[3], box[3])
        if left < right and top < bottom:
            clipped.append([left, top, right, bottom])
    return _rectangle_union_area(clipped) / _area(proposal)


def _rectangle_union_area(boxes: list[Box]) -> float:
    if not boxes:
        return 0.0
    x_values = sorted({coordinate for box in boxes for coordinate in (box[0], box[2])})
    area = 0.0
    for left, right in zip(x_values, x_values[1:]):
        if left == right:
            continue
        intervals = sorted(
            (box[1], box[3]) for box in boxes if box[0] < right and box[2] > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        area += (right - left) * covered
    return area


def _write_overlays(
    cases: list[dict[str, object]], overlays: Path, confidence_threshold: float
) -> None:
    if overlays.exists():
        raise FileExistsError(f"Overlay directory already exists: {overlays}")
    overlays.mkdir(parents=True)
    for case in cases:
        with Image.open(Path(case["image_path"])) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in case["current_boxes"]:
            draw.rectangle(box, outline=(80, 150, 210), width=1)
        for box in case["target_boxes"]:
            covered = max(
                (_target_coverage(box, current) for current in case["current_boxes"]),
                default=0,
            )
            color = (235, 170, 30) if covered >= COVERAGE_THRESHOLD else (220, 40, 40)
            draw.rectangle(box, outline=color, width=3)
        candidates = [
            detection
            for detection in case["detections"]
            if float(detection["score"]) >= confidence_threshold
            and _covered_fraction(detection["bbox"], case["current_boxes"])
            < NOVELTY_COVERAGE_THRESHOLD
        ]
        for detection in candidates:
            draw.rectangle(detection["bbox"], outline=(20, 185, 70), width=3)
        label = (
            f"{case['case_id']} | {case['role']} | novel proposals: {len(candidates)}"
        )
        text_box = draw.textbbox((0, 0), label)
        draw.rectangle((0, 0, text_box[2] + 8, text_box[3] + 8), fill=(255, 255, 255))
        draw.text((4, 4), label, fill=(0, 0, 0))
        image.save(overlays / f"{case['case_id']}.png")


def _public_case_summary(
    case: dict[str, object], confidence_threshold: float
) -> dict[str, object]:
    detections = [
        detection
        for detection in case["detections"]
        if float(detection["score"]) >= confidence_threshold
    ]
    candidates = [
        detection
        for detection in detections
        if _covered_fraction(detection["bbox"], case["current_boxes"])
        < NOVELTY_COVERAGE_THRESHOLD
    ]
    return {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "role": case["role"],
        "status": case["status"],
        "target_boxes": len(case["target_boxes"]),
        "current_regions": len(case["current_boxes"]),
        "raw_detections": len(detections),
        "novel_proposals": len(candidates),
    }


def _load_image(path: Path) -> tuple[np.ndarray, int, int, float]:
    started = time.perf_counter()
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        array = np.asarray(image).copy()
    return array, width, height, (time.perf_counter() - started) * 1000


def _cuda_memory(torch_module: Any, device: str) -> dict[str, float | None]:
    if not device.startswith("cuda"):
        return {
            "peak_allocated": None,
            "peak_reserved": None,
            "measurement": "unavailable without CUDA",
        }
    return {
        "peak_allocated": round(
            torch_module.cuda.max_memory_allocated(device) / 2**20, 3
        ),
        "peak_reserved": round(
            torch_module.cuda.max_memory_reserved(device) / 2**20, 3
        ),
        "measurement": "PyTorch process peak after one warmup page",
    }


def _synchronize(torch_module: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch_module.cuda.synchronize(device)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "observed": len(values),
        "p50": _rounded_percentile(values, 0.5),
        "p95": _rounded_percentile(values, 0.95),
        "max": round(max(values), 3) if values else None,
    }


def _rounded_percentile(values: Iterable[int | float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    value = ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    return round(value, 3)


def _evenly_spaced(values: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    if len(values) < count:
        raise ValueError(f"Cannot select {count} cases from {len(values)} candidates")
    if count == 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def _unique_path(root: Path, pattern: str, suffixes: set[str]) -> Path:
    matches = [
        path
        for path in root.rglob(pattern)
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _confidence_key(value: float) -> str:
    return f"score_{value:.2f}".replace(".", "_")


def _inference_console_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": payload["benchmark"],
        "model": payload["model"],
        "coverage": payload["coverage"],
        "operations": payload["operations"],
    }


def _evaluation_console_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": payload["benchmark"],
        "panel": payload["panel"],
        "coverage": payload["coverage"],
        "default_threshold": payload["default_threshold"],
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
