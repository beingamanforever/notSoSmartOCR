"""Failure-inclusive PubTables-1M table detection benchmark."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from experiments import pubtables_benchmark as pubtables

MODEL_ID = "microsoft/table-transformer-detection"
MODEL_REVISION = "2357cbe2b5a5d1c03e54f32764f06058933b65ab"
MODEL_LICENSE = "MIT"
MODEL_FILE = "pubtables1m_detection_detr_r18.pth"
SOURCE_REVISION = pubtables.SCORER_REVISION
SCORER_REVISION = "v1"
MIN_CASES = 30
DEFAULT_CASES = 60
IOU_THRESHOLDS = (0.5, 0.75)
CONFIDENCE_THRESHOLD = 0.5

Prediction = dict[str, Any]
Predictor = Callable[[pubtables.Case], Prediction]


class TatrDetector:
    """Run Microsoft's official full-page Table Transformer detector."""

    def __init__(self, source_root: Path, checkpoint: Path, device: str) -> None:
        if not checkpoint.is_file():
            raise ValueError(f"TATR detection checkpoint is missing: {checkpoint}")
        source = source_root / "src" if (source_root / "src").is_dir() else source_root
        config = source / "detection_config.json"
        if not config.is_file():
            raise ValueError(f"TATR detection config is missing: {config}")
        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")

        started = time.perf_counter()
        inference = pubtables._load_tatr_inference(source_root)
        self.pipeline = inference.TableExtractionPipeline(
            det_device=device,
            det_config_path=config,
            det_model_path=checkpoint,
        )
        self.device = device
        self.load_ms = round((time.perf_counter() - started) * 1000, 3)

    def predict(self, case: pubtables.Case) -> Prediction:
        if case.image_path is None or not case.image_path.is_file():
            return _failed("missing_page_image")

        started = time.perf_counter()
        try:
            _sync(self.device)
            with Image.open(case.image_path) as source:
                image = source.convert("RGB")
            result = self.pipeline.detect(image, out_objects=True)
            _sync(self.device)
            if not isinstance(result, dict) or not isinstance(
                result.get("objects"), list
            ):
                raise ValueError("TATR detection output lacks objects")
            detections = []
            for item in result["objects"]:
                if not isinstance(item, dict):
                    raise ValueError("TATR detection object is invalid")
                if item.get("label") not in {
                    "table",
                    "table rotated",
                }:
                    continue
                score = _score(item.get("score"))
                if score < CONFIDENCE_THRESHOLD:
                    continue
                detections.append({"bbox": _box(item.get("bbox")), "score": score})
            return _prediction(
                "success",
                detections,
                (time.perf_counter() - started) * 1000,
            )
        except Exception as error:
            _sync(self.device)
            return _failed(
                f"tatr_detection_error: {type(error).__name__}: {error}",
                (time.perf_counter() - started) * 1000,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score failure-inclusive PubTables-1M table detection"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--tatr-checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=pubtables._positive_int, default=DEFAULT_CASES)
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        detector = None
        if args.tatr_checkpoint is not None:
            if args.source_root is None:
                raise ValueError("TATR checkpoint execution requires --source-root")
            detector = TatrDetector(args.source_root, args.tatr_checkpoint, args.device)
        elif args.source_root is not None:
            raise ValueError("--source-root requires --tatr-checkpoint")
        report = run_benchmark(
            args.dataset_root,
            predictions_root=args.predictions,
            predictor=None if detector is None else detector.predict,
            model_metadata=_model_metadata(
                args.tatr_checkpoint,
                args.checkpoint_revision,
                args.source_root,
                args.source_revision,
            )
            if detector is not None
            else None,
            model_load_ms=None if detector is None else detector.load_ms,
            limit=args.limit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset_root: Path,
    *,
    predictions_root: Path | None = None,
    predictor: Predictor | None = None,
    model_metadata: dict[str, object] | None = None,
    model_load_ms: float | None = None,
    limit: int = DEFAULT_CASES,
) -> dict[str, object]:
    if (predictions_root is None) == (predictor is None):
        raise ValueError("Provide exactly one prediction directory or predictor")
    cases = pubtables._load_cases(dataset_root, limit)
    if len(cases) < MIN_CASES:
        raise ValueError(f"PubTables detection requires at least {MIN_CASES} cases")

    records = []
    totals = {threshold: Counter() for threshold in IOU_THRESHOLDS}
    best_ious: list[float] = []
    for case in cases:
        gold = _gold_boxes(case)
        prediction = _case_prediction(case, predictions_root, predictor)
        detections = prediction["detections"]
        case_metrics = {}
        for threshold in IOU_THRESHOLDS:
            counts = _match(gold, detections, threshold)
            totals[threshold].update(counts)
            case_metrics[_threshold_key(threshold)] = _metrics(counts)
        case_best = [
            max((_iou(box, item["bbox"]) for item in detections), default=0.0)
            for box in gold
        ]
        best_ious.extend(case_best)
        records.append(
            {
                "case_id": case.case_id,
                "status": prediction["status"],
                "ground_truth_tables": len(gold),
                "predicted_tables": len(detections),
                "ground_truth_boxes": gold,
                "detections": detections,
                "mean_best_iou": round(sum(case_best) / len(case_best), 6),
                "latency_ms": prediction["latency_ms"],
                "failure": prediction["failure"],
                "metrics": case_metrics,
            }
        )

    status_counts = Counter(item["status"] for item in records)
    latencies = [
        float(item["latency_ms"]) for item in records if item["latency_ms"] is not None
    ]
    return {
        "benchmark": "PubTables-1M full-page table detection",
        "status": "complete",
        "dataset": {
            "id": pubtables.DATASET_ID,
            "revision": pubtables.DATASET_REVISION,
            "revision_evidence": (
                "public repository metadata verified; prepared cases do not embed "
                "their source revision"
            ),
            "license": pubtables.DATASET_LICENSE,
            "split": "test",
            "root": str(dataset_root),
            "attempted_cases": len(cases),
            "case_ids": [case.case_id for case in cases],
            "panel": (
                f"{limit} deterministic evenly spaced cases over the supplied "
                "test panel's lexical order"
            ),
        },
        "model": model_metadata
        or {
            "id": "precomputed predictions",
            "revision": None,
            "license": None,
            "provenance": "model identity was not supplied",
        },
        "scorer": {
            "id": "notSoSmartOCR PubTables detection protocol",
            "revision": SCORER_REVISION,
            "license": None,
            "license_status": "no separate license declared for this local protocol",
            "implementation": "experiments/pubtables_detection_benchmark.py",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "matching": (
                "score-descending one-to-one table matching to the highest-IoU "
                "unmatched truth box"
            ),
        },
        "metric_definitions": {
            "precision_recall_f1": (
                "micro table detection precision, recall, and F1 at IoU 0.50 and 0.75"
            ),
            "mean_best_iou": (
                "mean over every truth table's best predicted IoU; missing and "
                "failed predictions contribute zero"
            ),
        },
        "coverage": {
            "attempted": len(cases),
            "valid": status_counts["success"],
            "failed": status_counts["failed"],
            "abstained": status_counts["abstained"],
            "coverage_rate": round(status_counts["success"] / len(cases), 6),
            "failure_policy": (
                "failed, abstained, missing, and invalid cases retain all truth "
                "tables as false negatives"
            ),
        },
        "metrics": {
            **{
                _threshold_key(threshold): _metrics(totals[threshold])
                for threshold in IOU_THRESHOLDS
            },
            "mean_best_iou": round(sum(best_ious) / len(best_ious), 6),
        },
        "operations": {
            "latency_ms": {
                "observed": len(latencies),
                "missing": len(cases) - len(latencies),
                "p50": _rounded_percentile(latencies, 0.5),
                "p95": _rounded_percentile(latencies, 0.95),
            },
            "model_load_ms": model_load_ms,
        },
        "cases": records,
    }


def _case_prediction(
    case: pubtables.Case,
    root: Path | None,
    predictor: Predictor | None,
) -> Prediction:
    try:
        if predictor is not None:
            return _normalize_prediction(predictor(case))
        assert root is not None
        path = root / f"{case.case_id}.json"
        if not path.is_file():
            return _failed("missing_prediction")
        return _normalize_prediction(json.loads(path.read_text(encoding="utf-8")))
    except Exception as error:
        return _failed(f"invalid_prediction: {error}")


def _normalize_prediction(value: object) -> Prediction:
    if not isinstance(value, dict):
        raise ValueError("prediction must be an object")
    status = value.get("status", "success")
    if status not in {"success", "failed", "abstained"}:
        raise ValueError(f"invalid status: {status!r}")
    raw = value.get("detections", [])
    if not isinstance(raw, list):
        raise ValueError("detections must be a list")
    detections = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each detection must be an object")
        score = _score(item.get("score", 1.0))
        if score >= CONFIDENCE_THRESHOLD:
            detections.append({"bbox": _box(item.get("bbox")), "score": score})
    latency = pubtables._optional_number(value.get("latency_ms"))
    failure = value.get("failure")
    if failure is not None and not isinstance(failure, str):
        raise ValueError("failure must be text or null")
    if status != "success":
        detections = []
        failure = failure or str(status)
    return _prediction(str(status), detections, latency, failure)


def _gold_boxes(case: pubtables.Case) -> list[list[float]]:
    structure = pubtables._mapping(case.target.get("structure"), "structure")
    boxes, labels = pubtables._xml_objects(structure.get("xml"))
    result = [box for box, label in zip(boxes, labels, strict=True) if label == 0]
    if not result:
        raise ValueError(f"PubTables case has no table box: {case.case_id}")
    return result


def _match(
    gold: list[list[float]], detections: list[dict[str, Any]], threshold: float
) -> Counter[str]:
    remaining = set(range(len(gold)))
    true_positives = 0
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        best = max(
            remaining,
            key=lambda index: _iou(gold[index], detection["bbox"]),
            default=None,
        )
        if best is not None and _iou(gold[best], detection["bbox"]) >= threshold:
            remaining.remove(best)
            true_positives += 1
    return Counter(
        true_positives=true_positives,
        false_positives=len(detections) - true_positives,
        false_negatives=len(gold) - true_positives,
    )


def _metrics(counts: Counter[str]) -> dict[str, float | int]:
    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    predicted = true_positives + false_positives
    truth = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _iou(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _box(value: object) -> list[float]:
    box = pubtables._bbox(value)
    if box is None or box[0] == box[2] or box[1] == box[3]:
        raise ValueError("detection bbox must have positive finite area")
    return box


def _score(value: object) -> float:
    score = pubtables._optional_number(value)
    if score is None or score > 1:
        raise ValueError("detection score must be from 0 to 1")
    return score


def _prediction(
    status: str,
    detections: list[dict[str, Any]],
    latency_ms: float | None,
    failure: str | None = None,
) -> Prediction:
    return {
        "status": status,
        "detections": detections,
        "latency_ms": latency_ms,
        "failure": failure,
    }


def _failed(reason: str, latency_ms: float | None = None) -> Prediction:
    return _prediction("failed", [], latency_ms, reason)


def _model_metadata(
    checkpoint: Path | None,
    checkpoint_revision: str | None,
    source_root: Path | None,
    source_revision: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "revision": checkpoint_revision,
        "source_root": str(source_root),
        "source_revision": source_revision,
        "license": None,
        "provenance": "caller-supplied paths and revisions",
    }
    if checkpoint_revision == MODEL_REVISION and source_revision == SOURCE_REVISION:
        metadata.update(
            {
                "id": MODEL_ID,
                "license": MODEL_LICENSE,
                "architecture": "Table Transformer with ResNet-18 backbone",
                "origin": "Microsoft, using Facebook DETR and ResNet-18",
            }
        )
    return metadata


def _sync(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


def _threshold_key(value: float) -> str:
    return f"iou_{value:.2f}".replace(".", "_")


def _rounded_percentile(values: list[float], quantile: float) -> float | None:
    value = pubtables._percentile(values, quantile)
    return None if value is None else round(value, 3)


if __name__ == "__main__":
    raise SystemExit(main())
