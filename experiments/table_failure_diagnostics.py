"""Isolate table detection, structure, and cell-text failures."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol

from PIL import Image

from experiments import pubtables_benchmark as pubtables

MODES = (
    "ground_truth_crop_to_structure",
    "predicted_crop_to_structure",
    "ground_truth_structure_to_cell_text",
)
STATUSES = frozenset({"success", "failed", "abstained"})


@dataclass(frozen=True)
class TableCropInput:
    case: pubtables.Case
    document_group_id: str
    kind: str
    image: Image.Image
    source_bbox: tuple[int, int, int, int]
    tokens: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GroundTruthStructureInput:
    case: pubtables.Case
    document_group_id: str
    image: Image.Image
    cells: tuple[dict[str, Any], ...]


class TableScorer(Protocol):
    def gold_cells(self, target: dict[str, Any]) -> list[dict[str, Any]]: ...

    def score(
        self,
        gold: list[dict[str, Any]],
        predicted: Any,
    ) -> dict[str, float]: ...


Detector = Callable[[pubtables.Case, Image.Image], Mapping[str, Any]]
StructurePredictor = Callable[[TableCropInput], Mapping[str, Any]]
CellRecognizer = Callable[[GroundTruthStructureInput], Mapping[str, Any]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run failure-isolating table evaluation on PubTables cases"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("predictions_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--scorer-revision")
    parser.add_argument("--limit", type=pubtables._positive_int, default=60)
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        replay = PredictionReplay(args.predictions_root)
        report = run_benchmark(
            args.dataset_root,
            scorer=pubtables.OfficialScorer(args.scorer_root),
            detector=replay.detect,
            structure_predictor=replay.predict_structure,
            cell_recognizer=replay.recognize_cells,
            scorer_metadata=pubtables._scorer_metadata(
                args.scorer_root, args.scorer_revision
            ),
            limit=args.limit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset_root: Path,
    *,
    scorer: TableScorer,
    detector: Detector,
    structure_predictor: StructurePredictor,
    cell_recognizer: CellRecognizer,
    scorer_metadata: Mapping[str, Any] | None = None,
    limit: int = 60,
) -> dict[str, Any]:
    """Run the three controlled paths without exporting table text."""
    cases = pubtables._load_cases(dataset_root, limit)
    if not cases:
        raise ValueError("Table diagnostics require at least one case")

    records = [
        _evaluate_case(case, scorer, detector, structure_predictor, cell_recognizer)
        for case in cases
    ]
    group_ids = sorted({record["document_group_id"] for record in records})
    return {
        "benchmark": "table failure isolation",
        "status": "complete",
        "dataset": {
            "id": pubtables.DATASET_ID,
            "revision": pubtables.DATASET_REVISION,
            "license": pubtables.DATASET_LICENSE,
            "split": "test",
            "root": str(dataset_root),
            "attempted_tables": len(records),
            "document_group_count": len(group_ids),
            "document_group_ids": group_ids,
            "case_ids": [record["case_id"] for record in records],
        },
        "protocol": {
            "paths": {
                MODES[0]: "ground-truth table crop to predicted structure",
                MODES[1]: "predicted table crop to predicted structure",
                MODES[2]: (
                    "ground-truth cell geometry to deployed cell recognition and fusion"
                ),
            },
            "coordinate_space": "source-page pixels",
            "predicted_crop_selection": "highest-confidence detection",
            "failure_policy": (
                "failed and abstained cases remain in every all-case denominator"
            ),
            "privacy": "case metrics and identifiers only; cell text is not exported",
        },
        "scorer": dict(scorer_metadata or {"interface": "injected GriTS scorer"}),
        "modes": {mode: _mode_summary(records, mode) for mode in MODES},
        "cases": records,
    }


def _evaluate_case(
    case: pubtables.Case,
    scorer: TableScorer,
    detector: Detector,
    structure_predictor: StructurePredictor,
    cell_recognizer: CellRecognizer,
) -> dict[str, Any]:
    group_id = _document_group_id(case)
    result = {
        "case_id": case.case_id,
        "document_group_id": group_id,
        "detector": _failure("not_attempted"),
        **{mode: _failure("not_attempted") for mode in MODES},
    }
    try:
        image = _case_image(case)
        gold = scorer.gold_cells(case.target)
        ground_truth_crop = _ground_truth_crop(case, image, group_id)
    except (OSError, TypeError, ValueError) as error:
        failure = _failure(f"case_setup_failed: {type(error).__name__}: {error}")
        return {**result, **{mode: failure for mode in MODES}}

    result[MODES[0]] = _run_structure(
        structure_predictor,
        ground_truth_crop,
        scorer,
        gold,
    )

    detection = _run_detector(detector, case, image)
    result["detector"] = _public_result(detection)
    if detection["status"] == "success":
        try:
            predicted_crop = _predicted_crop(
                case,
                image,
                group_id,
                detection["bbox"],
            )
            result[MODES[1]] = _run_structure(
                structure_predictor,
                predicted_crop,
                scorer,
                gold,
                added_latency_ms=detection["latency_ms"],
            )
        except (TypeError, ValueError) as error:
            result[MODES[1]] = _failure(
                f"predicted_crop_failed: {type(error).__name__}: {error}",
                latency_ms=detection["latency_ms"],
            )
    else:
        result[MODES[1]] = _failure(
            f"detector_{detection['status']}: {detection['failure']}",
            status=detection["status"],
            latency_ms=detection["latency_ms"],
        )

    result[MODES[2]] = _run_cell_recognizer(
        cell_recognizer,
        GroundTruthStructureInput(
            case=case,
            document_group_id=group_id,
            image=image,
            cells=tuple(gold),
        ),
        scorer,
        gold,
    )
    return result


def _run_detector(
    detector: Detector,
    case: pubtables.Case,
    image: Image.Image,
) -> dict[str, Any]:
    try:
        value = detector(case, image)
        status, latency, failure = _result_header(value)
        if status != "success":
            return _failure(failure or status, status=status, latency_ms=latency)
        raw = value.get("detections")
        if raw is None and "bbox" in value:
            raw = [{"bbox": value["bbox"], "score": value.get("score", 1.0)}]
        if not isinstance(raw, list) or not raw:
            raise ValueError("detector returned no table crop")
        detections = [_detection(item) for item in raw]
        selected = max(detections, key=lambda item: item["score"])
        return {
            "status": "success",
            "bbox": selected["bbox"],
            "score": selected["score"],
            "latency_ms": latency,
            "failure": None,
        }
    except Exception as error:
        return _failure(f"detector_error: {type(error).__name__}: {error}")


def _run_structure(
    predictor: StructurePredictor,
    crop: TableCropInput,
    scorer: TableScorer,
    gold: list[dict[str, Any]],
    *,
    added_latency_ms: float | None = None,
) -> dict[str, Any]:
    try:
        value = predictor(crop)
        prediction = _scored_result(value, scorer, gold, crop.source_bbox[:2])
    except Exception as error:
        prediction = _failure(f"structure_error: {type(error).__name__}: {error}")
    prediction["latency_ms"] = _sum_latency(added_latency_ms, prediction["latency_ms"])
    return prediction


def _run_cell_recognizer(
    recognizer: CellRecognizer,
    value: GroundTruthStructureInput,
    scorer: TableScorer,
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return _scored_result(recognizer(value), scorer, gold, (0, 0))
    except Exception as error:
        return _failure(f"cell_text_error: {type(error).__name__}: {error}")


def _scored_result(
    value: Mapping[str, Any],
    scorer: TableScorer,
    gold: list[dict[str, Any]],
    offset: tuple[int, int],
) -> dict[str, Any]:
    status, latency, failure = _result_header(value)
    if status != "success":
        return _failure(failure or status, status=status, latency_ms=latency)
    cells = value.get("cells")
    if value.get("coordinate_space", "source") == "crop":
        cells = pubtables._translate_cells(cells, offset)
    elif value.get("coordinate_space", "source") != "source":
        raise ValueError("coordinate_space must be source or crop")
    metrics = scorer.score(gold, cells)
    return {
        "status": "success",
        "latency_ms": latency,
        "failure": None,
        "metrics": metrics,
    }


def _mode_summary(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    values = [record[mode] for record in records]
    counts = Counter(value["status"] for value in values)
    successful = [value for value in values if value["status"] == "success"]
    metric_names = pubtables.METRICS
    attempted = len(values)
    all_cases = {
        name: round(
            sum(value["metrics"][name] for value in successful) / attempted,
            6,
        )
        for name in metric_names
    }
    served_only = {
        name: round(
            sum(value["metrics"][name] for value in successful) / len(successful),
            6,
        )
        if successful
        else 0.0
        for name in metric_names
    }
    latencies = [
        float(value["latency_ms"])
        for value in values
        if value["latency_ms"] is not None
    ]
    return {
        "coverage": {
            "attempted": attempted,
            "successful": counts["success"],
            "failed": counts["failed"],
            "abstained": counts["abstained"],
            "coverage_rate": round(counts["success"] / attempted, 6),
        },
        "metrics": {"all_cases": all_cases, "served_only": served_only},
        "latency_ms": {
            "observed": len(latencies),
            "missing": attempted - len(latencies),
            "p50": _rounded_percentile(latencies, 0.5),
            "p95": _rounded_percentile(latencies, 0.95),
        },
    }


def _ground_truth_crop(
    case: pubtables.Case,
    image: Image.Image,
    group_id: str,
) -> TableCropInput:
    crop, offset = pubtables._tight_crop(image, case.target)
    return _crop_input(case, group_id, "ground_truth", crop, offset)


def _predicted_crop(
    case: pubtables.Case,
    image: Image.Image,
    group_id: str,
    bbox: list[float],
) -> TableCropInput:
    left = max(0, math.floor(bbox[0]) - pubtables.TATR_CROP_MARGIN)
    top = max(0, math.floor(bbox[1]) - pubtables.TATR_CROP_MARGIN)
    right = min(image.width, math.ceil(bbox[2]) + pubtables.TATR_CROP_MARGIN)
    bottom = min(image.height, math.ceil(bbox[3]) + pubtables.TATR_CROP_MARGIN)
    if right <= left or bottom <= top:
        raise ValueError("predicted table crop is empty")
    return _crop_input(
        case,
        group_id,
        "predicted",
        image.crop((left, top, right, bottom)),
        (left, top),
    )


def _crop_input(
    case: pubtables.Case,
    group_id: str,
    kind: str,
    crop: Image.Image,
    offset: tuple[int, int],
) -> TableCropInput:
    left, top = offset
    return TableCropInput(
        case=case,
        document_group_id=group_id,
        kind=kind,
        image=crop,
        source_bbox=(left, top, left + crop.width, top + crop.height),
        tokens=tuple(pubtables._crop_tokens(case.target, offset, crop.size)),
    )


def _case_image(case: pubtables.Case) -> Image.Image:
    if case.image_path is None or not case.image_path.is_file():
        raise ValueError("case image is missing")
    with Image.open(case.image_path) as source:
        return source.convert("RGB")


def _document_group_id(case: pubtables.Case) -> str:
    explicit = case.target.get("document_group_id", case.target.get("document_id"))
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    for pattern in (
        r"^(.+-D\d+)-P\d+(?:-.+)?$",
        r"^(.+?)(?:[_-]table[_-]?\d+)$",
    ):
        match = re.match(pattern, case.case_id, re.IGNORECASE)
        if match:
            return match.group(1)
    return case.case_id


def _result_header(
    value: Mapping[str, Any],
) -> tuple[str, float | None, str | None]:
    if not isinstance(value, Mapping):
        raise TypeError("component output must be an object")
    status = value.get("status", "success")
    if status not in STATUSES:
        raise ValueError(f"invalid component status: {status!r}")
    latency = pubtables._optional_number(value.get("latency_ms"))
    failure = value.get("failure")
    if failure is not None and not isinstance(failure, str):
        raise ValueError("failure must be text or null")
    return str(status), latency, failure


def _detection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each detection must be an object")
    bbox = pubtables._bbox(value.get("bbox"))
    score = pubtables._optional_number(value.get("score", 1.0))
    if (
        bbox is None
        or bbox[0] == bbox[2]
        or bbox[1] == bbox[3]
        or score is None
        or score > 1
    ):
        raise ValueError("detection requires a positive box and score from 0 to 1")
    return {"bbox": bbox, "score": score}


def _failure(
    reason: str,
    *,
    status: str = "failed",
    latency_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "latency_ms": latency_ms,
        "failure": reason,
        "metrics": None,
    }


def _public_result(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": value["status"],
        "latency_ms": value["latency_ms"],
        "failure": value["failure"],
    }


def _sum_latency(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left + right


def _rounded_percentile(values: list[float], quantile: float) -> float | None:
    value = pubtables._percentile(values, quantile)
    return None if value is None else round(value, 3)


class PredictionReplay:
    """Replay per-case component outputs while the harness owns crop selection."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise ValueError(f"Prediction directory is missing: {root}")
        self.root = root
        self._cache: dict[str, dict[str, Any]] = {}

    def detect(self, case: pubtables.Case, image: Image.Image) -> Mapping[str, Any]:
        del image
        return self._component(case.case_id, "detector")

    def predict_structure(self, value: TableCropInput) -> Mapping[str, Any]:
        key = (
            "ground_truth_crop_structure"
            if value.kind == "ground_truth"
            else "predicted_crop_structure"
        )
        return self._component(value.case.case_id, key)

    def recognize_cells(self, value: GroundTruthStructureInput) -> Mapping[str, Any]:
        return self._component(
            value.case.case_id,
            "ground_truth_structure_recognition",
        )

    def _component(self, case_id: str, key: str) -> Mapping[str, Any]:
        record = self._record(case_id)
        value = record.get(key)
        if not isinstance(value, Mapping):
            return {"status": "failed", "failure": f"missing_{key}"}
        return value

    def _record(self, case_id: str) -> dict[str, Any]:
        if case_id not in self._cache:
            path = self.root / f"{case_id}.json"
            if not path.is_file():
                self._cache[case_id] = {}
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"Prediction record is not an object: {path}")
                self._cache[case_id] = value
        return self._cache[case_id]


if __name__ == "__main__":
    raise SystemExit(main())
