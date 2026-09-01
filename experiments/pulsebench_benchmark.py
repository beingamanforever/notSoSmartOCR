"""Score precomputed PulseBench-Select predictions with the pinned scorer."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

DATASET_ID = "pulse-ai/PulseBench-Select"
DATASET_REVISION = "fc657a3ebe7215fe7cfecd2cfb8d14a77fffeea8"
SCORER_REVISION = "9e068e0d31f5f30cd99863bb5560e2c8713f431d"
EXPECTED_CASES = 485
DEV_CASES = 60
EVAL_CASES = 425
EXPECTED_POSITIVE_CASES = 459
EXPECTED_NEGATIVE_CASES = 26
LICENSE = "CC BY-NC-ND 4.0"
OFFICIAL_PROVENANCE = (
    "Pinned official PulseBench-Select metrics.compute_metrics with a project "
    "input adapter; no official scorer code is copied or modified"
)
PREDICTION_KEYS = {"sample_id", "status", "latency_ms", "items", "failures"}
ITEM_KEYS = {"page", "bbox", "content", "selected"}
STATUS = {"success", "failed", "abstained"}


@dataclass(frozen=True)
class Control:
    page: int
    bbox: tuple[float, ...]
    content: str
    selected: bool


OfficialScore = Callable[
    [dict[str, list[Control]], dict[str, list[Control]], float], dict[str, object]
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score precomputed predictions on all PulseBench-Select pages"
    )
    parser.add_argument("dataset_export", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--role", choices=("dev", "eval", "full"), required=True)
    parser.add_argument(
        "--scorer-root",
        type=Path,
        required=True,
        help="Pinned checkout of the official PulseBench-Select scorer",
    )
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        official_score = load_official_scorer(args.scorer_root)
        payload = run_benchmark(
            args.dataset_export,
            args.predictions,
            official_score,
            role=args.role,
        )
        _write_new_json(args.output, payload)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    dataset_export: Path,
    predictions_path: Path,
    official_score: OfficialScore,
    *,
    role: str,
) -> dict[str, object]:
    dataset_meta, dataset_rows = _load_records(dataset_export, "cases")
    prediction_meta, prediction_rows = _load_records(predictions_path, "cases")
    _validate_metadata(dataset_meta, prediction_meta, role)
    panel_rows, split_ids = _select_panel(dataset_rows, role)
    references, direct_references, direct_reason, reference_mismatches = (
        _load_references(panel_rows, len(panel_rows))
    )
    prediction_ids = [_sample_id(row) for row in prediction_rows]
    unknown = set(prediction_ids) - set(references)
    if unknown:
        raise ValueError(
            f"Predictions contain IDs outside the {role} panel: {sorted(unknown)[:3]}"
        )
    supplied = _load_predictions(prediction_rows)

    case_records = []
    official_predictions: dict[str, list[Control]] = {}
    for sample_id in split_ids:
        record = supplied.get(sample_id) or _missing_prediction(sample_id)
        official_predictions[sample_id] = record["items"]
        case_records.append(
            {
                "sample_id": sample_id,
                "status": record["status"],
                "latency_ms": record["latency_ms"],
                "predicted_items": len(record["items"]),
                "reference_selected_items": len(references[sample_id]),
                "reference_controls": (
                    len(direct_references[sample_id])
                    if direct_references is not None
                    else None
                ),
                "failures": record["failures"],
            }
        )

    latencies = [
        float(record["latency_ms"])
        for record in case_records
        if record["latency_ms"] is not None
    ]
    total_latency_seconds = sum(latencies) / 1000
    official = official_score(
        official_predictions,
        references,
        total_latency_seconds,
    )
    direct_metrics = _direct_metrics(
        official_predictions,
        direct_references,
        direct_reason,
        reference_mismatches,
    )
    failure_codes = Counter(
        failure["code"] for record in case_records for failure in record["failures"]
    )
    status_counts = Counter(str(record["status"]) for record in case_records)
    panel_cases = len(split_ids)
    return {
        "benchmark": "PulseBench-Select",
        "status": "complete",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "train",
            "license": LICENSE,
            "official_panel_cases": EXPECTED_CASES,
            "role": role,
            "role_cases": panel_cases,
            "split_protocol": (
                "full is the official 485-page panel; dev is a deterministic "
                "60-page custom panel with 56 positive and 4 negative pages; eval "
                "is the remaining frozen 425-page primary panel"
            ),
            "evaluation_policy": {
                "dev": "tuning and policy selection only",
                "eval": "frozen primary evaluation after dev-only tuning",
                "full": "official public panel; not used for tuning claims",
                "active_role": role,
            },
        },
        "scorer": {
            "revision": SCORER_REVISION,
            "entrypoint": "metrics.compute_metrics",
            "provenance": OFFICIAL_PROVENANCE,
        },
        "prediction_protocol": "precomputed_only_no_inference_in_evaluator",
        "case_ids": split_ids,
        "summary": {
            "cases": panel_cases,
            "status_counts": dict(sorted(status_counts.items())),
            "failed_cases": status_counts["failed"],
            "abstained_cases": status_counts["abstained"],
            "failure_codes": dict(sorted(failure_codes.items())),
            "latency_ms": {
                "observed_cases": len(latencies),
                "missing_cases": panel_cases - len(latencies),
                "p50": round(_percentile(latencies, 0.50), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
            },
        },
        "official_selection_f1": {
            "metric": "positive_class_selection_f1",
            "provenance": OFFICIAL_PROVENANCE,
            "values": official,
            "note": ("This is not state macro-F1 or control-to-label association F1"),
        },
        "project_defined_controls": direct_metrics,
        "cases": case_records,
    }


def load_official_scorer(root: Path) -> OfficialScore:
    if _git_revision(root) != SCORER_REVISION:
        raise ValueError(
            f"Official scorer must be checked out at revision {SCORER_REVISION}"
        )
    metrics_path = root / "metrics.py"
    result_path = root / "result.py"
    if not metrics_path.is_file() or not result_path.is_file():
        raise ValueError("Pinned scorer checkout is missing metrics.py or result.py")

    result_module = _load_module("pulsebench_pinned_result", result_path)
    previous_result = sys.modules.get("result")
    sys.modules["result"] = result_module
    try:
        metrics_module = _load_module("pulsebench_pinned_metrics", metrics_path)
    finally:
        if previous_result is None:
            sys.modules.pop("result", None)
        else:
            sys.modules["result"] = previous_result
    compute_metrics = getattr(metrics_module, "compute_metrics", None)
    result_type = getattr(result_module, "ExtractionResult", None)
    if not callable(compute_metrics) or result_type is None:
        raise ValueError("Pinned scorer does not expose its documented metric API")

    def handle_score(
        predictions: dict[str, list[Control]],
        references: dict[str, list[Control]],
        latency_seconds: float,
    ) -> dict[str, object]:
        adapted_predictions = _adapt_controls(predictions, result_type)
        adapted_references = _adapt_controls(references, result_type)
        result = compute_metrics(
            predictions=adapted_predictions,
            ground_truth=adapted_references,
            provider_name="not-so-smart-ocr",
            latency_seconds=latency_seconds,
        )
        if not dataclasses.is_dataclass(result):
            raise ValueError("Official scorer returned an unsupported result")
        return dataclasses.asdict(result)

    return handle_score


def _load_references(
    rows: list[dict[str, object]], expected_cases: int
) -> tuple[
    dict[str, list[Control]],
    dict[str, list[Control]] | None,
    str | None,
    list[str],
]:
    if len(rows) != expected_cases:
        raise ValueError(f"Dataset panel must contain all {expected_cases} cases")
    selected: dict[str, list[Control]] = {}
    candidates: dict[str, list[Control]] = {}
    direct_available = True
    direct_reason = ""
    state_count_mismatches = []
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id in selected:
            raise ValueError(f"Duplicate dataset case ID: {sample_id}")
        ground_truth = row.get("ground_truth")
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid ground truth for {sample_id}: {error}"
                ) from error
        if not isinstance(ground_truth, dict):
            raise ValueError(f"Missing ground truth for {sample_id}")
        if ground_truth.get("page_count") != 1:
            raise ValueError(
                f"PulseBench case must contain exactly one page: {sample_id}"
            )
        selected_items = ground_truth.get("selected_items")
        if not isinstance(selected_items, list):
            raise ValueError(f"Missing selected_items for {sample_id}")
        selected[sample_id] = [
            _control(item, selected=True, context=f"{sample_id}/selected_items")
            for item in selected_items
        ]
        if row.get("selected_count") is not None and int(row["selected_count"]) != len(
            selected[sample_id]
        ):
            raise ValueError(f"Selected count mismatch for {sample_id}")

        annotations = ground_truth.get("annotations")
        if not isinstance(annotations, list):
            if direct_available:
                direct_reason = "Public export does not contain per-page annotations"
            direct_available = False
            candidates[sample_id] = []
            continue
        page_candidates = []
        for index, annotation in enumerate(_selection_candidates(annotations)):
            if not isinstance(annotation.get("selected"), bool):
                if direct_available:
                    direct_reason = (
                        "Selection candidates do not expose an explicit state"
                    )
                direct_available = False
                continue
            try:
                page_candidates.append(
                    _control(
                        annotation,
                        selected=bool(annotation["selected"]),
                        context=f"{sample_id}/annotations/{index}",
                        require_label_bbox=True,
                        allow_reference_text=True,
                    )
                )
            except ValueError:
                if direct_available:
                    direct_reason = (
                        "Selection candidates do not directly pair label content and "
                        "geometry"
                    )
                direct_available = False
        candidates[sample_id] = page_candidates
        candidate_count = row.get("selection_candidate_count")
        if (
            direct_available
            and candidate_count is not None
            and (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count != len(page_candidates)
            )
        ):
            direct_available = False
            direct_reason = "Selection candidate count does not match annotations"
        if sum(item.selected for item in page_candidates) != len(selected[sample_id]):
            state_count_mismatches.append(sample_id)
    if not direct_available:
        return selected, None, direct_reason, state_count_mismatches
    return selected, candidates, None, state_count_mismatches


def _selection_candidates(value: object) -> list[dict[str, object]]:
    candidates = []
    if isinstance(value, list):
        for item in value:
            candidates.extend(_selection_candidates(item))
        return candidates
    if not isinstance(value, dict):
        return candidates
    if value.get("selection_candidate") is True:
        candidates.append(value)
    for key, item in value.items():
        if key != "selected_items":
            candidates.extend(_selection_candidates(item))
    return candidates


def _load_predictions(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    predictions = {}
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id in predictions:
            raise ValueError(f"Duplicate prediction case ID: {sample_id}")
        unknown = set(row) - PREDICTION_KEYS
        if unknown:
            raise ValueError(
                f"Prediction {sample_id} has unsupported fields: {sorted(unknown)}"
            )
        status = row.get("status")
        if status not in STATUS:
            raise ValueError(f"Prediction {sample_id} has invalid status")
        latency = row.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0
        ):
            raise ValueError(f"Prediction {sample_id} has invalid latency_ms")
        items = row.get("items")
        failures = row.get("failures")
        if not isinstance(items, list) or not isinstance(failures, list):
            raise ValueError(f"Prediction {sample_id} has invalid items or failures")
        if status in {"failed", "abstained"} and not failures:
            raise ValueError(f"Prediction {sample_id} must explain its {status} status")
        if status in {"failed", "abstained"} and items:
            raise ValueError(
                f"{status.capitalize()} prediction {sample_id} must not contain items"
            )
        parsed_failures = [_failure(value, sample_id) for value in failures]
        parsed_items = [
            _control(value, context=f"{sample_id}/items/{index}")
            for index, value in enumerate(items)
        ]
        predictions[sample_id] = {
            "status": status,
            "latency_ms": round(float(latency), 3),
            "items": parsed_items,
            "failures": parsed_failures,
        }
    return predictions


def _direct_metrics(
    predictions: dict[str, list[Control]],
    references: dict[str, list[Control]] | None,
    unavailable_reason: str | None,
    reference_mismatches: list[str],
) -> dict[str, object]:
    if references is None:
        return {
            "status": "not_computed",
            "reason": unavailable_reason,
            "provenance": "project_defined_not_official_selection_f1",
        }
    matched = []
    unmatched_predictions = []
    unmatched_references = []
    for sample_id in references:
        case_matches, remaining_predictions, remaining_references = _match_controls(
            predictions[sample_id], references[sample_id]
        )
        matched.extend(case_matches)
        unmatched_predictions.extend(remaining_predictions)
        unmatched_references.extend(remaining_references)

    association_tp = len(matched)
    association_fp = len(unmatched_predictions)
    association_fn = len(unmatched_references)
    association = _prf(association_tp, association_fp, association_fn)
    class_metrics = {}
    for state in (True, False):
        true_positives = sum(
            prediction.selected == reference.selected == state
            for prediction, reference in matched
        )
        false_positives = sum(
            prediction.selected == state and reference.selected != state
            for prediction, reference in matched
        ) + sum(item.selected == state for item in unmatched_predictions)
        false_negatives = sum(
            reference.selected == state and prediction.selected != state
            for prediction, reference in matched
        ) + sum(item.selected == state for item in unmatched_references)
        class_metrics["selected" if state else "unselected"] = {
            "reference_controls": sum(
                item.selected == state
                for values in references.values()
                for item in values
            ),
            "predicted_controls": sum(
                item.selected == state
                for values in predictions.values()
                for item in values
            ),
            **_prf(true_positives, false_positives, false_negatives),
        }
    return {
        "status": "computed",
        "provenance": (
            "project_defined_not_official_selection_f1; public selection_candidate "
            "annotations directly pair explicit selected state, label content, and bbox"
        ),
        "matching": (
            "same page, token-set overlap >=0.80, and normalized bbox centroid "
            "distance <=0.35"
        ),
        "reference_diagnostics": {
            "candidate_source": (
                "recursive public annotations, including nested table cells"
            ),
            "selected_count_mismatch_cases": reference_mismatches,
        },
        "control_to_label_association": {
            "reference_controls": sum(map(len, references.values())),
            "predicted_controls": sum(map(len, predictions.values())),
            **association,
        },
        "state": {
            "classes": class_metrics,
            "macro_f1": round(
                (
                    float(class_metrics["selected"]["f1"])
                    + float(class_metrics["unselected"]["f1"])
                )
                / 2,
                6,
            ),
        },
    }


def _match_controls(
    predictions: list[Control], references: list[Control]
) -> tuple[list[tuple[Control, Control]], list[Control], list[Control]]:
    remaining = set(range(len(references)))
    matches = []
    unmatched_predictions = []
    for prediction in sorted(predictions, key=_control_key):
        candidates = []
        for index in remaining:
            reference = references[index]
            if prediction.page != reference.page:
                continue
            overlap = _token_overlap(prediction.content, reference.content)
            distance = _centroid_distance(prediction.bbox, reference.bbox)
            if overlap >= 0.80 and distance <= 0.35:
                candidates.append((-overlap, distance, index))
        if not candidates:
            unmatched_predictions.append(prediction)
            continue
        _, _, index = min(candidates)
        remaining.remove(index)
        matches.append((prediction, references[index]))
    return (
        matches,
        unmatched_predictions,
        [references[index] for index in sorted(remaining)],
    )


def _load_records(
    path: Path, records_key: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not path.is_file():
        raise ValueError(f"Input not found: {path}")
    if path.suffix.casefold() == ".jsonl":
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"Empty JSONL input: {path}")
        try:
            values = [json.loads(line) for line in lines]
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL input {path}: {error}") from error
        metadata = values[0]
        rows = values[1:]
        if not isinstance(metadata, dict) or metadata.get("type") != "metadata":
            raise ValueError("JSONL input must start with a metadata record")
        if any(
            not isinstance(value, dict) or value.get("type") != "case" for value in rows
        ):
            raise ValueError("JSONL case records must have type=case")
        return (
            {key: value for key, value in metadata.items() if key != "type"},
            [
                {key: value for key, value in row.items() if key != "type"}
                for row in rows
            ],
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON input {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get(records_key), list):
        raise ValueError(f"JSON input must contain a {records_key} list")
    metadata = {key: value for key, value in payload.items() if key != records_key}
    rows = payload[records_key]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{records_key} entries must be objects")
    return metadata, rows


def _validate_metadata(
    dataset: dict[str, object], predictions: dict[str, object], role: str
) -> None:
    if (
        dataset.get("dataset") != DATASET_ID
        or dataset.get("revision") != DATASET_REVISION
        or dataset.get("split") != "train"
    ):
        raise ValueError("Dataset export revision, ID, or split does not match the pin")
    if predictions.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Prediction dataset revision does not match the pin")
    if predictions.get("scorer_revision") != SCORER_REVISION:
        raise ValueError("Prediction scorer revision does not match the pin")
    if predictions.get("role") != role:
        raise ValueError("Prediction role does not match the requested panel")


def _select_panel(
    rows: list[dict[str, object]], role: str
) -> tuple[list[dict[str, object]], list[str]]:
    if role not in {"dev", "eval", "full"}:
        raise ValueError(f"Unsupported benchmark role: {role}")
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"Dataset export must contain all {EXPECTED_CASES} cases")
    ordered = sorted(rows, key=_sample_id)
    ids = [_sample_id(row) for row in ordered]
    if len(set(ids)) != EXPECTED_CASES:
        raise ValueError("Dataset export case IDs must be unique")
    positive = []
    negative = []
    for row in ordered:
        selected_count = row.get("selected_count")
        if (
            isinstance(selected_count, bool)
            or not isinstance(selected_count, int)
            or selected_count < 0
        ):
            raise ValueError(
                f"Dataset case {_sample_id(row)} has invalid selected_count"
            )
        (positive if selected_count > 0 else negative).append(row)
    if (
        len(positive) != EXPECTED_POSITIVE_CASES
        or len(negative) != EXPECTED_NEGATIVE_CASES
    ):
        raise ValueError(
            "Dataset class counts do not match the pinned 459-positive/26-negative panel"
        )
    dev_ids = {
        *(_sample_id(row) for row in _spaced(positive, 56)),
        *(_sample_id(row) for row in _spaced(negative, 4)),
    }
    if len(dev_ids) != DEV_CASES:
        raise ValueError("Deterministic development panel is invalid")
    if role == "dev":
        panel = [row for row in ordered if _sample_id(row) in dev_ids]
    elif role == "eval":
        panel = [row for row in ordered if _sample_id(row) not in dev_ids]
    else:
        panel = ordered
    expected = {"dev": DEV_CASES, "eval": EVAL_CASES, "full": EXPECTED_CASES}[role]
    if len(panel) != expected:
        raise ValueError(f"{role} panel must contain {expected} cases")
    return panel, [_sample_id(row) for row in panel]


def _spaced(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count < 2 or len(rows) < count:
        raise ValueError("Invalid proportional split request")
    denominator = count - 1
    indices = [
        (index * (len(rows) - 1) + denominator // 2) // denominator
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise ValueError("Proportional split produced duplicate positions")
    return [rows[index] for index in indices]


def _control(
    value: object,
    *,
    context: str,
    selected: bool | None = None,
    require_label_bbox: bool = False,
    allow_reference_text: bool = False,
) -> Control:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid control item: {context}")
    if selected is None and set(value) - ITEM_KEYS:
        raise ValueError(f"Unsupported control fields: {context}")
    page = value.get("page")
    content = value.get("content")
    if content is None and allow_reference_text:
        content = value.get("text")
    bbox = value.get("bbox", [])
    state = value.get("selected") if selected is None else selected
    if isinstance(page, bool) or not isinstance(page, int) or page != 1:
        raise ValueError(f"Invalid page for {context}")
    if not isinstance(content, str) or (require_label_bbox and not content.strip()):
        raise ValueError(f"Invalid label content for {context}")
    if not isinstance(state, bool):
        raise ValueError(f"Invalid selected state for {context}")
    if not isinstance(bbox, list) or len(bbox) not in (
        {8} if require_label_bbox else {0, 8}
    ):
        raise ValueError(f"Invalid bbox for {context}")
    coordinates = tuple(_coordinate(coordinate, context) for coordinate in bbox)
    return Control(page=page, bbox=coordinates, content=content, selected=state)


def _failure(value: object, sample_id: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) - {"code", "message", "stage"}:
        raise ValueError(f"Invalid failure for {sample_id}")
    if not isinstance(value.get("code"), str) or not value["code"]:
        raise ValueError(f"Invalid failure code for {sample_id}")
    if not isinstance(value.get("message"), str):
        raise ValueError(f"Invalid failure message for {sample_id}")
    if value.get("stage") is not None and not isinstance(value["stage"], str):
        raise ValueError(f"Invalid failure stage for {sample_id}")
    return {
        key: str(value[key])
        for key in ("code", "message", "stage")
        if value.get(key) is not None
    }


def _missing_prediction(sample_id: str) -> dict[str, object]:
    return {
        "status": "failed",
        "latency_ms": None,
        "items": [],
        "failures": [
            {
                "code": "missing_prediction",
                "message": f"No prediction was supplied for {sample_id}",
                "stage": "prediction_load",
            }
        ],
    }


def _sample_id(row: dict[str, object]) -> str:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Every record requires a non-empty sample_id")
    return sample_id


def _adapt_controls(
    values: dict[str, list[Control]], result_type: type
) -> dict[str, list[object]]:
    return {
        sample_id: [
            result_type(
                page=item.page,
                bbox=list(item.bbox),
                selected=item.selected,
                content=item.content,
            )
            for item in items
        ]
        for sample_id, items in values.items()
    }


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"Cannot inspect official scorer revision: {error}") from error
    if result.returncode != 0:
        raise ValueError("Official scorer root is not a readable Git checkout")
    return result.stdout.strip()


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import official scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        raise ValueError(
            f"Cannot import official scorer module {path}: {error}"
        ) from error
    return module


def _coordinate(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"Invalid normalized bbox coordinate for {context}")
    return float(value)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", _normalize(left)))
    right_tokens = set(re.findall(r"\w+", _normalize(right)))
    largest = max(len(left_tokens), len(right_tokens))
    return len(left_tokens & right_tokens) / largest if largest else 1.0


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _centroid_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != 8 or len(right) != 8:
        return math.inf
    left_x = sum(left[0::2]) / 4
    left_y = sum(left[1::2]) / 4
    right_x = sum(right[0::2]) / 4
    right_y = sum(right[1::2]) / 4
    return math.hypot(left_x - right_x, left_y - right_y)


def _control_key(item: Control) -> tuple[object, ...]:
    return (item.page, _normalize(item.content), item.bbox, item.selected)


def _prf(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, object]:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision_denominator": precision_denominator,
        "recall_denominator": recall_denominator,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


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


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary = Path(stream.name)
        os.link(temporary, path)
    except FileExistsError as error:
        raise ValueError(f"Output already exists: {path}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
