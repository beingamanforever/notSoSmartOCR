"""Train the optional Heron table-to-form-companion region router."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from experiments.serve_falcon_layout import (
    HERON_REGION_ROLE_FEATURE_NAMES,
    ROUTE_EXTRACTOR_VERSION,
    _model_revision,
)
from ocr_pipeline.heron_layout import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

LABELS = {"table_only": 0, "add_form_source": 1, "unresolved": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--heron-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--heron-model-revision",
        type=_model_revision,
        default=DEFAULT_MODEL_REVISION,
    )
    parser.add_argument("--layout-score-threshold", type=float, default=0.3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if not 0 <= args.layout_score_threshold <= 1:
        parser.error("--layout-score-threshold must be from 0 to 1")

    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line]
    model = _build_model(
        rows,
        args.model_id,
        {
            "version": ROUTE_EXTRACTOR_VERSION,
            "heron_model_id": args.heron_model_id,
            "heron_model_revision": args.heron_model_revision,
            "layout_score_threshold": args.layout_score_threshold,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps(model["development"], sort_keys=True))
    return 0


def _build_model(rows: list[dict], model_id: str, extractor: dict) -> dict:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    records = _validated_records(rows)
    used = [row for row in records if row["target"] is not None]
    train = [row for row in used if row["split"] == "train"]
    calibration = [row for row in used if row["split"] == "calibration"]
    matrix, targets, groups = _arrays(train)
    _require_grouped_training(targets, groups)

    logo_probabilities = _grouped_probabilities(matrix, targets, groups)
    scaler, estimator = _fit(matrix, targets)
    threshold, selects_positive = _calibration_threshold(calibration, scaler, estimator)
    status = (
        "calibrated"
        if _calibration_ready(used, train, calibration, selects_positive)
        else "development_only"
    )
    model = {
        "version": 1,
        "id": model_id,
        "status": status,
        "feature_names": list(HERON_REGION_ROLE_FEATURE_NAMES),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "threshold": threshold,
        "extractor": extractor,
        "development": {
            "input_records": len(rows),
            "physical_groups": len(records),
            "used_records": len(used),
            "unresolved_records": len(records) - len(used),
            "train_source_groups": len(set(groups)),
            "calibration_source_groups": len(
                {row["source_group"] for row in calibration}
            ),
            "logo": _metrics(targets, logo_probabilities, 0.5),
            "calibration": _calibration_metrics(
                calibration, scaler, estimator, threshold
            ),
        },
    }
    return model


def _validated_records(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("Training records are empty")
    records: dict[str, dict] = {}
    candidate_groups: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    feature_names = set(HERON_REGION_ROLE_FEATURE_NAMES)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each training record must be a JSON object")
        candidate_id = row.get("candidate_id")
        physical_group = row.get("physical_group")
        source_group = row.get("source_group")
        split = row.get("split")
        label = row.get("label")
        label_source = row.get("label_source")
        feature_values = row.get("feature_values")
        if any(
            not isinstance(value, str) or not value
            for value in (candidate_id, physical_group, source_group)
        ):
            raise ValueError("Each record needs candidate, physical, and source groups")
        if split not in {"train", "calibration"}:
            raise ValueError("Only train and calibration records may be loaded")
        if label not in LABELS:
            raise ValueError("Unknown region-role label")
        if label_source is not None and (
            not isinstance(label_source, str) or not label_source
        ):
            raise ValueError("label_source must be a nonempty string when supplied")
        if not isinstance(feature_values, dict) or set(feature_values) != feature_names:
            raise ValueError("Each record needs the exact named Heron feature mapping")
        if any(not _finite_number(feature_values[name]) for name in feature_names):
            raise ValueError("Heron feature values must be finite numbers")
        if (
            candidate_id in candidate_groups
            and candidate_groups[candidate_id] != physical_group
        ):
            raise ValueError("candidate_id cannot identify multiple physical groups")
        candidate_groups[candidate_id] = physical_group
        if source_group in source_splits and source_splits[source_group] != split:
            raise ValueError("source_group cannot leak across splits")
        source_splits[source_group] = split

        normalized = {
            "physical_group": physical_group,
            "source_group": source_group,
            "split": split,
            "label": label,
            "label_source": label_source,
            "target": LABELS[label],
            "features": tuple(
                float(feature_values[name]) for name in HERON_REGION_ROLE_FEATURE_NAMES
            ),
        }
        previous = records.get(physical_group)
        if previous is not None and previous["label"] != label:
            raise ValueError("Conflicting labels for physical_group")
        if previous is not None and previous != normalized:
            raise ValueError("Conflicting observations for physical_group")
        records.setdefault(physical_group, normalized)
    return list(records.values())


def _arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Training requires labeled train records")
    return (
        np.asarray([row["features"] for row in records], dtype=np.float64),
        np.asarray([row["target"] for row in records], dtype=np.int64),
        np.asarray([row["source_group"] for row in records]),
    )


def _require_grouped_training(targets: np.ndarray, groups: np.ndarray) -> None:
    if len(set(groups)) < 2 or any(
        len(set(groups[targets == target])) < 2 for target in (0, 1)
    ):
        raise ValueError(
            "Each train class requires at least two source groups for LOGO"
        )


def _fit(
    matrix: np.ndarray, targets: np.ndarray
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
    ).fit(scaler.transform(matrix), targets)
    return scaler, estimator


def _probabilities(
    matrix: np.ndarray, scaler: StandardScaler, estimator: LogisticRegression
) -> np.ndarray:
    return estimator.predict_proba(scaler.transform(matrix))[:, 1]


def _grouped_probabilities(
    matrix: np.ndarray, targets: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    probabilities = np.empty(len(matrix), dtype=np.float64)
    for fit, held_out in LeaveOneGroupOut().split(matrix, targets, groups):
        scaler, estimator = _fit(matrix[fit], targets[fit])
        probabilities[held_out] = _probabilities(matrix[held_out], scaler, estimator)
    return probabilities


def _calibration_threshold(
    records: list[dict], scaler: StandardScaler, estimator: LogisticRegression
) -> tuple[float, bool]:
    if not records:
        return 0.5, False
    matrix, targets, _ = _arrays(records)
    probabilities = _probabilities(matrix, scaler, estimator)
    negatives = probabilities[targets == 0]
    positives = probabilities[targets == 1]
    eligible = positives[positives > max(negatives, default=1.0)]
    return (float(eligible.min()), True) if len(eligible) else (1.0, False)


def _calibration_ready(
    used: list[dict], train: list[dict], calibration: list[dict], selects_positive: bool
) -> bool:
    if not train or not calibration or not selects_positive:
        return False
    if any(row["label_source"] != "human_review" for row in used):
        return False
    if {row["target"] for row in calibration} != {0, 1}:
        return False
    return all(
        len({row["source_group"] for row in used if row["target"] == target}) >= 3
        for target in (0, 1)
    )


def _metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    selected = probabilities >= threshold
    return {
        "average_precision": float(average_precision_score(targets, probabilities)),
        "precision": float(precision_score(targets, selected, zero_division=0)),
        "recall": float(recall_score(targets, selected, zero_division=0)),
    }


def _calibration_metrics(
    records: list[dict],
    scaler: StandardScaler,
    estimator: LogisticRegression,
    threshold: float,
) -> dict[str, float] | None:
    if not records:
        return None
    matrix, targets, _ = _arrays(records)
    probabilities = _probabilities(matrix, scaler, estimator)
    if len(set(targets)) < 2:
        return None
    return _metrics(targets, probabilities, threshold)


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


if __name__ == "__main__":
    raise SystemExit(main())
