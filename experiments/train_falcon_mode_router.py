"""Fit the optional Falcon layout-versus-plain router from grouped observations.

Each JSONL record supplies only pre-decode page evidence plus failure-inclusive
quality and expert cost for both observed actions. ``layout`` means the existing
layout pipeline including its selective Qwen correction. ``plain`` means native
full-page Falcon without Qwen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from experiments.serve_falcon_layout import (
    HERON_ROUTE_FEATURE_NAMES,
    ROUTE_EXTRACTOR_VERSION,
    ROUTE_TRAINING_ACTIONS,
    _model_revision,
)
from ocr_pipeline.heron_layout import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


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

    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line]
    feature_names = list(HERON_ROUTE_FEATURE_NAMES)
    if not 0 <= args.layout_score_threshold <= 1:
        parser.error("--layout-score-threshold must be from 0 to 1")
    matrix, targets, groups, qualities, costs = _training_arrays(rows, feature_names)
    cost_targets = costs[:, 0] - costs[:, 1]
    quality_oof, cost_oof = _grouped_predictions(matrix, targets, cost_targets, groups)
    tradeoff, threshold = _select_policy(quality_oof, cost_oof, qualities, costs)
    selected_plain = quality_oof + tradeoff * cost_oof > threshold
    nested_selected = _nested_grouped_policy(
        matrix, targets, cost_targets, groups, qualities, costs
    )
    scaler, quality_estimator, cost_estimator = _fit_heads(
        matrix, targets, cost_targets
    )
    model = {
        "version": 1,
        "id": args.model_id,
        "training_actions": ROUTE_TRAINING_ACTIONS,
        "feature_names": feature_names,
        "coefficients": quality_estimator.coef_.tolist(),
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "intercept": float(quality_estimator.intercept_),
        "threshold": threshold,
        "cost_tradeoff": tradeoff,
        "cost_head": {
            "coefficients": cost_estimator.coef_.tolist(),
            "intercept": float(cost_estimator.intercept_),
        },
        "extractor": {
            "version": ROUTE_EXTRACTOR_VERSION,
            "heron_model_id": args.heron_model_id,
            "heron_model_revision": args.heron_model_revision,
            "layout_score_threshold": args.layout_score_threshold,
        },
        "status": "development_only_optional_separate_calibration_required",
        "estimator": {
            "kind": "ridge",
            "alpha": 100.0,
            "solver": "lsqr",
            "selection": "fixed development-selected configuration; not final evaluation",
        },
        "development": {
            "records": len(rows),
            "source_groups": len(set(groups)),
            "development_selected_oof": {
                "quality_mae": float(np.mean(np.abs(quality_oof - targets))),
                "cost_mae": float(np.mean(np.abs(cost_oof - cost_targets))),
                "cost_tradeoff": tradeoff,
                "threshold": threshold,
                **_totals(selected_plain, qualities, costs),
            },
            "nested_grouped_development_estimate": _totals(
                nested_selected, qualities, costs
            ),
            "always_layout": _totals(np.zeros(len(rows), dtype=bool), qualities, costs),
            "always_plain": _totals(np.ones(len(rows), dtype=bool), qualities, costs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps(model["development"], sort_keys=True))
    return 0


def _training_arrays(
    rows: list[dict], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("Training records are empty")
    values, targets, groups, qualities, costs = [], [], [], [], []
    for row in rows:
        if row.get("split") != "development":
            raise ValueError(
                "This development-only trainer requires development records"
            )
        group = row.get("source_group")
        feature_values = row.get("feature_values")
        metrics = [
            row.get("layout_quality"),
            row.get("plain_quality"),
            row.get("layout_expert_cost"),
            row.get("plain_expert_cost"),
        ]
        if not isinstance(group, str) or not group:
            raise ValueError("Each record needs a source group")
        if any(not _finite_number(value) for value in metrics) or any(
            value < 0 for value in metrics[2:]
        ):
            raise ValueError(
                "Each action needs finite quality and nonnegative expert cost"
            )
        if not isinstance(feature_values, dict) or set(feature_values) != set(
            feature_names
        ):
            raise ValueError("Each record needs the exact named Heron feature mapping")
        if any(not _finite_number(feature_values[name]) for name in feature_names):
            raise ValueError("Heron feature values must be finite numbers")
        layout_quality, plain_quality, layout_cost, plain_cost = map(float, metrics)
        values.append([float(feature_values[name]) for name in feature_names])
        targets.append(plain_quality - layout_quality)
        groups.append(group)
        qualities.append([layout_quality, plain_quality])
        costs.append([layout_cost, plain_cost])
    if len(set(groups)) < 2:
        raise ValueError("Training requires at least two source groups")
    return (
        np.asarray(values),
        np.asarray(targets),
        np.asarray(groups),
        np.asarray(qualities),
        np.asarray(costs),
    )


def _fit(values: np.ndarray, targets: np.ndarray) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(values)
    estimator = Ridge(alpha=100.0, solver="lsqr").fit(scaler.transform(values), targets)
    return scaler, estimator


def _fit_heads(
    values: np.ndarray, quality_targets: np.ndarray, cost_targets: np.ndarray
) -> tuple[StandardScaler, Ridge, Ridge]:
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    return (
        scaler,
        Ridge(alpha=100.0, solver="lsqr").fit(scaled, quality_targets),
        Ridge(alpha=100.0, solver="lsqr").fit(scaled, cost_targets),
    )


def _predict(values: np.ndarray, estimator: Ridge) -> np.ndarray:
    return np.einsum("ij,j->i", values, estimator.coef_) + float(estimator.intercept_)


def _grouped_predictions(
    matrix: np.ndarray,
    quality_targets: np.ndarray,
    cost_targets: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quality_values = np.empty(len(matrix), dtype=np.float64)
    cost_values = np.empty(len(matrix), dtype=np.float64)
    for fit, held_out in LeaveOneGroupOut().split(matrix, quality_targets, groups):
        scaler, quality_estimator, cost_estimator = _fit_heads(
            matrix[fit], quality_targets[fit], cost_targets[fit]
        )
        held_values = scaler.transform(matrix[held_out])
        quality_values[held_out] = _predict(held_values, quality_estimator)
        cost_values[held_out] = _predict(held_values, cost_estimator)
    return quality_values, cost_values


def _nested_grouped_policy(
    matrix: np.ndarray,
    quality_targets: np.ndarray,
    cost_targets: np.ndarray,
    groups: np.ndarray,
    qualities: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    if len(set(groups)) < 3:
        raise ValueError(
            "Nested grouped development estimation needs three source groups"
        )
    selected = np.zeros(len(matrix), dtype=bool)
    for outer_group in sorted(set(groups)):
        inner = groups != outer_group
        outer = ~inner
        inner_quality, inner_cost = _grouped_predictions(
            matrix[inner], quality_targets[inner], cost_targets[inner], groups[inner]
        )
        tradeoff, threshold = _select_policy(
            inner_quality, inner_cost, qualities[inner], costs[inner]
        )
        scaler, quality_estimator, cost_estimator = _fit_heads(
            matrix[inner], quality_targets[inner], cost_targets[inner]
        )
        outer_values = scaler.transform(matrix[outer])
        selected[outer] = (
            _predict(outer_values, quality_estimator)
            + tradeoff * _predict(outer_values, cost_estimator)
            > threshold
        )
    return selected


def _threshold(values: np.ndarray, qualities: np.ndarray, costs: np.ndarray) -> float:
    ordered = sorted(set(float(value) for value in values))
    candidates = [ordered[0] - 1.0, ordered[-1] + 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    return max(
        candidates,
        key=lambda value: (
            _totals(values > value, qualities, costs)["quality"],
            -_totals(values > value, qualities, costs)["expert_cost"],
            -abs(value),
        ),
    )


def _select_policy(
    quality_values: np.ndarray,
    cost_values: np.ndarray,
    qualities: np.ndarray,
    costs: np.ndarray,
) -> tuple[float, float]:
    policies = []
    for tradeoff in _tradeoffs(quality_values, cost_values):
        values = quality_values + tradeoff * cost_values
        threshold = _threshold(values, qualities, costs)
        totals = _totals(values > threshold, qualities, costs)
        policies.append(
            (
                totals["quality"],
                -totals["expert_cost"],
                -tradeoff,
                -abs(threshold),
                tradeoff,
                threshold,
            )
        )
    *_, tradeoff, threshold = max(policies)
    return tradeoff, threshold


def _tradeoffs(quality_values: np.ndarray, cost_values: np.ndarray) -> list[float]:
    crossings = {0.0}
    for left in range(len(quality_values)):
        for right in range(left + 1, len(quality_values)):
            denominator = cost_values[left] - cost_values[right]
            if denominator == 0:
                continue
            crossing = (quality_values[right] - quality_values[left]) / denominator
            if math.isfinite(crossing) and crossing > 0:
                crossings.add(float(crossing))
    ordered = sorted(crossings)
    candidates = set(ordered)
    candidates.update((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    candidates.add(ordered[-1] + 1.0)
    return sorted(candidates)


def _totals(
    selected_plain: np.ndarray, qualities: np.ndarray, costs: np.ndarray
) -> dict[str, float | dict[str, int]]:
    selected = selected_plain.astype(int)
    indices = np.arange(len(selected))
    return {
        "quality": float(qualities[indices, selected].sum()),
        "expert_cost": float(costs[indices, selected].sum()),
        "pages": {
            "layout": int((~selected_plain).sum()),
            "plain": int(selected_plain.sum()),
        },
    }


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and bool(np.isfinite(value))


if __name__ == "__main__":
    raise SystemExit(main())
