"""Compose frozen orientation predictions with cached four-angle OCR outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from experiments.public_benchmark import _score, _summarize
else:
    from public_benchmark import _score, _summarize

ANGLES = (0, 90, 180, 270)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen orientation predictions using cached OCR views"
    )
    parser.add_argument("predictions", type=Path, help="Orientation prediction JSON")
    parser.add_argument("four_angle", type=Path, help="Four-angle OCR JSON")
    parser.add_argument("output", type=Path, help="Selected benchmark JSON")
    args = parser.parse_args(argv)

    try:
        payload = compose_files(args.predictions, args.four_angle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def compose_files(predictions_path: Path, four_angle_path: Path) -> dict[str, object]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    four_angle = json.loads(four_angle_path.read_text(encoding="utf-8"))
    return {
        "sources": {
            "predictions": str(predictions_path),
            "four_angle": str(four_angle_path),
        },
        **compose_payloads(predictions, four_angle),
    }


def compose_payloads(
    predictions_payload: object,
    four_angle_payload: object,
) -> dict[str, object]:
    predictions = _predictions(predictions_payload)
    runs = _angle_runs(four_angle_payload)
    cases_by_angle = {angle: _cases(runs[angle], f"angle {angle}") for angle in ANGLES}
    case_ids = set(cases_by_angle[0])
    if set(predictions) != case_ids:
        raise ValueError("Prediction and cached OCR case IDs differ")
    for angle in ANGLES[1:]:
        if set(cases_by_angle[angle]) != case_ids:
            raise ValueError(f"Cached OCR case IDs differ for angle {angle}")

    dataset = runs[0].get("dataset")
    normalization = runs[0].get("normalization")
    if not isinstance(dataset, str) or not isinstance(normalization, str):
        raise ValueError("Cached OCR run has invalid benchmark metadata")
    for angle in ANGLES[1:]:
        if (
            runs[angle].get("dataset") != dataset
            or runs[angle].get("normalization") != normalization
        ):
            raise ValueError(f"Cached OCR metadata differs for angle {angle}")

    selected_cases = []
    exact_oracle = 0
    catastrophic = 0
    regrets = []
    for case_id in sorted(case_ids):
        prediction = predictions[case_id]
        angle = int(prediction["lossless_rotation"])
        angle_cases = {value: cases_by_angle[value][case_id] for value in ANGLES}
        scored_cases = _score_angle_cases(case_id, angle_cases)
        oracle_rate = min(_cer_rate(case) for case in scored_cases.values())
        oracle_angles = [
            value
            for value, case in scored_cases.items()
            if _cer_rate(case) == oracle_rate
        ]
        selected = scored_cases[angle]
        selected_rate = _cer_rate(selected)
        regret = selected_rate - oracle_rate
        exact_oracle += int(angle in oracle_angles)
        catastrophic += int(regret >= 0.2)
        regrets.append(regret)
        selected["orientation_prediction"] = {
            "angle": angle,
            "confidence": prediction.get("confidence"),
            "oracle_angles": oracle_angles,
            "exact_oracle_angle_selection": angle in oracle_angles,
            "cer_regret": round(regret, 6),
        }
        selected_cases.append(selected)

    summary = _summarize(selected_cases)
    summary.pop("latency_ms", None)
    return {
        "experiment": "cached_orientation_policy",
        "dataset": dataset,
        "normalization": normalization,
        "summary": summary,
        "selection_summary": {
            "cases": len(selected_cases),
            "exact_oracle_angle_selection_count": exact_oracle,
            "exact_oracle_angle_selection_rate": round(
                exact_oracle / len(selected_cases), 6
            ),
            "mean_cer_regret": round(sum(regrets) / len(regrets), 6),
            "maximum_cer_regret": round(max(regrets), 6),
            "catastrophic_regret_threshold": 0.2,
            "catastrophic_regret_count": catastrophic,
            "oracle_is_deployable": False,
        },
        "latency": {
            "available": False,
            "reason": "Orientation and OCR views were measured in separate runs",
        },
        "cases": selected_cases,
    }


def _predictions(payload: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Orientation predictions must contain a cases array")
    predictions = {}
    for case in payload["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Orientation prediction has no valid id")
        case_id = case["id"]
        if case_id in predictions:
            raise ValueError(f"Duplicate orientation prediction: {case_id}")
        angle = case.get("lossless_rotation")
        if not isinstance(angle, int) or isinstance(angle, bool) or angle not in ANGLES:
            raise ValueError(f"Invalid orientation prediction for {case_id}")
        predictions[case_id] = case
    if not predictions:
        raise ValueError("Orientation predictions are empty")
    return predictions


def _angle_runs(payload: object) -> dict[int, Mapping[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError("Four-angle OCR JSON must contain a runs object")
    runs = payload["runs"]
    if set(runs) != {str(angle) for angle in ANGLES}:
        raise ValueError("Four-angle OCR JSON must contain 0, 90, 180, and 270")
    return {angle: runs[str(angle)] for angle in ANGLES}


def _cases(run: object, label: str) -> dict[str, Mapping[str, object]]:
    if not isinstance(run, dict) or not isinstance(run.get("cases"), list):
        raise ValueError(f"{label} has no cases array")
    cases = {}
    for case in run["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError(f"{label} has a case without a valid id")
        case_id = case["id"]
        if case_id in cases:
            raise ValueError(f"Duplicate {label} case: {case_id}")
        cases[case_id] = case
    return cases


def _score_angle_cases(
    case_id: str,
    angle_cases: Mapping[int, Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    reference = angle_cases[0].get("reference")
    cluster_id = angle_cases[0].get("cluster_id")
    if not isinstance(reference, str):
        raise ValueError(f"Cached OCR case {case_id!r} has an invalid reference")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise ValueError(f"Cached OCR case {case_id!r} has an invalid cluster ID")

    scored_cases = {}
    for angle, case in angle_cases.items():
        if case.get("reference") != reference:
            raise ValueError(
                f"Cached OCR reference differs for {case_id!r} at angle {angle}"
            )
        if case.get("cluster_id") != cluster_id:
            raise ValueError(
                f"Cached OCR cluster ID differs for {case_id!r} at angle {angle}"
            )
        prediction = case.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(
                f"Cached OCR prediction is invalid for {case_id!r} at angle {angle}"
            )
        scored_case = dict(case)
        scored_case["metrics"] = _score(prediction, reference)
        scored_cases[angle] = scored_case
    return scored_cases


def _cer_rate(case: Mapping[str, object]) -> float:
    metrics = case.get("metrics")
    cer = metrics.get("cer") if isinstance(metrics, dict) else None
    if not isinstance(cer, dict):
        raise ValueError(f"Cached OCR case {case.get('id')!r} has no CER metrics")
    edits = cer.get("edits")
    units = cer.get("reference_units")
    if (
        not isinstance(edits, int)
        or isinstance(edits, bool)
        or edits < 0
        or not isinstance(units, int)
        or isinstance(units, bool)
        or units <= 0
    ):
        raise ValueError(f"Cached OCR case {case.get('id')!r} has invalid CER")
    return edits / units


if __name__ == "__main__":
    raise SystemExit(main())
