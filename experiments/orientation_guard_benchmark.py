"""Evaluate the production two-view orientation guard from cached OCR runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from experiments.cached_orientation_policy import (
        ANGLES,
        _angle_runs,
        _cases,
        _cer_rate,
        _predictions,
        _score_angle_cases,
    )
    from experiments.public_benchmark import _summarize
else:
    from cached_orientation_policy import (  # type: ignore[no-redef]
        ANGLES,
        _angle_runs,
        _cases,
        _cer_rate,
        _predictions,
        _score_angle_cases,
    )
    from public_benchmark import _summarize  # type: ignore[no-redef]

COVERAGE_RECOVERY_RATIO = 2.0
COVERAGE_CONFIDENCE_TOLERANCE = 0.03
MIN_SUPPORTING_WORDS = 10
CATASTROPHIC_REGRET_THRESHOLD = 0.2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the cached docTR and Tesseract OSD orientation guard"
    )
    parser.add_argument("doctr", type=Path, help="docTR orientation prediction JSON")
    parser.add_argument("osd", type=Path, help="Tesseract OSD automatic run JSON")
    parser.add_argument("view_scores", type=Path, help="Full automatic view-score JSON")
    parser.add_argument("four_angle", type=Path, help="Four-angle OCR JSON")
    parser.add_argument("output", type=Path, help="Selected benchmark JSON")
    args = parser.parse_args(argv)

    try:
        payload = compose_files(
            args.doctr,
            args.osd,
            args.view_scores,
            args.four_angle,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def compose_files(
    doctr_path: Path,
    osd_path: Path,
    view_scores_path: Path,
    four_angle_path: Path,
) -> dict[str, object]:
    payload = compose_payloads(
        json.loads(doctr_path.read_text(encoding="utf-8")),
        json.loads(osd_path.read_text(encoding="utf-8")),
        json.loads(view_scores_path.read_text(encoding="utf-8")),
        json.loads(four_angle_path.read_text(encoding="utf-8")),
    )
    return {
        "sources": {
            "doctr": str(doctr_path),
            "osd": str(osd_path),
            "view_scores": str(view_scores_path),
            "four_angle": str(four_angle_path),
        },
        **payload,
    }


def compose_payloads(
    doctr_payload: object,
    osd_payload: object,
    view_scores_payload: object,
    four_angle_payload: object,
) -> dict[str, object]:
    predictions = _predictions(doctr_payload)
    osd_run = _automatic_run(osd_payload, "OSD")
    score_run = _automatic_run(view_scores_payload, "View-score")
    osd_cases = _cases(osd_run, "OSD automatic run")
    score_cases = _cases(score_run, "full automatic run")
    angle_runs = _angle_runs(four_angle_payload)
    cases_by_angle = {
        angle: _cases(angle_runs[angle], f"angle {angle}") for angle in ANGLES
    }
    case_ids = set(cases_by_angle[0])
    sources = {
        "docTR predictions": set(predictions),
        "OSD automatic run": set(osd_cases),
        "full automatic run": set(score_cases),
    }
    for label, ids in sources.items():
        if ids != case_ids:
            raise ValueError(f"{label} and cached OCR case IDs differ")
    for angle in ANGLES[1:]:
        if set(cases_by_angle[angle]) != case_ids:
            raise ValueError(f"Cached OCR case IDs differ for angle {angle}")

    dataset, normalization = _benchmark_metadata(
        osd_run,
        score_run,
        angle_runs,
    )
    selected_cases = []
    exact_oracle = 0
    catastrophic = 0
    coverage_recoveries = 0
    osd_fallbacks = 0
    agreements = 0
    regrets = []
    for case_id in sorted(case_ids):
        angle_cases = {angle: cases_by_angle[angle][case_id] for angle in ANGLES}
        _validate_case_identity(case_id, osd_cases[case_id], angle_cases[0], "OSD")
        _validate_case_identity(
            case_id,
            score_cases[case_id],
            angle_cases[0],
            "view-score",
        )
        scored_cases = _score_angle_cases(case_id, angle_cases)
        doctr_angle = int(predictions[case_id]["lossless_rotation"])
        osd_angle, osd_available = _osd_angle(osd_cases[case_id], case_id)
        candidate_angles = list(dict.fromkeys((doctr_angle, osd_angle)))
        scores, failures = _candidate_scores(
            score_cases[case_id],
            candidate_angles,
            case_id,
        )
        selected_angle, reason = select_angle(scores)

        oracle_rate = min(_cer_rate(case) for case in scored_cases.values())
        oracle_angles = [
            angle
            for angle, case in scored_cases.items()
            if _cer_rate(case) == oracle_rate
        ]
        selected = scored_cases[selected_angle]
        regret = _cer_rate(selected) - oracle_rate
        exact = selected_angle in oracle_angles
        exact_oracle += int(exact)
        catastrophic += int(regret >= CATASTROPHIC_REGRET_THRESHOLD)
        coverage_recoveries += int(reason == "coverage_recovery")
        osd_fallbacks += int(not osd_available)
        agreements += int(doctr_angle == osd_angle)
        regrets.append(regret)
        selected["orientation_selection"] = {
            "doctr_angle": doctr_angle,
            "doctr_confidence": predictions[case_id].get("confidence"),
            "osd_angle": osd_angle,
            "osd_available": osd_available,
            "candidate_angles": candidate_angles,
            "candidate_view_failures": failures,
            "selected_angle": selected_angle,
            "selection_reason": reason,
            "selected_score": scores[selected_angle],
            "oracle_angles": oracle_angles,
            "exact_oracle_angle_selection": exact,
            "cer_regret": round(regret, 6),
        }
        selected_cases.append(selected)

    summary = _summarize(selected_cases)
    summary.pop("latency_ms", None)
    case_count = len(selected_cases)
    return {
        "experiment": "cached_two_view_orientation_guard",
        "dataset": dataset,
        "normalization": normalization,
        "policy": {
            "candidates": "docTR angle and recorded OSD angle, or 0 without OSD",
            "coverage_recovery_ratio": COVERAGE_RECOVERY_RATIO,
            "coverage_confidence_tolerance": COVERAGE_CONFIDENCE_TOLERANCE,
            "minimum_supporting_words": MIN_SUPPORTING_WORDS,
            "otherwise": "retain confidence rank ordering",
        },
        "summary": summary,
        "selection_summary": {
            "cases": case_count,
            "doctr_osd_agreement_count": agreements,
            "osd_fallback_zero_count": osd_fallbacks,
            "coverage_recovery_count": coverage_recoveries,
            "exact_oracle_angle_selection_count": exact_oracle,
            "exact_oracle_angle_selection_rate": round(exact_oracle / case_count, 6),
            "mean_cer_regret": round(sum(regrets) / case_count, 6),
            "maximum_cer_regret": round(max(regrets), 6),
            "catastrophic_regret_threshold": CATASTROPHIC_REGRET_THRESHOLD,
            "catastrophic_regret_count": catastrophic,
            "oracle_is_deployable": False,
        },
        "latency": {
            "available": False,
            "reason": "Orientation signals and OCR views were measured in separate runs",
        },
        "cases": selected_cases,
    }


def select_angle(
    scores: Mapping[int, Mapping[str, object]],
) -> tuple[int, str]:
    if not scores:
        raise ValueError("No candidate orientation view succeeded")
    ranked_angle = max(scores, key=lambda angle: _rank(scores[angle]))
    broadest_angle = max(
        scores,
        key=lambda angle: (_number(scores[angle], "characters"), _rank(scores[angle])),
    )
    if broadest_angle == ranked_angle:
        return ranked_angle, "confidence_rank"

    ranked = scores[ranked_angle]
    broadest = scores[broadest_angle]
    broadest_characters = _number(broadest, "characters")
    ranked_characters = _number(ranked, "characters")
    coverage_recovery = (
        _number(broadest, "supporting_words") >= MIN_SUPPORTING_WORDS
        and _number(broadest, "mean_confidence") + COVERAGE_CONFIDENCE_TOLERANCE
        >= _number(ranked, "mean_confidence")
        and broadest_characters >= COVERAGE_RECOVERY_RATIO * ranked_characters
    )
    return (
        (broadest_angle, "coverage_recovery")
        if coverage_recovery
        else (ranked_angle, "confidence_rank")
    )


def _automatic_run(payload: object, label: str) -> Mapping[str, object]:
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError(f"{label} JSON must contain a runs object")
    runs = payload["runs"]
    if set(runs) != {"auto"} or not isinstance(runs["auto"], dict):
        raise ValueError(f"{label} JSON must contain only an auto run")
    return runs["auto"]


def _benchmark_metadata(
    osd_run: Mapping[str, object],
    score_run: Mapping[str, object],
    angle_runs: Mapping[int, Mapping[str, object]],
) -> tuple[str, str]:
    dataset = angle_runs[0].get("dataset")
    normalization = angle_runs[0].get("normalization")
    if not isinstance(dataset, str) or not isinstance(normalization, str):
        raise ValueError("Cached OCR run has invalid benchmark metadata")
    for label, run in (("OSD", osd_run), ("view-score", score_run)):
        if run.get("dataset") != dataset or run.get("normalization") != normalization:
            raise ValueError(f"{label} benchmark metadata differs")
    for angle in ANGLES[1:]:
        run = angle_runs[angle]
        if run.get("dataset") != dataset or run.get("normalization") != normalization:
            raise ValueError(f"Cached OCR metadata differs for angle {angle}")
    return dataset, normalization


def _validate_case_identity(
    case_id: str,
    candidate: Mapping[str, object],
    expected: Mapping[str, object],
    label: str,
) -> None:
    if candidate.get("reference") != expected.get("reference"):
        raise ValueError(f"{label} reference differs for {case_id!r}")
    if candidate.get("cluster_id") != expected.get("cluster_id"):
        raise ValueError(f"{label} cluster ID differs for {case_id!r}")


def _osd_angle(case: Mapping[str, object], case_id: str) -> tuple[int, bool]:
    selection = case.get("orientation_selection")
    if not isinstance(selection, dict):
        raise ValueError(f"OSD case {case_id!r} has no orientation selection")
    osd = selection.get("osd")
    if osd is None:
        return 0, False
    if not isinstance(osd, dict):
        raise ValueError(f"OSD case {case_id!r} has invalid OSD evidence")
    angle = osd.get("angle")
    if not isinstance(angle, int) or isinstance(angle, bool) or angle not in ANGLES:
        raise ValueError(f"OSD case {case_id!r} has an invalid angle")
    return angle, True


def _candidate_scores(
    case: Mapping[str, object],
    angles: list[int],
    case_id: str,
) -> tuple[dict[int, Mapping[str, object]], dict[str, object]]:
    selection = case.get("orientation_selection")
    if not isinstance(selection, dict):
        raise ValueError(f"View-score case {case_id!r} has no orientation selection")
    view_scores = selection.get("view_scores")
    view_failures = selection.get("view_failures")
    if not isinstance(view_scores, dict) or not isinstance(view_failures, dict):
        raise ValueError(f"View-score case {case_id!r} has invalid view evidence")
    scores = {}
    failures = {}
    for angle in angles:
        score = view_scores.get(str(angle))
        if isinstance(score, dict):
            _rank(score)
            scores[angle] = score
            continue
        failure = view_failures.get(str(angle))
        if isinstance(failure, dict):
            failures[str(angle)] = failure
            continue
        raise ValueError(
            f"View-score case {case_id!r} has no evidence for angle {angle}"
        )
    if not scores:
        raise ValueError(f"No candidate orientation view succeeded for {case_id!r}")
    return scores, failures


def _rank(score: Mapping[str, object]) -> tuple[int, float, float, float]:
    supporting_words = _number(score, "supporting_words")
    return (
        int(supporting_words >= MIN_SUPPORTING_WORDS),
        _number(score, "selection_value"),
        _number(score, "confidence_evidence"),
        _number(score, "characters"),
    )


def _number(score: Mapping[str, object], field: str) -> float:
    value = score.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"Orientation score has invalid {field}")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
