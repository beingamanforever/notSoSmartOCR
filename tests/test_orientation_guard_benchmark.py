from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import orientation_guard_benchmark
from experiments.public_benchmark import NORMALIZATION, _score


def test_cached_guard_reproduces_selection_and_failure_inclusive_metrics() -> None:
    doctr, osd, scores, four_angle = _payloads()

    result = orientation_guard_benchmark.compose_payloads(
        doctr,
        osd,
        scores,
        four_angle,
    )

    assert [case["prediction"] for case in result["cases"]] == [
        "wrong",
        "alpha",
        "charlie",
    ]
    selections = [case["orientation_selection"] for case in result["cases"]]
    assert [selection["selected_angle"] for selection in selections] == [90, 0, 180]
    assert [selection["selection_reason"] for selection in selections] == [
        "confidence_rank",
        "coverage_recovery",
        "confidence_rank",
    ]
    assert selections[2]["osd_angle"] == 0
    assert selections[2]["osd_available"] is False
    assert result["summary"]["failed_cases"] == 1
    assert result["summary"]["failure_codes"] == {"cached_failure": 1}
    assert result["summary"]["cer"] == {
        "case_mean": 0.266667,
        "micro": 0.235294,
        "edits": 4,
        "reference_units": 17,
    }
    assert result["summary"]["wer"] == {
        "case_mean": 0.333333,
        "micro": 0.333333,
        "edits": 1,
        "reference_units": 3,
    }
    assert result["selection_summary"] == {
        "cases": 3,
        "doctr_osd_agreement_count": 0,
        "osd_fallback_zero_count": 1,
        "coverage_recovery_count": 1,
        "exact_oracle_angle_selection_count": 2,
        "exact_oracle_angle_selection_rate": 0.666667,
        "mean_cer_regret": 0.266667,
        "maximum_cer_regret": 0.8,
        "catastrophic_regret_threshold": 0.2,
        "catastrophic_regret_count": 1,
        "oracle_is_deployable": False,
    }


def test_selector_requires_both_coverage_and_near_tied_confidence() -> None:
    broad = _view_score(mean=0.88, characters=250)
    narrow = _view_score(mean=0.92, characters=100)

    assert orientation_guard_benchmark.select_angle({0: broad, 90: narrow}) == (
        90,
        "confidence_rank",
    )
    broad["mean_confidence"] = 0.89
    broad["selection_value"] = 0.89
    assert orientation_guard_benchmark.select_angle({0: broad, 90: narrow}) == (
        0,
        "coverage_recovery",
    )


def test_main_writes_reproducible_cached_result(tmp_path: Path) -> None:
    payloads = _payloads()
    inputs = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"input-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        inputs.append(path)
    output = tmp_path / "result.json"

    assert (
        orientation_guard_benchmark.main([*(str(path) for path in inputs), str(output)])
        == 0
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["experiment"] == "cached_two_view_orientation_guard"
    assert result["sources"] == {
        "doctr": str(inputs[0]),
        "osd": str(inputs[1]),
        "view_scores": str(inputs[2]),
        "four_angle": str(inputs[3]),
    }
    assert result["selection_summary"]["cases"] == 3


def test_rejects_mismatched_cached_case_ids() -> None:
    doctr, osd, scores, four_angle = _payloads()
    scores["runs"]["auto"]["cases"].pop()

    with pytest.raises(
        ValueError, match="full automatic run and cached OCR case IDs differ"
    ):
        orientation_guard_benchmark.compose_payloads(
            doctr,
            osd,
            scores,
            four_angle,
        )


def _payloads() -> tuple[dict[str, object], ...]:
    case_ids = ("coverage", "confidence", "no-osd")
    references = {"coverage": "alpha", "confidence": "bravo", "no-osd": "charlie"}
    doctr = {
        "cases": [
            {"id": "coverage", "lossless_rotation": 90, "confidence": 0.9},
            {"id": "confidence", "lossless_rotation": 90, "confidence": 0.9},
            {"id": "no-osd", "lossless_rotation": 180, "confidence": 0.9},
        ]
    }
    osd_cases = []
    score_cases = []
    scores = {
        "coverage": {
            "0": _view_score(mean=0.90, characters=250),
            "90": _view_score(mean=0.92, characters=100),
        },
        "confidence": {
            "0": _view_score(mean=0.87, characters=300),
            "90": _view_score(mean=0.92, characters=100),
        },
        "no-osd": {
            "0": _view_score(mean=0.80, characters=100),
            "180": _view_score(mean=0.92, characters=120),
        },
    }
    for case_id in case_ids:
        identity = {
            "id": case_id,
            "cluster_id": case_id,
            "reference": references[case_id],
        }
        osd_selection = {} if case_id == "no-osd" else {"osd": {"angle": 0}}
        osd_cases.append({**identity, "orientation_selection": osd_selection})
        score_cases.append(
            {
                **identity,
                "orientation_selection": {
                    "view_scores": scores[case_id],
                    "view_failures": {},
                },
            }
        )
    auto_metadata = {"dataset": "clinocr", "normalization": NORMALIZATION}
    osd = {"runs": {"auto": {**auto_metadata, "cases": osd_cases}}}
    view_scores = {"runs": {"auto": {**auto_metadata, "cases": score_cases}}}

    predictions = {
        0: {"coverage": "alpha", "confidence": "bravo", "no-osd": "wrongxx"},
        90: {"coverage": "wrong", "confidence": "wrong", "no-osd": "wrongxx"},
        180: {"coverage": "wrong", "confidence": "bravo", "no-osd": "charlie"},
        270: {"coverage": "wrong", "confidence": "bravo", "no-osd": "wrongxx"},
    }
    runs = {}
    for angle in orientation_guard_benchmark.ANGLES:
        cases = []
        for case_id in case_ids:
            prediction = predictions[angle][case_id]
            status = "failed" if case_id == "confidence" and angle == 90 else "success"
            failures = [{"code": "cached_failure"}] if status == "failed" else []
            cases.append(
                {
                    "id": case_id,
                    "cluster_id": case_id,
                    "subset": "rotated",
                    "reference": references[case_id],
                    "prediction": prediction,
                    "status": status,
                    "failures": failures,
                    "metrics": _score(prediction, references[case_id]),
                }
            )
        runs[str(angle)] = {**auto_metadata, "cases": cases}
    return doctr, osd, view_scores, {"runs": runs}


def _view_score(*, mean: float, characters: int) -> dict[str, object]:
    return {
        "selection_value": mean,
        "confidence_evidence": 20.0,
        "mean_confidence": mean,
        "characters": characters,
        "supporting_words": 20,
    }
