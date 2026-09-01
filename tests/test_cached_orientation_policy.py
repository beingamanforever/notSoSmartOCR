from __future__ import annotations

import pytest

from experiments import cached_orientation_policy
from experiments.public_benchmark import NORMALIZATION, _score


def test_composes_predictions_with_cached_ocr_views() -> None:
    predictions, runs = _payloads()

    result = cached_orientation_policy.compose_payloads(
        predictions,
        {"runs": runs},
    )

    assert result["summary"]["cases"] == 2
    assert "latency_ms" not in result["summary"]
    assert result["selection_summary"] == {
        "cases": 2,
        "exact_oracle_angle_selection_count": 1,
        "exact_oracle_angle_selection_rate": 0.5,
        "mean_cer_regret": 0.5,
        "maximum_cer_regret": 1.0,
        "catastrophic_regret_threshold": 0.2,
        "catastrophic_regret_count": 1,
        "oracle_is_deployable": False,
    }
    assert result["cases"][0]["prediction"] == "alpha"
    assert result["cases"][1]["prediction"] == "yyyy"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference", "different", "reference differs"),
        ("cluster_id", "other-cluster", "cluster ID differs"),
    ],
)
def test_rejects_cross_angle_case_identity_mismatch(
    field: str,
    value: str,
    message: str,
) -> None:
    predictions, runs = _payloads()
    runs["90"]["cases"][0][field] = value

    with pytest.raises(ValueError, match=message):
        cached_orientation_policy.compose_payloads(predictions, {"runs": runs})


def test_recomputes_cached_metrics_from_text_and_retains_failures() -> None:
    predictions, runs = _payloads()
    selected = runs["270"]["cases"][1]
    selected["metrics"] = _score("beta", "beta")
    selected["status"] = "failed"
    selected["failures"] = [{"code": "cached_failure"}]

    result = cached_orientation_policy.compose_payloads(predictions, {"runs": runs})

    assert result["summary"]["cer"] == {
        "case_mean": 0.5,
        "micro": 0.444444,
        "edits": 4,
        "reference_units": 9,
    }
    assert result["summary"]["wer"] == {
        "case_mean": 0.5,
        "micro": 0.5,
        "edits": 1,
        "reference_units": 2,
    }
    assert result["selection_summary"]["exact_oracle_angle_selection_count"] == 1
    assert result["selection_summary"]["mean_cer_regret"] == 0.5
    assert result["summary"]["failed_cases"] == 1
    assert result["summary"]["failure_codes"] == {"cached_failure": 1}
    assert result["cases"][1]["failures"] == [{"code": "cached_failure"}]


def test_rejects_unknown_prediction_angle() -> None:
    predictions = {"cases": [{"id": "page", "lossless_rotation": 45}]}

    try:
        cached_orientation_policy.compose_payloads(predictions, {"runs": {}})
    except ValueError as error:
        assert "Invalid orientation prediction" in str(error)
    else:
        raise AssertionError("invalid orientation angle was accepted")


def _payloads() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    predictions: dict[str, object] = {
        "cases": [
            {"id": "page-a", "lossless_rotation": 0, "confidence": 0.9},
            {"id": "page-b", "lossless_rotation": 270, "confidence": 0.8},
        ]
    }
    references = {"page-a": "alpha", "page-b": "beta"}
    text = {
        0: {"page-a": "alpha", "page-b": "xxxx"},
        90: {"page-a": "xxxxx", "page-b": "beta"},
        180: {"page-a": "xxxxx", "page-b": "xxxx"},
        270: {"page-a": "xxxxx", "page-b": "yyyy"},
    }
    runs: dict[str, dict[str, object]] = {}
    for angle in cached_orientation_policy.ANGLES:
        cases = []
        for case_id, reference in references.items():
            prediction = text[angle][case_id]
            cases.append(
                {
                    "id": case_id,
                    "cluster_id": case_id,
                    "subset": "rotated",
                    "reference": reference,
                    "prediction": prediction,
                    "status": "success",
                    "failures": [],
                    "latency_ms": 1.0,
                    "metrics": _score(prediction, reference),
                }
            )
        runs[str(angle)] = {
            "dataset": "clinocr",
            "normalization": NORMALIZATION,
            "cases": cases,
        }
    return predictions, runs
