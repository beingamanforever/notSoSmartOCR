from __future__ import annotations

import pytest

from experiments.consensus_risk_benchmark import benchmark_risk
from experiments.public_benchmark import NORMALIZATION


def test_consensus_risk_ranks_disagreement_without_selecting_text() -> None:
    primary = _run("primary", ("alpha", "WRONG"))
    first = _run("first", ("alpha", "right"))
    second = _run("second", ("alpha", "right"))

    report = benchmark_risk(primary, [first, second], severe_cer=0.2)

    assert report["selection_policy"] == "none"
    assert report["summary"]["primary"]["severe_errors"] == 1
    consensus = report["summary"]["signals"]["consensus"]
    assert consensus["severe_error_auroc"] == 1.0
    assert consensus["augrc_severe_error"] == 0.25
    assert consensus["cer_rank_concordance"] == 1.0
    assert consensus["risk_at_coverage"]["0.5"]["case_mean_cer"] == 0.0
    assert report["cases"][0]["candidate_scores"] == {
        "primary": 0.0,
        "first": 0.0,
        "second": 0.0,
    }


def test_consensus_risk_rejects_mismatched_case_panels() -> None:
    primary = _run("primary", ("alpha", "beta"))
    alternative = _run("alternative", ("alpha",))

    with pytest.raises(ValueError, match="case IDs"):
        benchmark_risk(primary, [alternative])


def test_consensus_risk_accepts_an_alternative_superset() -> None:
    primary = _run("primary", ("alpha",))
    alternative = _run("alternative", ("alpha", "right"))

    report = benchmark_risk(primary, [alternative])

    assert report["matched_panel"] == {
        "cases": 1,
        "runs": [
            {"reader": "primary", "source_cases": 1, "ignored_extra_cases": 0},
            {
                "reader": "alternative",
                "source_cases": 2,
                "ignored_extra_cases": 1,
            },
        ],
    }


def _run(reader: str, predictions: tuple[str, ...]) -> dict[str, object]:
    references = ("alpha", "right")
    return {
        "dataset": "clinocr",
        "normalization": NORMALIZATION,
        "reader": reader,
        "cases": [
            {
                "id": f"case-{index}",
                "cluster_id": f"template-{index}",
                "prediction": prediction,
                "reference": references[index],
            }
            for index, prediction in enumerate(predictions)
        ],
    }
