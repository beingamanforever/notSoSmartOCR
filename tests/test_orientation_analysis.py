from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import orientation_analysis


def _case(
    case_id: str,
    template: str,
    cer_edits: int,
    wer_edits: int,
    *,
    angle: int | None = None,
    margin: float | None = None,
) -> dict[str, object]:
    case: dict[str, object] = {
        "id": case_id,
        "cluster_id": template,
        "reference": f"reference {case_id}",
        "metrics": {
            "cer": {"edits": cer_edits, "reference_units": 100, "rate": 9.0},
            "wer": {"edits": wer_edits, "reference_units": 20, "rate": 9.0},
        },
    }
    if angle is not None and margin is not None:
        case["orientation_selection"] = {
            "angle": angle,
            "score_margin": margin,
        }
    return case


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    automatic_cases = [
        _case("page-b", "2", 30, 5, angle=90, margin=0.005),
        _case("page-a", "1", 5, 2, angle=0, margin=0.02),
    ]
    scores = {
        "0": [(20, 4), (5, 2)],
        "90": [(10, 3), (5, 3)],
        "180": [(10, 2), (40, 8)],
        "270": [(50, 9), (50, 9)],
    }
    four_angle_runs = {}
    for angle, values in scores.items():
        four_angle_runs[angle] = {
            "normalization": "test normalization",
            "cases": [
                _case("page-b", "2", values[0][0], values[0][1]),
                _case("page-a", "1", values[1][0], values[1][1]),
            ],
        }
    return (
        {
            "runs": {
                "auto": {
                    "normalization": "test normalization",
                    "cases": automatic_cases,
                }
            }
        },
        {"runs": four_angle_runs},
    )


def test_analysis_uses_non_deployable_cer_oracle_and_summed_metrics() -> None:
    automatic, four_angle = _payloads()

    report = orientation_analysis.analyze_payloads(automatic, four_angle)

    assert report["oracle"]["deployable"] is False
    assert report["aggregate"]["automatic"]["cer"] == {
        "edits": 35,
        "reference_units": 200,
        "rate": 0.175,
        "method": "summed_edits_over_reference_units",
    }
    assert report["aggregate"]["oracle"]["cer"]["edits"] == 15
    assert report["aggregate"]["oracle"]["wer"]["edits"] == 5
    assert report["pages"][0]["id"] == "page-a"
    assert report["pages"][0]["oracle_angles"] == ["0", "90"]
    assert report["pages"][0]["selected_oracle_angle"] == "0"


def test_analysis_accepts_tied_oracle_angles_and_reports_margin_association() -> None:
    automatic, four_angle = _payloads()

    report = orientation_analysis.analyze_payloads(
        automatic,
        four_angle,
        catastrophic_regret_threshold=0.15,
        low_margin_threshold=0.01,
    )

    assert report["angle_selection"] == {
        "exact_oracle_angle_selection_count": 2,
        "exact_oracle_angle_selection_rate": 1.0,
        "ties_accepted": True,
        "oracle_angle_counts": {"0": 1, "90": 2, "180": 1, "270": 0},
        "pages_with_oracle_ties": 2,
    }
    assert report["cer_regret"]["catastrophic_count"] == 1
    assert report["low_margin_association"]["low_margin"] == {
        "cases": 1,
        "exact_oracle_angle_selection_count": 1,
        "exact_oracle_angle_selection_rate": 1.0,
        "mean_cer_regret": 0.2,
        "catastrophic_regret_count": 1,
    }
    assert report["templates"]["1"]["exact_oracle_angle_selection_count"] == 1
    assert report["templates"]["2"]["maximum_cer_regret"] == 0.2


def test_analysis_rejects_mismatched_references_and_normalization() -> None:
    automatic, four_angle = _payloads()
    four_angle["runs"]["90"]["normalization"] = "different"

    with pytest.raises(ValueError, match="Normalization differs"):
        orientation_analysis.analyze_payloads(automatic, four_angle)

    automatic, four_angle = _payloads()
    four_angle["runs"]["90"]["cases"][0]["reference"] = "different"

    with pytest.raises(ValueError, match="References differ"):
        orientation_analysis.analyze_payloads(automatic, four_angle)


def test_cli_writes_analysis_json(tmp_path: Path) -> None:
    automatic, four_angle = _payloads()
    automatic_path = tmp_path / "automatic.json"
    four_angle_path = tmp_path / "four-angle.json"
    output_path = tmp_path / "nested" / "analysis.json"
    automatic_path.write_text(json.dumps(automatic), encoding="utf-8")
    four_angle_path.write_text(json.dumps(four_angle), encoding="utf-8")

    exit_code = orientation_analysis.main(
        [str(automatic_path), str(four_angle_path), str(output_path)]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["sources"] == {
        "automatic": str(automatic_path),
        "four_angle": str(four_angle_path),
    }
    assert report["case_count"] == 2
