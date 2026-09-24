import json
from pathlib import Path

import pytest

from experiments.build_falcon_mode_observations import build_observations
from experiments.serve_falcon_layout import HERON_ROUTE_FEATURE_NAMES


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_inputs(tmp_path: Path):
    features = tmp_path / "features.json"
    package = tmp_path / "package"
    plain = tmp_path / "plain"
    names = list(HERON_ROUTE_FEATURE_NAMES)
    write_json(
        features,
        [
            {
                "id": "doc-b-p001",
                "group": "doc-b",
                "features": [float(index + 1) for index in range(len(names))],
                "layout_exact": 999,
                "plain_exact": 999,
            },
            {
                "id": "doc-a-p001",
                "group": "doc-a",
                "features": [float(index) for index in range(len(names))],
                "layout_exact": 999,
                "plain_exact": 999,
            },
        ],
    )
    references = [
        {
            "id": "doc-a-p001",
            "file": "a.pdf",
            "page": 1,
            "references": [{"row": 1, "text": "Dose 5 mg", "scorable": True}],
        },
        {
            "id": "doc-b-p001",
            "file": "b.pdf",
            "page": 1,
            "references": [{"row": 2, "text": "Checked", "scorable": True}],
        },
    ]
    write_json(package / "ground-truth" / "references.json", references)
    write_json(
        package / "evaluation" / "current-report.json",
        {
            "pages": [
                {
                    "id": "doc-b-p001",
                    "arms": {"pipeline": {"status": "failed", "cost_usd": None}},
                },
                {
                    "id": "doc-a-p001",
                    "arms": {"pipeline": {"status": "success", "cost_usd": 0.01}},
                },
            ]
        },
    )
    (package / "saved-responses" / "doc-a-p001").mkdir(parents=True)
    (package / "saved-responses" / "doc-a-p001" / "notSoSmartOCR.md").write_text(
        "Dose 5 mg"
    )
    (package / "saved-responses" / "doc-b-p001").mkdir(parents=True)
    (package / "saved-responses" / "doc-b-p001" / "notSoSmartOCR.md").write_text(
        "Unchecked"
    )
    write_json(
        plain / "doc-a-p001" / "record.json",
        {"id": "doc-a-p001", "text": "Dose 4 mg"},
    )
    write_json(
        plain / "doc-b-p001" / "record.json",
        {"id": "doc-b-p001", "text": "", "error": "reader failed"},
    )
    return features, package, plain


def test_builds_deterministic_failure_inclusive_observations(tmp_path):
    features, package, plain = make_inputs(tmp_path)

    rows = build_observations(features, package, plain, unknown_layout_cost_usd=0.03)

    assert [row["id"] for row in rows] == ["doc-a-p001", "doc-b-p001"]
    assert rows[0]["layout_quality"] == 1
    assert rows[0]["plain_quality"] == 0
    assert rows[0]["layout_expert_cost"] == 0.01
    assert rows[0]["layout_expert_cost_observed"] is True
    assert rows[0]["layout_expert_cost_source"] == "observed"
    assert rows[1]["layout_quality"] == 0
    assert rows[1]["plain_quality"] == 0
    assert rows[1]["layout_expert_cost"] == 0.03
    assert rows[1]["layout_expert_cost_observed"] is False
    assert rows[1]["layout_expert_cost_source"] == "explicit_unknown_cost_bound"
    assert rows[1]["source_group"] == "doc-b"
    assert list(rows[0]["feature_values"]) == list(HERON_ROUTE_FEATURE_NAMES)
    assert set(rows[0]) == {
        "id",
        "split",
        "source_group",
        "feature_values",
        "layout_quality",
        "plain_quality",
        "layout_expert_cost",
        "layout_expert_cost_observed",
        "layout_expert_cost_source",
        "plain_expert_cost",
    }


def test_rejects_mismatched_page_sets(tmp_path):
    features, package, plain = make_inputs(tmp_path)
    (plain / "doc-b-p001" / "record.json").unlink()

    with pytest.raises(ValueError, match="plain_outputs page mismatch"):
        build_observations(features, package, plain)


def test_requires_explicit_unknown_cost_bound(tmp_path):
    features, package, plain = make_inputs(tmp_path)

    with pytest.raises(ValueError, match="--unknown-layout-cost-usd"):
        build_observations(features, package, plain)
