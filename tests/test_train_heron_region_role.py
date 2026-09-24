from __future__ import annotations

import json
import math
import sys

import pytest

from experiments import serve_falcon_layout
from experiments import train_heron_region_role as trainer


def _record(
    group: str,
    label: str,
    *,
    split: str = "train",
    label_source: str = "agent_draft",
    physical_group: str | None = None,
) -> dict:
    features = {name: 0.0 for name in trainer.HERON_REGION_ROLE_FEATURE_NAMES}
    features[f"class_score.{'form' if label == 'add_form_source' else 'table'}"] = 0.9
    return {
        "candidate_id": f"candidate-{group}-{label}",
        "physical_group": physical_group or f"physical-{group}-{label}",
        "source_group": group,
        "split": split,
        "label": label,
        "label_source": label_source,
        "feature_values": features,
    }


def _train_rows(label_source: str = "agent_draft") -> list[dict]:
    return [
        _record("table-a", "table_only", label_source=label_source),
        _record("table-b", "table_only", label_source=label_source),
        _record("form-a", "add_form_source", label_source=label_source),
        _record("form-b", "add_form_source", label_source=label_source),
    ]


def test_exact_physical_duplicates_are_collapsed() -> None:
    rows = _train_rows()
    duplicate = dict(rows[0], candidate_id="duplicate-candidate")

    records = trainer._validated_records([*rows, duplicate])

    assert len(records) == len(rows)


def test_conflicting_physical_labels_are_rejected() -> None:
    rows = _train_rows()
    conflict = _record(
        "table-a",
        "add_form_source",
        physical_group=rows[0]["physical_group"],
    )

    with pytest.raises(ValueError, match="Conflicting labels"):
        trainer._validated_records([*rows, conflict])


def test_each_class_requires_two_training_source_groups() -> None:
    rows = _train_rows()
    rows[3]["source_group"] = rows[2]["source_group"]

    with pytest.raises(ValueError, match="two source groups"):
        trainer._build_model(rows, "test", {"version": 1})


def test_agent_labels_keep_artifact_in_development() -> None:
    rows = [
        *_train_rows(),
        _record("table-cal", "table_only", split="calibration"),
        _record("form-cal", "add_form_source", split="calibration"),
    ]

    model = trainer._build_model(rows, "test", {"version": 1})

    assert model["status"] == "development_only"
    assert set(model["development"]["logo"]) == {
        "average_precision",
        "precision",
        "recall",
    }


def test_cli_exports_dependency_free_coefficient_schema(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        *_train_rows("human_review"),
        _record(
            "table-cal",
            "table_only",
            split="calibration",
            label_source="human_review",
        ),
        _record(
            "form-cal",
            "add_form_source",
            split="calibration",
            label_source="human_review",
        ),
    ]
    records = tmp_path / "records.jsonl"
    output = tmp_path / "model.json"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_heron_region_role.py",
            "--records",
            str(records),
            "--output",
            str(output),
            "--model-id",
            "role-test",
        ],
    )

    assert trainer.main() == 0
    model = json.loads(output.read_text())
    size = len(trainer.HERON_REGION_ROLE_FEATURE_NAMES)
    assert model["status"] == "calibrated"
    assert len(model["coefficients"]) == size
    assert len(model["means"]) == size
    assert len(model["scales"]) == size
    assert all(math.isfinite(value) for value in model["coefficients"])
    assert all(scale > 0 for scale in model["scales"])
    assert math.isfinite(model["intercept"])
    assert 0 <= model["threshold"] <= 1
    assert model["extractor"]["version"] == trainer.ROUTE_EXTRACTOR_VERSION
    assert serve_falcon_layout.load_heron_region_role_model(output) == model
