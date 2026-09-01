from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import pulsebench_benchmark as pulse


BBOX = [0.1, 0.1, 0.3, 0.1, 0.3, 0.2, 0.1, 0.2]


def test_full_panel_scores_485_perfect_cases_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _dataset_rows()
    predictions = _prediction_rows(rows)
    dataset_path = _write_json(tmp_path / "dataset.json", _dataset(rows))
    prediction_path = _write_json(
        tmp_path / "predictions.json", _predictions(predictions, "full")
    )
    output_path = tmp_path / "result.json"
    scorer_root = tmp_path / "official-scorer"
    monkeypatch.setattr(
        pulse,
        "load_official_scorer",
        lambda root: _fake_official_score if root == scorer_root else None,
    )

    assert (
        pulse.main(
            [
                str(dataset_path),
                str(prediction_path),
                str(output_path),
                "--role",
                "full",
                "--scorer-root",
                str(scorer_root),
            ]
        )
        == 0
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["summary"]["cases"] == 485
    assert result["summary"]["status_counts"] == {"success": 485}
    assert result["summary"]["latency_ms"] == {
        "observed_cases": 485,
        "missing_cases": 0,
        "p50": 242.0,
        "p95": 459.8,
    }
    assert result["official_selection_f1"]["values"]["f1"] == 1
    controls = result["project_defined_controls"]
    assert controls["control_to_label_association"]["f1"] == 1
    assert controls["control_to_label_association"]["precision_denominator"] == 944
    assert controls["control_to_label_association"]["recall_denominator"] == 944
    assert controls["state"]["macro_f1"] == 1
    assert controls["reference_diagnostics"] == {
        "candidate_source": "recursive public annotations, including nested table cells",
        "selected_count_mismatch_cases": [],
    }
    assert result["case_ids"] == sorted(result["case_ids"])
    assert len(result["case_ids"]) == len(set(result["case_ids"]))

    with pytest.raises(SystemExit):
        pulse.main(
            [
                str(dataset_path),
                str(prediction_path),
                str(output_path),
                "--role",
                "full",
                "--scorer-root",
                str(scorer_root),
            ]
        )


def test_custom_dev_and_eval_panels_are_deterministic_and_stratified() -> None:
    rows = _dataset_rows()

    dev, dev_ids = pulse._select_panel(rows, "dev")
    second_dev, second_ids = pulse._select_panel(list(reversed(rows)), "dev")
    evaluation, evaluation_ids = pulse._select_panel(rows, "eval")

    assert dev_ids == second_ids
    assert [_id(row) for row in dev] == [_id(row) for row in second_dev]
    assert len(dev_ids) == 60
    assert sum(int(row["selected_count"]) > 0 for row in dev) == 56
    assert sum(int(row["selected_count"]) == 0 for row in dev) == 4
    assert len(evaluation_ids) == 425
    assert set(dev_ids).isdisjoint(evaluation_ids)
    assert set(dev_ids) | set(evaluation_ids) == {_id(row) for row in rows}


def test_dev_run_does_not_parse_held_out_eval_ground_truth(tmp_path: Path) -> None:
    rows = _dataset_rows()
    dev, dev_ids = pulse._select_panel(rows, "dev")
    _, evaluation_ids = pulse._select_panel(rows, "eval")
    eval_row = next(row for row in rows if _id(row) == evaluation_ids[0])
    eval_row["ground_truth"] = "held-out and deliberately unreadable"

    result = pulse.run_benchmark(
        _write_json(tmp_path / "dataset.json", _dataset(rows)),
        _write_json(
            tmp_path / "predictions.json",
            _predictions(_prediction_rows(dev), "dev"),
        ),
        _fake_official_score,
        role="dev",
    )

    assert result["case_ids"] == dev_ids
    assert result["summary"]["cases"] == 60
    assert result["official_selection_f1"]["values"]["f1"] == 1


def test_missing_predictions_remain_in_failure_inclusive_denominator(
    tmp_path: Path,
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    supplied = _prediction_rows(dev)[30:]
    supplied[0]["status"] = "abstained"
    supplied[0]["items"] = []
    supplied[0]["failures"] = [
        {
            "code": "low_confidence",
            "message": "Control extraction confidence was below policy",
            "stage": "extraction",
        }
    ]

    result = pulse.run_benchmark(
        _write_json(tmp_path / "dataset.json", _dataset(rows)),
        _write_json(tmp_path / "predictions.json", _predictions(supplied, "dev")),
        _fake_official_score,
        role="dev",
    )

    assert result["summary"]["cases"] == 60
    assert result["summary"]["failed_cases"] == 30
    assert result["summary"]["abstained_cases"] == 1
    assert result["summary"]["failure_codes"] == {
        "low_confidence": 1,
        "missing_prediction": 30,
    }
    assert result["summary"]["latency_ms"]["missing_cases"] == 30
    assert result["official_selection_f1"]["values"]["f1"] < 1
    controls = result["project_defined_controls"]
    assert controls["control_to_label_association"]["f1"] < 1
    assert controls["state"]["macro_f1"] < 1


def test_failed_prediction_items_are_rejected_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    predictions = _prediction_rows(dev)
    failed = predictions[0]
    failed["status"] = "failed"
    failed["failures"] = [{"code": "provider_error", "message": "request failed"}]
    output_path = tmp_path / "result.json"
    scorer_root = tmp_path / "official-scorer"
    monkeypatch.setattr(
        pulse, "load_official_scorer", lambda root: _fake_official_score
    )

    with pytest.raises(SystemExit):
        pulse.main(
            [
                str(_write_json(tmp_path / "dataset.json", _dataset(rows))),
                str(
                    _write_json(
                        tmp_path / "predictions.json",
                        _predictions(predictions, "dev"),
                    )
                ),
                str(output_path),
                "--role",
                "dev",
                "--scorer-root",
                str(scorer_root),
            ]
        )

    assert not output_path.exists()
    assert (
        f"Failed prediction {failed['sample_id']} must not contain items"
        in capsys.readouterr().err
    )


def test_jsonl_inputs_and_broken_states_use_project_metric_only(
    tmp_path: Path,
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    predictions = _prediction_rows(dev)
    predictions[0]["items"][0]["selected"] = not predictions[0]["items"][0]["selected"]

    result = pulse.run_benchmark(
        _write_jsonl(tmp_path / "dataset.jsonl", _dataset(rows)),
        _write_jsonl(tmp_path / "predictions.jsonl", _predictions(predictions, "dev")),
        _fake_official_score,
        role="dev",
    )

    assert result["summary"]["cases"] == 60
    assert result["project_defined_controls"]["state"]["macro_f1"] < 1
    assert result["project_defined_controls"]["control_to_label_association"]["f1"] == 1
    assert "not state macro-F1" in result["official_selection_f1"]["note"]


def test_direct_metrics_are_withheld_without_explicit_public_association(
    tmp_path: Path,
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    predictions = _prediction_rows(dev)
    first = dev[0]
    ground_truth = first["ground_truth"]
    assert isinstance(ground_truth, dict)
    annotations = ground_truth["annotations"]
    assert isinstance(annotations, list)
    annotations[0].pop("content")

    result = pulse.run_benchmark(
        _write_json(tmp_path / "dataset.json", _dataset(rows)),
        _write_json(
            tmp_path / "predictions.json",
            _predictions(predictions, "dev"),
        ),
        _fake_official_score,
        role="dev",
    )

    assert result["project_defined_controls"] == {
        "status": "not_computed",
        "reason": (
            "Selection candidates do not directly pair label content and geometry"
        ),
        "provenance": "project_defined_not_official_selection_f1",
    }


def test_nested_table_candidates_are_included_in_direct_metrics(tmp_path: Path) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    first = dev[0]
    ground_truth = first["ground_truth"]
    assert isinstance(ground_truth, dict)
    annotations = ground_truth["annotations"]
    assert isinstance(annotations, list)
    nested = {
        "row": 0,
        "col": 0,
        "text": "Nested choice",
        "bbox": BBOX,
        "page": 1,
        "selected": False,
        "selection_candidate": True,
    }
    annotations.append({"table": {"cells": [nested]}})
    first["selection_candidate_count"] = int(first["selection_candidate_count"]) + 1
    predictions = _prediction_rows(dev)
    first_prediction = next(
        row for row in predictions if row["sample_id"] == first["sample_id"]
    )
    first_prediction["items"].append(
        {
            "page": 1,
            "bbox": BBOX,
            "content": "Nested choice",
            "selected": False,
        }
    )

    result = pulse.run_benchmark(
        _write_json(tmp_path / "dataset.json", _dataset(rows)),
        _write_json(tmp_path / "predictions.json", _predictions(predictions, "dev")),
        _fake_official_score,
        role="dev",
    )

    controls = result["project_defined_controls"]
    assert controls["status"] == "computed"
    assert controls["control_to_label_association"]["f1"] == 1


@pytest.mark.parametrize(
    ("metadata_key", "bad_value", "message"),
    [
        ("dataset_revision", "wrong", "Prediction dataset revision"),
        ("scorer_revision", "wrong", "Prediction scorer revision"),
        ("role", "eval", "Prediction role"),
    ],
)
def test_prediction_metadata_must_match_pins_and_role(
    tmp_path: Path, metadata_key: str, bad_value: str, message: str
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    prediction_payload = _predictions(_prediction_rows(dev), "dev")
    prediction_payload[metadata_key] = bad_value

    with pytest.raises(ValueError, match=message):
        pulse.run_benchmark(
            _write_json(tmp_path / "dataset.json", _dataset(rows)),
            _write_json(tmp_path / "predictions.json", prediction_payload),
            _fake_official_score,
            role="dev",
        )


def test_dataset_revision_and_scorer_checkout_must_match_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    dataset = _dataset(rows)
    dataset["revision"] = "wrong"
    with pytest.raises(ValueError, match="Dataset export revision"):
        pulse.run_benchmark(
            _write_json(tmp_path / "dataset.json", dataset),
            _write_json(
                tmp_path / "predictions.json",
                _predictions(_prediction_rows(dev), "dev"),
            ),
            _fake_official_score,
            role="dev",
        )

    monkeypatch.setattr(pulse, "_git_revision", lambda root: "wrong")
    with pytest.raises(ValueError, match="must be checked out at revision"):
        pulse.load_official_scorer(tmp_path)


def test_pinned_official_scorer_is_loaded_through_its_public_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "result.py").write_text(
        """
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    page: int
    bbox: list[float]
    selected: bool
    content: str

@dataclass
class MetricResult:
    f1: float
    cases: int
""",
        encoding="utf-8",
    )
    (tmp_path / "metrics.py").write_text(
        """
from result import MetricResult

def compute_metrics(predictions, ground_truth, provider_name, latency_seconds):
    assert provider_name == "not-so-smart-ocr"
    assert latency_seconds == 0.25
    assert predictions["case"][0].content == "checked"
    assert ground_truth["case"][0].selected is True
    return MetricResult(f1=1.0, cases=len(ground_truth))
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(pulse, "_git_revision", lambda root: pulse.SCORER_REVISION)

    scorer = pulse.load_official_scorer(tmp_path)
    control = pulse.Control(1, tuple(BBOX), "checked", True)

    assert scorer({"case": [control]}, {"case": [control]}, 0.25) == {
        "f1": 1.0,
        "cases": 1,
    }


def test_predictions_outside_role_are_rejected_before_item_parsing(
    tmp_path: Path,
) -> None:
    rows = _dataset_rows()
    dev, _ = pulse._select_panel(rows, "dev")
    _, evaluation_ids = pulse._select_panel(rows, "eval")
    outside = {
        "sample_id": evaluation_ids[0],
        "ground_truth": "must not be parsed as a prediction",
    }

    with pytest.raises(ValueError, match="outside the dev panel"):
        pulse.run_benchmark(
            _write_json(tmp_path / "dataset.json", _dataset(rows)),
            _write_json(
                tmp_path / "predictions.json",
                _predictions([*_prediction_rows(dev), outside], "dev"),
            ),
            _fake_official_score,
            role="dev",
        )


def _dataset_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(459):
        sample_id = f"positive-{index:03d}"
        selected = _item(sample_id, "selected", True)
        unselected = _item(sample_id, "unselected", False)
        rows.append(
            {
                "sample_id": sample_id,
                "selected_count": 1,
                "selection_candidate_count": 2,
                "ground_truth": {
                    "page_count": 1,
                    "annotations": [selected, unselected],
                    "selected_items": [selected],
                },
            }
        )
    for index in range(26):
        sample_id = f"negative-{index:03d}"
        unselected = _item(sample_id, "unselected", False)
        rows.append(
            {
                "sample_id": sample_id,
                "selected_count": 0,
                "selection_candidate_count": 1,
                "ground_truth": {
                    "page_count": 1,
                    "annotations": [unselected],
                    "selected_items": [],
                },
            }
        )
    return list(reversed(rows))


def _prediction_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    predictions = []
    for index, row in enumerate(sorted(rows, key=_id)):
        ground_truth = row["ground_truth"]
        assert isinstance(ground_truth, dict)
        annotations = ground_truth["annotations"]
        assert isinstance(annotations, list)
        predictions.append(
            {
                "sample_id": _id(row),
                "status": "success",
                "latency_ms": index,
                "items": [
                    {key: item[key] for key in ("page", "bbox", "content", "selected")}
                    for item in annotations
                    if isinstance(item, dict) and item.get("selection_candidate")
                ],
                "failures": [],
            }
        )
    return predictions


def _item(sample_id: str, label: str, selected: bool) -> dict[str, object]:
    return {
        "page": 1,
        "bbox": BBOX,
        "content": f"{sample_id} {label}",
        "selected": selected,
        "selection_candidate": True,
    }


def _dataset(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "dataset": pulse.DATASET_ID,
        "revision": pulse.DATASET_REVISION,
        "split": "train",
        "cases": rows,
    }


def _predictions(rows: list[dict[str, object]], role: str) -> dict[str, object]:
    return {
        "dataset_revision": pulse.DATASET_REVISION,
        "scorer_revision": pulse.SCORER_REVISION,
        "role": role,
        "cases": rows,
    }


def _fake_official_score(
    predictions: dict[str, list[pulse.Control]],
    references: dict[str, list[pulse.Control]],
    latency_seconds: float,
) -> dict[str, object]:
    predicted = {
        (sample_id, item.page, item.content, item.bbox)
        for sample_id, items in predictions.items()
        for item in items
        if item.selected
    }
    reference = {
        (sample_id, item.page, item.content, item.bbox)
        for sample_id, items in references.items()
        for item in items
        if item.selected
    }
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    values = pulse._prf(true_positives, false_positives, false_negatives)
    return {"f1": values["f1"], "latency_seconds": latency_seconds}


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, payload: dict[str, object]) -> Path:
    rows = payload["cases"]
    assert isinstance(rows, list)
    metadata = {key: value for key, value in payload.items() if key != "cases"}
    lines = [json.dumps({"type": "metadata", **metadata})]
    lines.extend(json.dumps({"type": "case", **row}) for row in rows)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _id(row: dict[str, object]) -> str:
    sample_id = row["sample_id"]
    assert isinstance(sample_id, str)
    return sample_id
