from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from experiments import pubtables_benchmark as pubtables


def test_oracle_and_empty_controls_cover_30_tables_end_to_end(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", 30)
    scorer = _official_scorer(tmp_path / "table-transformer")
    output = tmp_path / "oracle.json"

    assert (
        pubtables.main(
            [
                str(dataset),
                str(output),
                "--scorer-root",
                str(scorer),
                "--control",
                "oracle",
                "--limit",
                "30",
            ]
        )
        == 0
    )
    oracle = json.loads(output.read_text(encoding="utf-8"))
    empty = pubtables.run_benchmark(
        dataset,
        scorer_root=scorer,
        control="empty",
        limit=30,
    )

    assert oracle["dataset"]["attempted_tables"] == 30
    assert oracle["coverage"]["valid"] == 30
    assert set(oracle["metrics"]["all_cases"].values()) == {1.0}
    assert set(empty["metrics"]["all_cases"].values()) == {0.0}
    assert oracle["operations"]["latency_ms"]["p50"] is None
    assert oracle["operations"]["throughput_tables_per_second"] is None
    assert oracle["operations"]["cost_missing"] == 30
    assert oracle["operations"]["total_cost_usd"] is None
    assert oracle["scorer"] == {
        "path": str(scorer),
        "revision": None,
        "interface": "src/grits.py cell-grid functions",
        "provenance": "caller-supplied scorer path; revision not supplied",
    }
    assert oracle["dataset"]["revision_evidence"].endswith(
        "prepared cases do not embed their source revision"
    )
    assert "reference_model" not in oracle
    assert "reference_model" not in empty

    with pytest.raises(SystemExit):
        pubtables.main(
            [
                str(dataset),
                str(output),
                "--scorer-root",
                str(scorer),
                "--control",
                "oracle",
                "--limit",
                "30",
            ]
        )


def test_missing_predictions_stay_in_the_failure_inclusive_denominator(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", 30)
    scorer = _official_scorer(tmp_path / "table-transformer")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for index in range(29):
        box = [0, 0, 9, 10] if index == 28 else [0, 0, 10, 10]
        _write_json(
            predictions / f"table-{index:02d}.json",
            {
                "prediction": {
                    "cells": [
                        {
                            "row_nums": [0],
                            "column_nums": [0],
                            "bbox": box,
                            "cell text": f"value-{index:02d}",
                        }
                    ]
                },
                "operations": {
                    "latency_ms": index + 1,
                    "peak_gpu_memory_mb": 100 + index,
                    "cost_usd": 0.001,
                },
            },
        )

    result = pubtables.run_benchmark(
        dataset,
        scorer_root=scorer,
        predictions_root=predictions,
        limit=30,
    )

    assert result["coverage"] == {
        "attempted": 30,
        "valid": 29,
        "failed": 1,
        "abstained": 0,
        "coverage_rate": 0.966667,
        "failure_policy": "failed, abstained, missing, and invalid cases score zero",
    }
    assert result["metrics"]["all_cases"] == {
        "grits_top": 0.966667,
        "grits_con": 0.966667,
        "grits_loc": 0.933333,
        "cell_exact_match": 0.966667,
    }
    assert result["metrics"]["served_only"] == {
        "grits_top": 1.0,
        "grits_con": 1.0,
        "grits_loc": 0.965517,
        "cell_exact_match": 1.0,
    }
    assert result["operations"] == {
        "latency_ms": {
            "observed": 29,
            "missing": 1,
            "p50": 15.0,
            "p95": 27.6,
        },
        "peak_gpu_memory_mb": 128.0,
        "throughput_tables_per_second": 66.666667,
        "cost_observed": 29,
        "cost_missing": 1,
        "total_cost_usd": None,
        "cost_per_attempted_table_usd": None,
        "failure_rate": 0.033333,
        "abstention_rate": 0.0,
        "model_load_ms": None,
    }
    assert result["cases"][-1]["failure"] == "missing_prediction"


def test_invalid_cells_fail_closed_without_allocating_an_unbounded_grid(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", 30)
    scorer = _official_scorer(tmp_path / "table-transformer")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    invalid = [
        {
            "row_nums": [10_000_000],
            "column_nums": [0],
            "bbox": [0, 0, 10, 10],
            "cell_text": "unsafe",
        }
    ]
    for index in range(30):
        cells = invalid if index == 0 else _cell(index)
        if index == 2:
            cells = cells * 2
        value: object = cells
        if index == 1:
            value = {
                "prediction": {"cells": cells},
                "operations": {"abstained": True},
            }
        _write_json(
            predictions / f"table-{index:02d}.json",
            value,
        )
    result = pubtables.run_benchmark(
        dataset,
        scorer_root=scorer,
        predictions_root=predictions,
        limit=30,
    )

    assert result["coverage"]["failed"] == 1
    assert result["coverage"]["abstained"] == 1
    assert "valid span" in result["cases"][0]["failure"]
    assert result["cases"][2]["status"] == "success"
    assert result["metrics"]["all_cases"]["grits_top"] == pytest.approx(28 / 30)


def test_checkpoint_identity_requires_explicit_revision_metadata(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", 30)
    scorer = _official_scorer(tmp_path / "table-transformer")
    model = tmp_path / "tatr-model"
    model.mkdir()
    (model / pubtables.MODEL_FILE).write_bytes(b"fixture")
    result = pubtables.run_benchmark(
        dataset,
        scorer_root=scorer,
        tatr_model_root=model,
        device="cpu",
        limit=30,
    )

    assert result["prediction_source"] == (f"checkpoint:{model / pubtables.MODEL_FILE}")
    assert result["reference_model"] == {
        "checkpoint": str(model / pubtables.MODEL_FILE),
        "revision": None,
        "provenance": "caller-supplied checkpoint path; revision not supplied",
        "preprocessing": (
            "supplied structure transform after a fixed 5 px tight crop; "
            "predicted boxes are translated to source-image coordinates"
        ),
    }
    assert result["coverage"]["valid"] == 30
    assert set(result["metrics"]["all_cases"].values()) == {1.0}
    assert result["operations"]["latency_ms"]["observed"] == 30
    assert result["operations"]["throughput_tables_per_second"] > 0
    assert result["operations"]["model_load_ms"] >= 0

    established = pubtables.run_benchmark(
        dataset,
        scorer_root=scorer,
        tatr_model_root=model,
        checkpoint_revision=pubtables.MODEL_REVISION,
        device="cpu",
        limit=30,
    )
    assert established["reference_model"]["id"] == pubtables.MODEL_ID
    assert established["reference_model"]["revision"] == pubtables.MODEL_REVISION

    all_model = tmp_path / "tatr-all"
    all_model.mkdir()
    (all_model / pubtables.ALL_MODEL_FILE).write_bytes(b"fixture")
    cross_domain = pubtables.run_benchmark(
        dataset,
        scorer_root=scorer,
        tatr_model_root=all_model,
        checkpoint_revision=pubtables.ALL_MODEL_REVISION,
        checkpoint_file=pubtables.ALL_MODEL_FILE,
        device="cpu",
        limit=30,
    )
    assert cross_domain["reference_model"]["id"] == pubtables.ALL_MODEL_ID
    assert cross_domain["reference_model"]["training_data"] == (
        "PubTables-1M and FinTabNet.c"
    )


def test_checkpoint_file_rejects_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a filename"):
        pubtables._checkpoint_path(tmp_path, "nested/model.pth")


def _dataset(root: Path, cases: int) -> Path:
    (root / "test").mkdir(parents=True)
    (root / "words").mkdir()
    (root / "images").mkdir()
    for index in range(cases):
        case_id = f"table-{index:02d}"
        (root / "test" / f"{case_id}.xml").write_text(_xml(case_id), encoding="utf-8")
        _write_json(
            root / "words" / f"{case_id}_words.json",
            [{"text": f"value-{index:02d}", "bbox": [0, 0, 10, 10]}],
        )
        Image.new("RGB", (20, 20), "white").save(root / "images" / f"{case_id}.jpg")
    return root


def _xml(case_id: str) -> str:
    return f"""<annotation><filename>{case_id}.jpg</filename>
    <object><name>table</name><bndbox><xmin>0</xmin><ymin>0</ymin>
    <xmax>10</xmax><ymax>10</ymax></bndbox></object>
    <object><name>table row</name><bndbox><xmin>0</xmin><ymin>0</ymin>
    <xmax>10</xmax><ymax>10</ymax></bndbox></object>
    <object><name>table column</name><bndbox><xmin>0</xmin><ymin>0</ymin>
    <xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>"""


def _cell(index: int) -> list[dict[str, object]]:
    return [
        {
            "row_nums": [0],
            "column_nums": [0],
            "bbox": [0, 0, 10, 10],
            "cell_text": f"value-{index:02d}",
        }
    ]


def _official_scorer(root: Path) -> Path:
    source = root / "src"
    source.mkdir(parents=True)
    (root / "detr").mkdir()
    (source / "structure_config.json").write_text("{}", encoding="utf-8")
    (source / "postprocess.py").write_text(
        """
def apply_class_thresholds(boxes, labels, scores, names, thresholds):
    return boxes, scores, labels

def iob(first, second):
    return 1.0

def objects_to_cells(table, objects, tokens, names, thresholds):
    return {}, [{
        'row_nums': [0],
        'column_nums': [0],
        'bbox': [0, 0, 10, 10],
        'cell_text': tokens[0]['text'],
    }], 1.0
""",
        encoding="utf-8",
    )
    (source / "grits.py").write_text(
        """
def cells_to_relspan_grid(cells):
    return [[cells[0]['row_nums'] + cells[0]['column_nums']]] if cells else [[]]

def cells_to_grid(cells, key='bbox'):
    return [[cells[0][key]]] if cells else [[]]

def _score(first, second):
    value = float(first.tolist() == second.tolist())
    return value, value, value, value

def grits_top(first, second):
    return _score(first, second)

def grits_con(first, second):
    return _score(first, second)

def grits_loc(first, second):
    return _score(first, second)
""",
        encoding="utf-8",
    )
    (source / "inference.py").write_text(
        """
class TableExtractionPipeline:
    def __init__(self, **kwargs):
        pass

    def recognize(self, image, tokens, out_cells=False):
        assert out_cells
        return {'cells': [[{
            'row_nums': [0],
            'column_nums': [0],
            'bbox': tokens[0]['bbox'],
            'cell text': tokens[0]['text'],
        }]]}
""",
        encoding="utf-8",
    )
    return root


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
