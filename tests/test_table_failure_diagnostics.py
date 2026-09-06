from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from experiments import table_failure_diagnostics as diagnostics


class _Scorer:
    def gold_cells(self, target: dict[str, Any]) -> list[dict[str, Any]]:
        return target["fixture_cells"]

    def score(
        self,
        gold: list[dict[str, Any]],
        predicted: Any,
    ) -> dict[str, float]:
        exact = float(predicted == gold)
        return {name: exact for name in diagnostics.pubtables.METRICS}


def test_three_paths_keep_failures_and_abstentions_in_denominators(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    seen: list[tuple[str, str, tuple[int, int, int, int]]] = []

    def detect(case: Any, image: Image.Image) -> dict[str, Any]:
        assert image.size == (40, 30)
        if case.case_id.endswith("P002"):
            return {"status": "abstained", "failure": "no_table", "latency_ms": 2}
        if case.case_id.endswith("P003"):
            return {"status": "failed", "failure": "detector_failed", "latency_ms": 3}
        return {
            "detections": [{"bbox": [12, 5, 28, 25], "score": 0.9}],
            "latency_ms": 1,
        }

    def structure(value: diagnostics.TableCropInput) -> dict[str, Any]:
        seen.append((value.case.case_id, value.kind, value.source_bbox))
        if value.kind == "ground_truth" and value.case.case_id.endswith("P003"):
            return {"status": "failed", "failure": "structure_failed"}
        left, top = value.source_bbox[:2]
        cells = [
            {
                **value.case.target["fixture_cells"][0],
                "bbox": [10 - left, 5 - top, 30 - left, 25 - top],
            }
        ]
        return {
            "cells": cells,
            "coordinate_space": "crop",
            "latency_ms": 4,
        }

    def recognize(
        value: diagnostics.GroundTruthStructureInput,
    ) -> dict[str, Any]:
        if value.case.case_id.endswith("P002"):
            return {"status": "abstained", "failure": "text_uncertain"}
        cells = list(value.cells)
        if value.case.case_id.endswith("P003"):
            cells = [{**cells[0], "cell_text": "wrong"}]
        return {"cells": cells, "latency_ms": 5}

    report = diagnostics.run_benchmark(
        dataset,
        scorer=_Scorer(),
        detector=detect,
        structure_predictor=structure,
        cell_recognizer=recognize,
        limit=3,
    )

    assert report["dataset"]["document_group_ids"] == ["C01-D001", "C01-D002"]
    assert report["modes"][diagnostics.MODES[0]]["coverage"] == {
        "attempted": 3,
        "successful": 2,
        "failed": 1,
        "abstained": 0,
        "coverage_rate": 0.666667,
    }
    assert report["modes"][diagnostics.MODES[0]]["metrics"]["all_cases"] == {
        name: 0.666667 for name in diagnostics.pubtables.METRICS
    }
    assert report["modes"][diagnostics.MODES[1]]["coverage"] == {
        "attempted": 3,
        "successful": 1,
        "failed": 1,
        "abstained": 1,
        "coverage_rate": 0.333333,
    }
    assert report["modes"][diagnostics.MODES[1]]["latency_ms"] == {
        "observed": 3,
        "missing": 0,
        "p50": 3.0,
        "p95": 4.8,
    }
    assert report["modes"][diagnostics.MODES[2]]["metrics"]["all_cases"] == {
        name: 0.333333 for name in diagnostics.pubtables.METRICS
    }
    assert report["cases"][1][diagnostics.MODES[1]]["status"] == "abstained"
    assert report["cases"][2][diagnostics.MODES[1]]["status"] == "failed"
    assert seen == [
        ("C01-D001-P001", "ground_truth", (5, 0, 35, 30)),
        ("C01-D001-P001", "predicted", (7, 0, 33, 30)),
        ("C01-D001-P002", "ground_truth", (5, 0, 35, 30)),
        ("C01-D002-P003", "ground_truth", (5, 0, 35, 30)),
    ]
    assert "fixture_cells" not in json.dumps(report)


def test_cli_replays_saved_component_outputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for case in diagnostics.pubtables._load_cases(dataset, 3):
        cells = case.target["fixture_cells"]
        (predictions / f"{case.case_id}.json").write_text(
            json.dumps(
                {
                    "detector": {"bbox": [10, 5, 30, 25], "latency_ms": 1},
                    "ground_truth_crop_structure": {
                        "cells": cells,
                        "latency_ms": 2,
                    },
                    "predicted_crop_structure": {
                        "cells": cells,
                        "latency_ms": 3,
                    },
                    "ground_truth_structure_recognition": {
                        "cells": cells,
                        "latency_ms": 4,
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(diagnostics.pubtables, "OfficialScorer", lambda root: _Scorer())
    output = tmp_path / "report.json"

    assert (
        diagnostics.main(
            [
                str(dataset),
                str(predictions),
                str(output),
                "--scorer-root",
                str(tmp_path / "scorer"),
                "--limit",
                "3",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert all(
        report["modes"][mode]["metrics"]["all_cases"]["grits_top"] == 1.0
        for mode in diagnostics.MODES
    )
    assert report["modes"][diagnostics.MODES[1]]["latency_ms"]["p50"] == 4.0


def _dataset(root: Path) -> Path:
    cases = (
        ("C01-D001-P001", "C01-D001"),
        ("C01-D001-P002", "C01-D001"),
        ("C01-D002-P003", "C01-D002"),
    )
    for case_id, group_id in cases:
        case_root = root / case_id
        case_root.mkdir(parents=True)
        Image.new("RGB", (40, 30), "white").save(case_root / "page.png")
        target = {
            "representation": "pubtables_structure_and_words",
            "document_group_id": group_id,
            "structure": {
                "representation": "pascal_voc_xml",
                "xml": _xml(case_id),
            },
            "words": {
                "representation": "word_boxes_json",
                "data": [{"text": case_id, "bbox": [10, 5, 30, 25]}],
            },
            "fixture_cells": [
                {
                    "row_nums": [0],
                    "column_nums": [0],
                    "bbox": [10.0, 5.0, 30.0, 25.0],
                    "cell_text": case_id,
                }
            ],
        }
        (case_root / "ground_truth.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "target": target,
                    "input_files": ["page.png"],
                }
            ),
            encoding="utf-8",
        )
    return root


def _xml(case_id: str) -> str:
    return f"""<annotation><filename>{case_id}.png</filename>
    <object><name>table</name><bndbox><xmin>10</xmin><ymin>5</ymin>
    <xmax>30</xmax><ymax>25</ymax></bndbox></object>
    <object><name>table row</name><bndbox><xmin>10</xmin><ymin>5</ymin>
    <xmax>30</xmax><ymax>25</ymax></bndbox></object>
    <object><name>table column</name><bndbox><xmin>10</xmin><ymin>5</ymin>
    <xmax>30</xmax><ymax>25</ymax></bndbox></object></annotation>"""
