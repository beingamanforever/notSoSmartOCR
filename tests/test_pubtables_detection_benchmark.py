from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from experiments import pubtables_detection_benchmark as detection


def test_failure_inclusive_detection_scores_30_cases_end_to_end(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", 30)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for index in range(29):
        boxes = [[0, 0, 10, 6]] if index == 1 else [[0, 0, 10, 10]]
        if index == 0:
            boxes.append([10, 10, 20, 20])
        _write_json(
            predictions / f"table-{index:02d}.json",
            {
                "status": "success",
                "detections": [
                    {"bbox": box, "score": 0.9 - order * 0.1}
                    for order, box in enumerate(boxes)
                ],
                "latency_ms": index + 1,
            },
        )

    output = tmp_path / "detection.json"
    assert (
        detection.main(
            [
                str(dataset),
                str(output),
                "--predictions",
                str(predictions),
                "--limit",
                "30",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["coverage"] == {
        "attempted": 30,
        "valid": 29,
        "failed": 1,
        "abstained": 0,
        "coverage_rate": 0.966667,
        "failure_policy": (
            "failed, abstained, missing, and invalid cases retain all truth "
            "tables as false negatives"
        ),
    }
    assert report["metrics"] == {
        "iou_0_50": {
            "true_positives": 29,
            "false_positives": 1,
            "false_negatives": 1,
            "precision": 0.966667,
            "recall": 0.966667,
            "f1": 0.966667,
        },
        "iou_0_75": {
            "true_positives": 28,
            "false_positives": 2,
            "false_negatives": 2,
            "precision": 0.933333,
            "recall": 0.933333,
            "f1": 0.933333,
        },
        "mean_best_iou": 0.953333,
    }
    assert report["operations"] == {
        "latency_ms": {
            "observed": 29,
            "missing": 1,
            "p50": 15.0,
            "p95": 27.6,
        },
        "model_load_ms": None,
    }
    assert report["cases"][0]["predicted_tables"] == 2
    assert report["cases"][1]["mean_best_iou"] == 0.6
    assert report["cases"][-1]["failure"] == "missing_prediction"
    assert report["dataset"]["revision"] == detection.pubtables.DATASET_REVISION
    assert report["scorer"]["revision"] == detection.SCORER_REVISION

    injected = detection.run_benchmark(
        dataset,
        predictor=lambda case: {
            "status": "success",
            "detections": [{"bbox": [0, 0, 10, 10], "score": 1.0}],
            "latency_ms": 1,
        },
        model_metadata={"id": "fixture", "revision": "1", "license": "MIT"},
        limit=30,
    )
    assert injected["coverage"]["valid"] == 30
    assert injected["metrics"]["iou_0_75"]["f1"] == 1.0
    assert injected["model"] == {"id": "fixture", "revision": "1", "license": "MIT"}


def _dataset(root: Path, cases: int) -> Path:
    (root / "test").mkdir(parents=True)
    (root / "words").mkdir()
    (root / "images").mkdir()
    for index in range(cases):
        case_id = f"table-{index:02d}"
        (root / "test" / f"{case_id}.xml").write_text(_xml(case_id), encoding="utf-8")
        _write_json(
            root / "words" / f"{case_id}_words.json",
            [{"text": case_id, "bbox": [0, 0, 10, 10]}],
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


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
