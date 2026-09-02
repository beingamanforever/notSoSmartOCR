from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments import benchmark_doctr_proposals as benchmark


def test_cli_scores_fixed_panel_and_keeps_private_text_out_of_results(
    tmp_path: Path,
) -> None:
    challenge = tmp_path / "challenge"
    cases = {
        "C14-D001-P001": {
            "role": "target",
            "handwriting": [[0, 0, 10, 10], [20, 0, 30, 10]],
            "current": [[0, 0, 10, 10]],
            "detections": [
                {"bbox": [20, 0, 30, 10], "score": 0.9},
                {"bbox": [40, 0, 50, 10], "score": 0.8},
            ],
        },
        "C14-D002-P001": {
            "role": "target",
            "handwriting": [[0, 20, 10, 30]],
            "current": [[0, 20, 10, 30]],
            "detections": [],
        },
        "C01-D001-P001": {
            "role": "negative",
            "handwriting": [],
            "current": [[0, 0, 10, 10]],
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.9},
                {"bbox": [20, 0, 30, 10], "score": 0.8},
            ],
        },
        "C01-D002-P001": {
            "role": "negative",
            "handwriting": [],
            "current": [],
            "detections": [],
        },
    }
    target_predictions = []
    negative_predictions = []
    for case_id, case in cases.items():
        _challenge_case(challenge, case_id, case)
        predictions_for_role = (
            target_predictions if case["role"] == "target" else negative_predictions
        )
        predictions_for_role.append(
            {
                "case_id": case_id,
                "status": "success",
                "detections": case["detections"],
                "failure": None,
            }
        )
    predictions = tmp_path / "predictions.json"
    _write_json(
        predictions,
        {
            "model": {"architecture": "fast_base"},
            "runtime": {"gpu": "fixture"},
            "operations": {"inference_ms": {"p50": 1.0, "p95": 2.0}},
            "cases": target_predictions,
        },
    )
    additional_predictions = tmp_path / "additional-predictions.json"
    _write_json(
        additional_predictions,
        {
            "model": {"architecture": "fast_base"},
            "runtime": {"gpu": None},
            "operations": {"inference_ms": {"p50": 10.0, "p95": 20.0}},
            "cases": negative_predictions,
        },
    )

    output = tmp_path / "evaluation.json"
    overlays = tmp_path / "overlays"
    assert (
        benchmark.main(
            [
                "evaluate",
                str(challenge),
                str(predictions),
                str(output),
                "--additional-predictions",
                str(additional_predictions),
                "--overlays",
                str(overlays),
                "--negative-pages",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    score = result["default_threshold"]
    assert result["coverage"] == {
        "attempted": 4,
        "succeeded": 4,
        "failed": 0,
        "failure_types": {},
        "failure_policy": "failed and missing pages retain zero proposals",
    }
    assert len(result["prediction_runs"]) == 2
    assert score["raw_detections"] == 4
    assert score["novel_proposals"] == 3
    assert score["redundant_detections"] == 1
    assert score["negative_false_proposals_per_page"] == {
        "pages": 2,
        "total": 1,
        "mean": 0.5,
        "p50": 0.5,
        "p95": 0.95,
        "max": 1,
    }
    for metric in ("iou_0_50", "coverage_0_50"):
        values = score["metrics"][metric]
        assert values["baseline"] == {
            "matched": 2,
            "targets": 3,
            "recall": 0.666667,
        }
        assert values["union"] == {
            "matched": 3,
            "targets": 3,
            "recall": 1.0,
            "absolute_recall_gain_points": 33.333,
        }
        assert values["novel_proposals"]["precision"] == 0.333333
        assert values["novel_proposals"]["recall"] == 1.0
    assert output.is_file()
    assert len(list(overlays.glob("*.png"))) == 4
    assert "PRIVATE TEXT" not in output.read_text(encoding="utf-8")


def test_normalizes_straight_docTR_boxes_and_union_coverage() -> None:
    detections = benchmark._normalize_doctr_prediction(
        [{"words": np.array([[0.1, 0.2, 0.5, 0.6, 0.75]])}],
        width=200,
        height=100,
    )

    assert detections == [{"bbox": [20.0, 20.0, 100.0, 60.0], "score": 0.75}]
    assert benchmark._rectangle_union_area([[0, 0, 10, 5], [0, 2.5, 10, 10]]) == 100
    assert (
        benchmark._covered_fraction([0, 0, 10, 10], [[0, 0, 10, 5], [0, 2.5, 10, 10]])
        == 1.0
    )


def _challenge_case(challenge: Path, case_id: str, case: dict[str, object]) -> None:
    category = case_id.split("-", 1)[0]
    source_dir = challenge / "sources" / f"{category}-fixture"
    source_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(source_dir / f"{case_id}.png")

    annotation_dir = challenge / "annotations" / "primary" / category
    annotation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        annotation_dir / f"{case_id}.json",
        {
            "case_id": case_id,
            "handwriting": [
                {
                    "bbox": box,
                    "legibility": "legible",
                    "text": "PRIVATE TEXT",
                }
                for box in case["handwriting"]
            ],
            "transcription": {"reading_order_text": "PRIVATE TEXT"},
        },
    )

    version = "specialist-v10" if case["role"] == "target" else "specialist-v8"
    output_dir = challenge / "runs" / version / "model-output" / f"{category}-fixture"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / f"{case_id}.json",
        {
            "result": {
                "pages": [
                    {
                        "regions": [
                            {
                                "kind": "text",
                                "text": "PRIVATE TEXT",
                                "bounding_box": {
                                    "left": box[0],
                                    "top": box[1],
                                    "right": box[2],
                                    "bottom": box[3],
                                },
                            }
                            for box in case["current"]
                        ]
                    }
                ]
            }
        },
    )
    other_version = "specialist-v8" if version == "specialist-v10" else "specialist-v10"
    (challenge / "runs" / other_version / "model-output").mkdir(
        parents=True, exist_ok=True
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
