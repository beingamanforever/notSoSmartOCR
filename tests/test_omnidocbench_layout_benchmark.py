from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.omnidocbench_layout_benchmark import (
    EVAL_CATEGORIES,
    build_control_predictions,
    main,
    run_benchmark,
    score_classwise,
)


def test_thirty_one_page_controls_keep_failures_in_denominator(tmp_path: Path) -> None:
    records = [_record(index) for index in range(31)]
    annotations = tmp_path / "OmniDocBench.json"
    annotations.write_text(json.dumps(records), encoding="utf-8")

    oracle = run_benchmark(
        annotations,
        evaluator_root=tmp_path,
        control="oracle",
        score_official=_official_score,
    )
    empty = run_benchmark(
        annotations,
        evaluator_root=tmp_path,
        control="empty",
        score_official=_official_score,
    )

    assert oracle["coverage"] == {
        "attempted_pages": 31,
        "covered_pages": 31,
        "abstained_pages": 0,
        "coverage_rate": 1.0,
        "failure_policy": "missing or empty page predictions remain in the denominator",
    }
    assert oracle["metrics"]["iou_0_5_detection"]["micro"] == {
        "true_positives": 31,
        "false_positives": 0,
        "false_negatives": 0,
        "predicted_boxes": 31,
        "ground_truth_boxes": 31,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert empty["coverage"]["attempted_pages"] == 31
    assert empty["coverage"]["abstained_pages"] == 31
    assert empty["metrics"]["iou_0_5_detection"]["micro"]["f1"] == 0.0
    assert empty["metrics"]["iou_0_5_detection"]["micro"]["false_negatives"] == 31


def test_matching_is_classwise_and_one_to_one() -> None:
    records = [_record(index) for index in range(30)]
    predictions = build_control_predictions(records, "oracle")
    predictions["results"].append(dict(predictions["results"][0]))
    predictions["results"][1]["bbox"] = [50.0, 50.0, 60.0, 60.0]

    metrics = score_classwise(records, predictions)

    text = metrics["by_class"]["text"]
    assert text["true_positives"] == 29
    assert text["false_positives"] == 2
    assert text["false_negatives"] == 1
    assert text["precision"] == round(29 / 31, 6)
    assert text["recall"] == round(29 / 30, 6)


def test_cli_refuses_small_panel_and_existing_output(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([_record(index) for index in range(29)]))
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                str(annotations),
                str(output),
                "--evaluator-root",
                str(tmp_path),
                "--control",
                "empty",
            ]
        )
    assert not output.exists()

    annotations.write_text(json.dumps([_record(index) for index in range(30)]))
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                str(annotations),
                str(output),
                "--evaluator-root",
                str(tmp_path),
                "--control",
                "empty",
            ]
        )
    assert output.read_text(encoding="utf-8") == "keep"


def _official_score(
    records: list[dict[str, object]],
    predictions: dict[str, object],
    evaluator_root: Path,
) -> dict[str, object]:
    assert len(records) >= 30
    del evaluator_root
    has_predictions = bool(predictions["results"])
    value = 1.0 if has_predictions else 0.0
    return {
        "summary": {"bbox_mAP": value},
        "class_ap": {category: value for category in EVAL_CATEGORIES},
    }


def _record(index: int) -> dict[str, object]:
    return {
        "page_info": {
            "image_path": f"page-{index:02d}.png",
            "width": 100,
            "height": 100,
            "page_attribute": {
                "subset": "v1.5",
                "data_source": "book",
                "language": "english",
                "layout": "single_column",
            },
        },
        "layout_dets": [
            {
                "category_type": "text_block",
                "poly": [10, 10, 40, 10, 40, 30, 10, 30],
                "ignore": False,
            }
        ],
    }
