from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.omnidocbench_layout_benchmark import (
    EVAL_CATEGORIES,
    PREDICTION_CATEGORIES,
    build_control_predictions,
    main,
    run_benchmark,
    score_classwise,
    score_cote,
    score_union_area_iou,
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
    assert oracle["metrics"]["union_area_iou"]["dataset_median_iou"] == 1.0
    assert oracle["metrics"]["union_area_iou"]["scored_pages"] == 31
    assert set(oracle["metrics"]["union_area_iou"]["per_page_iou"].values()) == {1.0}
    assert oracle["metrics"]["union_area_iou"]["area_weighted_micro_iou"] == {
        "intersection_area": 18600.0,
        "union_area": 18600.0,
        "iou": 1.0,
    }
    assert oracle["metrics"]["cote"]["dataset_mean"] == {
        "coverage": 1.0,
        "overlap": 0.0,
        "trespass": 0.0,
        "excess": 0.0,
        "cote": 1.0,
    }
    assert oracle["metrics"]["cote"]["scored_pages"] == 31
    assert {
        page["cote"] for page in oracle["metrics"]["cote"]["per_page"].values()
    } == {1.0}
    assert empty["coverage"]["attempted_pages"] == 31
    assert empty["coverage"]["abstained_pages"] == 31
    assert empty["metrics"]["iou_0_5_detection"]["micro"]["f1"] == 0.0
    assert empty["metrics"]["iou_0_5_detection"]["micro"]["false_negatives"] == 31
    assert empty["metrics"]["union_area_iou"]["dataset_median_iou"] == 0.0
    assert empty["metrics"]["union_area_iou"]["scored_pages"] == 31
    assert empty["metrics"]["union_area_iou"]["area_weighted_micro_iou"] == {
        "intersection_area": 0.0,
        "union_area": 18600.0,
        "iou": 0.0,
    }
    assert empty["metrics"]["cote"]["dataset_mean"] == {
        "coverage": 0.0,
        "overlap": 0.0,
        "trespass": 0.0,
        "excess": 0.0,
        "cote": 0.0,
    }
    assert empty["metrics"]["cote"]["scored_pages"] == 31


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


def test_union_area_iou_accepts_exact_tiling_fragmentation() -> None:
    records = [_record(0)]
    predictions = _predictions(
        ("plain text", [10.0, 10.0, 25.0, 30.0]),
        ("plain text", [25.0, 10.0, 40.0, 30.0]),
    )

    metrics = score_union_area_iou(records, predictions)

    assert metrics["dataset_median_iou"] == 1.0
    assert metrics["per_page_iou"] == {"page-00": 1.0}
    assert metrics["area_weighted_micro_iou"] == {
        "intersection_area": 600.0,
        "union_area": 600.0,
        "iou": 1.0,
    }


def test_union_area_iou_deduplicates_prediction_overlap() -> None:
    records = [_record(0)]
    predictions = _predictions(
        ("plain text", [10.0, 10.0, 40.0, 30.0]),
        ("plain text", [10.0, 10.0, 40.0, 30.0]),
    )

    metrics = score_union_area_iou(records, predictions)

    assert metrics["dataset_median_iou"] == 1.0
    assert metrics["area_weighted_micro_iou"]["iou"] == 1.0
    assert metrics["area_weighted_micro_iou"]["union_area"] == 600.0


def test_union_area_iou_isolates_classes_and_reports_empty_classes() -> None:
    records = [_record(0)]
    predictions = _predictions(("table", [10.0, 10.0, 40.0, 30.0]))

    metrics = score_union_area_iou(records, predictions)

    assert metrics["dataset_median_iou"] == 0.0
    assert metrics["area_weighted_micro_iou"] == {
        "intersection_area": 0.0,
        "union_area": 1200.0,
        "iou": 0.0,
    }
    assert metrics["by_class"]["text"] == {
        "intersection_area": 0.0,
        "union_area": 600.0,
        "iou": 0.0,
    }
    assert metrics["by_class"]["table"] == {
        "intersection_area": 0.0,
        "union_area": 600.0,
        "iou": 0.0,
    }
    assert metrics["by_class"]["figure"] == {
        "intersection_area": 0.0,
        "union_area": 0.0,
        "iou": 0.0,
    }


def test_union_area_iou_weights_active_categories_equally_per_page() -> None:
    records = [_record(0)]
    records[0]["layout_dets"][0]["poly"] = [0, 0, 10, 0, 10, 10, 0, 10]
    records[0]["layout_dets"].append(
        {
            "category_type": "table",
            "poly": [20, 20, 21, 20, 21, 21, 20, 21],
            "ignore": False,
        }
    )
    predictions = _predictions(("plain text", [0.0, 0.0, 10.0, 10.0]))

    metrics = score_union_area_iou(records, predictions)

    assert metrics["per_page_iou"] == {"page-00": 0.5}
    assert metrics["dataset_median_iou"] == 0.5
    assert metrics["area_weighted_micro_iou"]["iou"] == round(100 / 101, 6)


def test_union_area_iou_takes_failure_inclusive_median_across_pages() -> None:
    records = [_record(0), _record(1)]
    records[1]["layout_dets"][0]["poly"] = [0, 0, 1, 0, 1, 1, 0, 1]
    predictions = _predictions(("plain text", [10.0, 10.0, 40.0, 30.0]))

    metrics = score_union_area_iou(records, predictions)

    assert metrics["scored_pages"] == 2
    assert metrics["per_page_iou"] == {"page-00": 1.0, "page-01": 0.0}
    assert metrics["dataset_median_iou"] == 0.5
    assert metrics["area_weighted_micro_iou"]["iou"] == round(600 / 601, 6)


def test_cote_accepts_exact_tiling_fragmentation_without_overlap() -> None:
    metrics = score_cote(
        [_record(0)],
        _predictions(
            ("plain text", [10.0, 10.0, 25.0, 30.0]),
            ("plain text", [25.0, 10.0, 40.0, 30.0]),
        ),
    )

    assert metrics["per_page"]["page-00"] == {
        "coverage": 1.0,
        "overlap": 0.0,
        "trespass": 0.0,
        "excess": 0.0,
        "cote": 1.0,
        "ground_truth_regions": 1,
        "predicted_regions": 2,
    }


def test_cote_penalizes_duplicate_prediction_overlap() -> None:
    metrics = score_cote(
        [_record(0)],
        _predictions(
            ("plain text", [10.0, 10.0, 40.0, 30.0]),
            ("plain text", [10.0, 10.0, 40.0, 30.0]),
        ),
    )

    assert metrics["per_page"]["page-00"] == {
        "coverage": 1.0,
        "overlap": 1.0,
        "trespass": 0.0,
        "excess": 0.0,
        "cote": 0.0,
        "ground_truth_regions": 1,
        "predicted_regions": 2,
    }


def test_cote_penalizes_cross_region_trespass() -> None:
    record = _record(0)
    record["layout_dets"][0]["poly"] = [0, 0, 10, 0, 10, 10, 0, 10]
    record["layout_dets"].append(
        {
            "category_type": "table",
            "poly": [10, 0, 20, 0, 20, 10, 10, 10],
            "ignore": False,
        }
    )

    metrics = score_cote([record], _predictions(("plain text", [0.0, 0.0, 12.0, 10.0])))

    assert metrics["classes"] == list(EVAL_CATEGORIES)
    assert metrics["per_page"]["page-00"] == {
        "coverage": 0.6,
        "overlap": 0.0,
        "trespass": 0.1,
        "excess": 0.0,
        "cote": 0.5,
        "ground_truth_regions": 2,
        "predicted_regions": 1,
    }


def test_cote_reports_excess_against_page_background() -> None:
    record = _record(0)
    record["layout_dets"][0]["poly"] = [0, 0, 10, 0, 10, 10, 0, 10]

    metrics = score_cote([record], _predictions(("plain text", [0.0, 0.0, 20.0, 10.0])))

    page = metrics["per_page"]["page-00"]
    assert page["coverage"] == 1.0
    assert page["overlap"] == 0.0
    assert page["trespass"] == 0.0
    assert page["excess"] == round(100 / 9900, 6)
    assert page["cote"] == 1.0


def test_cote_dataset_mean_keeps_failed_pages_in_denominator() -> None:
    metrics = score_cote(
        [_record(0), _record(1)],
        _predictions(("plain text", [10.0, 10.0, 40.0, 30.0])),
    )

    assert metrics["scored_pages"] == 2
    assert metrics["per_page"]["page-00"]["cote"] == 1.0
    assert metrics["per_page"]["page-01"]["cote"] == 0.0
    assert metrics["dataset_mean"] == {
        "coverage": 0.5,
        "overlap": 0.0,
        "trespass": 0.0,
        "excess": 0.0,
        "cote": 0.5,
    }


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


def _predictions(*items: tuple[str, list[float]]) -> dict[str, object]:
    categories = {
        str(index): category for index, category in enumerate(PREDICTION_CATEGORIES)
    }
    category_ids = {category: int(index) for index, category in categories.items()}
    return {
        "categories": categories,
        "results": [
            {
                "image_name": "page-00",
                "bbox": bbox,
                "category_id": category_ids[category],
                "score": 1.0,
            }
            for category, bbox in items
        ],
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
