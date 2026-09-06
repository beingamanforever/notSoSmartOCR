from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from experiments.evaluate_challenge_set import evaluate_challenge_set, main


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _annotation(
    case_id: str,
    text: str,
    *,
    legibility: str = "complete",
    unresolved: list[str] | None = None,
    controls: list[dict[str, object]] | None = None,
    control_annotation_scope: str | None = None,
    tables: list[dict[str, object]] | None = None,
    handwriting: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    annotation: dict[str, object] = {
        "case_id": case_id,
        "category_id": case_id.split("-", 1)[0],
        "source_only": True,
        "page_legibility": legibility,
        "transcription": {
            "reading_order_text": text,
            "unresolved_spans": unresolved or [],
        },
        "controls": controls or [],
        "tables": tables or [],
        "handwriting": handwriting or [],
        "notes": "PRIVATE NOTE MUST NEVER LEAK",
    }
    if control_annotation_scope is not None:
        annotation["control_annotation_scope"] = control_annotation_scope
    return annotation


def _output(
    filename: str,
    text: str,
    *,
    status: str = "success",
    route: str = "accept_local",
    regions: list[dict[str, object]] | None = None,
    total_seconds: float = 2.0,
) -> dict[str, object]:
    return {
        "filename": filename,
        "elapsed_seconds": total_seconds,
        "timing": {
            "total_seconds": total_seconds,
            "pipeline_steps": {"reader": total_seconds / 2, "prepare": 0.1},
        },
        "result": {
            "status": status,
            "failures": [] if status == "success" else [{"code": "reader_failed"}],
            "pages": [
                {
                    "route": route,
                    "text": {"value": text},
                    "regions": regions or [],
                }
            ],
        },
    }


def test_evaluator_is_failure_inclusive_and_never_exports_private_content(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    private_reference = "Patient Alpha takes ten pills"
    private_prediction = "Patient Alpha takes nine pills extra"
    _write(
        annotations / "C01" / "first.json",
        _annotation(
            "C01-D001-P001",
            private_reference,
            controls=[
                {"kind": "checkbox", "label": "Private label", "state": "checked"}
            ],
            tables=[{"row_count": 2, "column_count": 3}],
            handwriting=[{"text": "Patient Alpha", "legibility": "legible"}],
        ),
    )
    _write(
        annotations / "C01" / "second.json",
        _annotation("C01-D002-P001", "Missing output text"),
    )
    _write(
        annotations / "C01" / "partial.json",
        _annotation(
            "C01-D003-P001",
            "Partial private text",
            legibility="partial",
            unresolved=["private unresolved value"],
        ),
    )
    regions = [
        {"kind": "word", "reading_order": 1},
        {
            "kind": "table",
            "reading_order": 2,
            "structure": {"row_count": 2, "column_count": 3},
        },
        {
            "kind": "checkbox",
            "reading_order": 3,
            "structure": {
                "control_type": "checkbox",
                "label": "Private label",
                "state": "checked",
            },
        },
    ]
    _write(
        outputs / "category" / "renamed.json",
        _output(
            "C01-D001-P001.png",
            private_prediction,
            regions=regions,
            total_seconds=1.0,
        ),
    )
    _write(
        outputs / "category" / "unannotated.json",
        _output(
            "C01-D004-P001.png",
            "Unannotated patient value",
            route="review",
            regions=[{"kind": "word"}],
            total_seconds=3.0,
        ),
    )

    report = evaluate_challenge_set(annotations, outputs)

    assert report["cases"] == {
        "total": 4,
        "annotated": 3,
        "model_outputs": 2,
        "paired": 1,
        "missing_model_output": 2,
        "unannotated_model_output": 1,
    }
    assert report["transcription"]["scored_cases"] == 2
    assert report["transcription"]["excluded"]["partial"] == 1
    assert report["transcription"]["missed_text_rate"] > 0
    assert report["transcription"]["hallucinated_text_rate"] > 0
    assert report["coverage"]["covered"] == 1
    assert report["coverage"]["failed"] == 2
    assert report["regions"]["kind_counts"] == {
        "checkbox": 1,
        "table": 1,
        "word": 1,
    }
    assert report["regions"]["route_counts"] == {"accept_local": 1}
    assert report["latency_seconds"]["total"] == {
        "count": 1,
        "p50": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }

    serialized = json.dumps(report)
    for private_value in (
        private_reference,
        private_prediction,
        "Patient Alpha",
        "Private label",
        "PRIVATE NOTE MUST NEVER LEAK",
        "C01-D001-P001",
        "renamed.json",
    ):
        assert private_value not in serialized


def test_evaluator_runs_without_scipy(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    report_path = tmp_path / "report.json"
    _write(annotations / "case.json", _annotation("C01-D001-P001", "Text"))
    _write(outputs / "case.json", _output("C01-D001-P001.png", "Text"))
    script = f"""
import builtins

original_import = builtins.__import__

def block_scipy(name, *args, **kwargs):
    if name == "scipy" or name.startswith("scipy."):
        raise ModuleNotFoundError("SciPy is unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_scipy
from experiments.evaluate_challenge_set import main

raise SystemExit(main([
    {str(annotations)!r},
    {str(outputs)!r},
    {str(report_path)!r},
]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["cases"]["paired"] == 1


def test_evaluator_reports_table_control_and_handwriting_metrics(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C02-D001-P001",
            "Printed words handwritten dose",
            controls=[
                {"kind": "checkbox", "label": "Choice A", "state": "checked"},
                {"kind": "checkbox", "label": "Choice B", "state": "unchecked"},
            ],
            tables=[{"row_count": 4, "column_count": 2}],
            handwriting=[
                {
                    "text": "handwritten dose",
                    "legibility": "legible",
                    "bbox": [40, 40, 140, 65],
                },
                {"text": "uncertain dose", "legibility": "partial"},
                {"text": "", "legibility": "illegible"},
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C02-D001-P001.png",
            "Printed words handwritten dose",
            regions=[
                {
                    "kind": "table",
                    "reading_order": 1,
                    "structure": {"row_count": 4, "column_count": 2},
                },
                {
                    "kind": "checkbox",
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Choice A",
                        "state": "selected",
                    },
                },
                {
                    "kind": "checkbox",
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Choice B",
                        "state": "selected",
                    },
                },
                {
                    "id": "handwriting",
                    "kind": "text",
                    "text": "handwritten dose",
                    "bounding_box": {
                        "left": 42,
                        "top": 41,
                        "right": 138,
                        "bottom": 64,
                    },
                },
            ],
        ),
    )

    report = evaluate_challenge_set(annotations, outputs)

    assert report["tables"]["presence"]["f1"] == 1.0
    assert report["tables"]["descriptors"]["row_count_accuracy"] == 1.0
    assert report["tables"]["descriptors"]["column_count_accuracy"] == 1.0
    assert report["controls"]["safely_matched"] == 2
    assert report["controls"]["state_accuracy_on_safe_matches"] == 0.5
    assert report["controls"]["state_macro_f1_on_safe_matches"] == 0.333333
    assert report["handwriting"]["legible_exact_recovery"]["rate"] == 1.0
    assert report["handwriting"]["partial_exact_recovery"]["rate"] is None
    assert report["handwriting"]["page_presence"]["partial"]["rate"] == 0.0
    assert report["handwriting"]["unlocalized_scorable"] == 1
    assert report["handwriting"]["localized_edit_metrics_supported"] is False


def test_transcription_scores_selected_table_cells_without_markdown_markup(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation("C02-D002-P001", "Heading First value Second value"),
    )
    table = {
        "id": "table",
        "kind": "table",
        "text": "| First value | Second value |\n| --- | --- |",
        "reading_order": 2,
        "structure": {
            "row_count": 1,
            "column_count": 2,
            "cells": [
                {"text": "Second value", "row_nums": [0], "column_nums": [1]},
                {"text": "First value", "row_nums": [0], "column_nums": [0]},
            ],
        },
    }
    output = _output(
        "C02-D002-P001.png",
        "Heading | First value | Second value | | --- | --- |",
        regions=[
            {
                "id": "heading",
                "kind": "text",
                "text": "Heading",
                "reading_order": 1,
            },
            table,
            {
                "id": "table-source",
                "kind": "word",
                "text": "First value",
                "reading_order": 2,
                "structure": {"role": "table_source", "parent_id": "table"},
            },
        ],
    )
    output["result"]["pages"][0]["text"]["evidence_ids"] = ["heading", "table"]
    _write(outputs / "case.json", output)

    transcription = evaluate_challenge_set(annotations, outputs)["transcription"]

    assert transcription["prediction_policy"] == "evidence_text_with_table_cells_once"
    assert transcription["cer"] == 0.0
    assert transcription["wer"] == 0.0
    assert transcription["hallucinated_text_rate"] == 0.0


def test_transcription_reports_token_multiset_omissions_and_additions(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation("C02-D003-P001", "Q1 100K Q2 200K"),
    )
    _write(
        outputs / "case.json",
        _output("C02-D003-P001.png", "Q1 Q2 100K 300K 100K"),
    )

    token_metrics = evaluate_challenge_set(annotations, outputs)["transcription"][
        "token_multiset"
    ]

    assert token_metrics == {
        "policy": "normalized_whitespace_token_multiset",
        "reference_tokens": 4,
        "prediction_tokens": 5,
        "found_tokens": 3,
        "added_tokens": 2,
        "tokens_found": 0.75,
        "tokens_added": 0.4,
    }


def test_evaluator_buckets_private_aggregate_labels(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    private_route = "PRIVATE ROUTE VALUE"
    private_kind = "PRIVATE KIND VALUE"
    private_failure = "PRIVATE FAILURE VALUE"
    _write(annotations / "case.json", _annotation("C03-D001-P001", "Text"))
    output = _output(
        "C03-D001-P001.png",
        "",
        status="failed",
        route=private_route,
        regions=[{"kind": private_kind}],
    )
    output["result"]["failures"] = [{"code": private_failure}]
    _write(outputs / "case.json", output)

    report = evaluate_challenge_set(annotations, outputs)

    assert report["coverage"]["failure_codes"] == {"other": 1}
    assert report["regions"] == {
        "kind_counts": {"other": 1},
        "route_counts": {"other": 1},
    }
    serialized = json.dumps(report)
    assert private_route not in serialized
    assert private_kind not in serialized
    assert private_failure not in serialized


def test_complete_transcription_with_unresolved_spans_is_excluded(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C04-D001-P001",
            "Complete but unresolved",
            unresolved=["private unresolved text"],
        ),
    )
    _write(
        outputs / "case.json",
        _output("C04-D001-P001.png", "Complete but unresolved"),
    )

    transcription = evaluate_challenge_set(annotations, outputs)["transcription"]

    assert transcription["scored_cases"] == 0
    assert transcription["excluded"]["unresolved"] == 1
    assert transcription["excluded"]["missing_reference"] == 0


def test_control_states_are_canonicalized_between_annotation_and_producer(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C05-D001-P001",
            "Controls",
            controls=[
                {"kind": "checkbox", "label": "A", "state": "checked"},
                {"kind": "checkbox", "label": "B", "state": "unchecked"},
                {"kind": "checkbox", "label": "C", "state": "uncertain"},
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C05-D001-P001.png",
            "Controls",
            regions=[
                {
                    "kind": "checkbox",
                    "structure": {
                        "control_type": "checkbox",
                        "label": label,
                        "state": state,
                    },
                }
                for label, state in (
                    ("A", "selected"),
                    ("B", "unselected"),
                    ("C", "ambiguous"),
                )
            ],
        ),
    )

    controls = evaluate_challenge_set(annotations, outputs)["controls"]

    assert controls["safely_matched"] == 3
    assert controls["state_accuracy_on_safe_matches"] == 1.0
    assert controls["state_macro_f1_on_safe_matches"] == 1.0


def test_exhaustive_control_scope_reports_bbox_state_metrics(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C05-D002-P001",
            "Controls",
            control_annotation_scope="exhaustive",
            controls=[
                {
                    "kind": "checkbox",
                    "label": "A",
                    "state": "checked",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                },
                {
                    "kind": "checkbox",
                    "label": "B",
                    "state": "unchecked",
                    "bounding_box": {
                        "left": 20,
                        "top": 0,
                        "right": 30,
                        "bottom": 10,
                    },
                },
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C05-D002-P001.png",
            "Controls",
            regions=[
                {
                    "kind": "checkbox",
                    "bounding_box": {"left": 1, "top": 0, "right": 11, "bottom": 10},
                    "structure": {
                        "control_type": "checkbox",
                        "label": "A",
                        "state": "selected",
                    },
                },
                {
                    "kind": "checkbox",
                    "bounding_box": {
                        "left": 20,
                        "top": 0,
                        "right": 30,
                        "bottom": 10,
                    },
                    "structure": {
                        "control_type": "checkbox",
                        "label": "B",
                        "state": "selected",
                    },
                },
                {
                    "kind": "checkbox",
                    "bounding_box": {
                        "left": 40,
                        "top": 0,
                        "right": 50,
                        "bottom": 10,
                    },
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Extra",
                        "state": "selected",
                    },
                },
            ],
        ),
    )

    bbox = evaluate_challenge_set(annotations, outputs)["controls"]["bbox"]

    assert bbox["iou_threshold"] == 0.5
    assert bbox["exhaustive"] == {
        "pages": 1,
        "reference_count": 2,
        "predicted_count": 3,
        "matched": 2,
        "precision": 0.666667,
        "recall": 1.0,
        "f1": 0.8,
        "checked": {
            "reference_count": 1,
            "predicted_count": 3,
            "matched": 1,
            "precision": 0.333333,
            "recall": 1.0,
            "f1": 0.5,
        },
        "unchecked_recall": {"reference_count": 1, "matched": 0, "recall": 0.0},
        "state_accuracy": {"eligible": 2, "correct": 1, "accuracy": 0.5},
    }


def test_selected_only_control_scope_reports_recall_without_false_positives(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C05-D003-P001",
            "Controls",
            control_annotation_scope="selected_only",
            controls=[
                {
                    "kind": "checkbox",
                    "label": "Selected",
                    "state": "checked",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                }
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C05-D003-P001.png",
            "Controls",
            regions=[
                {
                    "kind": "checkbox",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Selected",
                        "state": "selected",
                    },
                },
                {
                    "kind": "checkbox",
                    "bounding_box": {
                        "left": 20,
                        "top": 0,
                        "right": 30,
                        "bottom": 10,
                    },
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Unannotated empty",
                        "state": "unselected",
                    },
                },
                {
                    "kind": "checkbox",
                    "bounding_box": {
                        "left": 40,
                        "top": 0,
                        "right": 50,
                        "bottom": 10,
                    },
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Unannotated selected",
                        "state": "selected",
                    },
                },
            ],
        ),
    )

    bbox = evaluate_challenge_set(annotations, outputs)["controls"]["bbox"]

    assert bbox["selected_only"] == {
        "pages": 1,
        "checked_reference_count": 1,
        "checked_matched": 1,
        "checked_recall": 1.0,
    }
    assert bbox["exhaustive"]["pages"] == 0
    assert "precision" not in bbox["selected_only"]


def test_exhaustive_control_bbox_metrics_include_missing_output(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _write(
        annotations / "case.json",
        _annotation(
            "C05-D004-P001",
            "Controls",
            control_annotation_scope="exhaustive",
            controls=[
                {
                    "kind": "radio",
                    "label": "A",
                    "state": "checked",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                }
            ],
        ),
    )

    exhaustive = evaluate_challenge_set(annotations, outputs)["controls"]["bbox"][
        "exhaustive"
    ]

    assert exhaustive["precision"] is None
    assert exhaustive["recall"] == 0.0
    assert exhaustive["f1"] == 0.0
    assert exhaustive["checked"]["recall"] == 0.0


def test_control_bbox_metrics_report_optional_missing_boxes(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C05-D005-P001",
            "Controls",
            control_annotation_scope="exhaustive",
            controls=[
                {
                    "kind": "checkbox",
                    "label": "Localized",
                    "state": "checked",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                },
                {"kind": "checkbox", "label": "Legacy", "state": "unchecked"},
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C05-D005-P001.png",
            "Controls",
            regions=[
                {
                    "kind": "checkbox",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Localized",
                        "state": "selected",
                    },
                },
                {
                    "kind": "checkbox",
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Unlocalized prediction",
                        "state": "selected",
                    },
                },
            ],
        ),
    )

    bbox = evaluate_challenge_set(annotations, outputs)["controls"]["bbox"]

    assert bbox["missing_bounding_box"] == {"reference": 1, "prediction": 1}
    assert bbox["exhaustive"]["reference_count"] == 1
    assert bbox["exhaustive"]["predicted_count"] == 2
    assert bbox["exhaustive"]["precision"] == 0.5
    assert bbox["exhaustive"]["recall"] == 1.0


def test_control_bbox_metrics_use_optimal_one_to_one_assignment(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C05-D006-P001",
            "Controls",
            control_annotation_scope="exhaustive",
            controls=[
                {
                    "kind": "checkbox",
                    "label": "A",
                    "state": "checked",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 30},
                },
                {
                    "kind": "checkbox",
                    "label": "B",
                    "state": "checked",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 20},
                },
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C05-D006-P001.png",
            "Controls",
            regions=[
                {
                    "kind": "checkbox",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 30},
                    "structure": {
                        "control_type": "checkbox",
                        "label": "X",
                        "state": "selected",
                    },
                },
                {
                    "kind": "checkbox",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 50},
                    "structure": {
                        "control_type": "checkbox",
                        "label": "Y",
                        "state": "selected",
                    },
                },
            ],
        ),
    )

    bbox = evaluate_challenge_set(annotations, outputs)["controls"]["bbox"]

    assert bbox["matching_policy"] == ("hungarian_one_to_one_max_cardinality_then_iou")
    assert bbox["iou_thresholds"] == [0.5, 0.7]
    assert bbox["exhaustive"]["matched"] == 2
    assert bbox["iou_0_7"]["exhaustive"]["matched"] == 1
    assert bbox["length_penalty"]["value"] == 0.0


def test_table_descriptors_keep_valid_pairs_when_counts_differ(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "missing.json",
        _annotation(
            "C06-D001-P001",
            "Missing table output",
            tables=[{"row_count": 2, "column_count": 3}],
        ),
    )
    _write(
        annotations / "mismatch.json",
        _annotation(
            "C06-D002-P001",
            "Mismatched table count",
            tables=[
                {"row_count": 4, "column_count": 2},
                {"row_count": 5, "column_count": 6},
            ],
        ),
    )
    _write(
        outputs / "mismatch.json",
        _output(
            "C06-D002-P001.png",
            "Mismatched table count",
            regions=[
                {
                    "kind": "table",
                    "reading_order": 1,
                    "structure": {"row_count": 4, "column_count": 2},
                }
            ],
        ),
    )

    descriptors = evaluate_challenge_set(annotations, outputs)["tables"]["descriptors"]

    assert descriptors["pairing"] == "annotation-and-prediction-reading-order"
    assert descriptors["spatial_matching_supported"] is False
    assert descriptors["pairing_limitation"] == (
        "reference table boxes are unavailable; declared annotation and prediction "
        "reading order is used"
    )
    assert descriptors["count_semantics"] == "annotation-declared-not-geometric"
    assert descriptors["paired_tables"] == 1
    assert descriptors["missed_reference_tables"] == 2
    assert descriptors["extra_predicted_tables"] == 0
    assert descriptors["row_count_eligible"] == 3
    assert descriptors["row_count_accuracy"] == 0.333333
    assert descriptors["row_count_accuracy_on_pairs"] == 1.0
    assert descriptors["row_count_mae_on_numeric_pairs"] == 0.0
    assert descriptors["column_count_eligible"] == 3
    assert descriptors["column_count_accuracy"] == 0.333333
    assert descriptors["column_count_accuracy_on_pairs"] == 1.0
    assert descriptors["column_count_mae_on_numeric_pairs"] == 0.0


def test_table_descriptors_report_extra_predictions(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C06-D003-P001",
            "Extra table output",
            tables=[{"row_count": 3, "column_count": 4}],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C06-D003-P001.png",
            "Extra table output",
            regions=[
                {
                    "kind": "table",
                    "reading_order": 1,
                    "structure": {"row_count": 3, "column_count": 4},
                },
                {
                    "kind": "table",
                    "reading_order": 2,
                    "structure": {"row_count": 8, "column_count": 9},
                },
            ],
        ),
    )

    descriptors = evaluate_challenge_set(annotations, outputs)["tables"]["descriptors"]

    assert descriptors["paired_tables"] == 1
    assert descriptors["missed_reference_tables"] == 0
    assert descriptors["extra_predicted_tables"] == 1
    assert descriptors["row_count_accuracy"] == 1.0
    assert descriptors["column_count_accuracy"] == 1.0


def test_handwriting_recovery_uses_token_boundaries_and_unique_mentions(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C07-D001-P001",
            "Handwriting",
            handwriting=[
                {"text": "dose", "legibility": "legible"},
                {"text": "dose", "legibility": "legible"},
                {"text": "mg", "legibility": "partial"},
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output("C07-D001-P001.png", "overdose dose dosage 5mg"),
    )

    handwriting = evaluate_challenge_set(annotations, outputs)["handwriting"]

    assert handwriting["legible_exact_recovery"] == {
        "eligible": 0,
        "recovered": 0,
        "rate": None,
    }
    assert handwriting["partial_exact_recovery"] == {
        "eligible": 0,
        "recovered": 0,
        "rate": None,
    }
    assert handwriting["page_presence"] == {
        "legible": {"eligible": 2, "recovered": 1, "rate": 0.5},
        "partial": {"eligible": 1, "recovered": 0, "rate": 0.0},
    }
    assert handwriting["unlocalized_scorable"] == 3


def test_handwriting_recovery_rejects_matching_text_outside_annotated_box(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C07-D002-P001",
            "Printed 9 and handwritten dose",
            handwriting=[
                {"text": "9", "legibility": "legible", "bbox": [0, 0, 10, 10]},
                {
                    "text": "dose",
                    "legibility": "legible",
                    "bbox": [40, 40, 70, 60],
                },
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C07-D002-P001.png",
            "Printed 9 and handwritten dose",
            regions=[
                {
                    "id": "printed-digit",
                    "kind": "text",
                    "text": "9",
                    "bounding_box": {
                        "left": 80,
                        "top": 80,
                        "right": 90,
                        "bottom": 90,
                    },
                },
                {
                    "id": "handwritten-dose",
                    "kind": "text",
                    "text": "dose",
                    "bounding_box": {
                        "left": 41,
                        "top": 41,
                        "right": 69,
                        "bottom": 59,
                    },
                },
            ],
        ),
    )

    handwriting = evaluate_challenge_set(annotations, outputs)["handwriting"]

    assert handwriting["legible_exact_recovery"] == {
        "eligible": 2,
        "recovered": 1,
        "rate": 0.5,
    }
    assert handwriting["page_presence"]["legible"] == {
        "eligible": 2,
        "recovered": 2,
        "rate": 1.0,
    }


def test_handwriting_recovery_expands_canonical_layout_block_evidence(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C07-D003-P001",
            "Drug Jardiance",
            handwriting=[
                {
                    "text": "Jardiance",
                    "legibility": "legible",
                    "bbox": [40, 40, 80, 60],
                }
            ],
        ),
    )
    output = _output(
        "C07-D003-P001.png",
        "Drug Jardiance",
        regions=[
            {
                "id": "field",
                "kind": "layout_block",
                "text": "Drug Jardiance",
                "bounding_box": {"left": 0, "top": 0, "right": 100, "bottom": 80},
                "structure": {
                    "role": "layout_block",
                    "block_type": "form_row",
                    "child_evidence_ids": ["label", "value"],
                },
            },
            {
                "id": "label",
                "kind": "text",
                "text": "Drug",
                "bounding_box": {"left": 0, "top": 40, "right": 30, "bottom": 60},
            },
            {
                "id": "value",
                "kind": "text",
                "text": "Jardiance",
                "bounding_box": {"left": 40, "top": 40, "right": 80, "bottom": 60},
            },
        ],
    )
    output["result"]["pages"][0]["text"]["evidence_ids"] = ["field"]
    _write(outputs / "case.json", output)

    handwriting = evaluate_challenge_set(annotations, outputs)["handwriting"]

    assert handwriting["legible_exact_recovery"] == {
        "eligible": 1,
        "recovered": 1,
        "rate": 1.0,
    }


def test_handwriting_bbox_metrics_use_optimal_one_to_one_assignment(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(
        annotations / "case.json",
        _annotation(
            "C07-D003-P001",
            "dose dose",
            handwriting=[
                {"text": "dose", "legibility": "legible", "bbox": [0, 0, 10, 30]},
                {"text": "dose", "legibility": "legible", "bbox": [0, 0, 10, 20]},
            ],
        ),
    )
    _write(
        outputs / "case.json",
        _output(
            "C07-D003-P001.png",
            "dose dose",
            regions=[
                {
                    "id": "x",
                    "kind": "text",
                    "text": "dose",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 30},
                },
                {
                    "id": "y",
                    "kind": "text",
                    "text": "dose",
                    "bounding_box": {"left": 0, "top": 0, "right": 10, "bottom": 50},
                },
            ],
        ),
    )

    handwriting = evaluate_challenge_set(annotations, outputs)["handwriting"]

    assert handwriting["matching_policy"] == (
        "hungarian_one_to_one_exact_text_max_cardinality_then_iou"
    )
    assert handwriting["legible_exact_recovery"] == {
        "eligible": 2,
        "recovered": 2,
        "rate": 1.0,
    }
    assert handwriting["localized_exact_recovery"]["iou_0_7"]["legible"] == {
        "eligible": 2,
        "recovered": 1,
        "rate": 0.5,
    }


def test_latency_buckets_invalid_values_on_annotated_outputs(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    _write(annotations / "case.json", _annotation("C08-D001-P001", "Text"))
    output = _output("C08-D001-P001.png", "Text", total_seconds=-1.0)
    output["timing"]["pipeline_steps"] = {
        "reader": float("nan"),
        "prepare": 0.25,
    }
    _write(outputs / "case.json", output)

    latency = evaluate_challenge_set(annotations, outputs)["latency_seconds"]

    assert latency["total"]["count"] == 0
    assert latency["pipeline_steps"] == {
        "prepare": {"count": 1, "p50": 0.25, "p95": 0.25, "max": 0.25}
    }
    assert latency["invalid"] == {
        "total": 1,
        "pipeline_steps": {"reader": 1},
    }


def test_cli_writes_aggregate_json_without_stdout(tmp_path: Path, capsys) -> None:
    annotations = tmp_path / "annotations"
    outputs = tmp_path / "outputs"
    report_path = tmp_path / "report" / "aggregate.json"
    _write(annotations / "case.json", _annotation("C03-D001-P001", "Exact"))
    _write(outputs / "case.json", _output("C03-D001-P001.png", "Exact"))

    exit_code = main([str(annotations), str(outputs), str(report_path)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["transcription"]["cer"] == 0.0
    assert "Exact" not in report_path.read_text(encoding="utf-8")
