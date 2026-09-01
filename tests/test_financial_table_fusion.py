from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import financial_table_fusion as fusion


def test_fusion_preserves_raw_text_and_fills_only_missing_cells() -> None:
    cells = [_cell(0, 0, [0, 0, 50, 20]), _cell(0, 1, [50, 0, 100, 20])]
    raw = [_token("raw", [5, 2, 30, 18], 0)]
    enhanced = [
        _token("changed", [5, 2, 30, 18], 0),
        _token("recovered", [60, 2, 95, 18], 1),
    ]

    result = fusion.fuse_cells(cells, raw, enhanced)

    assert [cell["cell text"] for cell in result] == ["raw", "recovered"]
    assert [cell["text_source"] for cell in result] == ["raw", "enhanced"]


def test_fusion_replaces_low_confidence_raw_text_only_when_safer() -> None:
    cells = [_cell(0, 0, [0, 0, 50, 20]), _cell(0, 1, [50, 0, 100, 20])]
    raw = [
        {**_token("wrong", [5, 2, 30, 18], 0), "confidence": 0.2},
        {**_token("stable", [60, 2, 95, 18], 1), "confidence": 0.9},
    ]
    enhanced = [
        {**_token("fixed", [5, 2, 30, 18], 0), "confidence": 0.95},
        {**_token("changed", [60, 2, 95, 18], 1), "confidence": 0.99},
    ]

    result = fusion.fuse_cells(cells, raw, enhanced)

    assert [cell["cell text"] for cell in result] == ["fixed", "stable"]
    assert [cell["text_source"] for cell in result] == ["enhanced", "raw"]


def test_tri_fusion_repairs_conflicts_but_preserves_strong_primary_text() -> None:
    cells = [
        _cell(0, 0, [0, 0, 100, 20]),
        _cell(0, 1, [100, 0, 200, 20]),
        _cell(0, 2, [200, 0, 300, 20]),
    ]
    primary = [
        _conf_token("6,186", [10, 2, 45, 18], 0, 0.94),
        _conf_token("6,186,374", [10, 2, 90, 18], 1, 0.95),
        _conf_token("(6,247,590", [110, 2, 190, 18], 2, 0.88),
        _conf_token("SaaS GM%", [210, 2, 290, 18], 3, 0.94),
    ]
    raw = [
        _conf_token("6,186,374", [10, 2, 90, 18], 0, 0.96),
        _conf_token("(6,247,596)", [110, 2, 190, 18], 1, 0.95),
        _conf_token("aaS GM%", [210, 2, 290, 18], 2, 0.99),
    ]
    enhanced = [
        _conf_token("6,186,374", [10, 2, 90, 18], 0, 0.97),
        _conf_token("(6,247,596)", [110, 2, 190, 18], 1, 0.94),
        _conf_token("aaS GM%", [210, 2, 290, 18], 2, 0.99),
    ]

    result = fusion.tri_fuse_cells(cells, primary, raw, enhanced)

    assert [cell["cell text"] for cell in result] == [
        "6,186,374",
        "(6,247,596)",
        "SaaS GM%",
    ]
    assert [cell["decision_reason"] for cell in result] == [
        "overlap_conflict",
        "low_primary_confidence",
        "primary",
    ]


def test_translate_tokens_keeps_input_and_applies_crop_offset() -> None:
    tokens = [_token("value", [1, 2, 3, 4], 0)]

    translated = fusion.translate_tokens(tokens, 10.5, -1.5)

    assert translated[0]["bbox"] == [11.5, 0.5, 13.5, 2.5]
    assert tokens[0]["bbox"] == [1, 2, 3, 4]


def test_overlapping_token_is_assigned_once_to_best_cell() -> None:
    cells = [_cell(0, 0, [0, 0, 60, 20]), _cell(0, 1, [40, 0, 100, 20])]
    token = _token("value", [45, 2, 55, 18], 0)

    assigned = fusion.assign_tokens(cells, [token])

    assert sum(len(tokens) for tokens in assigned.values()) == 1
    assert assigned[(0, 0)] == [token]


def test_span_grid_avoids_overlapping_union_box_row_errors() -> None:
    cells = [
        _spanned_cell(0, [0, 0, 100, 20], [10, 2, 30, 18]),
        _spanned_cell(1, [0, 0, 100, 60], [10, 22, 30, 38]),
        _spanned_cell(2, [0, 0, 100, 60], [10, 42, 30, 58]),
    ]
    token = _token("middle", [12, 24, 35, 36], 0)

    assigned = fusion.assign_tokens(cells, [token])

    assert assigned == {(1, 0): [token]}


def test_score_keeps_missing_blank_cells_in_denominator() -> None:
    truth = (("", "A"), ("row", ""))
    cells = [
        {**_cell(0, 0, [0, 0, 10, 10]), "cell text": ""},
        {**_cell(0, 1, [10, 0, 20, 10]), "cell text": "A"},
    ]

    result = fusion.score_table(cells, truth)

    assert result["cell_exact_match"] == 0.5
    assert result["exact_cells"] == 2
    assert result["structural_missing_cells"] == 2
    assert result["missed_text_cells"] == 0


def test_row_labels_align_a_structure_with_missing_middle_rows() -> None:
    truth = (("", "h"), ("first", "1"), ("light", "2"), ("last", "3"))
    cells = [
        {**_cell(0, 0, [0, 0, 10, 10]), "cell text": ""},
        {**_cell(0, 1, [10, 0, 20, 10]), "cell text": "h"},
        {**_cell(1, 0, [0, 10, 10, 20]), "cell text": "first"},
        {**_cell(1, 1, [10, 10, 20, 20]), "cell text": "1"},
        {**_cell(2, 0, [0, 30, 10, 40]), "cell text": "last"},
        {**_cell(2, 1, [10, 30, 20, 40]), "cell text": "3"},
    ]

    result = fusion.score_table(cells, truth)

    assert result["cell_exact_match"] == 0.75
    assert result["structural_missing_cells"] == 2
    assert result["shape_match"] is False


def test_normalization_drops_only_known_currency_and_spacing_noise() -> None:
    assert fusion.normalize_cell(" § $ 1,234 ") == "1,234"
    assert fusion.normalize_cell("(1,234)") != fusion.normalize_cell("1,234")
    assert fusion.normalize_cell("+32%") != fusion.normalize_cell("32%")


def test_cli_scores_raw_enhanced_and_fused_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = (("", "Header"), ("Light", "42"))
    monkeypatch.setattr(fusion, "GROUND_TRUTH", {"case": truth})
    header = [
        _result_cell(0, 0, [0, 0, 50, 20], "", []),
        _result_cell(0, 1, [50, 0, 100, 20], "Header", [[60, 2, 90, 18]]),
    ]
    enhanced = header + [
        _result_cell(1, 0, [0, 20, 50, 40], "Light", [[5, 22, 35, 38]]),
        _result_cell(1, 1, [50, 20, 100, 40], "42", [[60, 22, 90, 38]]),
    ]
    results = tmp_path / "results.json"
    _write_json(
        results,
        [
            {"model": "all", "case": "case", "view": "raw", "cells": [header]},
            {
                "model": "all",
                "case": "case",
                "view": "sauvola",
                "cells": [enhanced],
            },
        ],
    )
    case_root = tmp_path / "tables" / "case"
    case_root.mkdir(parents=True)
    _write_json(
        case_root / "raw-tokens.json",
        [{**_token("Header", [60, 2, 90, 18], 0), "confidence": 0.9}],
    )
    _write_json(
        case_root / "sauvola-tokens.json",
        [
            {**_token("Header", [60, 2, 90, 18], 0), "confidence": 0.9},
            {**_token("Light", [5, 22, 35, 38], 1), "confidence": 0.9},
            {**_token("42", [60, 22, 90, 38], 2), "confidence": 0.9},
        ],
    )
    _write_json(case_root / "metadata.json", {"source_box": [0, 0, 100, 40]})
    output = tmp_path / "report.json"
    nemotron = tmp_path / "nemotron.raw"
    nemotron.write_text(
        "model log\n"
        + json.dumps(
            [
                {
                    "case": "case",
                    "elapsed_seconds": 0.5,
                    "detection_objects": [{"bbox": [5, 5, 95, 35]}],
                    "tables": [
                        {
                            "cells": [enhanced],
                            "tokens": [
                                _conf_token("Header", [60, 2, 90, 18], 0, 0.9),
                                _conf_token("Light", [5, 22, 35, 38], 1, 0.9),
                                _conf_token("42", [60, 22, 90, 38], 2, 0.9),
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert (
        fusion.main(
            [
                str(results),
                str(tmp_path / "tables"),
                str(output),
                "--nemotron-results",
                str(nemotron),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate"]["raw"]["cell_exact_match"] == 0.5
    assert report["aggregate"]["enhanced"]["cell_exact_match"] == 1.0
    assert report["aggregate"]["fused"]["cell_exact_match"] == 1.0
    assert report["aggregate"]["nemotron_raw"]["cell_exact_match"] == 1.0
    assert report["aggregate"]["tri_fused"]["cell_exact_match"] == 1.0
    assert report["cases"][0]["nemotron_operations"] == {
        "elapsed_seconds": 0.5,
        "challenger_offset": [0.0, 0.0],
    }


def _cell(row: int, column: int, bbox: list[int]) -> dict[str, object]:
    return {
        "row_nums": [row],
        "column_nums": [column],
        "bbox": bbox,
        "cell text": "",
    }


def _token(text: str, bbox: list[int], span: int) -> dict[str, object]:
    return {"text": text, "bbox": bbox, "span_num": span}


def _conf_token(
    text: str, bbox: list[int], span: int, confidence: float
) -> dict[str, object]:
    return {**_token(text, bbox, span), "confidence": confidence}


def _spanned_cell(row: int, bbox: list[int], span_bbox: list[int]) -> dict[str, object]:
    return {
        **_cell(row, 0, bbox),
        "spans": [{"text": "anchor", "bbox": span_bbox}],
    }


def _result_cell(
    row: int,
    column: int,
    bbox: list[int],
    text: str,
    span_boxes: list[list[int]],
) -> dict[str, object]:
    return {
        **_cell(row, column, bbox),
        "cell text": text,
        "spans": [{"text": text, "bbox": span_box} for span_box in span_boxes],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
