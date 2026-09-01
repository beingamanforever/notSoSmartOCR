"""Measure geometry-first OCR fusion on two manually transcribed tables."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Sequence

PROJECTED_2026 = (
    ("", "1Q26", "2Q26", "3Q26E", "4Q26E"),
    ("Adj. CARR", "25,971,901", "29,471,901", "32,450,651", "37,951,401"),
    ("CARR", "24,013,833", "27,513,833", "30,303,833", "35,804,583"),
    ("Total Revenue", "7,695,223", "7,667,380", "9,476,244", "11,167,018"),
    ("SaaS revenue", "6,186,374", "6,003,456", "6,396,583", "7,500,542"),
    ("Imp revenue", "1,322,916", "1,211,583", "2,471,830", "3,058,647"),
    ("Services Revenue", "185,933", "452,340", "607,830", "607,830"),
    ("COGS-SaaS", "665,609", "649,343", "704,062", "854,062"),
    ("COGS-Imp & T&M", "1,473,156", "1,817,688", "1,851,102", "1,941,102"),
    ("SaaS GM%", "89%", "89%", "89%", "89%"),
    ("Overall GM%", "72%", "68%", "73%", "75%"),
    ("Gross Profit", "5,556,458", "5,200,349", "6,921,081", "8,371,855"),
    ("S&M", "1,451,201", "1,515,391", "1,544,808", "1,604,808"),
    ("R&D", "4,697,613", "4,581,522", "4,558,322", "4,558,322"),
    ("G&A", "1,359,642", "1,798,663", "1,605,451", "1,755,451"),
    ("EBITDA", "(1,951,997)", "(2,695,227)", "(787,500)", "453,274"),
    ("EBITDA %", "-25%", "-35%", "-8%", "4%"),
)

THREE_YEAR = (
    (
        "",
        "CY 2024",
        "CY 2026",
        "Growth (24 vs. 25)",
        "CY 2026E",
        "Growth (25 vs. 26)",
    ),
    ("Adj. CARR", "21,237,344", "28,050,651", "+32%", "37,951,401", "+35%"),
    ("CARR", "19,536,914", "25,283,833", "+29%", "35,804,583", "+42%"),
    ("Total Revenue", "18,896,979", "27,919,313", "+48%", "36,005,865", "+29%"),
    ("SaaS revenue", "11,853,868", "21,393,997", "+80%", "26,086,956", "+22%"),
    ("Imp revenue", "4,728,242", "3,100,779", "", "8,064,976", ""),
    ("Services Revenue", "2,314,070", "3,424,537", "", "1,853,933", ""),
    ("COGS-SaaS", "1,451,159", "2,341,437", "", "2,873,074", ""),
    ("COGS-Imp & T&M", "3,656,927", "5,805,938", "", "7,083,047", ""),
    ("SaaS GM%", "88%", "89%", "", "89%", ""),
    ("Overall GM%", "73%", "71%", "", "72%", ""),
    ("Gross Profit", "13,788,094", "19,771,938", "", "26,049,743", ""),
    ("S&M", "4,866,384", "4,871,873", "", "6,116,208", ""),
    ("R&D", "10,624,826", "14,366,911", "", "18,395,779", ""),
    ("G&A", "4,544,479", "5,974,154", "", "6,519,207", ""),
    ("EBITDA", "(6,247,596)", "(5,441,001)", "", "(4,981,449)", ""),
    ("EBITDA %", "-33%", "-19%", "", "-14%", ""),
)

GROUND_TRUTH = {
    "projected-2026": PROJECTED_2026,
    "three-year": THREE_YEAR,
}
MIN_RAW_CONFIDENCE = 0.7
MIN_PRIMARY_CONFIDENCE = 0.9
DETECTION_CROP_PADDING = 5
VALUE_PATTERN = re.compile(r"[+-]?\(?\d[\d,]*(?:\.\d+)?%?\)?")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare raw, enhanced, and fused financial table OCR"
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("tables_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nemotron-results", type=Path)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        report = run_experiment(
            args.results,
            args.tables_root,
            nemotron_results_path=args.nemotron_results,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


def run_experiment(
    results_path: Path,
    tables_root: Path,
    *,
    nemotron_results_path: Path | None = None,
) -> dict[str, Any]:
    records = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("TATR results must be a JSON list")
    nemotron_records = (
        _load_last_json(nemotron_results_path)
        if nemotron_results_path is not None
        else None
    )

    cases = []
    for case_name, truth in GROUND_TRUTH.items():
        raw_record = _record(records, case_name, "raw")
        enhanced_record = _record(records, case_name, "sauvola")
        raw_cells = _single_table(raw_record)
        enhanced_cells = _single_table(enhanced_record)
        case_root = tables_root / case_name
        raw_tokens = _load_tokens(case_root / "raw-tokens.json")
        enhanced_tokens = _load_tokens(case_root / "sauvola-tokens.json")
        fused_cells = fuse_cells(enhanced_cells, raw_tokens, enhanced_tokens)
        case_result = {
            "case": case_name,
            "ground_truth_shape": [len(truth), len(truth[0])],
            "raw": score_table(raw_cells, truth),
            "enhanced": score_table(enhanced_cells, truth),
            "fused": score_table(fused_cells, truth),
            "fusion_sources": _source_counts(fused_cells),
            "fusion_changes": _fusion_changes(fused_cells),
        }
        if nemotron_records is not None:
            (
                nemotron_cells,
                nemotron_tokens,
                detection_box,
                elapsed_seconds,
            ) = _nemotron_table(nemotron_records, case_name)
            source_box = _source_box(case_root / "metadata.json")
            offset = (
                source_box[0] - (detection_box[0] - DETECTION_CROP_PADDING),
                source_box[1] - (detection_box[1] - DETECTION_CROP_PADDING),
            )
            tri_cells = tri_fuse_cells(
                nemotron_cells,
                nemotron_tokens,
                translate_tokens(raw_tokens, *offset),
                translate_tokens(enhanced_tokens, *offset),
            )
            case_result["nemotron_raw"] = score_table(nemotron_cells, truth)
            case_result["tri_fused"] = score_table(tri_cells, truth)
            case_result["tri_sources"] = _source_counts(tri_cells)
            case_result["tri_changes"] = _tri_changes(tri_cells)
            case_result["nemotron_operations"] = {
                "elapsed_seconds": elapsed_seconds,
                "challenger_offset": [round(value, 6) for value in offset],
            }
        cases.append(case_result)

    modes = ["raw", "enhanced", "fused"]
    if nemotron_records is not None:
        modes.extend(("nemotron_raw", "tri_fused"))

    report = {
        "method": (
            "TATR-v1.1-All enhanced-view cell geometry; raw tokens retained "
            "unless absent or below 0.7 mean confidence with a more confident "
            "enhanced candidate"
        ),
        "normalization": (
            "Unicode NFKC, case-folding, whitespace removal, and removal of "
            "dollar or section-sign currency glyphs"
        ),
        "cases": cases,
        "aggregate": {mode: _aggregate(cases, mode) for mode in modes},
    }
    if nemotron_records is not None:
        report["tri_method"] = (
            "full-page TATR detection and TATR-v1.1-All geometry; Nemotron raw "
            "tokens are primary; translated Tesseract raw and Sauvola tokens "
            "challenge only missing, low-confidence, or overlapping primary text"
        )
    return report


def fuse_cells(
    structure_cells: list[dict[str, Any]],
    raw_tokens: list[dict[str, Any]],
    enhanced_tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill fixed cell geometry without letting enhancement overwrite raw text."""
    raw_by_cell = assign_tokens(structure_cells, raw_tokens)
    enhanced_by_cell = assign_tokens(structure_cells, enhanced_tokens)
    fused = []
    for cell in structure_cells:
        key = _cell_key(cell)
        raw_text = join_tokens(raw_by_cell.get(key, []))
        enhanced_text = join_tokens(enhanced_by_cell.get(key, []))
        raw_confidence = _mean_confidence(raw_by_cell.get(key, []))
        enhanced_confidence = _mean_confidence(enhanced_by_cell.get(key, []))
        use_enhanced = (
            normalize_cell(enhanced_text)
            and raw_confidence is not None
            and enhanced_confidence is not None
            and raw_confidence < MIN_RAW_CONFIDENCE
            and enhanced_confidence > raw_confidence
        )
        if normalize_cell(raw_text) and not use_enhanced:
            text = raw_text
            source = "raw"
        elif normalize_cell(enhanced_text):
            text = enhanced_text
            source = "enhanced"
        else:
            text = ""
            source = "blank"
        fused.append(
            {
                "bbox": cell["bbox"],
                "row_nums": cell["row_nums"],
                "column_nums": cell["column_nums"],
                "cell text": text,
                "text_source": source,
                "raw text": raw_text,
                "enhanced text": enhanced_text,
                "raw confidence": raw_confidence,
                "enhanced confidence": enhanced_confidence,
            }
        )
    return fused


def tri_fuse_cells(
    structure_cells: list[dict[str, Any]],
    primary_tokens: list[dict[str, Any]],
    raw_tokens: list[dict[str, Any]],
    enhanced_tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Challenge Nemotron cell text only with stronger local evidence."""
    primary_by_cell = assign_tokens(structure_cells, primary_tokens)
    raw_by_cell = assign_tokens(structure_cells, raw_tokens)
    enhanced_by_cell = assign_tokens(structure_cells, enhanced_tokens)
    fused = []
    for cell in structure_cells:
        key = _cell_key(cell)
        primary = _candidate("nemotron", primary_by_cell.get(key, []))
        raw = _candidate("tesseract_raw", raw_by_cell.get(key, []))
        enhanced = _candidate("sauvola", enhanced_by_cell.get(key, []))
        challenger = _best_challenger(raw, enhanced)
        selected = primary
        reason = "primary"
        if not normalize_cell(primary["text"]) and challenger["agreed"]:
            selected = challenger
            reason = "primary_missing"
        elif _overlap_conflict(primary["tokens"]) and _supports_primary_value(
            primary, challenger
        ):
            selected = challenger
            reason = "overlap_conflict"
        elif _low_confidence_challenge(primary, challenger):
            selected = challenger
            reason = "low_primary_confidence"

        fused.append(
            {
                "bbox": cell["bbox"],
                "row_nums": cell["row_nums"],
                "column_nums": cell["column_nums"],
                "cell text": selected["text"],
                "text_source": selected["source"],
                "decision_reason": reason,
                "primary text": primary["text"],
                "raw text": raw["text"],
                "enhanced text": enhanced["text"],
                "primary confidence": primary["confidence"],
                "raw confidence": raw["confidence"],
                "enhanced confidence": enhanced["confidence"],
            }
        )
    return fused


def translate_tokens(
    tokens: list[dict[str, Any]], offset_x: float, offset_y: float
) -> list[dict[str, Any]]:
    translated = []
    for token in tokens:
        box = _box(token.get("bbox"))
        translated.append(
            {
                **token,
                "bbox": [
                    box[0] + offset_x,
                    box[1] + offset_y,
                    box[2] + offset_x,
                    box[3] + offset_y,
                ],
            }
        )
    return translated


def _candidate(source: str, tokens: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": source,
        "text": join_tokens(tokens),
        "confidence": _mean_confidence(tokens),
        "tokens": tokens,
        "agreed": False,
    }


def _best_challenger(raw: dict[str, Any], enhanced: dict[str, Any]) -> dict[str, Any]:
    raw_text = normalize_cell(raw["text"])
    enhanced_text = normalize_cell(enhanced["text"])
    if not raw_text and not enhanced_text:
        return {**raw, "agreed": False}
    if not raw_text:
        return {**enhanced, "agreed": True}
    if not enhanced_text:
        return {**raw, "agreed": True}
    agreed = raw_text == enhanced_text or (
        _value_signature(raw["text"])
        and _value_signature(raw["text"]) == _value_signature(enhanced["text"])
    )
    selected = max((raw, enhanced), key=lambda item: _confidence(item["confidence"]))
    return {**selected, "agreed": bool(agreed)}


def _supports_primary_value(
    primary: dict[str, Any], challenger: dict[str, Any]
) -> bool:
    primary_value = _value_signature(primary["text"])
    challenger_value = _value_signature(challenger["text"])
    return bool(
        normalize_cell(challenger["text"])
        and challenger["agreed"]
        and primary_value
        and primary_value == challenger_value
    )


def _low_confidence_challenge(
    primary: dict[str, Any], challenger: dict[str, Any]
) -> bool:
    primary_confidence = primary["confidence"]
    challenger_confidence = challenger["confidence"]
    return bool(
        challenger["agreed"]
        and primary_confidence is not None
        and challenger_confidence is not None
        and primary_confidence < MIN_PRIMARY_CONFIDENCE
        and challenger_confidence > primary_confidence
    )


def _overlap_conflict(tokens: list[dict[str, Any]]) -> bool:
    meaningful = [
        token
        for token in tokens
        if normalize_cell(token.get("text", ""))
        and normalize_cell(token.get("text", "")) not in {"$", "§"}
    ]
    for index, first in enumerate(meaningful):
        first_box = _box(first.get("bbox"))
        first_text = normalize_cell(first["text"])
        for second in meaningful[index + 1 :]:
            second_box = _box(second.get("bbox"))
            second_text = normalize_cell(second["text"])
            smaller_area = min(_area(first_box), _area(second_box))
            if smaller_area <= 0:
                continue
            overlap = _intersection_area(first_box, second_box) / smaller_area
            nested_text = first_text in second_text or second_text in first_text
            if overlap >= 0.3 and nested_text:
                return True
    return False


def _value_signature(text: str) -> str:
    matches = VALUE_PATTERN.findall(str(text))
    if not matches:
        return ""
    value = max(
        matches, key=lambda item: (sum(char.isdigit() for char in item), len(item))
    )
    return normalize_cell(value)


def _confidence(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else -1.0


def assign_tokens(
    cells: list[dict[str, Any]], tokens: list[dict[str, Any]]
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Assign each token once to the cell covering most of its area."""
    assigned: dict[tuple[int, int], list[dict[str, Any]]] = {}
    grid = _span_grid(cells)
    if grid is not None:
        row_centers, column_centers, table_box = grid
        for token in tokens:
            token_box = _box(token.get("bbox"))
            if not str(token.get("text", "")).strip():
                continue
            center_x = (token_box[0] + token_box[2]) / 2
            center_y = (token_box[1] + token_box[3]) / 2
            if not _contains(table_box, center_x, center_y):
                continue
            row = min(row_centers, key=lambda key: abs(row_centers[key] - center_y))
            column = min(
                column_centers,
                key=lambda key: abs(column_centers[key] - center_x),
            )
            assigned.setdefault((row, column), []).append(token)
        return assigned

    valid_cells = [(cell, _box(cell.get("bbox"))) for cell in cells]
    for token in tokens:
        token_box = _box(token.get("bbox"))
        token_area = _area(token_box)
        if token_area <= 0 or not str(token.get("text", "")).strip():
            continue
        best: tuple[float, float, dict[str, Any]] | None = None
        center_x = (token_box[0] + token_box[2]) / 2
        center_y = (token_box[1] + token_box[3]) / 2
        for cell, cell_box in valid_cells:
            overlap = _intersection_area(token_box, cell_box) / token_area
            contains_center = (
                cell_box[0] <= center_x <= cell_box[2]
                and cell_box[1] <= center_y <= cell_box[3]
            )
            if overlap < 0.5 and not contains_center:
                continue
            rank = (float(contains_center) + overlap, -_area(cell_box), cell)
            if best is None or rank[:2] > best[:2]:
                best = rank
        if best is None:
            continue
        key = _cell_key(best[2])
        assigned.setdefault(key, []).append(token)
    return assigned


def _span_grid(
    cells: list[dict[str, Any]],
) -> (
    tuple[dict[int, float], dict[int, float], tuple[float, float, float, float]] | None
):
    rows: dict[int, list[float]] = {}
    columns: dict[int, list[float]] = {}
    cell_boxes = []
    for cell in cells:
        row, column = _cell_key(cell)
        cell_boxes.append(_box(cell.get("bbox")))
        spans = cell.get("spans")
        if not isinstance(spans, list):
            return None
        for span in spans:
            span_box = _box(span.get("bbox"))
            rows.setdefault(row, []).append((span_box[1] + span_box[3]) / 2)
            columns.setdefault(column, []).append((span_box[0] + span_box[2]) / 2)
    expected_rows = {_cell_key(cell)[0] for cell in cells}
    expected_columns = {_cell_key(cell)[1] for cell in cells}
    if rows.keys() != expected_rows or columns.keys() != expected_columns:
        return None
    table_box = (
        min(box[0] for box in cell_boxes),
        min(box[1] for box in cell_boxes),
        max(box[2] for box in cell_boxes),
        max(box[3] for box in cell_boxes),
    )
    return (
        {key: median(values) for key, values in rows.items()},
        {key: median(values) for key, values in columns.items()},
        table_box,
    )


def join_tokens(tokens: list[dict[str, Any]]) -> str:
    return " ".join(
        str(token["text"]).strip()
        for token in sorted(tokens, key=_token_order)
        if str(token.get("text", "")).strip()
    )


def score_table(
    cells: list[dict[str, Any]], truth: tuple[tuple[str, ...], ...]
) -> dict[str, Any]:
    predicted, present, extra_cells, source_shape = _aligned_matrix(cells, truth)
    failures = []
    exact = 0
    nonempty_exact = 0
    nonempty_total = 0
    numeric_exact = 0
    numeric_total = 0
    structural_missing = 0
    missed_text = 0
    hallucinated_text = 0
    wrong_text = 0

    for row_index, truth_row in enumerate(truth):
        for column_index, expected in enumerate(truth_row):
            key = (row_index, column_index)
            actual = predicted[row_index][column_index]
            expected_norm = normalize_cell(expected)
            actual_norm = normalize_cell(actual)
            if expected_norm:
                nonempty_total += 1
                if column_index > 0:
                    numeric_total += 1
            if key in present and expected_norm == actual_norm:
                exact += 1
                if expected_norm:
                    nonempty_exact += 1
                    if column_index > 0:
                        numeric_exact += 1
                continue

            if key not in present:
                reason = "structural_missing"
                structural_missing += 1
            elif expected_norm and not actual_norm:
                reason = "missed_text"
                missed_text += 1
            elif not expected_norm and actual_norm:
                reason = "hallucinated_text"
                hallucinated_text += 1
            else:
                reason = "wrong_text"
                wrong_text += 1
            failures.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "expected": expected,
                    "predicted": actual,
                    "reason": reason,
                }
            )

    expected_cells = len(truth) * len(truth[0])
    return {
        "source_shape": list(source_shape),
        "expected_cells": expected_cells,
        "predicted_cells": len(cells),
        "extra_cells": extra_cells,
        "shape_match": source_shape == (len(truth), len(truth[0])),
        "cell_exact_match": _ratio(exact, expected_cells),
        "nonempty_cell_exact_match": _ratio(nonempty_exact, nonempty_total),
        "value_cell_exact_match": _ratio(numeric_exact, numeric_total),
        "exact_cells": exact,
        "structural_missing_cells": structural_missing,
        "missed_text_cells": missed_text,
        "hallucinated_text_cells": hallucinated_text,
        "wrong_text_cells": wrong_text,
        "failures": failures,
    }


def normalize_cell(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = value.replace("−", "-").replace("–", "-")
    value = re.sub(r"[$§\s]", "", value)
    return value


def _aligned_matrix(
    cells: list[dict[str, Any]], truth: tuple[tuple[str, ...], ...]
) -> tuple[list[list[str]], set[tuple[int, int]], int, tuple[int, int]]:
    row_count = max((_cell_key(cell)[0] for cell in cells), default=-1) + 1
    column_count = max((_cell_key(cell)[1] for cell in cells), default=-1) + 1
    source_rows: dict[int, dict[int, dict[str, Any]]] = {}
    for cell in cells:
        row_index, column_index = _cell_key(cell)
        source_rows.setdefault(row_index, {})[column_index] = cell

    row_map = _align_rows(source_rows, truth)
    matrix = [["" for _ in row] for row in truth]
    present: set[tuple[int, int]] = set()
    extra_cells = 0
    for source_row, columns in source_rows.items():
        target_row = row_map.get(source_row)
        if target_row is None:
            extra_cells += len(columns)
            continue
        for column_index, cell in columns.items():
            if column_index >= len(truth[target_row]):
                extra_cells += 1
                continue
            matrix[target_row][column_index] = str(cell.get("cell text", ""))
            present.add((target_row, column_index))
    return matrix, present, extra_cells, (row_count, column_count)


def _align_rows(
    rows: dict[int, dict[int, dict[str, Any]]],
    truth: tuple[tuple[str, ...], ...],
) -> dict[int, int]:
    if not rows:
        return {}
    mapping = {min(rows): 0}
    last_target = 0
    for source_row in sorted(rows)[1:]:
        label = str(rows[source_row].get(0, {}).get("cell text", ""))
        choices = range(last_target + 1, len(truth))
        scored = [
            (
                SequenceMatcher(
                    None, normalize_cell(label), normalize_cell(truth[index][0])
                ).ratio(),
                index,
            )
            for index in choices
        ]
        if not scored:
            continue
        score, target_row = max(scored)
        if score < 0.55:
            continue
        mapping[source_row] = target_row
        last_target = target_row
    return mapping


def _record(records: list[dict[str, Any]], case_name: str, view: str) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record.get("model") == "all"
        and record.get("case") == case_name
        and record.get("view") == view
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one TATR-v1.1-All {case_name}/{view} result, got {len(matches)}"
        )
    return matches[0]


def _single_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    tables = record.get("cells")
    if not isinstance(tables, list) or len(tables) != 1:
        raise ValueError("Each result must contain exactly one table")
    cells = tables[0]
    if not isinstance(cells, list):
        raise ValueError("Table cells must be a list")
    return cells


def _load_tokens(path: Path) -> list[dict[str, Any]]:
    tokens = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tokens, list):
        raise ValueError(f"Token file must contain a list: {path}")
    return tokens


def _load_last_json(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"Nemotron result file is empty: {path}")
    records = json.loads(lines[-1])
    if not isinstance(records, list):
        raise ValueError("Nemotron results must end with a JSON list")
    return records


def _nemotron_table(
    records: list[dict[str, Any]], case_name: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[float, float, float, float],
    float,
]:
    source_name = "projected" if case_name == "projected-2026" else case_name
    matches = [record for record in records if record.get("case") == source_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Nemotron {source_name} result, got {len(matches)}"
        )
    record = matches[0]
    tables = record.get("tables")
    if not isinstance(tables, list) or len(tables) != 1:
        raise ValueError(f"Nemotron {source_name} must contain exactly one table")
    table = tables[0]
    cells = _single_table(table)
    tokens = table.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"Nemotron {source_name} lacks table tokens")
    detections = record.get("detection_objects")
    if not isinstance(detections, list) or len(detections) != 1:
        raise ValueError(f"Nemotron {source_name} must have one table detection")
    detection_box = _box(detections[0].get("bbox"))
    elapsed = record.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)):
        raise ValueError(f"Nemotron {source_name} lacks elapsed_seconds")
    return cells, tokens, detection_box, float(elapsed)


def _source_box(path: Path) -> tuple[float, float, float, float]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Table metadata must be an object: {path}")
    return _box(metadata.get("source_box"))


def _source_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        source = str(cell["text_source"])
        counts[source] = counts.get(source, 0) + 1
    return counts


def _fusion_changes(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row": _cell_key(cell)[0],
            "column": _cell_key(cell)[1],
            "source": cell["text_source"],
            "raw": cell["raw text"],
            "enhanced": cell["enhanced text"],
            "raw_confidence": cell["raw confidence"],
            "enhanced_confidence": cell["enhanced confidence"],
        }
        for cell in cells
        if cell["text_source"] == "enhanced"
        and normalize_cell(cell["raw text"]) != normalize_cell(cell["enhanced text"])
    ]


def _tri_changes(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row": _cell_key(cell)[0],
            "column": _cell_key(cell)[1],
            "source": cell["text_source"],
            "reason": cell["decision_reason"],
            "primary": cell["primary text"],
            "selected": cell["cell text"],
            "raw": cell["raw text"],
            "enhanced": cell["enhanced text"],
            "primary_confidence": cell["primary confidence"],
            "raw_confidence": cell["raw confidence"],
            "enhanced_confidence": cell["enhanced confidence"],
        }
        for cell in cells
        if cell["text_source"] != "nemotron"
    ]


def _mean_confidence(tokens: list[dict[str, Any]]) -> float | None:
    values = [
        float(token["confidence"])
        for token in tokens
        if isinstance(token.get("confidence"), (int, float))
    ]
    return sum(values) / len(values) if values else None


def _aggregate(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    expected = sum(case[mode]["expected_cells"] for case in cases)
    exact = sum(case[mode]["exact_cells"] for case in cases)
    return {
        "tables": len(cases),
        "shape_matches": sum(case[mode]["shape_match"] for case in cases),
        "expected_cells": expected,
        "exact_cells": exact,
        "cell_exact_match": _ratio(exact, expected),
        "structural_missing_cells": sum(
            case[mode]["structural_missing_cells"] for case in cases
        ),
        "missed_text_cells": sum(case[mode]["missed_text_cells"] for case in cases),
        "hallucinated_text_cells": sum(
            case[mode]["hallucinated_text_cells"] for case in cases
        ),
        "wrong_text_cells": sum(case[mode]["wrong_text_cells"] for case in cases),
    }


def _cell_key(cell: dict[str, Any]) -> tuple[int, int]:
    rows = cell.get("row_nums")
    columns = cell.get("column_nums")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("Fusion requires one row per cell")
    if not isinstance(columns, list) or len(columns) != 1:
        raise ValueError("Fusion requires one column per cell")
    return int(rows[0]), int(columns[0])


def _box(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Bounding boxes must contain four coordinates")
    box = tuple(float(coordinate) for coordinate in value)
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("Bounding box coordinates are reversed")
    return box


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _contains(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _token_order(token: dict[str, Any]) -> tuple[float, float, float]:
    span = token.get("span_num")
    box = _box(token.get("bbox"))
    if isinstance(span, int):
        return float(span), box[1], box[0]
    return float("inf"), box[1], box[0]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
