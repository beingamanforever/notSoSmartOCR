"""Adapt live structured OCR evidence to the private handwritten benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from clinical_ocr_benchmark.metrics import normalize_text
from ocr_pipeline.falcon_layout import _checkbox_label


def load_pages(raw_pages: Path, ground_truth: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every image of the case, keyed by image_id (a case may span pages)."""
    pages = {}
    for image in ground_truth["images"]:
        page_id = f"{ground_truth['source_reference']}-p{image['page']:03d}"
        response = json.loads((raw_pages / page_id / "response.json").read_text())
        pages[image["image_id"]] = response["result"]["pages"][0]
    return pages


def label_key(value: str) -> str:
    """Label identity: benchmark normalization without edge punctuation ("Diabetes:")."""
    return normalize_text(value).strip(" :;,.")


def adapt_form(
    ground_truth: dict[str, Any], pages: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    controls = []
    for image_id, page in pages.items():
        targets = [
            target
            for target in ground_truth["target"]["controls"]
            if image_id in target["image_ids"]
        ]
        regions = sorted(
            (region for region in page["regions"] if region["kind"] == "checkbox"),
            key=lambda region: (
                region["bounding_box"]["top"],
                region["bounding_box"]["left"],
            ),
        )
        # Controls sharing a label ("Yes" under several questions) pair in reading
        # order; the ground-truth array order is the canonical reading order.
        by_label: dict[str, list[dict[str, Any]]] = {}
        for region in regions:
            structure = region.get("structure") or {}
            if structure.get("state") not in {"selected", "unselected"}:
                continue
            label = _checkbox_label(structure.get("label") or region.get("text") or "")
            by_label.setdefault(label_key(label), []).append(structure)
        used: set[int] = set()
        for target in targets:
            keys = {label_key(v) for v in [target["label"], *target.get("aliases", [])]}
            found = next(
                (
                    structure
                    for key in keys
                    for structure in by_label.get(key, [])
                    if id(structure) not in used
                ),
                None,
            )
            if found is None:
                continue
            used.add(id(found))
            controls.append(
                {
                    "control_id": target["control_id"],
                    "state": "checked" if found["state"] == "selected" else "unchecked",
                }
            )
    return {"controls": controls}


def cell_grid(table: dict[str, Any]) -> dict[tuple[int, int], str]:
    grid = {}
    for cell in (table.get("structure") or {}).get("cells", []):
        for row in cell.get("row_nums", []):
            for column in cell.get("column_nums", []):
                grid[(row, column)] = str(cell.get("text") or "")
    return grid


def contained(inner: dict[str, int], outer: dict[str, int]) -> bool:
    return all(
        (
            inner[edge] >= outer[edge]
            if edge in {"left", "top"}
            else inner[edge] <= outer[edge]
        )
        for edge in ("left", "top", "right", "bottom")
    )


def unique_tables(page: dict[str, Any]) -> list[dict[str, Any]]:
    tables = [region for region in page["regions"] if region["kind"] == "table"]
    keep = []
    for table in tables:
        values = {normalize_text(value) for value in cell_grid(table).values() if value}
        duplicate = False
        for other in tables:
            if table is other or not contained(table["bounding_box"], other["bounding_box"]):
                continue
            other_values = {
                normalize_text(value) for value in cell_grid(other).values() if value
            }
            if values and values <= other_values:
                duplicate = True
                break
        if not duplicate:
            keep.append(table)
    return sorted(
        keep,
        key=lambda table: (
            table["bounding_box"]["top"],
            table["bounding_box"]["left"],
        ),
    )


def labels(item: dict[str, Any]) -> set[str]:
    return {
        label_key(value)
        for value in [item["label"], *item.get("aliases", [])]
        if not normalize_text(value).startswith(("row-", "column-"))
    }


# Axes are aligned by content, monotonically with gaps, as table metrics align grids: a
# missing blank row or a split header must not shift every later row. Alignment only
# places predicted cells; the evaluator still scores their text exactly.
MIN_AXIS_SIMILARITY = 0.5


def similarity(first: str, second: str) -> float:
    first, second = label_key(first), label_key(second)
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def align(
    targets: list[tuple[str, str]], candidates: list[tuple[str, str]]
) -> dict[int, int]:
    """Monotonic target-to-candidate index mapping with the largest total similarity.

    Each axis is (header, content); a pair is as similar as its closer view, so matching
    headers anchor a column whose cells differ and matching cells anchor an unlabelled row.
    """
    n, m = len(targets), len(candidates)
    pair = [
        [max(similarity(t[0], c[0]), similarity(t[1], c[1])) for c in candidates]
        for t in targets
    ]
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            matched = pair[i][j] + score[i + 1][j + 1] if pair[i][j] >= MIN_AXIS_SIMILARITY else 0.0
            score[i][j] = max(score[i + 1][j], score[i][j + 1], matched)
    mapping, i, j = {}, 0, 0
    while i < n and j < m:
        if pair[i][j] >= MIN_AXIS_SIMILARITY and score[i][j] == pair[i][j] + score[i + 1][j + 1]:
            mapping[i] = j
            i, j = i + 1, j + 1
        elif score[i][j] == score[i + 1][j]:
            i += 1
        else:
            j += 1
    return mapping


def table_alignment(
    target: dict[str, Any], grid: dict[tuple[int, int], str]
) -> tuple[dict[int, int], dict[int, int]]:
    """Target row and column index to predicted grid line."""
    cells = {(c.get("row_id"), c.get("column_id")): c.get("text") or "" for c in target.get("cells", [])}
    rows = sorted({r for r, _ in grid})
    columns = sorted({c for _, c in grid})

    def label(axis):
        return axis["label"] if labels(axis) else ""

    target_columns = [
        (label(col), " ".join(cells.get((row["row_id"], col["column_id"]), "") for row in target["rows"]))
        for col in target["columns"]
    ]
    predicted_columns = [
        (first_text(grid, c, 1), " ".join(grid.get((r, c), "") for r in rows)) for c in columns
    ]
    column_map = {i: columns[j] for i, j in align(target_columns, predicted_columns).items()}
    # Rows compare over the mapped columns only.
    mapped = sorted(column_map.items())
    target_rows = [
        (label(row), " ".join(cells.get((row["row_id"], target["columns"][i]["column_id"]), "") for i, _ in mapped))
        for row in target["rows"]
    ]
    predicted_rows = [
        (first_text(grid, r, 0), " ".join(grid.get((r, line), "") for _, line in mapped)) for r in rows
    ]
    row_map = {i: rows[j] for i, j in align(target_rows, predicted_rows).items()}
    return row_map, column_map


def table_match_score(target: dict[str, Any], candidate: dict[str, Any]) -> int:
    row_map, column_map = table_alignment(target, cell_grid(candidate))
    return len(row_map) + len(column_map)


def first_text(grid: dict[tuple[int, int], str], line: int, coordinate: int) -> str:
    """A predicted row or column's own label: its first non-empty cell."""
    return next(
        (value for position, value in sorted(grid.items()) if position[coordinate] == line and value),
        "",
    )


def table_label(page: dict[str, Any], table: dict[str, Any]) -> str:
    """The nearest heading or caption directly above the table: the table's name."""
    box = table["bounding_box"]
    headings = [
        region
        for region in page["regions"]
        if region["kind"] in {"section-header", "title", "caption"}
        and region["bounding_box"]["bottom"] <= box["top"] + 5
        and region["bounding_box"]["right"] > box["left"]
        and region["bounding_box"]["left"] < box["right"]
    ]
    nearest = max(headings, key=lambda r: r["bounding_box"]["bottom"], default=None)
    return (nearest or {}).get("text") or ""


def adapt_table(
    target: dict[str, Any], candidate: dict[str, Any], label: str = ""
) -> dict[str, Any]:
    grid = cell_grid(candidate)
    row_map, column_map = table_alignment(target, grid)
    rows = [
        {"row_id": target["rows"][i]["row_id"], "label": first_text(grid, line, 0)}
        for i, line in sorted(row_map.items())
    ]
    columns = [
        {"column_id": target["columns"][i]["column_id"], "label": first_text(grid, line, 1)}
        for i, line in sorted(column_map.items())
    ]
    cells = [
        {
            "row_id": row["row_id"],
            "column_id": column["column_id"],
            "text": grid.get((row_map[row_index], column_map[column_index]), ""),
        }
        for row_index, row in enumerate(target["rows"])
        for column_index, column in enumerate(target["columns"])
        if row_index in row_map and column_index in column_map
    ]
    return {
        "table_id": target["table_id"],
        "label": label,
        "image_id": target["image_id"],
        "rows": rows,
        "columns": columns,
        "cells": cells,
        "merged_ranges": [],
    }


def adapt_tables(
    ground_truth: dict[str, Any], pages: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    candidates = [
        (image_id, table) for image_id, page in pages.items() for table in unique_tables(page)
    ]
    available = set(range(len(candidates)))
    tables = []
    for target in ground_truth["target"]["tables"]:
        index = max(
            (i for i in available if candidates[i][0] == target["image_id"]),
            key=lambda candidate_index: (
                table_match_score(target, candidates[candidate_index][1]),
                -candidate_index,
            ),
            default=None,
        )
        if index is None or table_match_score(target, candidates[index][1]) == 0:
            continue
        available.remove(index)
        image_id, table = candidates[index]
        tables.append(adapt_table(target, table, table_label(pages[image_id], table)))
    return {"tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("raw_pages", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cohort", action="append", default=None)
    args = parser.parse_args()
    cohorts = set(args.cohort or ["handwritten"])
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    adapters = {
        "forms": adapt_form,
        "tables": adapt_tables,
    }
    for ground_truth_path in sorted(args.data.glob("**/ground_truth.json")):
        ground_truth = json.loads(ground_truth_path.read_text())
        if ground_truth.get("cohort") not in cohorts or ground_truth.get(
            "availability"
        ) != "ready":
            continue
        pages = load_pages(args.raw_pages, ground_truth)
        task = ground_truth["task"]
        if task not in adapters:
            continue
        prediction = {
            "case_id": ground_truth["case_id"],
            "task": task,
            "prediction": adapters[task](ground_truth, pages),
        }
        (args.output / f"{ground_truth['case_id']}.json").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
