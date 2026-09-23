"""Adapt live structured OCR evidence to the private handwritten benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from clinical_ocr_benchmark.metrics import normalize_text
from ocr_pipeline.falcon_layout import _checkbox_label


def load_page(raw_pages: Path, ground_truth: dict[str, Any]) -> dict[str, Any]:
    page_number = ground_truth["images"][0]["page"]
    page_id = f"{ground_truth['source_reference']}-p{page_number:03d}"
    response = json.loads((raw_pages / page_id / "response.json").read_text())
    return response["result"]["pages"][0]


def adapt_form(ground_truth: dict[str, Any], page: dict[str, Any]) -> dict[str, Any]:
    targets = ground_truth["target"]["controls"]
    by_label: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        for label in [target["label"], *target.get("aliases", [])]:
            by_label.setdefault(normalize_text(label), []).append(target)

    used: set[str] = set()
    controls = []
    regions = sorted(
        (region for region in page["regions"] if region["kind"] == "checkbox"),
        key=lambda region: (
            region["bounding_box"]["top"],
            region["bounding_box"]["left"],
        ),
    )
    for region in regions:
        structure = region.get("structure") or {}
        label = _checkbox_label(structure.get("label") or region.get("text") or "")
        candidates = [
            target
            for target in by_label.get(normalize_text(label), [])
            if target["control_id"] not in used
        ]
        if len(candidates) != 1:
            continue
        state = structure.get("state")
        if state not in {"selected", "unselected"}:
            continue
        target = candidates[0]
        used.add(target["control_id"])
        controls.append(
            {
                "control_id": target["control_id"],
                "state": "checked" if state == "selected" else "unchecked",
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
        normalize_text(value)
        for value in [item["label"], *item.get("aliases", [])]
        if not normalize_text(value).startswith(("row-", "column-"))
    }


def axis_matches(
    axes: list[dict[str, Any]], grid: dict[tuple[int, int], str], coordinate: int
) -> list[tuple[int, tuple[int, int], str]]:
    matches = []
    used_coordinates = set()
    for index, axis in enumerate(axes):
        candidates = sorted(
            (
                (position, value)
                for position, value in grid.items()
                if position[coordinate] not in used_coordinates
                and normalize_text(value) in labels(axis)
            ),
            key=lambda item: item[0],
        )
        if not candidates:
            continue
        position, value = candidates[0]
        used_coordinates.add(position[coordinate])
        matches.append((index, position, value))
    return matches


def table_match_score(target: dict[str, Any], candidate: dict[str, Any]) -> int:
    grid = cell_grid(candidate)
    return len(axis_matches(target["rows"], grid, 0)) + len(
        axis_matches(target["columns"], grid, 1)
    )


def axis_map(
    axes: list[dict[str, Any]], grid: dict[tuple[int, int], str], coordinate: int
) -> tuple[dict[int, int], list[dict[str, str]]]:
    mapped = {}
    output = []
    axis_name = "row" if coordinate == 0 else "column"
    for index, position, label in axis_matches(axes, grid, coordinate):
        candidate_index = position[coordinate]
        mapped[index] = candidate_index
        output.append(
            {f"{axis_name}_id": axes[index][f"{axis_name}_id"], "label": label}
        )
    return mapped, output


def adapt_table(target: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    grid = cell_grid(candidate)
    row_map, rows = axis_map(target["rows"], grid, 0)
    column_map, columns = axis_map(target["columns"], grid, 1)
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
        "label": "",
        "image_id": target["image_id"],
        "rows": rows,
        "columns": columns,
        "cells": cells,
        "merged_ranges": [],
    }


def adapt_tables(ground_truth: dict[str, Any], page: dict[str, Any]) -> dict[str, Any]:
    candidates = unique_tables(page)
    available = set(range(len(candidates)))
    tables = []
    for target in ground_truth["target"]["tables"]:
        index = max(
            available,
            key=lambda candidate_index: (
                table_match_score(target, candidates[candidate_index]),
                -candidate_index,
            ),
            default=None,
        )
        if index is None or table_match_score(target, candidates[index]) == 0:
            continue
        available.remove(index)
        tables.append(adapt_table(target, candidates[index]))
    return {"tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("raw_pages", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    adapters = {
        "forms": adapt_form,
        "tables": adapt_tables,
    }
    for ground_truth_path in sorted(args.data.glob("**/ground_truth.json")):
        ground_truth = json.loads(ground_truth_path.read_text())
        if ground_truth.get("cohort") != "handwritten" or ground_truth.get(
            "availability"
        ) != "ready":
            continue
        page = load_page(args.raw_pages, ground_truth)
        task = ground_truth["task"]
        if task not in adapters:
            continue
        prediction = {
            "case_id": ground_truth["case_id"],
            "task": task,
            "prediction": adapters[task](ground_truth, page),
        }
        (args.output / f"{ground_truth['case_id']}.json").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
