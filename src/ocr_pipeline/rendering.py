"""Render page text without losing its evidence links."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .contracts import EvidenceText, TextRegion

INLINE_MAX_GAP_HEIGHTS = 3


def render_evidence(regions: list[TextRegion]) -> EvidenceText:
    ordered = [
        region
        for _, region in sorted(
            enumerate(regions),
            key=lambda item: (item[1].reading_order, item[0]),
        )
    ]
    rendered = [
        region
        for region in ordered
        if region.resolution == "resolved"
        and region.kind != "checkbox"
        and (region.structure or {}).get("role") != "table_source"
    ]
    return EvidenceText(
        value=" ".join(region.text for region in rendered),
        evidence_ids=[region.id for region in rendered],
    )


def render_page_markdown(
    regions: Sequence[dict[str, Any]],
    evidence_ids: Iterable[str],
    *,
    fallback: str = "",
) -> str:
    """Render canonical evidence as readable Markdown for downloads."""
    rendered = _display_regions(regions, set(evidence_ids))
    blocks = [_markdown_block(group) for group in _group_inline_regions(rendered)]
    blocks = [block for block in blocks if block]
    return "\n\n".join(blocks) or fallback


def _group_inline_regions(
    regions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for region in regions:
        previous = groups[-1] if groups else None
        if previous is None or not _same_inline_line(previous, region):
            groups.append(region)
            continue
        groups[-1] = {
            **previous,
            "text": f"{_text(previous)} {_text(region)}",
            "bounding_box": region.get("bounding_box"),
        }
    return groups


def _same_inline_line(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if _resolution(first) != "resolved" or _resolution(second) != "resolved":
        return False
    if not _inline_text(first) or not _inline_text(second):
        return False
    if _semantic_kind(first) != _semantic_kind(second):
        return False
    first_box = _normalized_box(first.get("bounding_box"))
    second_box = _normalized_box(second.get("bounding_box"))
    if first_box is None or second_box is None:
        return False
    overlap = min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
    height = min(first_box[3] - first_box[1], second_box[3] - second_box[1])
    horizontal_gap = second_box[0] - first_box[2]
    return (
        height > 0
        and overlap / height >= 0.4
        and second_box[0] >= first_box[0]
        and -height <= horizontal_gap <= height * INLINE_MAX_GAP_HEIGHTS
    )


def _inline_text(region: dict[str, Any]) -> bool:
    kind = _kind(region).casefold()
    if kind in {"word", "token"} or kind.endswith(("_word", "_token")):
        return True
    provenance = region.get("text_provenance")
    return (
        kind == "text"
        and isinstance(provenance, dict)
        and provenance.get("merge_level") == "word"
    )


def _normalized_box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, list) and len(value) == 4:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    if not isinstance(value, dict):
        return None
    keys = ("left", "top", "right", "bottom")
    if not all(isinstance(value.get(key), int | float) for key in keys):
        return None
    return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]


def _display_regions(
    regions: Sequence[dict[str, Any]],
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    ordered = _ordered_regions(regions)
    included = [region for region in ordered if _renderable(region, evidence_ids)]
    linked_labels = {
        str(label_id)
        for region in included
        if _semantic_kind(region) == "control" and _text(region)
        for label_id in (_structure(region).get("label_evidence_ids") or [])
    }
    return [region for region in included if _id(region) not in linked_labels]


def _renderable(region: dict[str, Any], evidence_ids: set[str]) -> bool:
    structure = _structure(region)
    role = str(structure.get("role", ""))
    if role in {"table_source", "coverage_risk", "table_candidate"}:
        return False
    if _kind(region) in {"coverage_risk", "table_candidate"}:
        return False
    return (
        _id(region) in evidence_ids
        or _semantic_kind(region) == "control"
        or _resolution(region) != "resolved"
    )


def _markdown_block(region: dict[str, Any]) -> str:
    kind = _semantic_kind(region)
    text = _text(region).strip()
    resolution = _resolution(region)
    if kind == "table":
        block = _markdown_table(region) or text
    elif kind == "title":
        block = f"### {text}" if text else ""
    elif kind == "heading":
        block = f"#### {text}" if text else ""
    elif kind == "control":
        block = f"- {text}" if text else ""
    elif kind == "field":
        label = str(
            _structure(region).get("label")
            or _structure(region).get("field_name")
            or ""
        ).strip()
        block = f"**{label}:** {text}" if label and text else text
    else:
        block = text
    if resolution == "resolved":
        return block

    review = [f"> **{resolution.title()} {kind.replace('_', ' ')} evidence**"]
    if block:
        review.append(f"> {block}")
    alternatives = _alternatives(region)
    review.extend(f"> Alternative evidence: {value}" for value in alternatives)
    return "\n".join(review)


def _markdown_table(region: dict[str, Any]) -> str:
    structure = _structure(region)
    cells = structure.get("cells")
    if not isinstance(cells, list) or not cells:
        return ""
    normalized = _normalized_cells(cells)
    if not normalized:
        return ""
    row_count = max(
        int(structure.get("row_count") or 0),
        max(row for row, _, _ in normalized) + 1,
    )
    column_count = max(
        int(structure.get("column_count") or 0),
        max(column for _, column, _ in normalized) + 1,
    )
    grid = [["" for _ in range(column_count)] for _ in range(row_count)]
    for row, column, cell in normalized:
        if row >= row_count or column >= column_count:
            continue
        grid[row][column] = _escape_markdown_cell(str(cell.get("text", "")))

    lines = [_markdown_row(grid[0]), _markdown_row(["---"] * column_count)]
    lines.extend(_markdown_row(row) for row in grid[1:])
    for row, column, cell in normalized:
        resolution = str(cell.get("resolution", "resolved"))
        if resolution == "resolved":
            continue
        selected = str(cell.get("text", "")).strip()
        location = f"row {row + 1}, column {column + 1}"
        lines.extend(
            ["", f"> **{resolution.title()} table cell evidence ({location})**"]
        )
        if selected:
            lines.append(f"> {selected}")
        lines.extend(
            f"> Alternative evidence: {value}"
            for value in _mapping_alternatives(cell.get("alternatives"), selected)
        )
    return "\n".join(lines)


def _normalized_cells(cells: list[Any]) -> list[tuple[int, int, dict[str, Any]]]:
    raw: list[tuple[int, int, dict[str, Any]]] = []
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        rows = cell.get("row_nums")
        columns = cell.get("column_nums")
        if isinstance(rows, list) and rows and isinstance(columns, list) and columns:
            raw.append((int(min(rows)), int(min(columns)), cell))
            continue
        row = cell.get("row_index")
        column = cell.get("column_index")
        if isinstance(row, int) and isinstance(column, int):
            indexed.append((row, column, cell))
    if indexed and min(row for row, _, _ in indexed) >= 1:
        indexed = [(row - 1, column, cell) for row, column, cell in indexed]
    if indexed and min(column for _, column, _ in indexed) >= 1:
        indexed = [(row, column - 1, cell) for row, column, cell in indexed]
    return raw + indexed


def _ordered_regions(
    regions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        region
        for _, region in sorted(
            enumerate(regions),
            key=lambda item: (_reading_order(item[1]), item[0]),
        )
    ]


def _semantic_kind(region: dict[str, Any]) -> str:
    value = f"{_kind(region)} {_structure(region).get('role', '')}".casefold()
    if ("section" in value and "head" in value) or "heading" in value:
        return "heading"
    if "footer" in value:
        return "footer"
    if "title" in value:
        return "title"
    if "header" in value:
        return "header"
    if "table" in value:
        return "table"
    if "list" in value:
        return "list"
    if "figure" in value or "image" in value:
        return "figure"
    if "checkbox" in value or "radio" in value or "control" in value:
        return "control"
    if "form" in value or "field" in value:
        return "field"
    if "handwrit" in value:
        return "handwriting"
    return "text"


def _markdown_row(values: Sequence[str]) -> str:
    return f"| {' | '.join(values)} |"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _id(region: dict[str, Any]) -> str:
    return str(region.get("id", ""))


def _kind(region: dict[str, Any]) -> str:
    return str(region.get("kind", "text"))


def _text(region: dict[str, Any]) -> str:
    return str(region.get("text", ""))


def _resolution(region: dict[str, Any]) -> str:
    return str(region.get("resolution", "resolved"))


def _reading_order(region: dict[str, Any]) -> int:
    value = region.get("reading_order", 10**9)
    return value if isinstance(value, int) else 10**9


def _structure(region: dict[str, Any]) -> dict[str, Any]:
    value = region.get("structure")
    return value if isinstance(value, dict) else {}


def _alternatives(region: dict[str, Any]) -> list[str]:
    values = region.get("alternatives", [])
    result = []
    for value in values:
        if not isinstance(value, dict):
            continue
        text = value.get("text", "")
        text = str(text).strip()
        if text and text != _text(region).strip() and text not in result:
            result.append(text)
    return result


def _mapping_alternatives(values: Any, selected: str) -> list[str]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if not isinstance(value, dict):
            continue
        text = str(value.get("text", "")).strip()
        if text and text != selected and text not in result:
            result.append(text)
    return result
