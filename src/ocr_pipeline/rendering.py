"""Render page text without losing its evidence links."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from html import escape
from typing import Any

from .contracts import EvidenceText, TextRegion
from .table_topology import TableTopology, TableTopologyError, validate_table_topology

INLINE_MAX_GAP_HEIGHTS = 3
UNREADABLE_HANDWRITING = "[unreadable handwriting]"
MARK_GLYPHS = ("✓", "✗", "∅", "◯")
CONTROL_SYMBOLS = ("[x]", "[ ]", "[?]", *MARK_GLYPHS)


def render_evidence(regions: list[TextRegion]) -> EvidenceText:
    ordered = [
        region
        for _, region in sorted(
            enumerate(regions),
            key=lambda item: (_text_region_order(item[1]), item[0]),
        )
    ]
    rendered = [
        (region, _plain_text(region))
        for region in ordered
        if (
            region.resolution == "resolved"
            or region.kind in {"handwriting", "checkbox"}
        )
        and (region.structure or {}).get("role") != "table_source"
        and not (region.structure or {}).get("layout_owner_id")
    ]
    return EvidenceText(
        value=" ".join(text for _, text in rendered),
        evidence_ids=[region.id for region, _ in rendered],
    )


def _plain_text(region: TextRegion) -> str:
    if region.kind == "checkbox":
        # Only the state symbol: the label is already its own evidence region.
        return next(
            (symbol for symbol in CONTROL_SYMBOLS if region.text.startswith(symbol)),
            "",
        )
    if region.resolution == "resolved":
        return region.text
    if region.kind == "handwriting":
        return UNREADABLE_HANDWRITING
    return ""


def render_page_markdown(
    regions: Sequence[dict[str, Any]],
    evidence_ids: Iterable[str],
    *,
    fallback: str = "",
) -> str:
    """Render canonical evidence as readable Markdown for downloads."""
    source_index = {_id(region): region for region in regions if _id(region)}
    rendered = _display_regions(regions, set(evidence_ids))
    blocks = [
        _markdown_block(group, source_index)
        for group in _group_inline_regions(rendered)
    ]
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
    owner_ids = {
        _id(region)
        for region in regions
        if _structure(region).get("role") == "layout_block"
    }
    ordered = _ordered_regions(regions)
    included = [region for region in ordered if _renderable(region, evidence_ids)]
    linked_labels = {
        str(label_id)
        for region in included
        if _semantic_kind(region) == "control" and _text(region)
        for label_id in (_structure(region).get("label_evidence_ids") or [])
    }
    return [
        region
        for region in included
        if (
            not _owned_by_layout(region, owner_ids) and _id(region) not in linked_labels
        )
        or _semantic_kind(region) in {"table", "figure", "control"}
    ]


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


def _markdown_block(
    region: dict[str, Any],
    source_index: dict[str, dict[str, Any]],
) -> str:
    kind = _semantic_kind(region)
    text = _canonical_layout_text(region, source_index).strip()
    resolution = _resolution(region)
    if kind == "table":
        block = _markdown_table(region) or text
    elif kind == "title":
        block = f"### {text}" if text else ""
    elif kind == "heading":
        block = f"#### {text}" if text else ""
    elif kind == "formula":
        if resolution != "resolved":
            block = "[unreadable formula]"
        elif not text:
            block = ""
        elif _formula_math_ready(region):
            block = f"$$\n{text}\n$$"
        else:
            block = text
    elif kind == "control":
        if resolution == "resolved":
            block = f"- {text}" if text else ""
        else:
            label = str(_structure(region).get("label") or "").strip()
            # A slashed loop is evidence of what was drawn, so it keeps its glyph instead
            # of collapsing into the unknown-state marker.
            marker = str(_structure(region).get("mark_glyph") or "[?]")
            if not _structure(region).get("annotation_shape"):
                marker = "[?]"
            block = f"- {marker} {label}" if label else ""
    elif kind == "field":
        label = str(
            _structure(region).get("label")
            or _structure(region).get("field_name")
            or ""
        ).strip()
        value = text if resolution == "resolved" else ""
        block = f"**{label}:** {value}" if label else value
    else:
        block = _display_text(region)
    return block


def _markdown_table(region: dict[str, Any]) -> str:
    structure = _structure(region)
    try:
        topology = validate_table_topology(structure)
    except TableTopologyError as error:
        return _invalid_table(region, str(error))

    header_row_count = structure.get("header_row_count")
    block = (
        _html_table(topology)
        if topology.has_spans or header_row_count == 0
        else _simple_markdown_table(topology)
    )
    return block


def _simple_markdown_table(topology: TableTopology) -> str:
    grid = [
        ["" for _ in range(topology.column_count)] for _ in range(topology.row_count)
    ]
    for cell in topology.cells:
        grid[cell.row][cell.column] = _escape_markdown_cell(
            _display_cell_text(cell.value)
        )
    lines = [_markdown_row(grid[0]), _markdown_row(["---"] * topology.column_count)]
    lines.extend(_markdown_row(row) for row in grid[1:])
    return "\n".join(lines)


def _html_table(topology: TableTopology) -> str:
    anchors = {(cell.row, cell.column): cell for cell in topology.cells}
    lines = ["<table>", "<tbody>"]
    for row in range(topology.row_count):
        lines.append("<tr>")
        for column in range(topology.column_count):
            cell = anchors.get((row, column))
            if cell is None:
                continue
            tag = "th" if _header_cell(cell.value) else "td"
            attributes = []
            if len(cell.rows) > 1:
                attributes.append(f' rowspan="{len(cell.rows)}"')
            if len(cell.columns) > 1:
                attributes.append(f' colspan="{len(cell.columns)}"')
            text = escape(_display_cell_text(cell.value), quote=True)
            lines.append(f"<{tag}{''.join(attributes)}>{text}</{tag}>")
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _header_cell(cell: Any) -> bool:
    return bool(cell.get("column_header") or cell.get("projected_row_header"))


def _invalid_table(region: dict[str, Any], problem: str) -> str:
    del problem
    literal = _text(region)
    return "\n".join(f"    {line}" for line in literal.splitlines()) if literal else ""


def _display_cell_text(cell: dict[str, Any]) -> str:
    if str(cell.get("resolution", "resolved")) != "resolved":
        return ""
    return str(cell.get("text", ""))


def _control_label(text: str) -> str:
    stripped = text.strip()
    for prefix in ("[?]", "[x]", "[X]", "[ ]", *MARK_GLYPHS):
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()
    return stripped


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
    structure = _structure(region)
    block_type = str(structure.get("block_type", "")).casefold()
    if structure.get("role") == "layout_block" and block_type:
        return block_type
    value = f"{_kind(region)} {structure.get('role', '')}".casefold()
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


def _formula_math_ready(region: dict[str, Any]) -> bool:
    return _structure(region).get("formula_recognition") not in {
        "heuristic",
        "specialist_pending",
    }


def _canonical_layout_text(
    region: dict[str, Any],
    source_index: dict[str, dict[str, Any]],
) -> str:
    structure = _structure(region)
    if structure.get("role") != "layout_block":
        return _display_text(region)
    if structure.get("block_type") == "formula" and _resolution(region) == "resolved":
        return _text(region)
    groups = (
        structure.get("fields", structure.get("segments"))
        if structure.get("block_type") == "form_row"
        else structure.get("lines")
    )
    if not isinstance(groups, list):
        return ""
    rendered_lines = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        values = [
            value
            for evidence_id in group.get("evidence_ids", [])
            if isinstance(evidence_id, str)
            and (source := source_index.get(evidence_id)) is not None
            and (value := _display_text(source))
        ]
        rendered_lines.append(" ".join(values))
    separator = " | " if structure.get("block_type") == "form_row" else " "
    return separator.join(rendered_lines)


def _owned_by_layout(region: dict[str, Any], owner_ids: set[str]) -> bool:
    owner_id = _structure(region).get("layout_owner_id")
    return isinstance(owner_id, str) and owner_id in owner_ids


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


def _display_text(region: dict[str, Any]) -> str:
    if _resolution(region) == "resolved":
        return _text(region).strip()
    if _semantic_kind(region) == "handwriting":
        return UNREADABLE_HANDWRITING
    return ""


def _resolution(region: dict[str, Any]) -> str:
    return str(region.get("resolution", "resolved"))


def _reading_order(region: dict[str, Any]) -> int:
    structure = _structure(region)
    rank = structure.get("presentation_rank")
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank
    value = region.get("reading_order", 10**9)
    return value if isinstance(value, int) else 10**9


def _text_region_order(region: TextRegion) -> int:
    structure = region.structure if isinstance(region.structure, dict) else {}
    rank = structure.get("presentation_rank")
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank
    return region.reading_order


def _structure(region: dict[str, Any]) -> dict[str, Any]:
    value = region.get("structure")
    return value if isinstance(value, dict) else {}
