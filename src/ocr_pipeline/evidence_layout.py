"""Build readable spatial blocks without inventing document text."""

from __future__ import annotations

import copy
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "evidence-spatial-layout"
_TEXT_KINDS = frozenset(
    {
        "caption",
        "equation",
        "field",
        "footnote",
        "formula",
        "handwriting",
        "heading",
        "list_item",
        "math",
        "page_footer",
        "page_header",
        "page_text",
        "paragraph",
        "section_header",
        "text",
        "text_block",
        "title",
        "token",
        "word",
    }
)
_ATOMIC_TYPES = {
    "caption": "caption",
    "equation": "formula",
    "field": "field",
    "footnote": "footnote",
    "formula": "formula",
    "handwriting": "handwriting",
    "heading": "heading",
    "list_item": "list",
    "math": "formula",
    "page_footer": "footer",
    "page_header": "header",
    "section_header": "heading",
    "title": "title",
}
_EXCLUDED_ROLES = frozenset(
    {
        "coverage_risk",
        "layout_block",
        "presentation_challenger",
        "table",
        "table_candidate",
        "table_cell",
        "table_source",
    }
)
_PRESENTATION_KINDS = frozenset(
    {"checkbox", "control", "figure", "image", "radio", "table"}
)
_PRESENTATION_ROLES = frozenset({"control", "figure", "layout_block", "table"})
_PRESENTATION_EXCLUDED_ROLES = frozenset(
    {
        "coverage_risk",
        "presentation_challenger",
        "table_candidate",
        "table_cell",
        "table_source",
    }
)


@dataclass
class _Line:
    items: list[TextRegion]

    @property
    def box(self) -> BoundingBox:
        return _union(region.bounding_box for region in self.items)


@dataclass(frozen=True)
class _Separator:
    orientation: str
    position: int
    start: int
    end: int


class EvidenceLayoutStage:
    """Own canonical text once and expose geometry-backed display blocks."""

    name = "evidence-layout"

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        updated = copy.deepcopy(regions)
        excluded_ids = _structured_evidence_ids(updated)
        sources = [
            region for region in updated if _eligible_source(region, excluded_ids)
        ]
        if not sources:
            return updated

        separators = _detect_separators(image_path, sources)
        blocks = _build_blocks(
            sources,
            page_number,
            {region.id for region in updated},
            separators,
        )
        owners = {
            evidence_id: block.id
            for block in blocks
            for evidence_id in _child_ids(block)
        }
        for region in updated:
            owner_id = owners.get(region.id)
            if owner_id is None:
                continue
            structure = copy.deepcopy(region.structure) if region.structure else {}
            structure["layout_owner_id"] = owner_id
            region.structure = structure
        result = [*updated, *blocks]
        _assign_presentation_ranks(result, separators, excluded_ids)
        return result


def _build_blocks(
    sources: list[TextRegion],
    page_number: int,
    existing_ids: set[str],
    separators: tuple[_Separator, ...],
) -> list[TextRegion]:
    atomic = [region for region in sources if _source_type(region) != "text"]
    regular = [region for region in sources if _source_type(region) == "text"]
    lines = _spatial_lines(regular, separators)
    lines = _pre_cut_order(lines, separators, lambda line: line.box)
    groups: list[tuple[str, list[_Line]]] = []
    paragraph: list[_Line] = []
    for line in lines:
        if _form_row(line):
            if paragraph:
                groups.append((_paragraph_type(paragraph), paragraph))
                paragraph = []
            groups.append(("form_row", [line]))
            continue
        if paragraph and (
            not _same_paragraph(paragraph[-1], line)
            or _crosses_boundary(paragraph[-1].box, line.box, separators)
        ):
            groups.append((_paragraph_type(paragraph), paragraph))
            paragraph = []
        paragraph.append(line)
    if paragraph:
        groups.append((_paragraph_type(paragraph), paragraph))

    candidates: list[tuple[str, list[_Line]]] = groups + [
        (_source_type(region), [_Line([region])]) for region in atomic
    ]
    candidates = _pre_cut_order(
        candidates,
        separators,
        lambda item: _union(line.box for line in item[1]),
    )

    blocks = []
    next_index = 1
    for block_type, block_lines in candidates:
        while f"p{page_number}-layout-{next_index}" in existing_ids:
            next_index += 1
        block_id = f"p{page_number}-layout-{next_index}"
        existing_ids.add(block_id)
        next_index += 1
        ordered_lines = [
            sorted(
                line.items,
                key=lambda item: (item.bounding_box.left, item.reading_order),
            )
            for line in block_lines
        ]
        child_regions = [region for line in ordered_lines for region in line]
        resolved = [
            region for region in child_regions if region.resolution == "resolved"
        ]
        text = _block_text(block_type, ordered_lines)
        line_metadata = [
            {"evidence_ids": [region.id for region in line]} for line in ordered_lines
        ]
        structure: dict[str, Any] = {
            "role": "layout_block",
            "block_type": block_type,
            "child_evidence_ids": [region.id for region in child_regions],
            "source_reading_orders": [
                {
                    "evidence_id": region.id,
                    "reading_order": region.reading_order,
                }
                for region in child_regions
            ],
            "lines": line_metadata,
        }
        if separators:
            structure["pre_cut_boundaries"] = [
                {
                    "orientation": separator.orientation,
                    "position": separator.position,
                    "span": [separator.start, separator.end],
                }
                for separator in separators
            ]
        if block_type == "form_row":
            segments = _line_segments(ordered_lines[0])
            structure["segments"] = [
                {"evidence_ids": [region.id for region in segment]}
                for segment in segments
            ]
            structure["fields"] = [
                {"evidence_ids": [region.id for region in field]}
                for field in _form_fields(segments)
            ]
        blocks.append(
            TextRegion(
                id=block_id,
                kind="layout_block",
                text=text,
                confidence=(
                    statistics.fmean(
                        region.confidence
                        for region in resolved
                        if region.confidence is not None
                    )
                    if any(region.confidence is not None for region in resolved)
                    else None
                ),
                bounding_box=_union(region.bounding_box for region in child_regions),
                reading_order=min(region.reading_order for region in child_regions),
                provider=PROVIDER,
                text_provenance={
                    "method": "evidence_owned_spatial_grouping",
                    "source_region_ids": [region.id for region in child_regions],
                    "source_providers": list(
                        dict.fromkeys(region.provider for region in child_regions)
                    ),
                    "literal_policy": "resolved_canonical_evidence_only",
                },
                resolution="resolved" if text else "unreadable",
                structure=structure,
            )
        )
    return blocks


def _assign_presentation_ranks(
    regions: list[TextRegion],
    separators: tuple[_Separator, ...],
    excluded_ids: set[str],
) -> None:
    top_level = [
        region
        for region in regions
        if _top_level_presentation_item(region, excluded_ids)
    ]
    ordered = _pre_cut_order(
        top_level,
        separators,
        lambda region: region.bounding_box,
    )
    for rank, region in enumerate(ordered, start=1):
        structure = copy.deepcopy(region.structure) if region.structure else {}
        structure["presentation_rank"] = rank
        region.structure = structure
        if region.provider == PROVIDER:
            region.reading_order = rank


def _top_level_presentation_item(
    region: TextRegion,
    excluded_ids: set[str],
) -> bool:
    if region.id in excluded_ids:
        return False
    structure = region.structure if isinstance(region.structure, dict) else {}
    if structure.get("layout_owner_id"):
        return False
    role = _label(structure.get("role"))
    kind = _label(region.kind)
    if role in _PRESENTATION_EXCLUDED_ROLES or kind in _PRESENTATION_EXCLUDED_ROLES:
        return False
    return role in _PRESENTATION_ROLES or kind in _PRESENTATION_KINDS


def _spatial_lines(
    regions: list[TextRegion],
    separators: tuple[_Separator, ...],
) -> list[_Line]:
    lines: list[_Line] = []
    for region in sorted(
        regions,
        key=lambda item: (
            (item.bounding_box.top + item.bounding_box.bottom) / 2,
            item.bounding_box.left,
            item.reading_order,
        ),
    ):
        compatible = [
            line
            for line in lines
            if _same_line(line.box, region.bounding_box)
            and not _crosses_boundary(line.box, region.bounding_box, separators)
        ]
        if not compatible:
            lines.append(_Line([region]))
            continue
        line = min(
            compatible,
            key=lambda item: abs(_center(item.box) - _center(region.bounding_box)),
        )
        line.items.append(region)
    return lines


def _detect_separators(
    image_path: Path,
    regions: list[TextRegion],
) -> tuple[_Separator, ...]:
    if not image_path.is_file():
        return ()

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise ReaderError("layout_separator_unavailable", str(error)) from error

    try:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
            width, height = image.size
        detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    except (OSError, cv2.error) as error:
        raise ReaderError("layout_separator_failed", str(error)) from error
    if detected is None:
        return ()

    typical_height = statistics.median(
        _height(region.bounding_box) for region in regions
    )
    candidates = []
    for x1, y1, x2, y2 in detected.reshape(-1, 4):
        horizontal_length = abs(float(x2 - x1))
        vertical_length = abs(float(y2 - y1))
        if vertical_length <= max(2.0, horizontal_length * 0.03):
            if horizontal_length < max(width * 0.15, typical_height * 6):
                continue
            candidates.append(
                _Separator(
                    "horizontal",
                    round((float(y1) + float(y2)) / 2),
                    round(min(float(x1), float(x2))),
                    round(max(float(x1), float(x2))),
                )
            )
        elif horizontal_length <= max(2.0, vertical_length * 0.03):
            if vertical_length < max(height * 0.15, typical_height * 6):
                continue
            candidates.append(
                _Separator(
                    "vertical",
                    round((float(x1) + float(x2)) / 2),
                    round(min(float(y1), float(y2))),
                    round(max(float(y1), float(y2))),
                )
            )

    merged = _merge_separators(candidates, width, height)
    return tuple(
        separator
        for separator in merged
        if _separator_has_text_support(
            separator,
            merged,
            regions,
            typical_height,
        )
    )


def _merge_separators(
    separators: list[_Separator],
    width: int,
    height: int,
) -> list[_Separator]:
    merged: list[_Separator] = []
    for separator in sorted(
        separators,
        key=lambda item: (item.orientation, item.position, item.start),
    ):
        axis_length = width if separator.orientation == "horizontal" else height
        match = next(
            (
                item
                for item in reversed(merged)
                if item.orientation == separator.orientation
                and abs(item.position - separator.position) <= 4
                and separator.start <= item.end + max(4, round(axis_length * 0.01))
            ),
            None,
        )
        if match is None:
            merged.append(separator)
            continue
        merged.remove(match)
        merged.append(
            _Separator(
                separator.orientation,
                round((match.position + separator.position) / 2),
                min(match.start, separator.start),
                max(match.end, separator.end),
            )
        )
    return merged


def _separator_has_text_support(
    separator: _Separator,
    merged: list[_Separator],
    regions: list[TextRegion],
    typical_height: float,
) -> bool:
    if separator.orientation == "vertical":
        if _grid_intersection_count(separator, merged) >= 2:
            return False
        left = [
            region
            for region in regions
            if region.bounding_box.right <= separator.position
            and _overlap(
                region.bounding_box.top,
                region.bounding_box.bottom,
                separator.start,
                separator.end,
            )
            > 0
        ]
        right = [
            region
            for region in regions
            if region.bounding_box.left >= separator.position
            and _overlap(
                region.bounding_box.top,
                region.bounding_box.bottom,
                separator.start,
                separator.end,
            )
            > 0
        ]
        supported_pairs = [
            (first, second)
            for first in left
            for second in right
            if _same_line(first.bounding_box, second.bounding_box)
        ]
        if not supported_pairs:
            return False
        return (
            min(
                min(
                    separator.position - first.bounding_box.right,
                    second.bounding_box.left - separator.position,
                )
                for first, second in supported_pairs
            )
            >= typical_height
        )

    above = [
        region
        for region in regions
        if region.bounding_box.bottom <= separator.position
        and _overlap(
            region.bounding_box.left,
            region.bounding_box.right,
            separator.start,
            separator.end,
        )
        > 0
    ]
    below = [
        region
        for region in regions
        if region.bounding_box.top >= separator.position
        and _overlap(
            region.bounding_box.left,
            region.bounding_box.right,
            separator.start,
            separator.end,
        )
        > 0
    ]
    if not above or not below:
        return False
    return not _looks_like_underline(separator, above, typical_height)


def _grid_intersection_count(
    vertical: _Separator,
    separators: list[_Separator],
) -> int:
    return sum(
        vertical.start - 4 <= horizontal.position <= vertical.end + 4
        and horizontal.start - 4 <= vertical.position <= horizontal.end + 4
        for horizontal in separators
        if horizontal.orientation == "horizontal"
    )


def _looks_like_underline(
    separator: _Separator,
    above: list[TextRegion],
    typical_height: float,
) -> bool:
    adjacent = [
        region
        for region in above
        if 0 <= separator.position - region.bounding_box.bottom <= typical_height * 0.75
    ]
    if not adjacent:
        return False
    line_box = _union(region.bounding_box for region in adjacent)
    line_length = separator.end - separator.start
    overlap = _overlap(separator.start, separator.end, line_box.left, line_box.right)
    return (
        overlap / max(1, line_length) >= 0.7
        and line_length <= (line_box.right - line_box.left) * 1.25
    )


def _pre_cut_order(
    items: list[Any],
    separators: tuple[_Separator, ...],
    box_of: Callable[[Any], BoundingBox],
) -> list[Any]:
    if len(items) < 2:
        return items
    for separator in sorted(separators, key=lambda item: item.position):
        if separator.orientation != "vertical":
            continue
        before = [item for item in items if box_of(item).bottom <= separator.start]
        after = [item for item in items if box_of(item).top >= separator.end]
        scoped = [
            item
            for item in items
            if box_of(item).bottom > separator.start
            and box_of(item).top < separator.end
        ]
        left = [item for item in scoped if box_of(item).right <= separator.position]
        right = [item for item in scoped if box_of(item).left >= separator.position]
        if (
            not left
            or not right
            or len(before) + len(left) + len(right) + len(after) != len(items)
        ):
            continue
        return [
            *_pre_cut_order(before, separators, box_of),
            *_pre_cut_order(left, separators, box_of),
            *_pre_cut_order(right, separators, box_of),
            *_pre_cut_order(after, separators, box_of),
        ]
    return sorted(items, key=lambda item: (box_of(item).top, box_of(item).left))


def _crosses_boundary(
    first: BoundingBox,
    second: BoundingBox,
    separators: tuple[_Separator, ...],
) -> bool:
    for separator in separators:
        if separator.orientation == "horizontal":
            separated = (
                first.bottom <= separator.position <= second.top
                or second.bottom <= separator.position <= first.top
            )
            shared_start = max(first.left, second.left)
            shared_end = min(first.right, second.right)
        else:
            separated = (
                first.right <= separator.position <= second.left
                or second.right <= separator.position <= first.left
            )
            shared_start = max(first.top, second.top)
            shared_end = min(first.bottom, second.bottom)
        if (
            separated
            and _overlap(
                shared_start,
                shared_end,
                separator.start,
                separator.end,
            )
            > 0
        ):
            return True
    return False


def _overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> int:
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def _same_line(first: BoundingBox, second: BoundingBox) -> bool:
    overlap = min(first.bottom, second.bottom) - max(first.top, second.top)
    minimum_height = min(_height(first), _height(second))
    center_distance = abs(_center(first) - _center(second))
    return minimum_height > 0 and (
        overlap / minimum_height >= 0.5
        or center_distance <= min(_height(first), _height(second)) * 0.35
    )


def _same_paragraph(previous: _Line, current: _Line) -> bool:
    first = previous.box
    second = current.box
    typical_height = statistics.median((_height(first), _height(second)))
    vertical_gap = second.top - first.bottom
    return (
        -typical_height * 0.25 <= vertical_gap <= typical_height * 1.5
        and abs(second.left - first.left) <= typical_height * 2
        and min(first.right, second.right) > max(first.left, second.left)
    )


def _form_row(line: _Line) -> bool:
    return len(_line_segments(line.items)) > 1


def _line_segments(items: list[TextRegion]) -> list[list[TextRegion]]:
    ordered = sorted(items, key=lambda item: item.bounding_box.left)
    if not ordered:
        return []
    typical_height = statistics.median(_height(item.bounding_box) for item in ordered)
    segments = [[ordered[0]]]
    for region in ordered[1:]:
        gap = region.bounding_box.left - segments[-1][-1].bounding_box.right
        if gap > typical_height * 4:
            segments.append([region])
        else:
            segments[-1].append(region)
    return segments


def _block_text(block_type: str, lines: list[list[TextRegion]]) -> str:
    if block_type == "form_row":
        return " | ".join(
            _resolved_text(field) for field in _form_fields(_line_segments(lines[0]))
        )
    return " ".join(_resolved_text(line) for line in lines).strip()


def _resolved_text(regions: list[TextRegion]) -> str:
    return " ".join(
        region.text.strip()
        for region in regions
        if region.resolution == "resolved" and region.text.strip()
    )


def _form_fields(segments: list[list[TextRegion]]) -> list[list[TextRegion]]:
    split_fields = [
        field for segment in segments for field in _split_form_segment(segment)
    ]
    fields = []
    index = 0
    while index < len(split_fields):
        field = split_fields[index]
        text = _resolved_text(field)
        if index + 1 < len(split_fields) and _standalone_label(text):
            following = split_fields[index + 1]
            following_text = _resolved_text(following)
            if following_text and ":" not in following_text:
                fields.append([*field, *following])
                index += 2
                continue
        fields.append(field)
        index += 1
    return fields


def _split_form_segment(segment: list[TextRegion]) -> list[list[TextRegion]]:
    fields: list[list[TextRegion]] = []
    current: list[TextRegion] = []
    for region in segment:
        if current and _standalone_label(_resolved_text([region])):
            fields.append(current)
            current = []
        current.append(region)
    if current:
        fields.append(current)
    return fields


def _standalone_label(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(":") and stripped.count(":") == 1


def _paragraph_type(lines: list[_Line]) -> str:
    return "paragraph" if len(lines) > 1 else "line"


def _source_type(region: TextRegion) -> str:
    structure = region.structure if isinstance(region.structure, dict) else {}
    for value in (
        structure.get("semantic_class"),
        structure.get("role"),
        region.kind,
    ):
        label = _label(value)
        if label in _ATOMIC_TYPES:
            return _ATOMIC_TYPES[label]
    return "text"


def _eligible_source(region: TextRegion, excluded_ids: set[str]) -> bool:
    if region.id in excluded_ids or not region.text.strip():
        return False
    structure = region.structure if isinstance(region.structure, dict) else {}
    if structure.get("layout_owner_id"):
        return False
    role = _label(structure.get("role"))
    kind = _label(region.kind)
    return (
        kind in _TEXT_KINDS
        and role not in _EXCLUDED_ROLES
        and not role.endswith("_candidate")
        and _valid_box(region.bounding_box)
    )


def _structured_evidence_ids(regions: list[TextRegion]) -> set[str]:
    evidence_ids: set[str] = set()
    for region in regions:
        structure = region.structure if isinstance(region.structure, dict) else {}
        role = _label(structure.get("role"))
        if region.kind in {"checkbox", "control", "radio"} or role == "control":
            evidence_ids.update(_string_ids(structure.get("label_evidence_ids")))
        if region.kind == "table" or role == "table":
            for cell in structure.get("cells", []):
                if isinstance(cell, dict):
                    evidence_ids.update(_string_ids(cell.get("evidence_ids")))
    return evidence_ids


def _child_ids(region: TextRegion) -> list[str]:
    structure = region.structure if isinstance(region.structure, dict) else {}
    return _string_ids(structure.get("child_evidence_ids"))


def _string_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _union(boxes: Any) -> BoundingBox:
    items = list(boxes)
    return BoundingBox(
        min(box.left for box in items),
        min(box.top for box in items),
        max(box.right for box in items),
        max(box.bottom for box in items),
    )


def _height(box: BoundingBox) -> int:
    return box.bottom - box.top


def _center(box: BoundingBox) -> float:
    return (box.top + box.bottom) / 2


def _valid_box(box: BoundingBox) -> bool:
    return box.left < box.right and box.top < box.bottom


def _label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "_".join(value.strip().casefold().replace("-", " ").split())
