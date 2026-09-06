"""Build readable spatial blocks without inventing document text."""

from __future__ import annotations

import copy
import re
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
_SECTION_HEADINGS = frozenset(
    {
        "abstract",
        "conclusion",
        "conclusions",
        "discussion",
        "introduction",
        "methods",
        "references",
        "results",
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
        separators = (
            *separators,
            *_infer_sidebar_separators(image_path, sources, separators),
        )
        blocks = _build_blocks(
            sources,
            page_number,
            {region.id for region in updated},
            separators,
        )
        blocks = _group_formula_blocks(image_path, blocks)
        owners = {
            evidence_id: (block.id, block.structure["block_type"])
            for block in blocks
            for evidence_id in _child_ids(block)
        }
        for region in updated:
            owner = owners.get(region.id)
            if owner is None:
                continue
            structure = copy.deepcopy(region.structure) if region.structure else {}
            structure["layout_owner_id"], structure["layout_owner_type"] = owner
            region.structure = structure
        result = [*updated, *blocks]
        _assign_presentation_ranks(result, separators, excluded_ids)
        return result


def refresh_layout_owners(regions: list[TextRegion], owner_ids: set[str]) -> None:
    """Refresh affected owners after a child canonical revision."""
    if not owner_ids:
        return
    source_index = {region.id: region for region in regions}
    for owner_id in owner_ids:
        owner = source_index.get(owner_id)
        if owner is None:
            continue
        structure = copy.deepcopy(owner.structure) if owner.structure else {}
        block_type = structure.get("block_type")
        if structure.get("role") != "layout_block" or block_type == "formula":
            continue
        groups = structure.get("lines")
        if not isinstance(groups, list):
            continue
        lines = [
            [
                source_index[evidence_id]
                for evidence_id in _string_ids(group.get("evidence_ids"))
                if evidence_id in source_index
            ]
            for group in groups
            if isinstance(group, dict)
        ]
        if not lines:
            continue
        owner.text = _block_text(str(block_type), lines)
        owner.resolution = "resolved" if owner.text else "unreadable"
        if block_type == "form_row":
            segments = _line_segments(lines[0])
            structure["segments"] = [
                {"evidence_ids": [region.id for region in segment]}
                for segment in segments
            ]
            structure["fields"] = [
                _structured_form_field(owner.id, index, field)
                for index, field in enumerate(_form_fields(segments), start=1)
            ]
        owner.structure = structure
        provenance = (
            copy.deepcopy(owner.text_provenance) if owner.text_provenance else {}
        )
        provenance["source_providers"] = list(
            dict.fromkeys(
                source_index[evidence_id].provider
                for evidence_id in _child_ids(owner)
                if evidence_id in source_index
            )
        )
        owner.text_provenance = provenance


def _build_blocks(
    sources: list[TextRegion],
    page_number: int,
    existing_ids: set[str],
    separators: tuple[_Separator, ...],
) -> list[TextRegion]:
    atomic = [
        region
        for region in sources
        if _source_type(region) != "text" or _is_orientation_residual(region)
    ]
    regular = [
        region
        for region in sources
        if _source_type(region) == "text" and not _is_orientation_residual(region)
    ]
    lines = _spatial_lines(regular, separators)
    lines = _pre_cut_order(lines, separators, lambda line: line.box)
    groups, lines = _column_groups(lines)
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

    candidates: list[tuple[str, list[_Line]]] = groups + _atomic_candidates(atomic)
    candidates = _pre_cut_order(
        candidates,
        separators,
        lambda item: _union(line.box for line in item[1]),
    )
    candidates = _semantic_block_types(candidates)

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
        if block_type == "formula":
            structure["formula_recognition"] = "heuristic"
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
                _structured_form_field(block_id, index, field)
                for index, field in enumerate(_form_fields(segments), start=1)
            ]
        blocks.append(
            TextRegion(
                id=block_id,
                kind="layout_block",
                text=text,
                confidence=None,
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
    regions_by_id = {region.id: region for region in regions}
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
        if region.kind not in {"checkbox", "control", "radio"}:
            continue
        for label_id in _string_ids(structure.get("label_evidence_ids")):
            label = regions_by_id.get(label_id)
            if label is None:
                continue
            label_structure = copy.deepcopy(label.structure) if label.structure else {}
            label_structure["presentation_rank"] = rank
            label.structure = label_structure


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
            if all(
                _same_line(item.bounding_box, region.bounding_box)
                for item in line.items
            )
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
        left = [
            item
            for item in scoped
            if _horizontal_center(box_of(item)) <= separator.position
        ]
        right = [
            item
            for item in scoped
            if _horizontal_center(box_of(item)) > separator.position
        ]
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
            first_center = _center(first)
            second_center = _center(second)
            separated = (
                first_center <= separator.position < second_center
                or second_center <= separator.position < first_center
            )
            shared_start = max(first.left, second.left)
            shared_end = min(first.right, second.right)
        else:
            first_center = _horizontal_center(first)
            second_center = _horizontal_center(second)
            separated = (
                first_center <= separator.position < second_center
                or second_center <= separator.position < first_center
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
    first_height = _height(first)
    second_height = _height(second)
    if min(first_height, second_height) <= 0:
        return False
    if max(first_height, second_height) / min(first_height, second_height) > 1.8:
        return False
    typical_height = statistics.median((_height(first), _height(second)))
    vertical_gap = second.top - first.bottom
    return (
        -typical_height * 0.25 <= vertical_gap <= typical_height * 1.5
        and (
            abs(second.left - first.left) <= typical_height * 2
            or abs(_horizontal_center(second) - _horizontal_center(first))
            <= typical_height * 2
        )
        and min(first.right, second.right) > max(first.left, second.left)
    )


def _infer_sidebar_separators(
    image_path: Path,
    regions: list[TextRegion],
    separators: tuple[_Separator, ...],
) -> tuple[_Separator, ...]:
    regular = [
        region
        for region in regions
        if _source_type(region) == "text" and not _is_orientation_residual(region)
    ]
    if len(regular) < 6:
        return ()

    lines = _spatial_lines(regular, separators)
    typical_height = statistics.median(
        _height(region.bounding_box) for region in regular
    )
    candidates = []
    for line in lines:
        ordered = sorted(line.items, key=lambda item: item.bounding_box.left)
        if len(ordered) < 3 or any(_standalone_label(item.text) for item in ordered):
            continue
        line_height = statistics.median(_height(item.bounding_box) for item in ordered)
        split_indexes = [
            index
            for index in range(1, len(ordered))
            if ordered[index].bounding_box.left - ordered[index - 1].bounding_box.right
            >= line_height * 2
        ]
        for split_position, index in enumerate(split_indexes):
            if split_position == 0:
                continue
            next_index = (
                split_indexes[split_position + 1]
                if split_position + 1 < len(split_indexes)
                else len(ordered)
            )
            trailing = _union(item.bounding_box for item in ordered[index:next_index])
            if trailing.right - trailing.left < line_height * 6:
                continue
            left = ordered[index - 1].bounding_box
            right = ordered[index].bounding_box
            candidates.append(
                (
                    round((left.right + right.left) / 2),
                    line.box.top,
                    line.box.bottom,
                )
            )
    if len(candidates) < 2:
        return ()

    groups: list[list[tuple[int, int, int]]] = []
    tolerance = max(2, round(typical_height * 3))
    for candidate in sorted(candidates):
        group = next(
            (
                group
                for group in groups
                if abs(statistics.median(item[0] for item in group) - candidate[0])
                <= tolerance
            ),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)

    try:
        with Image.open(image_path) as source:
            gray = source.convert("L")
    except OSError:
        return ()
    page_width, page_height = gray.size
    inferred = []
    for group in groups:
        position = round(statistics.median(item[0] for item in group))
        for cluster in _continuous_separator_support(group, typical_height * 4):
            start = min(item[1] for item in cluster)
            end = max(item[2] for item in cluster)
            if (
                len(cluster) < 2
                or end - start < max(typical_height * 4, page_height * 0.05)
                or position < page_width * 0.6
                or not _low_ink_separator(gray, position, start, end)
            ):
                continue
            if any(
                separator.orientation == "vertical"
                and abs(separator.position - position) <= tolerance
                for separator in separators
            ):
                continue
            inferred.append(_Separator("vertical", position, start, end))
    return tuple(inferred)


def _continuous_separator_support(
    candidates: list[tuple[int, int, int]],
    maximum_gap: float,
) -> list[list[tuple[int, int, int]]]:
    clusters: list[list[tuple[int, int, int]]] = []
    for candidate in sorted(candidates, key=lambda item: item[1]):
        if not clusters or candidate[1] - clusters[-1][-1][2] > maximum_gap:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    return clusters


def _low_ink_separator(
    gray: Image.Image,
    position: int,
    start: int,
    end: int,
) -> bool:
    left = max(0, position - 4)
    right = min(gray.width, position + 5)
    top = max(0, start)
    bottom = min(gray.height, end)
    if left >= right or top >= bottom:
        return False
    pixels = gray.crop((left, top, right, bottom)).getdata()
    return statistics.mean(value < 220 for value in pixels) <= 0.05


def _column_groups(
    lines: list[_Line],
) -> tuple[list[tuple[str, list[_Line]]], list[_Line]]:
    groups: list[tuple[str, list[_Line]]] = []
    remaining = []
    index = 0
    while index < len(lines):
        run = [lines[index]]
        while index + len(run) < len(lines) and _aligned_columns(
            run[-1], lines[index + len(run)]
        ):
            run.append(lines[index + len(run)])
        if len(run) < 2:
            remaining.append(lines[index])
            index += 1
            continue
        ordered_rows = [
            sorted(line.items, key=lambda item: item.bounding_box.left) for line in run
        ]
        column_lines = [
            [_Line([row[column]]) for row in ordered_rows]
            for column in range(len(ordered_rows[0]))
        ]
        groups.extend(
            (
                "author"
                if any("@" in item.text for line in column for item in line.items)
                else "paragraph",
                column,
            )
            for column in column_lines
        )
        index += len(run)
    return groups, remaining


def _semantic_block_types(
    candidates: list[tuple[str, list[_Line]]],
) -> list[tuple[str, list[_Line]]]:
    heights = [
        _height(region.bounding_box)
        for block_type, lines in candidates
        if block_type != "aside_text"
        for line in lines
        for region in line.items
    ]
    if not heights:
        return candidates
    typical_height = statistics.median(heights)
    texts = [
        _resolved_text([item for line in lines for item in line.items])
        for _, lines in candidates
    ]
    math_flags = [
        _looks_like_math(text, lines)
        for text, (_, lines) in zip(texts, candidates, strict=True)
    ]
    formula_page = len(candidates) <= 40 and sum(math_flags) >= 3
    boxes = [_union(line.box for line in lines) for _, lines in candidates]
    typed = []
    for index, ((block_type, lines), text, math_flag) in enumerate(
        zip(candidates, texts, math_flags, strict=True)
    ):
        if formula_page and index == 0 and _running_header(text):
            typed.append(("header", lines))
        elif _numbered_formula(lines):
            typed.append((block_type, lines))
        elif formula_page and (
            block_type in {"line", "paragraph"}
            and math_flag
            or block_type in {"line", "paragraph", "form_row"}
            and _formula_context_fragment(text)
            and any(
                (
                    nearby_math
                    or other_index < len(typed)
                    and typed[other_index][0] == "formula"
                )
                and other_index != index
                and _near_formula_expression(
                    boxes[index], boxes[other_index], typical_height
                )
                for other_index, nearby_math in enumerate(math_flags)
            )
        ):
            typed.append(("formula", lines))
        elif block_type in {"line", "paragraph"} and math_flag:
            typed.append(("formula", lines))
        elif block_type in {"line", "paragraph"} and any(
            "@" in item.text for line in lines for item in line.items
        ):
            typed.append(("author", lines))
        else:
            typed.append((block_type, lines))
    title_candidates = []
    for index, (block_type, lines) in enumerate(typed):
        text = _resolved_text([item for line in lines for item in line.items])
        words = text.split()
        line_height = statistics.median(
            _height(item.bounding_box) for line in lines for item in line.items
        )
        if (
            block_type in {"line", "paragraph"}
            and 1 <= len(words) <= 20
            and any(character.isalpha() for character in text)
            and not _looks_like_math(text, lines)
            and (
                line_height >= typical_height * 1.35
                or len(words) <= 12
                and text.isupper()
            )
        ):
            title_candidates.append((index, line_height))
    if title_candidates:
        index, _ = max(title_candidates, key=lambda item: item[1])
        typed[index] = ("title", typed[index][1])

    for index, (block_type, lines) in enumerate(typed):
        text = _resolved_text([item for line in lines for item in line.items])
        if block_type in {"line", "paragraph"} and text.startswith(("*", "†", "‡")):
            typed[index] = ("footnote", lines)
            continue
        if block_type != "line" or index + 1 >= len(typed):
            continue
        words = text.split()
        line_height = statistics.median(
            _height(item.bounding_box) for item in lines[0].items
        )
        if (
            text.casefold() in _SECTION_HEADINGS
            or (
                1 <= len(words) <= 4
                and any(character.isalpha() for character in text)
                and not _looks_like_math(text, lines)
                and not text.endswith((".", ",", ";", ":"))
                and line_height >= typical_height
            )
        ) and typed[index + 1][0] == "paragraph":
            typed[index] = ("heading", lines)
    merged = []
    for block_type, block_lines in typed:
        line_groups = (
            [[line] for line in block_lines]
            if block_type == "formula"
            else [block_lines]
        )
        for lines in line_groups:
            if (
                block_type == "formula"
                and merged
                and merged[-1][0] == "formula"
                and not _numbered_formula(lines)
                and (
                    _formula_continuation(lines)
                    or _same_formula_expression(
                        _union(line.box for line in merged[-1][1]),
                        _union(line.box for line in lines),
                        typical_height,
                    )
                )
            ):
                merged[-1] = ("formula", [*merged[-1][1], *lines])
            else:
                merged.append((block_type, lines))
    return merged


def _running_header(text: str) -> bool:
    words = text.split()
    return any(character.isdigit() for character in text) and any(
        len(word) >= 4 and word.isupper() for word in words
    )


def _formula_context_fragment(text: str) -> bool:
    alpha_runs = re.findall(r"[A-Za-z]+", text)
    if not alpha_runs:
        return any(character.isdigit() for character in text)
    math_marker = any(
        character.isdigit() or character in "=+−±×÷∫√∑∂^²³|" for character in text
    )
    return all(len(token) <= 3 for token in alpha_runs) and (
        all(len(token) == 1 for token in alpha_runs)
        and len(alpha_runs) >= 2
        or math_marker
    )


def _near_formula_expression(
    first: BoundingBox,
    second: BoundingBox,
    typical_height: float,
) -> bool:
    vertical_gap = max(0, max(first.top, second.top) - min(first.bottom, second.bottom))
    return (
        vertical_gap <= typical_height * 5
        and _overlap(first.left, first.right, second.left, second.right) > 0
    )


def _same_formula_expression(
    first: BoundingBox,
    second: BoundingBox,
    typical_height: float,
) -> bool:
    vertical_gap = max(0, max(first.top, second.top) - min(first.bottom, second.bottom))
    horizontal_gap = max(
        0,
        max(first.left, second.left) - min(first.right, second.right),
    )
    vertical_overlap = _overlap(first.top, first.bottom, second.top, second.bottom)
    return vertical_gap <= typical_height * 0.75 and (
        _overlap(first.left, first.right, second.left, second.right) > 0
        or vertical_overlap > 0
        and horizontal_gap <= typical_height * 3
    )


def _numbered_formula(lines: list[_Line]) -> bool:
    if not lines:
        return False
    first = min(lines[0].items, key=lambda item: item.bounding_box.left).text.strip()
    if not first:
        return False
    token = first.split(maxsplit=1)[0].lstrip("([{")
    number = token.rstrip(".)]}:")
    return number != token and number.isdigit()


def _formula_continuation(lines: list[_Line]) -> bool:
    if not lines:
        return False
    first = min(lines[0].items, key=lambda item: item.bounding_box.left).text.strip()
    return first.startswith(("=", "+", "-", "−"))


def _atomic_candidates(regions: list[TextRegion]) -> list[tuple[str, list[_Line]]]:
    candidates = [
        (_source_type(region), [_Line([region])])
        for region in regions
        if not _is_orientation_residual(region)
    ]
    residuals: dict[tuple[object, object, object], list[TextRegion]] = {}
    for region in regions:
        if not _is_orientation_residual(region):
            continue
        metadata = region.text_provenance["orientation_residual"]  # type: ignore[index]
        key = (
            metadata.get("method"),
            metadata.get("source_margin"),
            tuple(metadata.get("source_crop", [])),
        )
        residuals.setdefault(key, []).append(region)
    candidates.extend(
        (
            "aside_text",
            [
                _Line([region])
                for region in sorted(items, key=lambda item: item.reading_order)
            ],
        )
        for items in residuals.values()
    )
    return candidates


def _aligned_columns(first: _Line, second: _Line) -> bool:
    first_items = sorted(first.items, key=lambda item: item.bounding_box.left)
    second_items = sorted(second.items, key=lambda item: item.bounding_box.left)
    if not 2 <= len(first_items) == len(second_items) <= 6:
        return False
    if any(":" in item.text for item in [*first_items, *second_items]):
        return False
    typical_height = statistics.median(
        _height(item.bounding_box) for item in [*first_items, *second_items]
    )
    vertical_gap = second.box.top - first.box.bottom
    if not -typical_height * 0.25 <= vertical_gap <= typical_height * 1.5:
        return False
    if any(
        right.bounding_box.left - left.bounding_box.right <= typical_height
        for row in (first_items, second_items)
        for left, right in zip(row, row[1:])
    ):
        return False
    return all(
        abs(
            _horizontal_center(left.bounding_box)
            - _horizontal_center(right.bounding_box)
        )
        <= typical_height * 3
        for left, right in zip(first_items, second_items, strict=True)
    )


def _form_row(line: _Line) -> bool:
    return not _looks_like_math(_resolved_text(line.items), [line]) and (
        len(_line_segments(line.items)) > 1
        or any(_standalone_label(region.text) for region in line.items)
    )


def _looks_like_math(text: str, lines: list[_Line]) -> bool:
    operators = frozenset("=+−±×÷∫√∑∂^²³{}|")
    items = [region.text.strip() for line in lines for region in line.items]
    compact = [item for item in items if item]
    words = [word for item in compact for word in item.split()]
    operator_count = sum(character in operators for character in text)
    long_words = sum(
        len("".join(character for character in word if character.isalpha())) >= 4
        for word in words
    )
    if operator_count >= 2 and long_words < max(2, len(words) // 3):
        return True
    if len(words) < 2 or any(_standalone_label(item) for item in compact):
        return False
    short = sum(
        len("".join(character for character in word if character.isalnum())) <= 3
        for word in words
    )
    return long_words <= 1 and short / len(words) >= 0.75 and operator_count > 0


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
        text for region in regions if (text := _canonical_region_text(region))
    )


def _canonical_region_text(region: TextRegion) -> str:
    if region.resolution == "resolved":
        return region.text.strip()
    if region.kind == "handwriting":
        return "[unreadable handwriting]"
    return ""


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


def _structured_form_field(
    block_id: str,
    index: int,
    regions: list[TextRegion],
) -> dict[str, Any]:
    label_regions, value_regions = _form_field_parts(regions)
    resolutions = {region.resolution for region in value_regions}
    if "conflicting" in resolutions:
        state = "conflicting_readings"
    elif "unreadable" in resolutions:
        state = "illegible"
    elif value_regions:
        state = "present"
    else:
        state = "not_located"
    return {
        "id": f"{block_id}-field-{index}",
        "label": _literal_text(label_regions),
        "raw_value": _literal_text(value_regions) or None,
        "evidence_ids": [region.id for region in regions],
        "label_evidence_ids": [region.id for region in label_regions],
        "value_evidence_ids": [region.id for region in value_regions],
        "normalization_history": [],
        "state": state,
        "source_geometry": _field_geometry(regions),
        "answer_geometry": _field_geometry(value_regions),
        "recognition_evidence": [
            {
                "evidence_id": region.id,
                "raw_text": region.text,
                "provider": region.provider,
                "confidence": region.confidence,
                "resolution": region.resolution,
                "alternatives": [
                    {
                        "raw_text": alternative.text,
                        "provider": alternative.provider,
                        "confidence": alternative.confidence,
                    }
                    for alternative in region.alternatives
                ],
            }
            for region in regions
        ],
        "association_status": "linked" if value_regions else "unmatched",
        "association_confidence": None,
    }


def _form_field_parts(
    regions: list[TextRegion],
) -> tuple[list[TextRegion], list[TextRegion]]:
    label_end = next(
        (
            index
            for index in range(len(regions) - 1, -1, -1)
            if _standalone_label(regions[index].text)
        ),
        None,
    )
    if label_end is not None:
        return regions[: label_end + 1], regions[label_end + 1 :]
    return (regions[:1], regions[1:]) if len(regions) > 1 else (regions, [])


def _literal_text(regions: list[TextRegion]) -> str:
    return " ".join(region.text.strip() for region in regions if region.text.strip())


def _field_geometry(regions: list[TextRegion]) -> dict[str, Any] | None:
    if not regions:
        return None
    box = _union(region.bounding_box for region in regions)
    return {
        "evidence_ids": [region.id for region in regions],
        "bounding_box": {
            "left": box.left,
            "top": box.top,
            "right": box.right,
            "bottom": box.bottom,
        },
    }


def _split_form_segment(segment: list[TextRegion]) -> list[list[TextRegion]]:
    fields: list[list[TextRegion]] = []
    current: list[TextRegion] = []
    for region in segment:
        if current and _standalone_label(region.text):
            label_index = next(
                (
                    index
                    for index, item in enumerate(current)
                    if _standalone_label(item.text)
                ),
                None,
            )
            if label_index is not None and label_index < len(current) - 1:
                gaps = [
                    (
                        index,
                        current[index].bounding_box.left
                        - current[index - 1].bounding_box.right,
                    )
                    for index in range(label_index + 2, len(current))
                ]
                typical_height = statistics.median(
                    _height(item.bounding_box) for item in current
                )
                split_at = (
                    max(gaps, key=lambda item: item[1])[0] if gaps else len(current)
                )
                if gaps and max(gap for _, gap in gaps) <= typical_height * 2:
                    split_at = len(current)
                fields.append(current[:split_at])
                current = current[split_at:]
        current.append(region)
    if current:
        fields.append(current)
    return fields


def _standalone_label(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.endswith(":")
        and stripped.count(":") == 1
        and any(character.isalnum() for character in stripped[:-1])
    )


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


def _is_orientation_residual(region: TextRegion) -> bool:
    provenance = region.text_provenance
    return isinstance(provenance, dict) and isinstance(
        provenance.get("orientation_residual"), dict
    )


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


def _group_formula_blocks(
    image_path: Path,
    blocks: list[TextRegion],
) -> list[TextRegion]:
    formulas = [
        block
        for block in blocks
        if (block.structure or {}).get("block_type") == "formula"
    ]
    if not formulas or not image_path.is_file():
        return blocks
    try:
        with Image.open(image_path) as source:
            image_size = source.size
            mask = source.convert("L").point(lambda value: 255 if value < 220 else 0)
    except OSError:
        return blocks

    consumed: set[str] = set()
    for formula in formulas:
        if formula.id in consumed:
            continue
        crop = formula_ink_box(mask, formula.bounding_box, image_size, 0)
        members = [
            block
            for block in blocks
            if block.id not in consumed
            and _formula_crop_member(block, crop, formula.bounding_box, formula.id)
        ]
        members.sort(
            key=lambda block: (
                block.bounding_box.top,
                block.bounding_box.left,
                block.reading_order,
            )
        )
        structure = dict(formula.structure or {})
        structure["child_evidence_ids"] = list(
            dict.fromkeys(
                evidence_id for member in members for evidence_id in _child_ids(member)
            )
        )
        structure["source_reading_orders"] = [
            copy.deepcopy(item)
            for member in members
            for item in (member.structure or {}).get("source_reading_orders", [])
            if isinstance(item, dict)
        ]
        structure["lines"] = [
            copy.deepcopy(line)
            for member in members
            for line in (member.structure or {}).get("lines", [])
            if isinstance(line, dict)
        ]
        structure["formula_crop"] = {
            "left": crop.left,
            "top": crop.top,
            "right": crop.right,
            "bottom": crop.bottom,
        }
        structure["formula_grouping"] = {
            "method": "source_ink_band",
            "source_owner_ids": [member.id for member in members],
        }
        formula.structure = structure
        provenance = dict(formula.text_provenance or {})
        provenance["source_region_ids"] = copy.deepcopy(structure["child_evidence_ids"])
        provenance["source_providers"] = list(
            dict.fromkeys(
                provider
                for member in members
                for provider in (member.text_provenance or {}).get(
                    "source_providers", []
                )
                if isinstance(provider, str)
            )
        )
        provenance["formula_grouping"] = copy.deepcopy(structure["formula_grouping"])
        formula.text_provenance = provenance
        formula.text = " ".join(
            member.text.strip() for member in members if member.text.strip()
        )
        formula.bounding_box = crop
        formula.reading_order = min(member.reading_order for member in members)
        consumed.update(member.id for member in members if member.id != formula.id)
    mask.close()
    return [block for block in blocks if block.id not in consumed]


def _formula_crop_member(
    block: TextRegion,
    crop: BoundingBox,
    seed: BoundingBox,
    formula_id: str,
) -> bool:
    if block.id == formula_id:
        return True
    block_type = (block.structure or {}).get("block_type")
    if block_type not in {"formula", "line"}:
        return False
    if block_type != "formula" and len(block.text.split()) > 3:
        return False
    center_x = (block.bounding_box.left + block.bounding_box.right) / 2
    center_y = (block.bounding_box.top + block.bounding_box.bottom) / 2
    if not (
        crop.left <= center_x <= crop.right and crop.top <= center_y <= crop.bottom
    ):
        return False
    horizontal_overlap = _overlap(
        seed.left,
        seed.right,
        block.bounding_box.left,
        block.bounding_box.right,
    )
    vertical_overlap = _overlap(
        seed.top,
        seed.bottom,
        block.bounding_box.top,
        block.bounding_box.bottom,
    )
    if (
        block_type != "formula"
        and vertical_overlap <= 0
        and not (_formula_context_fragment(block.text) and ":" not in block.text)
    ):
        return False
    horizontal_gap = max(
        0,
        max(seed.left, block.bounding_box.left)
        - min(seed.right, block.bounding_box.right),
    )
    seed_height = seed.bottom - seed.top
    vertical_gap = max(
        0,
        max(seed.top, block.bounding_box.top)
        - min(seed.bottom, block.bounding_box.bottom),
    )
    minimum_width = min(
        seed.right - seed.left,
        block.bounding_box.right - block.bounding_box.left,
    )
    return (
        vertical_gap <= seed_height and horizontal_overlap * 5 >= minimum_width
    ) or (vertical_overlap > 0 and horizontal_gap <= seed_height * 3)


def formula_ink_box(
    mask: Image.Image,
    box: BoundingBox,
    image_size: tuple[int, int],
    padding: int,
) -> BoundingBox:
    width, height = image_size
    seed_height = box.bottom - box.top
    vertical_limit = max(padding, seed_height * 4)
    horizontal_limit = max(padding, seed_height * 5)
    search_left = max(0, box.left - horizontal_limit)
    search_right = min(width, box.right + horizontal_limit)
    search_top = max(0, box.top - vertical_limit)
    search_bottom = min(height, box.bottom + vertical_limit)
    row_ink = []
    row_width = search_right - search_left
    for row in range(search_top, search_bottom):
        strip = mask.crop((search_left, row, search_right, row + 1))
        ink_pixels = strip.histogram()[255]
        strip.close()
        row_ink.append(0 < ink_pixels < row_width * 0.8)
    seed_rows = [row for row in range(box.top, box.bottom) if row_ink[row - search_top]]
    if not seed_rows:
        return _padded_box(box, image_size, padding)

    max_blank_rows = max(2, min(8, seed_height // 3))
    ink_top = min(seed_rows)
    blank_rows = 0
    for row in range(ink_top - 1, search_top - 1, -1):
        if row_ink[row - search_top]:
            ink_top = row
            blank_rows = 0
        else:
            blank_rows += 1
            if blank_rows > max_blank_rows:
                break
    ink_bottom = max(seed_rows) + 1
    blank_rows = 0
    for row in range(ink_bottom, search_bottom):
        if row_ink[row - search_top]:
            ink_bottom = row + 1
            blank_rows = 0
        else:
            blank_rows += 1
            if blank_rows > max_blank_rows:
                break

    band = mask.crop((search_left, ink_top, search_right, ink_bottom))
    column_ink = [
        band.crop((column, 0, column + 1, band.height)).getbbox() is not None
        for column in range(band.width)
    ]
    band.close()
    seed_columns = [
        column
        for column in range(box.left - search_left, box.right - search_left)
        if column_ink[column]
    ]
    if not seed_columns:
        return _padded_box(box, image_size, padding)

    max_blank_columns = max(padding, seed_height * 2)
    ink_left = min(seed_columns)
    blank_columns = 0
    for column in range(ink_left - 1, -1, -1):
        if column_ink[column]:
            ink_left = column
            blank_columns = 0
        else:
            blank_columns += 1
            if blank_columns > max_blank_columns:
                break
    ink_right = max(seed_columns) + 1
    blank_columns = 0
    for column in range(ink_right, len(column_ink)):
        if column_ink[column]:
            ink_right = column + 1
            blank_columns = 0
        else:
            blank_columns += 1
            if blank_columns > max_blank_columns:
                break
    return _padded_box(
        BoundingBox(
            ink_left + search_left,
            ink_top,
            ink_right + search_left,
            ink_bottom,
        ),
        image_size,
        padding,
    )


def _padded_box(
    box: BoundingBox,
    image_size: tuple[int, int],
    padding: int,
) -> BoundingBox:
    width, height = image_size
    return BoundingBox(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(width, box.right + padding),
        min(height, box.bottom + padding),
    )


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


def _horizontal_center(box: BoundingBox) -> float:
    return (box.left + box.right) / 2


def _valid_box(box: BoundingBox) -> bool:
    return box.left < box.right and box.top < box.bottom


def _label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "_".join(value.strip().casefold().replace("-", " ").split())
