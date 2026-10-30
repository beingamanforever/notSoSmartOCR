"""Arrange an OCR page's regions into deduplicated, ordered reading blocks.

Both the Markdown export and the LLM extractor's prompt rows read through
``arrange_page``/``block_text``/``arranged_markdown`` so the two outputs can
never disagree about what a page contains, in what order, or which text is a
duplicate. The input is a raw page dict from the OCR ``result.json`` (never
mutated); the output only ever repeats text that was already there.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from . import rendering

Box = tuple[float, float, float, float]

_SOURCE_ROLE_SUFFIX = "_source"
_NOT_CONTENT_KINDS = {"coverage_risk", "table_candidate"}
_LAYOUT_CONTAINER_KINDS = {"layout_block", "key-value-region", "form"}
_CONTROL_KINDS = {"checkbox", "radio"}
_HEADING_KINDS = {"title", "section-header", "page-header"}

_WORD_RE = re.compile(r"\w+")
# A leaf whose own text already reads as "Label: value" can never be someone
# else's value -- picking it up would duplicate it under two labels.
_SELF_CONTAINED_KV_RE = re.compile(r"^[^:]{1,40}:\s+\S")
_CAPTION_TAIL_RE = re.compile(r"^(?P<id>\[[^\]]+\] )?(?P<caption>[A-Za-z]+(?: [A-Za-z]+)?):\s+(?P<value>\S.*)$", re.S)
_PRINTED_BULLET_RE = re.compile(r"^[•●▪◦‣∙*\-–]\s+")
# "1 Name ...", "5 Address ...", "3a Federal ..." (a form line's printed caption, no period)
_FORM_LINE_LABEL_RE = re.compile(r"^\d{1,2}[a-z]?\s+[A-Z(]")
_NUMBERED_LINE_RE = re.compile(r"^\d{1,3}[a-z]?[.)]?\s")
# One digit per printed box: "1 2 - 3 4 5 6 7 8 9" reads as "12-3456789".
_DIGIT_BOXES_RE = re.compile(r"(?<![\w-])\d(?: (?:- )?\d){4,}(?![\w-])")
# A bracketed placeholder for a picture is never a section title.
_PLACEHOLDER_RE = re.compile(r"^\[[^\]]{1,30}\]$")
INLINE_LABEL_MAX_WORDS = 3
INLINE_LABEL_MAX_CHARS = 24
# A filled-in value (handwriting, a larger typed fill) is this much taller than its printed
# label: 2.1-3.8x on a handwritten fax cover; two printed captions differ by under 1.4x.
VALUE_TALLER_RATIO = 1.6
FILLED_VALUE_MAX_WORDS = 8
_NUMBERED_ITEM_RE = re.compile(r"^\d{1,3}[.)]\s")

ROW_OVERLAP_RATIO = 0.5
TALL_LEAF_HEIGHT_RATIO = 3.0
JUNK_SPECK_CONFIDENCE = 0.3
JUNK_SPECK_MAX_CHARS = 2
REPEATED_TOKEN_COVERAGE = 0.8
REPEATED_BOX_OVERLAP = 0.3
VALUE_RIGHT_MAX_WIDTH_RATIO = 0.25
VALUE_BELOW_TOP_HEIGHTS = 2.5
VALUE_BELOW_LEFT_HEIGHTS = 2.0
CONTINUATION_LEFT_LINE_HEIGHTS = 1.0
CONTINUATION_GAP_LINE_HEIGHTS = 0.5
GRID_LABEL_MAX_WORDS = 8
GRID_ROW_GAP_HEIGHTS = 2.5
CAPTION_HEIGHT_RATIO = 1.3  # value text this much taller than the row below: captions below
GRID_ALIGN_OVERLAP_RATIO = 0.5
CUT_GAP_HEIGHT_RATIO = 0.4
# Measured on the real MICA certificate: its label/value "columns" only share
# their first row, so the right column covers roughly half the band's height
# while a genuine two-column prose zone (W-9 General Instructions) covers
# nearly all of it. 0.7 sits cleanly between the two measured ratios.
COLUMN_SPAN_COVERAGE = 0.7
ADDRESS_LIKE_MAX_LINES = 4
ADDRESS_LIKE_MAX_WORDS = 8


def arrange_page(page: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate, classify and order one page's regions."""
    regions = [r for r in (page.get("regions") or []) if isinstance(r, dict)]
    all_ids = {r.get("id") for r in regions}

    suppressed: list[dict[str, Any]] = []
    leaves: list[dict[str, Any]] = []
    for region in regions:
        reason = _suppression_reason(region, all_ids)
        if reason:
            suppressed.append(
                {"region_id": region.get("id"), "reason": reason, "covered_by": None}
            )
        else:
            leaves.append(region)

    kept: list[dict[str, Any]] = []
    for leaf in leaves:
        if _is_junk_speck(leaf):
            suppressed.append(
                {"region_id": leaf.get("id"), "reason": "junk_speck", "covered_by": None}
            )
        else:
            kept.append(leaf)

    trimmed_lines, repeat_suppressed, leaves = _trim_repeated_text(kept)
    suppressed.extend(repeat_suppressed)

    if not leaves:
        return {"version": 1, "blocks": [], "suppressed": suppressed, "trimmed_lines": trimmed_lines}
    figure_boxes = [_box(region) for region in regions if region.get("kind") == "figure" and _box(region)]

    if any(_box(leaf) is None for leaf in leaves):
        # ponytail: a page with any un-boxed leaf skips geometry entirely rather than
        # mixing boxed and boxless placement; upgrade to partial geometry if that ever
        # loses real layout on a page that mostly does have boxes.
        units = [_standalone_unit(leaf) for leaf in sorted(leaves, key=_reading_order)]
    else:
        median_height = _median_height(leaves)
        # The layout detector's blocks are the page's groups (one signature column, one
        # form section): fields pair only inside a block, and each block reads whole.
        group_of = _layout_groups(leaves)
        groups: dict[Any, list[dict[str, Any]]] = {}
        for leaf in leaves:
            groups.setdefault(group_of[leaf["id"]], []).append(leaf)
        units = [unit for group in groups.values() for unit in _build_units(group, median_height, page)]
        units = _blocks_whole(_xy_cut(units, median_height), group_of)

    index = {region.get("id"): region for region in regions}
    for unit in units:
        if unit["kind"] == "heading" and _not_a_title(index[unit["region_ids"][0]], figure_boxes):
            unit["kind"] = "text"
    blocks = [_unit_to_block(unit) for unit in units]
    return {"version": 1, "blocks": blocks, "suppressed": suppressed, "trimmed_lines": trimmed_lines}


def _layout_groups(leaves: list[dict[str, Any]]) -> dict[Any, Any]:
    """Each leaf's layout block, or None when its block holds nothing else (lone regions
    pair with each other, as the detector did not group them)."""
    owner = {leaf["id"]: (leaf.get("structure") or {}).get("layout_owner_id") for leaf in leaves}
    sizes = Counter(owner.values())
    return {leaf_id: block if block is not None and sizes[block] > 1 else None for leaf_id, block in owner.items()}


def _blocks_whole(units: list[dict[str, Any]], group_of: dict[Any, Any]) -> list[dict[str, Any]]:
    """The units in reading order, each layout block's units kept together where its first
    unit reads; the reader's sections are ordered by the same first units."""
    first: dict[Any, int] = {}
    keys = []
    for position, unit in enumerate(units):
        group = group_of[(unit["label_ids"] or unit["region_ids"])[0]]
        keys.append(first.setdefault(group if group is not None else ("unit", position), position))
    return [unit for _, _, unit in sorted(zip(keys, range(len(units)), units), key=lambda item: item[:2])]


def _not_a_title(region: dict[str, Any], figure_boxes: list[Box]) -> bool:
    """"[diagram]" placeholders and words inside a picture ("SEAL" in a stamp) read as text."""
    text = (region.get("text") or "").strip()
    if _PLACEHOLDER_RE.match(text):
        return True
    box = _box(region)
    return box is not None and len(text.split()) <= 2 and any(
        _containment(box, figure) >= 0.8 for figure in figure_boxes if figure != box
    )


def _containment(inner: Box, outer: Box) -> float:
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0.0, width) * max(0.0, height) / area if area > 0 else 0.0


def block_text(
    page: dict[str, Any],
    block: dict[str, Any],
    *,
    with_ids: bool,
    arrangement: dict[str, Any] | None = None,
) -> str:
    """Render one block as plain text, with or without per-region evidence ids."""
    arrangement = arrangement if arrangement is not None else arrange_page(page)
    index = _region_index(page)
    if block["kind"] == "controls":
        label = _ids_text(block["label_ids"], index, _trimmed_by_region(arrangement), with_ids)
        # The tick is the value of a checkbox: "[p1-r-70] [x] C Corporation".
        options = [_control_option(index[i], with_ids) for i in block["region_ids"] if i in index]
        value = "  ".join(option for option in options if option)
        return f"{_as_label(label)} {value}" if label and value else label or value
    label, value = _label_and_value(block, index, arrangement, with_ids)
    return f"{label} {value}" if label and value else label or value


def _label_and_value(
    block: dict[str, Any], index: dict[Any, dict[str, Any]], arrangement: dict[str, Any], with_ids: bool
) -> tuple[str, str]:
    """A block's caption (ending in ':' when it has a value) and its value text."""
    trimmed = _trimmed_by_region(arrangement)
    borrowed = block.get("borrowed_label")
    if borrowed:  # the caption OCR read into the previous value
        label = f"[{borrowed['region_id']}] {borrowed['text']}" if with_ids else borrowed["text"]
    else:
        label = _ids_text(block["label_ids"], index, trimmed, with_ids)
    value = _value_text(
        block["region_ids"], index, trimmed, with_ids, labelled=bool(label),
        drop_last_words=block.get("drop_last_words") or {},
    )
    rest = _CAPTION_TAIL_RE.match(value) if label and not label.endswith(":") else None
    if rest:  # the caption ran on into the value's box: "Print" | "Name: JANE DOE"
        label, value = f"{label} {rest['caption']}", f"{rest['id'] or ''}{rest['value']}"
    return (_as_label(label) if label and value else label), value


def _as_label(label: str) -> str:
    return label.rstrip(" .:") + ":"


def arranged_markdown(page: dict[str, Any], arrangement: dict[str, Any] | None = None) -> str:
    """Render a page's arranged blocks as Markdown."""
    arrangement = arrangement if arrangement is not None else arrange_page(page)
    index = _region_index(page)
    parts = [_block_markdown(page, block, index, arrangement) for block in arrangement["blocks"]]
    return "\n\n".join(part for part in parts if part)


# --- Suppression (dedup + junk removal) -------------------------------------


def _suppression_reason(region: dict[str, Any], all_ids: set[Any]) -> str | None:
    structure = region.get("structure") or {}
    role = str(structure.get("role") or "")
    kind = region.get("kind", "text")
    if role.endswith(_SOURCE_ROLE_SUFFIX) or kind in _NOT_CONTENT_KINDS:
        return "not_content"
    if role == "layout_block" or kind in _LAYOUT_CONTAINER_KINDS:
        children = structure.get("child_evidence_ids") or []
        if any(child in all_ids for child in children):
            return "layout_container"
    if not (region.get("text") or "").strip() and not _has_image(region):
        if kind == "figure" and region.get("resolution", "resolved") == "resolved":
            return "empty_graphic"  # a seal or speck the LLM found no text in: nothing to show
        if kind == "figure" or region.get("resolution") == "unreadable":
            return "no_text"
    return None


def _has_image(region: dict[str, Any]) -> bool:
    """A resolved figure carrying its extracted image (the Markdown embeds it)."""
    image = (region.get("structure") or {}).get("image") or {}
    return region.get("resolution", "resolved") == "resolved" and bool(image.get("data"))


def _is_junk_speck(region: dict[str, Any]) -> bool:
    if _is_control(region) or region.get("resolution") != "resolved":
        return False
    if (region.get("text_provenance") or {}).get("method") == "qwen_crop_refinement":
        return False
    confidence = region.get("confidence")
    if not isinstance(confidence, int | float) or confidence >= JUNK_SPECK_CONFIDENCE:
        return False
    return len((region.get("text") or "").strip()) <= JUNK_SPECK_MAX_CHARS


def _trim_repeated_text(
    leaves: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [leaf for leaf in leaves if leaf.get("kind") not in {"table", "figure"}]
    trims_by_id: dict[Any, dict[int, dict[str, Any]]] = {}
    for leaf in eligible:
        box = _box(leaf)
        for line_index, line in enumerate((leaf.get("text") or "").split("\n")):
            tokens = Counter(_WORD_RE.findall(line.casefold()))
            if sum(tokens.values()) < 2:
                continue
            # ponytail: O(regions^2) line-vs-region scan; fine at page scale (<200 regions).
            covering = _find_covering_region(leaf, box, tokens, leaves)
            if covering is not None:
                trims_by_id.setdefault(leaf["id"], {})[line_index] = covering

    suppressed: list[dict[str, Any]] = []
    trimmed_lines: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for leaf in leaves:
        line_trims = dict(trims_by_id.get(leaf["id"], {}))
        total_lines = len((leaf.get("text") or "").split("\n"))
        fully_trimmed = total_lines > 0 and len(line_trims) == total_lines
        if _is_control(leaf):
            state = (leaf.get("structure") or {}).get("state")
            matched_by_control = any(
                _is_control(covering) and (covering.get("structure") or {}).get("state") == state
                for covering in line_trims.values()
            )
            if not (fully_trimmed and matched_by_control):
                # A control only disappears outright when a matching control replaces
                # it; otherwise it must keep showing the option it marks.
                line_trims.pop(0, None)
                fully_trimmed = False
        if fully_trimmed and line_trims:
            covering = line_trims[0]
            suppressed.append(
                {"region_id": leaf["id"], "reason": "repeated_text", "covered_by": covering["id"]}
            )
            continue
        survivors.append(leaf)
        for line_index, covering in sorted(line_trims.items()):
            trimmed_lines.append(
                {"region_id": leaf["id"], "line": line_index, "covered_by": covering["id"]}
            )
    return trimmed_lines, suppressed, survivors


def _find_covering_region(
    leaf: dict[str, Any],
    box: Box | None,
    tokens: Counter,
    leaves: list[dict[str, Any]],
) -> dict[str, Any] | None:
    token_total = sum(tokens.values())
    best: dict[str, Any] | None = None
    best_total = -1
    for other in leaves:
        if other is leaf or box is None:
            continue
        other_box = _box(other)
        if other_box is None or _box_overlap_ratio(box, other_box) < REPEATED_BOX_OVERLAP:
            continue
        other_tokens = Counter(_WORD_RE.findall((other.get("text") or "").casefold()))
        other_total = sum(other_tokens.values())
        if other_total <= token_total:
            continue
        covered = sum(min(count, other_tokens.get(word, 0)) for word, count in tokens.items())
        if covered / token_total < REPEATED_TOKEN_COVERAGE:
            continue
        if other_total > best_total or (
            other_total == best_total
            and best is not None
            and _reading_order(other) < _reading_order(best)
        ):
            best, best_total = other, other_total
    return best


# --- Units: key/value pairs, controls, grids, and standalone leaves ---------


def _build_units(
    leaves: list[dict[str, Any]], median_height: float, page: dict[str, Any]
) -> list[dict[str, Any]]:
    page_width = page.get("width") or 0
    all_leaves = leaves
    leaves = [leaf for leaf in leaves if not _is_code(leaf)]  # code is never a caption or a value
    pool = {leaf["id"]: leaf for leaf in leaves}
    consumed: set[Any] = set()
    units: list[dict[str, Any]] = []

    # Printed captions with a filled value on their right ("FROM | ACME" on a fax cover),
    # then numbered form lines with their value below ("1 Name ..." over the name).
    inline_units, inline_consumed = _pair_inline(leaves, page_width)
    units.extend(inline_units)
    consumed.update(inline_consumed)
    form_units, form_consumed = _pair_form_lines([leaf for leaf in leaves if leaf["id"] not in consumed])
    units.extend(form_units)
    consumed.update(form_consumed)

    labels = sorted(
        (leaf for leaf in leaves if _is_colon_label(leaf) and leaf["id"] not in consumed), key=_reading_order
    )
    for label in labels:
        if label["id"] in consumed:
            continue
        unit, used_ids = _pair_label(label, pool, consumed, median_height, page_width)
        units.append(unit)
        consumed.add(label["id"])
        consumed.update(used_ids)

    remaining = [leaf for leaf in leaves if leaf["id"] not in consumed]
    grid_units, grid_consumed = _pair_grid(remaining, median_height)
    units.extend(grid_units)
    consumed.update(grid_consumed)

    for leaf in all_leaves:
        if leaf["id"] not in consumed:
            units.append(_standalone_unit(leaf))
    return units


def _is_code(leaf: dict[str, Any]) -> bool:
    return leaf.get("kind") == "code" or (leaf.get("text_provenance") or {}).get("layout_category") in {"algorithm", "code"}


def _is_inline_label(leaf: dict[str, Any]) -> bool:
    """A short printed caption: 'FROM', 'FAX.', 'Date', 'To:' (no digits, one line)."""
    if _is_control(leaf) or leaf.get("kind") in {"table", "figure", "formula", "list-item"}:
        return False
    text = (leaf.get("text") or "").strip()
    if not text or "\n" in text or len(text) > INLINE_LABEL_MAX_CHARS or any(c.isdigit() for c in text):
        return False
    words = text.rstrip(":.").split()
    return 1 <= len(words) <= INLINE_LABEL_MAX_WORDS and all(any(c.isalpha() for c in w) for w in words)


def _looks_filled(value: dict[str, Any], label: dict[str, Any]) -> bool:
    """Written into the form rather than printed on it: taller than the caption, or set large."""
    value_box, label_box = _box(value), _box(label)
    if len((value.get("text") or "").split()) > FILLED_VALUE_MAX_WORDS:
        return False  # a paragraph beside a caption is prose, not its value
    return (
        value_box[3] - value_box[1] >= VALUE_TALLER_RATIO * (label_box[3] - label_box[1])
        or value.get("kind") in {"section-header", "title"}
    )


def _right_neighbours(label: dict[str, Any], leaves: list[dict[str, Any]], consumed: set[Any]) -> list[dict[str, Any]]:
    label_box = _box(label)
    # Overlap is judged against the smaller box: a tall handwritten value still shares
    # the printed caption's line.
    return sorted(
        (
            leaf
            for leaf in leaves
            if leaf is not label
            and leaf["id"] not in consumed
            and _box(leaf)[0] >= label_box[2] - 2
            and _axis_overlap_ratio(label_box[1], label_box[3], _box(leaf)[1], _box(leaf)[3]) >= ROW_OVERLAP_RATIO
        ),
        key=lambda leaf: _box(leaf)[0],
    )


def _pair_inline(leaves: list[dict[str, Any]], page_width: float) -> tuple[list[dict[str, Any]], set[Any]]:
    units: list[dict[str, Any]] = []
    consumed: set[Any] = set()
    for label in sorted(leaves, key=lambda leaf: (_box(leaf)[1], _box(leaf)[0])):
        if label["id"] in consumed or not (_is_inline_label(label) or _is_colon_label(label)):
            continue
        row = _right_neighbours(label, leaves, consumed)
        if not row:
            continue
        value = row[0]
        gap = _box(value)[0] - _box(label)[2]
        if _is_control(value) or gap > VALUE_RIGHT_MAX_WIDTH_RATIO * page_width or not _looks_filled(value, label):
            continue
        unit = _make_unit("key_value", [label], [value])
        consumed.update((label["id"], value["id"]))
        units.append(unit)
        # OCR sometimes reads the next printed caption into this value ("555-201-0147 Pages"
        # beside a filled "10"): the caption is shown as the next value's label instead.
        caption = _trailing_caption(value)
        following = _right_neighbours(value, leaves, consumed)
        if caption and following and not _is_inline_label(following[0]) and _looks_filled(following[0], label):
            unit["drop_last_words"] = {value["id"]: 1}
            borrowed = _make_unit("key_value", [], [following[0]])
            borrowed["borrowed_label"] = {"region_id": value["id"], "text": caption}
            borrowed["box"] = _union_box([borrowed["box"], _box(value)])
            consumed.add(following[0]["id"])
            units.append(borrowed)
    return units, consumed


def _looks_entered(value: dict[str, Any], label: dict[str, Any]) -> bool:
    """Typed or written into a form line: large, numeric or capitalised, unlike a caption."""
    text = (value.get("text") or "").strip()
    letters = [c for c in text if c.isalpha()]
    return (
        _looks_filled(value, label)
        or any(c.isdigit() for c in text)
        or (len(letters) >= 3 and sum(c.isupper() for c in letters) >= 0.7 * len(letters))
    )


def _trailing_caption(region: dict[str, Any]) -> str | None:
    words = (region.get("text") or "").strip().split()
    if len(words) < 2 or "\n" in (region.get("text") or "").strip():
        return None
    last = words[-1].rstrip(":.")
    return last if last.isalpha() and last[:1].isupper() and 3 <= len(last) <= 12 else None


def _pair_form_lines(leaves: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[Any]]:
    """'1 Name (as shown ...)' with the value printed on the line below it."""
    units: list[dict[str, Any]] = []
    consumed: set[Any] = set()
    for label in sorted(leaves, key=lambda leaf: (_box(leaf)[1], _box(leaf)[0])):
        text = (label.get("text") or "").strip()
        if label["id"] in consumed or not _FORM_LINE_LABEL_RE.match(text) or _is_control(label):
            continue
        label_box = _box(label)
        height = label_box[3] - label_box[1]
        below = [
            leaf
            for leaf in leaves
            if leaf["id"] not in consumed
            and leaf is not label
            and 0 <= _box(leaf)[1] - label_box[3] <= VALUE_BELOW_TOP_HEIGHTS * height
            and abs(_box(leaf)[0] - label_box[0]) <= VALUE_BELOW_LEFT_HEIGHTS * height
            and leaf.get("kind") not in {"list-item", "table", "figure"}
            and not _is_control(leaf)
            and not _NUMBERED_LINE_RE.match((leaf.get("text") or "").strip())
        ]
        if not below:
            continue
        value = min(below, key=lambda leaf: _box(leaf)[1])
        if not _looks_entered(value, label):
            continue  # "Exempt payee code (if any)" under "4 Exemptions" is another caption
        units.append(_make_unit("key_value", [label], [value]))
        consumed.update((label["id"], value["id"]))
    return units, consumed


def _pair_label(
    label: dict[str, Any],
    pool: dict[Any, dict[str, Any]],
    consumed: set[Any],
    median_height: float,
    page_width: float,
) -> tuple[dict[str, Any], set[Any]]:
    candidates = [
        region for region in pool.values() if region["id"] not in consumed and region is not label
    ]
    row_values = _row_value_candidates(label, candidates, median_height, page_width)
    if row_values and all(_is_control(region) for region in row_values):
        used = {region["id"] for region in row_values}
        return _make_unit("controls", [label], row_values), used

    chain: list[dict[str, Any]] = []
    used = set()
    first = row_values[0] if row_values else _below_value_candidate(label, candidates, median_height)
    if first is not None:
        chain.append(first)
        used.add(first["id"])
        current = first
        while True:
            available = [
                region
                for region in pool.values()
                if region["id"] not in consumed and region["id"] not in used
            ]
            following = _continuation_candidate(current, available, median_height)
            if following is None:
                break
            chain.append(following)
            used.add(following["id"])
            current = following
    return _make_unit("key_value", [label], chain), used


def _row_value_candidates(
    label: dict[str, Any],
    candidates: list[dict[str, Any]],
    median_height: float,
    page_width: float,
) -> list[dict[str, Any]]:
    label_box = _box(label)
    same_row = sorted(
        (
            region
            for region in candidates
            if _box(region) is not None
            and _box(region)[0] > label_box[0]
            and _same_row(label_box, _box(region), median_height)
        ),
        key=lambda region: _box(region)[0],
    )
    if not same_row:
        return []
    controls = [region for region in same_row if _is_control(region)]
    if controls:
        return controls
    for candidate in same_row:
        if _is_colon_label(candidate) or _looks_self_contained(candidate):
            continue
        candidate_box = _box(candidate)
        gap = candidate_box[0] - label_box[2]
        if gap > VALUE_RIGHT_MAX_WIDTH_RATIO * page_width:
            continue
        blockers = [
            other
            for other in same_row
            if other is not candidate and label_box[2] <= _box(other)[0] < candidate_box[0]
        ]
        if blockers:
            continue
        return [candidate]
    return []


def _below_value_candidate(
    label: dict[str, Any], candidates: list[dict[str, Any]], median_height: float
) -> dict[str, Any] | None:
    label_box = _box(label)
    label_height = label_box[3] - label_box[1]
    best: dict[str, Any] | None = None
    best_gap = None
    for candidate in candidates:
        if _is_colon_label(candidate):
            continue
        candidate_box = _box(candidate)
        if candidate_box is None or candidate_box[1] < label_box[3]:
            continue
        vertical_gap = candidate_box[1] - label_box[3]
        if vertical_gap > VALUE_BELOW_TOP_HEIGHTS * label_height:
            continue
        left_close = abs(candidate_box[0] - label_box[0]) <= VALUE_BELOW_LEFT_HEIGHTS * label_height
        aligned = left_close or _axis_overlap_ratio(
            label_box[0], label_box[2], candidate_box[0], candidate_box[2]
        ) >= 0.5
        if not aligned:
            continue
        # A leaf that already has its own label to its left in its row (e.g. a
        # sibling field under the same section heading) belongs to that label, not
        # to a heading-like colon label searching further down the page.
        if _has_row_label_to_left(candidate, candidates, median_height):
            continue
        if best is None or vertical_gap < best_gap:
            best, best_gap = candidate, vertical_gap
    return best


def _continuation_candidate(
    current: dict[str, Any], candidates: list[dict[str, Any]], line_height: float
) -> dict[str, Any] | None:
    box = _box(current)
    best: dict[str, Any] | None = None
    best_gap = None
    for candidate in candidates:
        if _is_colon_label(candidate):
            continue
        candidate_box = _box(candidate)
        if candidate_box is None:
            continue
        vertical_gap = candidate_box[1] - box[3]
        if vertical_gap < 0 or vertical_gap > CONTINUATION_GAP_LINE_HEIGHTS * line_height:
            continue
        if abs(candidate_box[0] - box[0]) > CONTINUATION_LEFT_LINE_HEIGHTS * line_height:
            continue
        # A leaf that is itself some other label's same-row value belongs to that
        # label's row, not to this value's continuation below.
        if _has_row_label_to_left(candidate, candidates, line_height):
            continue
        if best is None or vertical_gap < best_gap:
            best, best_gap = candidate, vertical_gap
    return best


def _has_row_label_to_left(
    candidate: dict[str, Any], candidates: list[dict[str, Any]], median_height: float
) -> bool:
    candidate_box = _box(candidate)
    return any(
        other is not candidate
        and _is_colon_label(other)
        and _box(other)[0] < candidate_box[0]
        and _same_row(_box(other), candidate_box, median_height)
        for other in candidates
    )


def _pair_grid(
    leaves: list[dict[str, Any]], median_height: float
) -> tuple[list[dict[str, Any]], set[Any]]:
    rows = _group_rows(leaves, median_height, box_of=_box)
    units: list[dict[str, Any]] = []
    consumed: set[Any] = set()
    skip_next = False
    for index in range(len(rows) - 1):
        if skip_next:
            skip_next = False
            continue
        upper, lower = rows[index], rows[index + 1]
        # Captions are small print: a row of clearly taller entries over a row that labels
        # nothing further down holds its captions below it ("Signature", "Date" under the
        # signed line). A lone heading over a label row is not such a row.
        captions_below = (
            len(upper) >= 2
            and _text_height(upper) > CAPTION_HEIGHT_RATIO * _text_height(lower)
            and not (
                index + 2 < len(rows)
                and _align_grid([leaf for leaf in lower if _is_grid_label(leaf)], rows[index + 2])
            )
        )
        labels, values = (lower, upper) if captions_below else (upper, lower)
        grid_labels = [leaf for leaf in labels if leaf["id"] not in consumed and _is_grid_label(leaf)]
        if len(grid_labels) < 2:
            continue
        upper_height = max(_box(leaf)[3] for leaf in upper) - min(_box(leaf)[1] for leaf in upper)
        gap = min(_box(leaf)[1] for leaf in lower) - max(_box(leaf)[3] for leaf in upper)
        if gap < 0 or gap > GRID_ROW_GAP_HEIGHTS * upper_height:
            continue
        pairing = _align_grid(grid_labels, [leaf for leaf in values if leaf["id"] not in consumed])
        if len(pairing) < 2:  # a row of short words over text is not a field grid
            continue
        captions = [label for label in grid_labels if label["id"] in pairing]
        if not any(_looks_entered(pairing[label["id"]], label) for label in captions):
            continue  # nothing written in (no digits, capitals or larger entries): plain text lines
        for label in grid_labels:
            value = pairing.get(label["id"])
            if value is None:
                continue  # stays a plain leaf; a colon is never invented for it
            units.append(_make_unit("key_value", [label], [value]))
            consumed.update((label["id"], value["id"]))
        skip_next = True
    return units, consumed


def _text_height(row: list[dict[str, Any]]) -> float:
    return statistics.median(_box(leaf)[3] - _box(leaf)[1] for leaf in row)


def _align_grid(
    grid_labels: list[dict[str, Any]], row_below: list[dict[str, Any]]
) -> dict[Any, dict[str, Any]]:
    # ponytail: an empty form's next label row (no answers filled in yet) can look
    # like a grid of values and get paired as if it were one; a filled-vs-blank
    # signal would be needed to tell the two apart.
    pairs: dict[Any, dict[str, Any]] = {}
    for value in row_below:
        value_box = _box(value)
        aligned = [
            label
            for label in grid_labels
            if _axis_overlap_ratio(
                _box(label)[0], _box(label)[2], value_box[0], value_box[2]
            )
            >= GRID_ALIGN_OVERLAP_RATIO
        ]
        if len(aligned) == 1:
            pairs[aligned[0]["id"]] = value
    return pairs


def _standalone_unit(leaf: dict[str, Any]) -> dict[str, Any]:
    return _make_unit(_leaf_block_kind(leaf), [], [leaf])


def _make_unit(
    kind: str, label_regions: list[dict[str, Any]], value_regions: list[dict[str, Any]]
) -> dict[str, Any]:
    members = [*label_regions, *value_regions]
    return {
        "kind": kind,
        "label_ids": [region["id"] for region in label_regions],
        "region_ids": [region["id"] for region in value_regions],
        "box": _union_box([_box(region) for region in members]),
        "reading_order": min((_reading_order(region) for region in members), default=10**9),
    }


def _unit_to_block(unit: dict[str, Any]) -> dict[str, Any]:
    left, top, right, bottom = unit["box"]
    block = {
        "kind": unit["kind"],
        "label_ids": unit["label_ids"],
        "region_ids": unit["region_ids"],
        "box": {"left": left, "top": top, "right": right, "bottom": bottom},
    }
    for key in ("borrowed_label", "drop_last_words"):
        if key in unit:
            block[key] = unit[key]
    return block


def _leaf_block_kind(leaf: dict[str, Any]) -> str:
    if _is_control(leaf):
        return "controls"
    if _is_code(leaf):
        return "code"
    kind = leaf.get("kind", "text")
    if kind == "table":
        return "table"
    if kind == "list-item":
        return "list"
    if kind in _HEADING_KINDS - {"page-header"}:  # a running header is not a section title
        return "heading"
    if kind == "figure":
        return "figure"
    if kind == "formula":
        return "formula"
    return "text"


def _is_colon_label(leaf: dict[str, Any]) -> bool:
    if _is_control(leaf) or leaf.get("kind") == "table":
        return False
    text = (leaf.get("text") or "").strip()
    if not text or "\n" in text or not text.endswith(":"):
        return False
    return len(text.split()) <= 6


def _is_grid_label(leaf: dict[str, Any]) -> bool:
    if _is_control(leaf) or leaf.get("kind") in {"table", "figure"} or leaf.get("kind") in _HEADING_KINDS:
        return False
    text = (leaf.get("text") or "").strip()
    if not text or "\n" in text or text.endswith((".", "!", "?")):
        # A label caption rarely ends a sentence; this also keeps wrapped prose
        # lines (which do) from being mistaken for form field captions.
        return False
    # A caption ending in a number already holds its value ("Federal Tax ID Number 123456789");
    # "SSN (last 4)" does not.
    return len(text.split()) <= GRID_LABEL_MAX_WORDS and not text[-1].isdigit()


def _looks_self_contained(region: dict[str, Any]) -> bool:
    first_line = (region.get("text") or "").strip().split("\n", 1)[0]
    return bool(_SELF_CONTAINED_KV_RE.match(first_line))


def _is_control(region: dict[str, Any]) -> bool:
    if region.get("kind") in _CONTROL_KINDS:
        return True
    return (region.get("structure") or {}).get("state") is not None


# --- Reading order: recursive XY-cut over units -----------------------------


def _xy_cut(units: list[dict[str, Any]], median_height: float) -> list[dict[str, Any]]:
    if len(units) <= 1:
        return list(units)
    threshold = CUT_GAP_HEIGHT_RATIO * median_height if median_height else 0.0
    horizontal_bands = _axis_bands(units, axis="y", threshold=threshold, median_height=median_height)
    if len(horizontal_bands) > 1:
        return [unit for band in horizontal_bands for unit in _xy_cut(band, median_height)]

    vertical_bands = _axis_bands(units, axis="x", threshold=threshold)
    if len(vertical_bands) > 1 and _columns_span_whole_band(units, vertical_bands):
        vertical_bands.sort(key=lambda group: min(unit["box"][0] for unit in group))
        return [unit for column in vertical_bands for unit in _xy_cut(column, median_height)]

    # Neither cut cleanly separates this scope (e.g. a form whose label/value
    # "columns" only share their first row): order by row instead of reading_order,
    # which keeps label-then-value rows together even when columns are uneven.
    return _row_major_order(units, median_height)


def _axis_bands(
    units: list[dict[str, Any]], axis: str, threshold: float, median_height: float = 0.0
) -> list[list[dict[str, Any]]]:
    start, end = (1, 3) if axis == "y" else (0, 2)
    ordered = sorted(units, key=lambda unit: unit["box"][start])
    bands: list[list[dict[str, Any]]] = [[ordered[0]]]
    band_end = _band_extension(ordered[0], start, end, axis, median_height)
    for unit in ordered[1:]:
        gap = unit["box"][start] - band_end
        if gap >= threshold:
            bands.append([unit])
        else:
            bands[-1].append(unit)
        band_end = max(band_end, _band_extension(unit, start, end, axis, median_height))
    return bands


def _band_extension(
    unit: dict[str, Any], start: int, end: int, axis: str, median_height: float
) -> float:
    """How far a unit can stretch a horizontal band before the next gap check.

    A visually tall outlier (e.g. a signature box whose border reaches well past
    its actual content) should not single-handedly bridge two otherwise-separate
    vertical sections; cap its reach the same way the "too tall to join a row"
    rule already caps leaf-level row joining.
    """
    box = unit["box"]
    extent = box[end] - box[start]
    if axis == "y" and median_height and extent > TALL_LEAF_HEIGHT_RATIO * median_height:
        return box[start] + TALL_LEAF_HEIGHT_RATIO * median_height
    return box[end]


def _columns_span_whole_band(
    units: list[dict[str, Any]], columns: list[list[dict[str, Any]]]
) -> bool:
    overall_top = min(unit["box"][1] for unit in units)
    overall_bottom = max(unit["box"][3] for unit in units)
    overall_height = overall_bottom - overall_top
    if overall_height <= 0:
        return False
    for column in columns:
        top = min(unit["box"][1] for unit in column)
        bottom = max(unit["box"][3] for unit in column)
        if (bottom - top) / overall_height < COLUMN_SPAN_COVERAGE:
            return False
    return True


def _row_major_order(units: list[dict[str, Any]], median_height: float) -> list[dict[str, Any]]:
    # Units are already-merged label+value boxes (often tall, e.g. a label next to a
    # 3-line address), so the leaf-level "too tall to join a row" rule does not apply
    # here: a short label legitimately rows with its tall value.
    rows = _group_rows(units, median_height, box_of=lambda unit: unit["box"], tall_check=False)
    return [unit for row in rows for unit in row]


def _group_rows(
    items: list[Any], median_height: float, box_of: Any, *, tall_check: bool = True
) -> list[list[Any]]:
    remaining = sorted(items, key=lambda item: box_of(item)[1])
    rows: list[list[Any]] = []
    while remaining:
        seed = remaining.pop(0)
        row = [seed]
        rest = []
        for item in remaining:
            if _same_row(box_of(seed), box_of(item), median_height, tall_check=tall_check):
                row.append(item)
            else:
                rest.append(item)
        remaining = rest
        row.sort(key=lambda item: box_of(item)[0])
        rows.append(row)
    rows.sort(key=lambda row: min(box_of(item)[1] for item in row))
    return rows


def _same_row(box_a: Box, box_b: Box, median_height: float, *, tall_check: bool = True) -> bool:
    height_a, height_b = box_a[3] - box_a[1], box_b[3] - box_b[1]
    if (
        tall_check
        and median_height
        and (
            height_a > TALL_LEAF_HEIGHT_RATIO * median_height
            or height_b > TALL_LEAF_HEIGHT_RATIO * median_height
        )
    ):
        return False
    return _axis_overlap_ratio(box_a[1], box_a[3], box_b[1], box_b[3]) >= ROW_OVERLAP_RATIO


# --- Geometry helpers --------------------------------------------------------


def _box(region: dict[str, Any]) -> Box | None:
    bounding_box = region.get("bounding_box")
    if not isinstance(bounding_box, dict):
        return None
    try:
        return (
            float(bounding_box["left"]),
            float(bounding_box["top"]),
            float(bounding_box["right"]),
            float(bounding_box["bottom"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _median_height(leaves: list[dict[str, Any]]) -> float:
    heights = [box[3] - box[1] for leaf in leaves if (box := _box(leaf)) is not None]
    return statistics.median(heights) if heights else 0.0


def _union_box(boxes: list[Box | None]) -> Box:
    present = [box for box in boxes if box is not None]
    if not present:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    )


def _axis_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    smaller = min(a_end - a_start, b_end - b_start)
    return overlap / smaller if smaller > 0 else 0.0


def _box_overlap_ratio(a: Box, b: Box) -> float:
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0
    intersection = x_overlap * y_overlap
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


def _reading_order(region: dict[str, Any]) -> int:
    value = region.get("reading_order")
    return value if isinstance(value, int) else 10**9


# --- Text rendering helpers ---------------------------------------------------


def _region_index(page: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    return {
        region["id"]: region
        for region in (page.get("regions") or [])
        if isinstance(region, dict) and region.get("id") is not None
    }


def _trimmed_by_region(arrangement: dict[str, Any]) -> dict[Any, set[int]]:
    trimmed: dict[Any, set[int]] = {}
    for entry in arrangement.get("trimmed_lines", []):
        trimmed.setdefault(entry["region_id"], set()).add(entry["line"])
    return trimmed


def _region_lines(
    region: dict[str, Any], trimmed_by_region: dict[Any, set[int]], drop_last_words: int = 0
) -> list[str]:
    """The region's shown lines: trimmed repeats left out, digit boxes joined, and the last
    `drop_last_words` words left out when they are shown as the next value's caption."""
    skip = trimmed_by_region.get(region["id"], set())
    text = region.get("text") or ""
    lines = [line for index, line in enumerate(text.split("\n")) if index not in skip and line.strip()]
    if drop_last_words and lines:
        lines[-1] = " ".join(lines[-1].split()[:-drop_last_words])
        lines = [line for line in lines if line.strip()]
    return [_DIGIT_BOXES_RE.sub(lambda match: match.group().replace(" ", ""), line) for line in lines]


def _ids_text(
    ids: list[Any], index: dict[Any, dict[str, Any]], trimmed_by_region: dict[Any, set[int]], with_ids: bool
) -> str:
    parts = []
    for region_id in ids:
        region = index.get(region_id)
        if region is None:
            continue
        text = " ".join(_region_lines(region, trimmed_by_region))
        if not text:
            continue
        parts.append(f"[{region_id}] {text}" if with_ids else text)
    return " ".join(parts)


def _value_text(
    region_ids: list[Any],
    index: dict[Any, dict[str, Any]],
    trimmed_by_region: dict[Any, set[int]],
    with_ids: bool,
    labelled: bool = False,
    drop_last_words: dict[Any, int] | None = None,
) -> str:
    lines: list[tuple[Any, str]] = []
    for region_id in region_ids:
        region = index.get(region_id)
        if region is None:
            continue
        dropped = (drop_last_words or {}).get(region_id, 0)
        lines.extend((region_id, line) for line in _region_lines(region, trimmed_by_region, dropped))
    if not lines:
        return ""
    rendered = [
        f"[{region_id}] {line}" if with_ids and (index_ == 0 or lines[index_ - 1][0] != region_id) else line
        for index_, (region_id, line) in enumerate(lines)
    ]
    # Only a labelled value's short lines are an address ("Insured: X, 3010 Highland ...");
    # commas between the lines of a heading or a sentence would change its text.
    address_like = labelled and len(lines) <= ADDRESS_LIKE_MAX_LINES and all(
        len(line.split()) <= ADDRESS_LIKE_MAX_WORDS for _, line in lines
    )
    return ", ".join(rendered) if address_like else "\n".join(rendered)


def _block_markdown(
    page: dict[str, Any],
    block: dict[str, Any],
    index: dict[Any, dict[str, Any]],
    arrangement: dict[str, Any],
) -> str:
    kind = block["kind"]
    if kind == "table":
        return _table_markdown(page, block, index, arrangement)
    if kind == "heading":
        return _heading_markdown(page, block, index, arrangement)
    if kind == "list":
        text = rendering.RAW_HTML_START.sub("&lt;", block_text(page, block, with_ids=False, arrangement=arrangement))
        # The printed marker is the list marker: "• Form 1099-DIV" is "- Form 1099-DIV",
        # and a numbered item already reads as a Markdown ordered item.
        text = _PRINTED_BULLET_RE.sub("", text)
        return text if _NUMBERED_ITEM_RE.match(text) else f"- {text}" if text else ""
    if kind == "controls":
        return _controls_markdown(block, index, arrangement)
    if kind == "formula":
        region = index.get(block["region_ids"][0]) if block["region_ids"] else None
        text = (region or {}).get("text", "").strip()
        if region is None or not text or region.get("resolution", "resolved") != "resolved":
            return "[unreadable formula]" if region is not None else ""
        math_ready = rendering._formula_math_ready(region)
        return text if not math_ready or (text.startswith("$$") and text.endswith("$$")) else f"$$\n{text}\n$$"
    if kind == "key_value":
        return _key_value_markdown(block, index, arrangement)
    if kind == "figure":
        region = index.get(block["region_ids"][0]) if block["region_ids"] else None
        return rendering._markdown_block(region, index) if region is not None else ""
    if kind == "code":
        return rendering._fenced_code(block_text(page, block, with_ids=False, arrangement=arrangement))
    return rendering.RAW_HTML_START.sub("&lt;", block_text(page, block, with_ids=False, arrangement=arrangement))


def _table_markdown(
    page: dict[str, Any],
    block: dict[str, Any],
    index: dict[Any, dict[str, Any]],
    arrangement: dict[str, Any],
) -> str:
    region = index.get(block["region_ids"][0]) if block["region_ids"] else None
    table_markdown = rendering._markdown_table(region, always_html=True, cell_text=_cell_text) if region is not None else ""
    if table_markdown:
        return table_markdown
    text = block_text(page, block, with_ids=False, arrangement=arrangement)
    return rendering.RAW_HTML_START.sub("&lt;", text)


def _heading_markdown(
    page: dict[str, Any],
    block: dict[str, Any],
    index: dict[Any, dict[str, Any]],
    arrangement: dict[str, Any],
) -> str:
    region = index.get(block["region_ids"][0]) if block["region_ids"] else None
    mark = "#" if region is not None and region.get("kind") == "title" else "##"
    # A Markdown heading is one line; the OCR line breaks inside a title are layout.
    text = " ".join(block_text(page, block, with_ids=False, arrangement=arrangement).split("\n"))
    return f"{mark} {rendering.RAW_HTML_START.sub('&lt;', text)}" if text else ""


# A boxed form cell whose caption OCR glued to its value: "Tax ID (TIN)86-4419027".
_GLUED_CAPTION_RE = re.compile(r"^(?P<label>\S+(?:\s+\S+)+?(?:\)|[a-z]{4}))(?P<value>[A-Z0-9].*)$", re.S)


def _cell_text(text: str) -> str:
    """Display text for a table cell: a glued caption gets its own line, digit boxes join."""
    match = _GLUED_CAPTION_RE.match(text)
    if match and len(match["label"].split()) <= 8:
        text = f"{match['label']}\n{match['value']}"
    return _DIGIT_BOXES_RE.sub(lambda found: found.group().replace(" ", ""), text)


def _key_value_markdown(
    block: dict[str, Any], index: dict[Any, dict[str, Any]], arrangement: dict[str, Any]
) -> str:
    label, value = _label_and_value(block, index, arrangement, with_ids=False)
    if not label or not value:
        return rendering.RAW_HTML_START.sub("&lt;", label or value)
    label = rendering.RAW_HTML_START.sub("&lt;", label)
    value = rendering.RAW_HTML_START.sub("&lt;", value)
    return f"**{label}** {value}".strip()


def _controls_markdown(
    block: dict[str, Any], index: dict[Any, dict[str, Any]], arrangement: dict[str, Any]
) -> str:
    trimmed = _trimmed_by_region(arrangement)
    lines = []
    if block["label_ids"]:
        label = _ids_text(block["label_ids"], index, trimmed, False)
        if label:
            if not label.endswith(":"):
                label = f"{label}:"
            lines.append(f"**{rendering.RAW_HTML_START.sub('&lt;', label)}**")
    for region_id in block["region_ids"]:
        region = index.get(region_id)
        if region is not None:
            line = _control_markdown_line(region)
            if line:
                lines.append(line)
    return "\n".join(lines)


def _control_markdown_line(region: dict[str, Any]) -> str:
    option = _control_option(region, with_ids=False)
    return f"- {rendering.RAW_HTML_START.sub('&lt;', option)}" if option else ""


def _control_option(region: dict[str, Any], with_ids: bool) -> str:
    """A checkbox as its mark and label: "[x] C Corporation"; "[?]" when the state is unknown."""
    structure = region.get("structure") or {}
    label = str(structure.get("label") or region.get("text") or "").strip().split("\n")[0]
    if not label:
        return ""
    mark = {"selected": "x", "unselected": " "}.get(structure.get("state"), "?")
    return f"[{region['id']}] [{mark}] {label}" if with_ids else f"[{mark}] {label}"
