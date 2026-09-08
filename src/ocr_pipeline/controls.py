"""Evidence-linked checkbox detection without a learned model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "opencv-geometric-controls"
MODEL = {
    "id": "opencv-square-contours-v1",
    "origin": "Open Source Vision Foundation",
    "license": "Apache-2.0",
}
MARK_MODEL = {
    "id": "opencv-label-anchored-residual-v1",
    "origin": "Open Source Vision Foundation",
    "license": "Apache-2.0",
}
MATH_TYPES = frozenset({"equation", "formula", "math"})
# What a mark looks like, kept separate from what it means. A tick or cross beside a label
# selects an option; a slashed loop or a ring is an annotation whose meaning the form's
# conventions decide. U+2205 is what frontier readers emit for the null glyph.
MARK_GLYPHS = {
    "tick": "✓",
    "cross": "✗",
    "slashed_loop": "∅",
    "ring": "◯",
}
# A mark inside a printed box is a selection; the glyph still matters, because a reviewer
# checking a disputed field wants to see whether it was ticked or crossed.
BOXED_GLYPHS = {
    "tick": "☑",
    "cross": "☒",
    "empty": "☐",
    "unknown": "?",
}
NULL_MARK_GLYPH = MARK_GLYPHS["slashed_loop"]
# Shapes that annotate an answer area rather than select a listed option.
ANNOTATION_SHAPES = frozenset({"slashed_loop", "ring"})
# A hand-drawn ring encircles words, so it is wider than it is tall.
MINIMUM_RING_ASPECT = 1.6


@dataclass(frozen=True)
class ControlDetection:
    bounding_box: BoundingBox
    state: str
    confidence: float
    ink_ratio: float
    source: str = "square"
    shape: str = "unknown"
    label: str | None = None
    label_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    reading_order: int | None = None


class GeometricControlStage:
    """Add structured checkbox regions and flag uncertain controls for review."""

    name = "controls"

    def __init__(
        self,
        *,
        minimum_group_size: int = 6,
        label_provider: str | Sequence[str] | None = None,
    ) -> None:
        if minimum_group_size < 1:
            raise ValueError("minimum_group_size must be positive")
        self.minimum_group_size = minimum_group_size
        self.label_provider = label_provider
        self.label_providers = _normalized_providers(label_provider)

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        cv2, np, gray = _load_gray(image_path)
        detection_group_size = (
            1
            if _form_like_regions(regions, self.label_providers)
            else self.minimum_group_size
        )
        if _reader_transcribes_marks(regions):
            # The reader can transcribe a printed ballot glyph as text, but never a
            # hand-drawn ring or null glyph, so only those channels run here. The
            # square, anchored tick/cross and table channels would either duplicate
            # what the reader already wrote or need the square channel this path skips.
            null_marks = _anchored_marks(
                gray,
                regions,
                self.label_providers,
                [],
                cv2,
                np,
                null_only=True,
            )
            detections = _merge_detections(
                null_marks,
                _ring_marks(gray, regions, self.label_providers, null_marks, cv2, np),
            )
        else:
            detections = _square_detections(gray, cv2, np)
            detections = [
                detection
                for detection in detections
                if not _contains_readable_text(
                    detection.bounding_box,
                    regions,
                    self.label_providers,
                )
                and not _inside_math_region(detection.bounding_box, regions)
            ]
            anchored_marks = _anchored_marks(
                gray,
                regions,
                self.label_providers,
                detections,
                cv2,
                np,
            )
            anchored_marks = _supported_anchored_groups(
                anchored_marks,
                detection_group_size,
            )
            detections = _merge_detections(detections, anchored_marks)
            detections = _merge_detections(
                detections,
                _ring_marks(gray, regions, self.label_providers, detections, cv2, np),
            )
            table_marks = _table_marks(gray, regions, detections, cv2, np)
            detections = [
                detection
                for detection in detections
                if not _overlaps_detection(detection.bounding_box, table_marks)
            ]
            detections = _merge_detections(detections, table_marks)
        detections = [
            detection
            for detection in detections
            if detection.source in {"table_mark", "ring_mark"}
            or not _inside_text_region(
                detection.bounding_box,
                regions,
                self.label_providers,
            )
        ]
        labels = [
            _anchor_label(detection, regions)
            or detection.label
            or _nearest_label(detection.bounding_box, regions, self.label_providers)
            for detection in detections
        ]
        geometric_count = sum(
            detection.source != "table_mark" for detection in detections
        )
        if geometric_count < detection_group_size:
            option_row = _aligned_option_row(detections)
            supported = [
                (detection, label)
                for index, (detection, label) in enumerate(
                    zip(detections, labels, strict=True)
                )
                if detection.source != "square"
                or index in option_row
                or label is None
                or _explicit_control_label(label)
                or (
                    detection.state != "unselected"
                    and (
                        _descriptive_control_label(label)
                        or _supported_short_option_label(
                            detection,
                            label,
                            detections,
                            labels,
                        )
                    )
                )
            ]
            detections = [detection for detection, _ in supported]
            labels = [label for _, label in supported]
        if any(label is not None for label in labels):
            detections = [
                detection
                for detection, label in zip(detections, labels, strict=True)
                if label is not None
                or _supported_unmatched_detection(
                    detection,
                    detections,
                    labels,
                    gray.shape,
                )
            ]
        controls = [
            _control_region(
                detection,
                page_number,
                index,
                regions,
                self.label_providers,
                detections,
                gray.shape,
            )
            for index, detection in enumerate(detections, start=1)
        ]
        geometric_controls = [
            control
            for control in controls
            if control.text_provenance["method"] != "table_cell_residual_ink"
        ]
        coverage_missing = (
            bool(geometric_controls)
            and len(geometric_controls) < self.minimum_group_size
        )
        for control in controls:
            if control.structure["label"] is None:
                control.structure["coverage_status"] = "unmatched_label"
            else:
                control.structure["coverage_status"] = (
                    "insufficient_control_group" if coverage_missing else "detected"
                )
        _dedupe_shared_labels(controls)
        return sorted(
            regions + controls,
            key=lambda region: (
                region.reading_order,
                0 if region.kind == "checkbox" else 1,
                region.id,
            ),
        )


def _dedupe_shared_labels(controls: list[TextRegion]) -> None:
    """Print a label region shared by several controls once, not once per control.

    A reader that returns a whole option row as one region ("Urgent [ ] For Review
    [ ] Please Reply") makes every square on that row resolve to the same nearest
    label, and the row's text then renders once per square. The first control keeps
    the text; the rest keep their mark and their evidence link.
    """
    seen: dict[tuple[str, tuple[str, ...]], str] = {}
    for control in controls:
        label = control.structure.get("label")
        evidence_ids = tuple(control.structure.get("label_evidence_ids") or ())
        if not label or not evidence_ids:
            continue
        key = (label, evidence_ids)
        first = seen.setdefault(key, control.id)
        if first == control.id:
            continue
        symbol = control.text[: len(control.text) - len(label)].strip()
        control.text = symbol
        control.structure["label"] = None
        control.structure["association_status"] = "shared_label"
        control.structure["label_shared_with"] = first


# Unicode ballot glyphs a reader emits when it transcribes the box itself.
TRANSCRIBED_MARKS = ("\u2610", "\u2611", "\u2612", "\u2713", "\u2717")


def _reader_transcribes_marks(regions: list[TextRegion]) -> bool:
    """True when the page text already carries checkbox glyphs from the reader."""
    return any(
        symbol in region.text
        for region in regions
        if region.kind != "checkbox"
        for symbol in TRANSCRIBED_MARKS
    )


def _normalized_providers(
    label_provider: str | Sequence[str] | None,
) -> frozenset[str] | None:
    """None keeps every provider. A single provider still normalizes, so every call
    site can test membership instead of juggling both a scalar and a collection."""
    if label_provider is None:
        return None
    if isinstance(label_provider, str):
        return frozenset({label_provider})
    return frozenset(label_provider)


def detect_controls(image_path: Path) -> list[ControlDetection]:
    cv2, np, gray = _load_gray(image_path)
    return _square_detections(gray, cv2, np)


def _form_like_regions(
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
) -> bool:
    # A field label ends with its colon. Counting any word that merely contains one made
    # times, ratios and section references look like form fields, so a dense printed
    # contract was declared form-like and lone spurious marks lost their corroboration
    # requirement. This matches _field_anchor in anchored_ink.py.
    labels = {
        region.id
        for region in regions
        if region.resolution == "resolved"
        and region.text.rstrip().endswith((":", "："))
        and (label_providers is None or region.provider in label_providers)
    }
    return len(labels) >= 6


def _load_gray(image_path: Path) -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise ReaderError(
            "control_dependency_unavailable",
            "Checkbox detection requires OpenCV and NumPy",
        ) from error

    try:
        with Image.open(image_path) as source:
            gray = np.asarray(source.convert("L"))
    except (OSError, UnidentifiedImageError) as error:
        raise ReaderError("control_image_failed", str(error)) from error

    return cv2, np, gray


def _square_detections(gray: Any, cv2: Any, np: Any) -> list[ControlDetection]:

    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    faint = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1]
    boxes = _square_boxes(otsu, cv2) + _square_boxes(faint, cv2)
    boxes = _remove_input_grids(_deduplicate(boxes))

    detections = [
        _classify(gray, box, cv2, np)
        for box in sorted(boxes, key=lambda item: (item.top, item.left))
    ]
    minimum_reliable_side = round(min(gray.shape) * 0.01)
    return [
        ControlDetection(
            detection.bounding_box,
            "ambiguous",
            min(0.5, detection.confidence),
            detection.ink_ratio,
            detection.source,
            detection.shape,
        )
        if _side(detection.bounding_box) < minimum_reliable_side
        else detection
        for detection in detections
    ]


def _anchored_marks(
    gray: Any,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
    existing: list[ControlDetection],
    cv2: Any,
    np: Any,
    *,
    null_only: bool = False,
) -> list[ControlDetection]:
    height, width = gray.shape
    detections = []
    for region in regions:
        if region.kind in {"checkbox", "table", "coverage_risk", "page_text"}:
            continue
        if _math_region(region):
            continue
        if label_providers is not None and region.provider not in label_providers:
            continue
        if not _readable_label(region.text):
            continue
        confident = region.confidence is not None and region.confidence >= 0.9
        # A null glyph declares a field intentionally empty, so only a field label
        # can anchor one. A layout reader scores printed labels by detection, not
        # reading quality, so the colon stands in for confidence in null-only mode:
        # the glyph's own shape and emptiness tests carry the evidence.
        field_label = region.text.rstrip().endswith((":", "："))
        if null_only:
            if not field_label:
                continue
        elif not confident:
            continue
        if (region.structure or {}).get("role") == "table_source" and sum(
            character.isalpha() for character in region.text
        ) < 2:
            continue

        box = region.bounding_box
        line_height = max(1, box.bottom - box.top)
        vertical_pad = max(2, line_height // 2)
        top = max(0, box.top - vertical_pad)
        bottom = min(height, box.bottom + vertical_pad)
        slot_width = round(line_height * 1.75)
        # A layout reader's box for "Label:" often runs past the printed colon and
        # covers the writing space, so the value slot reaches back into the box's
        # tail. A null glyph found there is trusted; a tick there would be a letter.
        value_slot_left = (
            max(box.left, box.right - round(line_height * 2))
            if field_label
            else box.right
        )
        # A handwritten null glyph sits further from its label than a checkbox tick,
        # so the value window extends further but accepts that glyph only.
        slots = [
            (BoundingBox(max(0, box.left - slot_width), top, box.left, bottom), None),
            (
                BoundingBox(
                    value_slot_left,
                    top,
                    min(width, box.right + round(line_height * 4)),
                    bottom,
                ),
                box.right + slot_width,
            ),
        ]
        region_detections = []
        for slot, tick_limit in slots:
            for detection in _marks_in_slot(
                gray, slot, line_height, cv2, np, tick_limit=tick_limit
            ):
                null_mark = detection.source == "null_mark"
                if null_only and not null_mark:
                    continue
                if null_mark and not field_label:
                    continue
                if (
                    not null_mark
                    and value_slot_left <= detection.bounding_box.left < box.right
                ):
                    continue
                if _overlaps_detection(detection.bounding_box, existing + detections):
                    continue
                others = [
                    other
                    for other in regions
                    if other is not region
                    and not _transcribed_mark_region(other, detection.bounding_box)
                ]
                if _center_inside_reader_text(
                    detection.bounding_box,
                    others,
                    label_providers,
                ):
                    continue
                if _inside_text_region(
                    detection.bounding_box,
                    others,
                    label_providers,
                ):
                    continue
                region_detections.append(detection)
        if region_detections:
            nearest = min(
                region_detections,
                key=lambda detection: (
                    _horizontal_gap(detection.bounding_box, box),
                    abs(_center(detection.bounding_box)[1] - _center(box)[1]),
                ),
            )
            detections.append(replace(nearest, label_ids=(region.id,)))
    return detections


def _transcribed_mark_region(region: TextRegion, mark: BoundingBox) -> bool:
    """True when the reader region is its own misreading of the mark.

    The fused reader's under-read repair writes a handwritten glyph back as a stray
    character (the null glyph comes back as "8"). Only that repair channel is
    exempted: a single-character region from the main read is real page text, and a
    printed 8 or e passes the null shape test, so exempting it would re-detect it.
    Mutual centre containment says the region is the mark, not text the mark sits in.
    """
    if (region.text_provenance or {}).get("method") != "region_underread_repair":
        return False
    if len(region.text.strip()) != 1:
        return False
    region_x, region_y = _center(region.bounding_box)
    mark_x, mark_y = _center(mark)
    box = region.bounding_box
    return (
        mark.left <= region_x <= mark.right
        and mark.top <= region_y <= mark.bottom
        and box.left <= mark_x <= box.right
        and box.top <= mark_y <= box.bottom
    )


def _ring_marks(
    gray: Any,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
    existing: list[ControlDetection],
    cv2: Any,
    np: Any,
) -> list[ControlDetection]:
    """Detect a hand-drawn ring annotation enclosing printed text."""
    mask = np.where(gray < 180, 255, 0).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    detections = []
    for index in range(1, count):
        left, top, box_width, box_height, area = map(int, stats[index])
        if box_width < 24 or box_height < 8 or area < 60:
            continue
        if area / max(1, box_width * box_height) > 0.45:
            continue
        component = np.where(
            labels[top : top + box_height, left : left + box_width] == index,
            255,
            0,
        ).astype("uint8")
        if box_width < box_height * MINIMUM_RING_ASPECT:
            # An annotation is drawn around words, which are wider than they are tall.
            # A capital O or D at scan resolution otherwise satisfies every ring test.
            continue
        hole = _largest_hole_area(component, cv2)
        if hole <= area or not 0.45 <= hole / (box_width * box_height) <= 0.85:
            # An ellipse fills about 0.785 of its box. A ruled rectangle fills nearly
            # all of it, and a multi-cell grid frame far less.
            continue
        box = BoundingBox(left, top, left + box_width, top + box_height)
        if _overlaps_detection(box, existing + detections):
            continue
        enclosed = _enclosed_regions(box, regions, label_providers)
        if not enclosed:
            continue
        detections.append(
            ControlDetection(
                box,
                "selected",
                0.6,
                area / max(1, box_width * box_height),
                "ring_mark",
                # A ring circles a value, not its whole row. When the enclosed region
                # carries word geometry, the label is the words under the ring; the
                # full-row fallback is what rendered one row's text once per ring.
                label=_ring_word_label(box, enclosed),
                label_ids=tuple(region.id for region in enclosed),
            )
        )
    return detections


def _ring_word_label(ring: BoundingBox, enclosed: list[TextRegion]) -> str | None:
    """The words whose boxes sit inside the ring, in evidence order.

    A parent region and its repair child can carry the same word entry, so each
    word box is counted once.
    """
    words = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for region in enclosed:
        for entry in (region.structure or {}).get("word_evidence") or []:
            if not isinstance(entry, dict):
                continue
            box = entry.get("bbox")
            if not isinstance(box, dict):
                continue
            try:
                centre_x = (box["left"] + box["right"]) / 2
                centre_y = (box["top"] + box["bottom"]) / 2
            except (KeyError, TypeError):
                continue
            if (
                ring.left <= centre_x <= ring.right
                and ring.top <= centre_y <= ring.bottom
            ):
                text = str(entry.get("text") or "").strip()
                key = (box["left"], box["top"], box["right"], box["bottom"], text)
                if text and key not in seen:
                    seen.add(key)
                    words.append(text)
    return " ".join(words) or None


def _largest_hole_area(component: Any, cv2: Any) -> float:
    contours, hierarchy = cv2.findContours(
        component.copy(),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return 0.0
    return max(
        (
            cv2.contourArea(contours[index])
            for index, (*_, parent) in enumerate(hierarchy[0])
            if parent >= 0
        ),
        default=0.0,
    )


def _enclosed_regions(
    ring: BoundingBox,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
) -> list[TextRegion]:
    enclosed = []
    for region in regions:
        if region.kind in {"checkbox", "coverage_risk", "page_text"}:
            continue
        if label_providers is not None and region.provider not in label_providers:
            continue
        if region.resolution != "resolved" or not region.text.strip():
            continue
        center_x, center_y = _center(region.bounding_box)
        if (
            region.kind != "table"
            and ring.left <= center_x <= ring.right
            and ring.top <= center_y <= ring.bottom
        ):
            enclosed.append(region)
            continue
        # A table region can anchor a ring drawn on one of its rows (a form row the
        # reader typed as a table), but never by the centre test: a page-sized table
        # contains every ring's centre.
        # A ring drawn over one option of a long printed row overlaps only part of it.
        overlap = min(ring.right, region.bounding_box.right) - max(
            ring.left, region.bounding_box.left
        )
        vertical = min(ring.bottom, region.bounding_box.bottom) - max(
            ring.top, region.bounding_box.top
        )
        region_width = region.bounding_box.right - region.bounding_box.left
        if (
            vertical > 0
            and overlap >= min((ring.right - ring.left), region_width) * 0.6
        ):
            enclosed.append(region)
    return enclosed


def _aligned_option_row(
    detections: list[ControlDetection],
    minimum_members: int = 3,
) -> set[int]:
    """Indices of squares that sit in a row of evenly sized peers.

    Three or more same-sized boxes sharing a baseline are an option row by construction,
    which is the corroboration the page-wide group count is looking for. Without this a
    fax cover's untouched Urgent/For Review/Please Reply boxes are dropped outright, and
    the output cannot say the sender ticked nothing.
    """
    squares = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection.source == "square"
    ]
    members: set[int] = set()
    for index, detection in squares:
        row = [
            other_index
            for other_index, other in squares
            if _marks_share_axis(detection.bounding_box, other.bounding_box)
            and 0.75
            <= _side(other.bounding_box) / max(1, _side(detection.bounding_box))
            <= 1.33
        ]
        if len(row) >= minimum_members:
            members.update(row)
    return members


def _supported_anchored_groups(
    detections: list[ControlDetection],
    minimum_group_size: int,
) -> list[ControlDetection]:
    if minimum_group_size == 1:
        return detections

    remaining = set(range(len(detections)))
    supported = []
    while remaining:
        pending = [remaining.pop()]
        group = []
        while pending:
            index = pending.pop()
            group.append(index)
            linked = {
                candidate
                for candidate in remaining
                if any(
                    _marks_share_axis(
                        detections[candidate].bounding_box,
                        detections[member].bounding_box,
                    )
                    for member in group
                )
            }
            remaining.difference_update(linked)
            pending.extend(linked)
        if len(group) >= minimum_group_size:
            supported.extend(detections[index] for index in sorted(group))
    return supported


def _marks_share_axis(first: BoundingBox, second: BoundingBox) -> bool:
    first_side = _side(first)
    second_side = _side(second)
    if min(first_side, second_side) <= 0:
        return False
    if not 0.6 <= first_side / second_side <= 1.67:
        return False
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    tolerance = max(first_side, second_side)
    return abs(first_x - second_x) <= tolerance or abs(first_y - second_y) <= tolerance


def _horizontal_gap(first: BoundingBox, second: BoundingBox) -> int:
    if first.right <= second.left:
        return second.left - first.right
    if second.right <= first.left:
        return first.left - second.right
    return 0


def _marks_in_slot(
    gray: Any,
    slot: BoundingBox,
    line_height: int,
    cv2: Any,
    np: Any,
    *,
    tick_limit: int | None = None,
) -> list[ControlDetection]:
    if slot.right - slot.left < 4 or slot.bottom - slot.top < 4:
        return []

    crop = gray[slot.top : slot.bottom, slot.left : slot.right]
    mask = np.where(crop < 180, 255, 0).astype("uint8")
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(8, round(mask.shape[1] * 0.5)), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(8, round(mask.shape[0] * 0.5))),
        ),
    )
    residual = cv2.bitwise_and(
        mask,
        cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual)
    # A null glyph asserts an otherwise empty answer area, so it must be the only
    # significant ink in its slot. A letter passes the same shape test (a g or an 8
    # is a loop with a stroke), but its sibling letters are significant ink too.
    significant = sum(
        int(stats[index][4]) >= 8
        and min(int(stats[index][2]), int(stats[index][3])) >= line_height * 0.38
        for index in range(1, count)
    )
    detections = []
    for index in range(1, count):
        left, top, box_width, box_height, area = map(int, stats[index])
        if area < 8 or min(box_width, box_height) < 4:
            continue
        if min(box_width, box_height) < line_height * 0.38:
            continue
        if max(box_width, box_height) > line_height * 2:
            continue
        component = np.where(
            labels[top : top + box_height, left : left + box_width] == index,
            255,
            0,
        ).astype("uint8")
        beyond_tick_reach = tick_limit is not None and slot.left + left >= tick_limit
        shape = _mark_shape(component, cv2, np)
        null_mark = shape == "slashed_loop"
        if null_mark and significant > 1:
            continue
        if not null_mark and (beyond_tick_reach or shape == "unknown"):
            continue
        box = BoundingBox(
            slot.left + left,
            slot.top + top,
            slot.left + left + box_width,
            slot.top + top + box_height,
        )
        ink_ratio = area / max(1, box_width * box_height)
        detections.append(
            ControlDetection(
                box,
                "selected",
                min(0.9, 0.6 + ink_ratio * 0.5),
                ink_ratio,
                # A slashed loop is an annotation written in the answer area, not a tick in
                # a box, so it is carried through as its own kind rather than a selection.
                "null_mark" if null_mark else "anchored_mark",
                shape,
            )
        )
    return detections


def _table_marks(
    gray: Any,
    regions: list[TextRegion],
    existing: list[ControlDetection],
    cv2: Any,
    np: Any,
) -> list[ControlDetection]:
    detections = _candidate_table_marks(gray, regions, cv2, np)
    for table in (region for region in regions if region.kind == "table"):
        cells = (table.structure or {}).get("cells", [])
        indexed = _indexed_cells(cells)
        for (row, _), cell in indexed.items():
            label = _same_cell_label(cell)
            if label is None:
                continue
            cell_box = _cell_box(cell)
            if cell_box is None:
                continue
            mark = _same_cell_mark(gray, cell_box, cv2, np)
            if mark is None or _overlaps_detection(
                mark.bounding_box,
                detections,
            ):
                continue
            cell_id = cell.get("id")
            detections.append(
                ControlDetection(
                    bounding_box=mark.bounding_box,
                    state=mark.state,
                    confidence=mark.confidence,
                    ink_ratio=mark.ink_ratio,
                    source="table_mark",
                    label=label,
                    label_ids=(cell_id,) if cell_id else (),
                    source_ids=tuple(item for item in (table.id, cell_id) if item),
                    reading_order=table.reading_order + row,
                )
            )
        header_rows = _header_rows(indexed)
        for header_index, header_row in enumerate(header_rows):
            next_header = (
                header_rows[header_index + 1]
                if header_index + 1 < len(header_rows)
                else 10**9
            )
            headers = {
                column: cell
                for (row, column), cell in indexed.items()
                if row == header_row and _header_text(cell.get("text", ""))
            }
            control_columns = _control_columns(
                indexed,
                headers,
                header_row,
                next_header,
            )
            for (row, column), cell in indexed.items():
                if not header_row < row < next_header or column not in control_columns:
                    continue
                row_label = _row_label(indexed, row, column)
                if row_label is None or not _mark_cell_text(cell):
                    continue
                cell_box = _cell_box(cell)
                if cell_box is None:
                    continue
                mark = _mark_in_cell(gray, cell_box, cv2, np)
                if mark is None or _overlaps_detection(
                    mark.bounding_box, existing + detections
                ):
                    continue
                header = headers[column]
                label = f"{row_label['text'].strip()} ({header['text'].strip()})"
                label_ids = tuple(
                    item
                    for item in (
                        row_label.get("id"),
                        header.get("id"),
                    )
                    if item
                )
                source_ids = tuple(item for item in (table.id, cell.get("id")) if item)
                detections.append(
                    ControlDetection(
                        bounding_box=mark.bounding_box,
                        state="selected",
                        confidence=mark.confidence,
                        ink_ratio=mark.ink_ratio,
                        source="table_mark",
                        label=label,
                        label_ids=label_ids,
                        source_ids=source_ids,
                        reading_order=table.reading_order + row,
                    )
                )
    return detections


def _candidate_table_marks(
    gray: Any,
    regions: list[TextRegion],
    cv2: Any,
    np: Any,
) -> list[ControlDetection]:
    detections = []
    for candidate in (region for region in regions if region.kind == "table_candidate"):
        if (candidate.structure or {}).get("status") != "rejected":
            continue
        labels = _candidate_control_labels(candidate, regions)
        if len(labels) < 3:
            continue
        expected_side = round(
            float(
                np.median(
                    [
                        label.bounding_box.bottom - label.bounding_box.top
                        for label in labels
                    ]
                )
            )
        )
        marks = _candidate_scope_marks(
            gray,
            candidate.bounding_box,
            expected_side,
            cv2,
            np,
        )
        matches = _unique_candidate_matches(marks, labels)
        for mark, label in _clustered_candidate_matches(matches):
            detections.append(
                ControlDetection(
                    bounding_box=mark.bounding_box,
                    state=mark.state,
                    confidence=mark.confidence,
                    ink_ratio=mark.ink_ratio,
                    source="table_mark",
                    label=label.text.strip(),
                    label_ids=(label.id,),
                    source_ids=(candidate.id,),
                    reading_order=label.reading_order,
                )
            )
    return detections


def _candidate_control_labels(
    candidate: TextRegion,
    regions: list[TextRegion],
) -> list[TextRegion]:
    source_ids = set((candidate.text_provenance or {}).get("source_region_ids", []))
    labels = []
    for region in regions:
        if source_ids and region.id not in source_ids:
            continue
        if region.kind in {"checkbox", "table", "table_candidate", "coverage_risk"}:
            continue
        if (region.structure or {}).get("role") in {"control", "table_source"}:
            continue
        if region.resolution != "resolved":
            continue
        if region.confidence is None or region.confidence < 0.8:
            continue
        text = " ".join(region.text.split())
        if sum(character.isalpha() for character in text) < 4:
            continue
        if text.endswith(":") or text.isupper():
            continue
        center_x, center_y = _center(region.bounding_box)
        box = candidate.bounding_box
        if (
            not box.left <= center_x <= box.right
            or not box.top <= center_y <= box.bottom
        ):
            continue
        labels.append(region)

    normalized_counts: dict[str, int] = {}
    for label in labels:
        normalized = " ".join(label.text.casefold().split())
        normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
    return [
        label
        for label in labels
        if normalized_counts[" ".join(label.text.casefold().split())] == 1
    ]


def _candidate_scope_marks(
    gray: Any,
    scope: BoundingBox,
    expected_side: int,
    cv2: Any,
    np: Any,
) -> list[ControlDetection]:
    left = max(0, scope.left)
    top = max(0, scope.top)
    right = min(gray.shape[1], scope.right)
    bottom = min(gray.shape[0], scope.bottom)
    crop = gray[top:bottom, left:right].copy()
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return []

    mask = np.where(crop < 180, 255, 0).astype("uint8")
    rule_length = max(12, round(expected_side * 1.7))
    residual = _remove_long_rules(
        mask,
        max(rule_length, round(mask.shape[1] * 0.12)),
        rule_length,
        cv2,
    )
    contours = cv2.findContours(
        residual.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    minimum_side = max(4, round(expected_side * 0.35))
    maximum_side = max(minimum_side, round(expected_side * 1.35))
    boxes = []
    for contour in contours:
        box_left, box_top, box_width, box_height = cv2.boundingRect(contour)
        short_side = min(box_width, box_height)
        long_side = max(box_width, box_height)
        if short_side < minimum_side or long_side > maximum_side:
            continue
        if not 0.65 <= box_width / box_height <= 1.5:
            continue
        component = residual[
            box_top : box_top + box_height,
            box_left : box_left + box_width,
        ]
        if not _has_candidate_control_shape(component, cv2, np):
            continue
        boxes.append(
            BoundingBox(
                left + box_left,
                top + box_top,
                left + box_left + box_width,
                top + box_top + box_height,
            )
        )
    return [_classify(gray, box, cv2, np) for box in _deduplicate(boxes)]


def _has_candidate_control_shape(component: Any, cv2: Any, np: Any) -> bool:
    height, width = component.shape
    band = max(1, round(min(width, height) * 0.16))
    top = float(np.max(np.mean(component[:band] > 0, axis=1))) >= 0.58
    bottom = float(np.max(np.mean(component[-band:] > 0, axis=1))) >= 0.58
    left = float(np.max(np.mean(component[:, :band] > 0, axis=0))) >= 0.58
    right = float(np.max(np.mean(component[:, -band:] > 0, axis=0))) >= 0.58
    partial_square = (top or bottom) and (left or right)
    return partial_square or _has_mark_shape(component, cv2, np)


def _unique_candidate_matches(
    marks: list[ControlDetection],
    labels: list[TextRegion],
) -> list[tuple[ControlDetection, TextRegion]]:
    matches = []
    for mark in marks:
        possible = [
            label
            for label in labels
            if _candidate_label_matches(mark.bounding_box, label.bounding_box)
        ]
        if len(possible) == 1:
            matches.append((mark, possible[0]))

    label_counts: dict[str, int] = {}
    for _, label in matches:
        label_counts[label.id] = label_counts.get(label.id, 0) + 1
    return [(mark, label) for mark, label in matches if label_counts[label.id] == 1]


def _candidate_label_matches(mark: BoundingBox, label: BoundingBox) -> bool:
    side = max(mark.right - mark.left, mark.bottom - mark.top)
    label_height = max(1, label.bottom - label.top)
    mark_x, mark_y = _center(mark)
    _, label_y = _center(label)
    if not label_height * 0.35 <= side <= label_height * 1.25:
        return False
    if mark.left >= label.left or mark_x > label.left:
        return False
    gap = label.left - mark.right
    if not -side * 0.5 <= gap <= max(5, label_height * 0.8):
        return False
    return abs(mark_y - label_y) <= max(side, label_height) * 0.55


def _clustered_candidate_matches(
    matches: list[tuple[ControlDetection, TextRegion]],
) -> list[tuple[ControlDetection, TextRegion]]:
    clusters: list[list[tuple[ControlDetection, TextRegion]]] = []
    for match in sorted(
        matches,
        key=lambda item: (
            _center(item[0].bounding_box)[1],
            item[0].bounding_box.left,
        ),
    ):
        if not clusters:
            clusters.append([match])
            continue
        previous = clusters[-1][-1]
        previous_y = _center(previous[0].bounding_box)[1]
        current_y = _center(match[0].bounding_box)[1]
        previous_height = previous[1].bounding_box.bottom - previous[1].bounding_box.top
        current_height = match[1].bounding_box.bottom - match[1].bounding_box.top
        if current_y - previous_y <= max(previous_height, current_height) * 2.25:
            clusters[-1].append(match)
        else:
            clusters.append([match])
    return [match for cluster in clusters if len(cluster) >= 3 for match in cluster]


def _remove_long_rules(
    mask: Any,
    horizontal_length: int,
    vertical_length: int,
    cv2: Any,
) -> Any:
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1)),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    return cv2.bitwise_and(
        mask,
        cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)),
    )


def _same_cell_label(cell: dict[str, Any]) -> str | None:
    if cell.get("resolution") not in {None, "resolved"}:
        return None
    if cell.get("column_header") or cell.get("projected_row_header"):
        return None

    text = " ".join(str(cell.get("text", "")).split())
    if not text.startswith("["):
        return None
    marker_end = text.find("]", 1, 4)
    if marker_end < 0 or text[1:marker_end].strip().casefold() not in {"", "x"}:
        return None
    label = text[marker_end + 1 :].strip(" :-")
    if sum(character.isalpha() for character in label) < 2:
        return None
    return label


def _same_cell_mark(
    gray: Any,
    cell: BoundingBox,
    cv2: Any,
    np: Any,
) -> ControlDetection | None:
    left = max(0, cell.left)
    top = max(0, cell.top)
    right = min(gray.shape[1], cell.right)
    bottom = min(gray.shape[0], cell.bottom)
    crop = gray[top:bottom, left:right].copy()
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    mask = np.where(crop < 180, 255, 0).astype("uint8")
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(10, round(mask.shape[1] * 0.45)), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(10, round(mask.shape[0] * 0.65))),
        ),
    )
    residual = cv2.bitwise_and(
        mask,
        cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)),
    )
    contours = cv2.findContours(
        residual.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    minimum_side = max(6, round(mask.shape[0] * 0.1))
    maximum_side = min(
        max(minimum_side, round(mask.shape[0] * 0.55)),
        max(minimum_side, round(mask.shape[1] * 0.28)),
        48,
    )
    candidates = []
    for contour in contours:
        box_left, box_top, box_width, box_height = cv2.boundingRect(contour)
        short_side = min(box_width, box_height)
        long_side = max(box_width, box_height)
        if short_side < minimum_side or long_side > maximum_side:
            continue
        if not 0.72 <= box_width / box_height <= 1.35:
            continue
        if box_left + box_width / 2 > mask.shape[1] * 0.4:
            continue
        component = residual[
            box_top : box_top + box_height,
            box_left : box_left + box_width,
        ]
        if not _has_square_border(component, np):
            continue
        candidates.append((box_left, box_top, box_width, box_height))
    if len(candidates) != 1:
        return None

    box_left, box_top, box_width, box_height = candidates[0]
    box = BoundingBox(
        left + box_left,
        top + box_top,
        left + box_left + box_width,
        top + box_top + box_height,
    )
    return _classify(gray, box, cv2, np)


def _has_square_border(component: Any, np: Any) -> bool:
    height, width = component.shape
    band = max(1, round(min(width, height) * 0.16))
    edge_strengths = (
        float(np.max(np.mean(component[:band] > 0, axis=1))),
        float(np.max(np.mean(component[-band:] > 0, axis=1))),
        float(np.max(np.mean(component[:, :band] > 0, axis=0))),
        float(np.max(np.mean(component[:, -band:] > 0, axis=0))),
    )
    return sum(strength >= 0.65 for strength in edge_strengths) >= 3


def _indexed_cells(
    cells: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    indexed = {}
    for cell in cells:
        rows = cell.get("row_nums") or []
        columns = cell.get("column_nums") or []
        if len(rows) == 1 and len(columns) == 1:
            indexed[(int(rows[0]), int(columns[0]))] = cell
    return indexed


def _header_rows(
    cells: dict[tuple[int, int], dict[str, Any]],
) -> list[int]:
    rows = sorted({row for row, _ in cells})
    leading_rows = set(rows[:2])
    return [
        row
        for row in rows
        if any(
            _control_header(cell.get("text", ""))
            for (cell_row, _), cell in cells.items()
            if cell_row == row
        )
        or (
            row in leading_rows
            and sum(
                _header_text(cell.get("text", ""))
                for (cell_row, _), cell in cells.items()
                if cell_row == row
            )
            >= 3
        )
    ]


def _header_text(text: str) -> bool:
    normalized = " ".join(text.split())
    alpha_count = sum(character.isalpha() for character in normalized)
    return 2 <= alpha_count <= 24 and not any(
        character.isdigit() for character in normalized
    )


def _control_header(text: str) -> bool:
    words = [
        "".join(character for character in word.lower() if character.isalnum())
        for word in text.split()
    ]
    compact = "".join(words)
    return "checkbox" in words or "radio" in words or compact == "checkbox"


def _control_columns(
    cells: dict[tuple[int, int], dict[str, Any]],
    headers: dict[int, dict[str, Any]],
    header_row: int,
    next_header: int,
) -> set[int]:
    columns = {
        column
        for column, header in headers.items()
        if _control_header(header.get("text", ""))
    }
    header_columns = sorted(headers)
    inferred_grid = len(header_columns) >= 3 and header_columns == list(
        range(header_columns[0], header_columns[-1] + 1)
    )
    if not inferred_grid:
        return columns

    data_rows = sorted({row for row, _ in cells if header_row < row < next_header})
    for row in data_rows:
        mark_columns = [
            column
            for column in headers
            if (row, column) in cells
            and _mark_cell_text(cells[(row, column)])
            and _row_label(cells, row, column) is not None
        ]
        if len(mark_columns) >= 2:
            columns.update(mark_columns)
    return columns


def _row_label(
    cells: dict[tuple[int, int], dict[str, Any]],
    row: int,
    column: int,
) -> dict[str, Any] | None:
    candidates = [
        (candidate_column, cell)
        for (candidate_row, candidate_column), cell in cells.items()
        if candidate_row == row
        and candidate_column < column
        and sum(character.isalpha() for character in cell.get("text", "")) >= 2
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            sum(character.isalpha() for character in item[1].get("text", "")),
            item[0],
        ),
    )[1]


def _mark_cell_text(cell: dict[str, Any]) -> bool:
    text = str(cell.get("text", ""))
    normalized = "".join(character for character in text if character.isalnum())
    if any(character.isdigit() for character in normalized) or len(normalized) > 3:
        return False
    if not normalized:
        return True
    confidence = cell.get("confidence")
    return isinstance(confidence, int | float) and confidence < 0.6


def _cell_box(cell: dict[str, Any]) -> BoundingBox | None:
    """A cell's pixel box, or None when the table came without per-cell geometry.

    A reader that emits table structure as markup gives no box per cell, and a cell
    with no box cannot be examined for a mark.
    """
    box = cell.get("bbox")
    if not isinstance(box, dict):
        return None
    try:
        return BoundingBox(
            int(box["left"]),
            int(box["top"]),
            int(box["right"]),
            int(box["bottom"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _mark_in_cell(
    gray: Any,
    cell: BoundingBox,
    cv2: Any,
    np: Any,
) -> ControlDetection | None:
    width = cell.right - cell.left
    height = cell.bottom - cell.top
    pad = max(2, round(min(width, height) * 0.08))
    crop = gray[
        cell.top + pad : cell.bottom - pad,
        cell.left + pad : cell.right - pad,
    ]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    mask = np.where(crop < 180, 255, 0).astype("uint8")
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(8, round(mask.shape[1] * 0.55)), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(8, round(mask.shape[0] * 0.65))),
        ),
    )
    residual = cv2.bitwise_and(
        mask,
        cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual)
    candidates = []
    crop_area = max(1, crop.shape[0] * crop.shape[1])
    for index in range(1, count):
        left, top, box_width, box_height, area = map(int, stats[index])
        if area < 8 or box_width < crop.shape[1] * 0.3:
            continue
        if box_height < crop.shape[0] * 0.35 or area / crop_area > 0.35:
            continue
        component = np.where(
            labels[top : top + box_height, left : left + box_width] == index,
            255,
            0,
        ).astype("uint8")
        if not _has_table_mark_shape(component, cv2, np):
            continue
        candidates.append((area, left, top, box_width, box_height))
    if not candidates:
        return None

    area, left, top, box_width, box_height = max(candidates)
    box = BoundingBox(
        cell.left + pad + left,
        cell.top + pad + top,
        cell.left + pad + left + box_width,
        cell.top + pad + top + box_height,
    )
    ink_ratio = area / max(1, box_width * box_height)
    return ControlDetection(
        box,
        "selected",
        min(0.94, 0.78 + ink_ratio * 0.5),
        ink_ratio,
        "table_mark",
    )


def _has_table_mark_shape(component: Any, cv2: Any, np: Any) -> bool:
    height, width = component.shape
    lines = cv2.HoughLinesP(
        component,
        1,
        np.pi / 180,
        threshold=2,
        minLineLength=max(3, round(min(width, height) * 0.2)),
        maxLineGap=2,
    )
    if lines is None:
        return False
    diagonal_lengths = {False: 0.0, True: 0.0}
    for left, top, right, bottom in lines.reshape(-1, 4):
        if right < left:
            left, top, right, bottom = right, bottom, left, top
        angle = float(np.degrees(np.arctan2(bottom - top, right - left)))
        if 15 <= abs(angle) <= 80:
            length = float(np.hypot(right - left, bottom - top))
            diagonal_lengths[angle > 0] = max(diagonal_lengths[angle > 0], length)
    minimum_dimension = min(width, height)
    return max(diagonal_lengths.values()) >= minimum_dimension * 0.35 and all(
        length >= minimum_dimension * 0.18 for length in diagonal_lengths.values()
    )


def _mark_shape(component: Any, cv2: Any, np: Any) -> str:
    """Name the glyph. A cross fills all four corners; a tick does not.

    Returns "unknown" rather than guessing, so an unnamed mark stays visible evidence
    instead of being rendered as a shape nobody observed.
    """
    if _has_null_mark_shape(component, cv2, np):
        return "slashed_loop"
    if not _has_mark_shape(component, cv2, np):
        return "unknown"
    height, width = component.shape
    if not 0.5 <= width / height <= 2:
        return "tick"
    edge_height = max(1, height // 3)
    edge_width = max(1, width // 3)
    corners = (
        component[:edge_height, :edge_width],
        component[:edge_height, -edge_width:],
        component[-edge_height:, :edge_width],
        component[-edge_height:, -edge_width:],
    )
    return "cross" if all(np.any(corner) for corner in corners) else "tick"


def _has_mark_shape(component: Any, cv2: Any, np: Any) -> bool:
    height, width = component.shape
    if float(np.mean(component > 0)) >= 0.85:
        return False
    contours, hierarchy = cv2.findContours(
        component.copy(),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is not None and any(
        parent >= 0
        and cv2.contourArea(contours[index]) >= max(3, component.size * 0.01)
        for index, (*_, parent) in enumerate(hierarchy[0])
    ):
        return False

    corners = ()
    if 0.5 <= width / height <= 2:
        edge_height = max(1, height // 3)
        edge_width = max(1, width // 3)
        corners = (
            component[:edge_height, :edge_width],
            component[:edge_height, -edge_width:],
            component[-edge_height:, :edge_width],
            component[-edge_height:, -edge_width:],
        )
        if all(np.any(corner) for corner in corners):
            return True

    points = np.column_stack(np.where(component > 0))
    if len(points) < 4:
        return False
    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
    if eigenvalues[-1] <= 0 or eigenvalues[0] / eigenvalues[-1] < 0.04:
        return False

    lines = cv2.HoughLinesP(
        component,
        1,
        np.pi / 180,
        threshold=2,
        minLineLength=max(3, round(min(width, height) * 0.25)),
        maxLineGap=2,
    )
    if lines is None:
        return False
    longest_diagonal = 0.0
    diagonal_lengths = {False: 0.0, True: 0.0}
    for left, top, right, bottom in lines.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(bottom - top, right - left)))
        if 20 <= abs(angle) <= 75:
            length = float(np.hypot(right - left, bottom - top))
            longest_diagonal = max(longest_diagonal, length)
            diagonal_lengths[angle > 0] = max(diagonal_lengths[angle > 0], length)
    minimum_arm = max(width, height) * 0.35
    if all(length >= minimum_arm for length in diagonal_lengths.values()):
        return True
    corner_count = sum(np.any(corner) for corner in corners)
    return corner_count >= 3 and longest_diagonal >= max(width, height) * 0.7


def _has_null_mark_shape(component: Any, cv2: Any, np: Any) -> bool:
    """Accept the handwritten null glyph: a closed loop with a stroke through it."""
    height, width = component.shape
    fill = float(np.mean(component > 0))
    if not 0.2 <= fill <= 0.7:
        return False
    contours, hierarchy = cv2.findContours(
        component.copy(),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return False
    minimum_hole = max(2.0, component.size * 0.02)
    if not any(
        parent >= 0 and cv2.contourArea(contours[index]) >= minimum_hole
        for index, (*_, parent) in enumerate(hierarchy[0])
    ):
        return False
    # The null stroke crosses its own loop; an empty checkbox outline has a blank centre.
    core = component[
        height // 3 : -(height // 3) or None, width // 3 : -(width // 3) or None
    ]
    if core.size == 0 or float(np.mean(core > 0)) < 0.15:
        return False
    lines = cv2.HoughLinesP(
        component,
        1,
        np.pi / 180,
        threshold=2,
        minLineLength=max(3, round(min(width, height) * 0.25)),
        maxLineGap=2,
    )
    if lines is None:
        return False
    span = max(width, height)
    # A handwritten slash through a flat cursive loop can sit below 20 degrees, so
    # the floor is lower than the tick/cross band; a ruled line at 0 stays excluded.
    return any(
        15 <= abs(float(np.degrees(np.arctan2(bottom - top, right - left)))) <= 75
        and float(np.hypot(right - left, bottom - top)) >= span * 0.3
        for left, top, right, bottom in lines.reshape(-1, 4)
    )


def _merge_detections(
    primary: list[ControlDetection],
    recovered: list[ControlDetection],
) -> list[ControlDetection]:
    return sorted(
        primary + recovered,
        key=lambda item: (item.bounding_box.top, item.bounding_box.left),
    )


def _overlaps_detection(
    box: BoundingBox,
    detections: list[ControlDetection],
) -> bool:
    center_x, center_y = _center(box)
    for detection in detections:
        other = detection.bounding_box
        other_x, other_y = _center(other)
        if (
            other.left <= center_x <= other.right
            and other.top <= center_y <= other.bottom
        ):
            return True
        if box.left <= other_x <= box.right and box.top <= other_y <= box.bottom:
            return True
    return False


def _supported_unmatched_detection(
    detection: ControlDetection,
    detections: list[ControlDetection],
    labels: list[TextRegion | None],
    page_shape: tuple[int, ...],
) -> bool:
    side = _side(detection.bounding_box)
    center_x, center_y = _center(detection.bounding_box)
    maximum_gap = max(side * 8, min(page_shape) * 0.08)
    for peer, label in zip(detections, labels, strict=True):
        if label is None:
            continue
        peer_side = _side(peer.bounding_box)
        if not 0.75 <= peer_side / max(1, side) <= 1.25:
            continue
        peer_x, peer_y = _center(peer.bounding_box)
        aligned = (
            min(abs(peer_x - center_x), abs(peer_y - center_y))
            <= max(side, peer_side) * 0.75
        )
        if (
            aligned
            and max(abs(peer_x - center_x), abs(peer_y - center_y)) <= maximum_gap
        ):
            return True
    return False


def _square_boxes(mask: Any, cv2: Any) -> list[BoundingBox]:
    height, width = mask.shape
    min_side = max(4, round(min(height, width) * 0.006))
    max_side = max(20, round(min(height, width) * 0.08))
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return []

    boxes = []
    deformed_boxes = []
    for index, contour in enumerate(contours):
        left, top, box_width, box_height = cv2.boundingRect(contour)
        side = min(box_width, box_height)
        if side < min_side or max(box_width, box_height) > max_side:
            continue
        if not 0.85 <= box_width / box_height <= 1.2:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        rectangularity = cv2.contourArea(contour) / (box_width * box_height)
        if hierarchy[0][index][2] < 0:
            continue
        box = BoundingBox(
            left=left,
            top=top,
            right=left + box_width,
            bottom=top + box_height,
        )
        if (
            len(polygon) == 4
            and rectangularity >= 0.6
            and not _joins_text(mask, left, top, box_width, box_height)
        ):
            boxes.append(box)
        elif 5 <= len(polygon) <= 6 and rectangularity >= 0.45:
            # A mark drawn past the border notches the outer contour, so a ticked box
            # scores far below an empty one (0.49 against 0.78 on the wellness form).
            # Precision here comes from the same-row peer below, not from this floor.
            deformed_boxes.append(box)
    return boxes + [box for box in deformed_boxes if _has_square_row_peer(box, boxes)]


def _has_square_row_peer(candidate: BoundingBox, boxes: list[BoundingBox]) -> bool:
    candidate_side = _side(candidate)
    _, candidate_y = _center(candidate)
    for box in boxes:
        box_side = _side(box)
        if not 0.75 <= box_side / candidate_side <= 1.33:
            continue
        _, box_y = _center(box)
        if abs(box_y - candidate_y) <= max(box_side, candidate_side) * 0.35:
            return True
    return False


def _joins_text(
    mask: Any,
    left: int,
    top: int,
    width: int,
    height: int,
) -> bool:
    import numpy as np

    page_height, page_width = mask.shape
    crop = mask[top : top + height, left : left + width]
    if height <= 2:
        return True

    border_width = max(1, round(min(width, height) * 0.2))
    edge_ink = (
        float(np.mean(crop[:border_width] > 0, axis=1).max()),
        float(np.mean(crop[-border_width:] > 0, axis=1).max()),
        float(np.mean(crop[:, :border_width] > 0, axis=0).max()),
        float(np.mean(crop[:, -border_width:] > 0, axis=0).max()),
    )
    if min(edge_ink) >= 0.9:
        return _continues_grid(mask, left, top, width, height)

    scan = max(3, round(min(width, height) * 0.5))
    left_strip = mask[top : top + height, max(0, left - scan) : left]
    right_strip = mask[
        top : top + height,
        left + width : min(page_width, left + width + scan),
    ]
    above_strip = mask[max(0, top - scan) : top, left : left + width]
    below_strip = mask[
        top + height : min(page_height, top + height + scan),
        left : left + width,
    ]
    occupancy = [
        float(np.mean(strip > 0, axis=axis).max())
        for strip, axis in (
            (left_strip, 0),
            (right_strip, 0),
            (above_strip, 1),
            (below_strip, 1),
        )
        if strip.size
    ]
    return bool(occupancy and max(occupancy) >= 0.15)


def _continues_grid(
    mask: Any,
    left: int,
    top: int,
    width: int,
    height: int,
) -> bool:
    import numpy as np

    page_height, page_width = mask.shape
    reach = max(3, round(min(width, height) * 0.5))
    thickness = max(1, round(min(width, height) * 0.12))
    horizontal_bands = (
        (max(0, top - thickness), min(page_height, top + thickness + 1)),
        (
            max(0, top + height - thickness - 1),
            min(page_height, top + height + thickness),
        ),
    )
    vertical_bands = (
        (max(0, left - thickness), min(page_width, left + thickness + 1)),
        (
            max(0, left + width - thickness - 1),
            min(page_width, left + width + thickness),
        ),
    )
    patches = []
    for band_top, band_bottom in horizontal_bands:
        patches.extend(
            (
                mask[band_top:band_bottom, max(0, left - reach) : left],
                mask[
                    band_top:band_bottom,
                    left + width : min(page_width, left + width + reach),
                ],
            )
        )
    for band_left, band_right in vertical_bands:
        patches.extend(
            (
                mask[max(0, top - reach) : top, band_left:band_right],
                mask[
                    top + height : min(page_height, top + height + reach),
                    band_left:band_right,
                ],
            )
        )
    continuations = sum(
        float(np.mean(patch > 0)) >= 0.15 for patch in patches if patch.size
    )
    return continuations >= 2


def _deduplicate(boxes: list[BoundingBox]) -> list[BoundingBox]:
    kept = []
    for box in sorted(boxes, key=_area, reverse=True):
        center = _center(box)
        if any(
            abs(center[0] - _center(other)[0]) <= 2
            and abs(center[1] - _center(other)[1]) <= 2
            for other in kept
        ):
            continue
        kept.append(box)
    return kept


def _remove_input_grids(boxes: list[BoundingBox]) -> list[BoundingBox]:
    grid_ids: set[int] = set()
    for index, box in enumerate(boxes):
        side = max(box.right - box.left, box.bottom - box.top)
        neighbors = [
            other_index
            for other_index, other in enumerate(boxes)
            if other_index != index
            and abs(_center(box)[1] - _center(other)[1]) <= 2
            and 0 <= other.left - box.right <= side
        ]
        for neighbor in neighbors:
            next_box = boxes[neighbor]
            next_side = max(
                next_box.right - next_box.left,
                next_box.bottom - next_box.top,
            )
            tails = [
                tail_index
                for tail_index, tail in enumerate(boxes)
                if tail_index not in {index, neighbor}
                and abs(_center(next_box)[1] - _center(tail)[1]) <= 2
                and 0 <= tail.left - next_box.right <= max(side, next_side)
            ]
            if tails:
                grid_ids.update({index, neighbor, *tails})

    return [box for index, box in enumerate(boxes) if index not in grid_ids]


def _classify(gray: Any, box: BoundingBox, cv2: Any, np: Any) -> ControlDetection:
    side = min(box.right - box.left, box.bottom - box.top)
    margin = max(2, round(side * 0.4))
    inner = gray[
        box.top + margin : box.bottom - margin,
        box.left + margin : box.right - margin,
    ]
    if not inner.size:
        return ControlDetection(box, "ambiguous", 0.0, 0.0)

    pad = max(2, round(side * 0.5))
    top = max(0, box.top - pad)
    left = max(0, box.left - pad)
    bottom = min(gray.shape[0], box.bottom + pad)
    right = min(gray.shape[1], box.right + pad)
    context = gray[top:bottom, left:right]
    background = float(np.median(context)) if context.size else 255.0
    dark_ratio = float(np.mean(inner < background - 45))
    faint_ratio = float(np.mean(inner < background - 20))
    ink_ratio = max(dark_ratio, faint_ratio * 0.75)
    if dark_ratio >= 0.2 or faint_ratio >= 0.45:
        shape = _boxed_mark_shape(inner, background, cv2, np)
        return ControlDetection(
            box, "selected", min(0.99, 0.75 + ink_ratio), ink_ratio, shape=shape
        )
    if dark_ratio <= 0.05 and faint_ratio <= 0.15:
        return ControlDetection(box, "unselected", 0.98, ink_ratio, shape="empty")
    return ControlDetection(box, "ambiguous", 0.5, ink_ratio)


def _boxed_mark_shape(inner: Any, background: float, cv2: Any, np: Any) -> str:
    """Name the glyph drawn inside a checkbox, so ☑ and ☒ are told apart."""
    mask = np.where(inner < background - 20, 255, 0).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return "unknown"
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left, top, width, height = (int(stats[largest][index]) for index in range(4))
    if min(width, height) < 3:
        return "unknown"
    component = np.where(
        labels[top : top + height, left : left + width] == largest, 255, 0
    ).astype("uint8")
    return _mark_shape(component, cv2, np)


def _control_region(
    detection: ControlDetection,
    page_number: int,
    index: int,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
    detections: list[ControlDetection],
    page_shape: tuple[int, ...],
) -> TextRegion:
    # A ring circles one value, not its whole row. Falling back to the anchor/nearest
    # label for an unmatched ring_mark would render the row's full text once per ring.
    label = (
        None
        if detection.label is not None or detection.source == "ring_mark"
        else _anchor_label(detection, regions)
        or _nearest_label(detection.bounding_box, regions, label_providers)
    )
    label_regions = _label_regions(detection.bounding_box, label, regions)
    selection_supported = _selection_supported(
        detection,
        detections,
        page_shape,
    )
    observed_state = detection.state
    state = (
        "ambiguous"
        if observed_state == "selected"
        and ((label is None and detection.label is None) or not selection_supported)
        else observed_state
    )
    resolution = "resolved"
    if state == "ambiguous" or (label is None and detection.label is None):
        resolution = "unreadable"
    state_confidence = detection.confidence if resolution == "resolved" else None
    shape = detection.shape
    if detection.source == "ring_mark":
        shape = "ring"
    boxed = detection.source == "square"
    annotation = shape in ANNOTATION_SHAPES
    # The glyph records what was drawn. It does not claim the field means "none": the
    # form's conventions or a reviewer establish that, so interpretation stays unresolved.
    # Checkbox text stays "[x]"/"[ ]" because that is GitHub task-list syntax and the
    # Markdown export depends on it. The tick-versus-cross distinction rides in
    # structure.mark_glyph instead, where the UI and the inspector can use it.
    symbol = (
        MARK_GLYPHS[shape]
        if annotation
        else {"selected": "[x]", "unselected": "[ ]", "ambiguous": "[?]"}[state]
    )
    label_text = detection.label or _semantic_label(label_regions)
    label_ids = list(detection.label_ids) or [region.id for region in label_regions]
    source_ids = list(detection.source_ids)
    methods = {
        "anchored_mark": "label_anchored_residual_ink",
        "null_mark": "label_anchored_null_glyph",
        "ring_mark": "enclosing_ring_annotation",
        "table_mark": "table_cell_residual_ink",
    }
    method = methods.get(detection.source, "square_contour_with_line_cleanup")
    model = MARK_MODEL if detection.source != "square" else MODEL
    text_provenance = {
        "method": method,
        "label_evidence_ids": label_ids,
    }
    if source_ids:
        text_provenance["source_evidence_ids"] = source_ids
    structure = {
        "role": "control",
        "control_type": "annotation" if annotation else "checkbox",
        # What the mark looks like, recorded for every control so a reviewer can see the
        # tick or cross that produced a selection, not just the selection.
        "mark_shape": shape,
        "mark_glyph": (BOXED_GLYPHS if boxed else MARK_GLYPHS).get(shape),
        **(
            {
                "annotation_shape": shape,
                "interpretation_status": "unresolved",
                # A ring means the enclosed option was chosen, so its target is recorded
                # as a relationship rather than being conflated with the mark's label.
                **(
                    {
                        "annotation_target": {
                            "relation": "encloses",
                            "evidence_ids": list(detection.label_ids),
                        }
                    }
                    if shape == "ring" and detection.label_ids
                    else {}
                ),
            }
            if annotation
            else {}
        ),
        "state": state,
        "observed_state": observed_state,
        "state_confidence": state_confidence,
        "observed_mark_confidence": detection.confidence,
        "selection_supported": selection_supported,
        "label": label_text or None,
        "label_evidence_ids": label_ids,
        "association_status": "linked" if label_text else "unmatched",
        "association_confidence": None,
        "ink_ratio": round(detection.ink_ratio, 4),
        "model": model,
    }
    if source_ids:
        structure["source_evidence_ids"] = source_ids
    return TextRegion(
        id=f"p{page_number}-controls-checkbox-{index}",
        kind="checkbox",
        text=f"{symbol} {label_text}".strip(),
        confidence=state_confidence,
        bounding_box=detection.bounding_box,
        reading_order=(
            detection.reading_order
            if detection.reading_order is not None
            else min(region.reading_order for region in label_regions)
            if label_regions
            else 10**9 + index
        ),
        provider=PROVIDER,
        text_provenance=text_provenance,
        resolution=resolution,
        structure=structure,
    )


def _selection_supported(
    detection: ControlDetection,
    detections: list[ControlDetection],
    page_shape: tuple[int, ...],
) -> bool:
    if detection.state != "selected":
        return True
    if detection.source == "table_mark":
        return True
    if detection.source == "anchored_mark":
        return False

    side = _side(detection.bounding_box)
    maximum_intrinsic_side = max(18, round(min(page_shape) * 0.045))
    if side <= maximum_intrinsic_side:
        return True

    center_y = _center(detection.bounding_box)[1]
    for peer in detections:
        if peer is detection:
            continue
        peer_side = _side(peer.bounding_box)
        if not 0.75 <= peer_side / side <= 1.25:
            continue
        peer_center_y = _center(peer.bounding_box)[1]
        if abs(peer_center_y - center_y) <= max(side, peer_side) * 0.6:
            return True
    return False


def _contains_readable_text(
    control: BoundingBox,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
) -> bool:
    side = _side(control)
    for region in regions:
        if region.kind in {"checkbox", "table", "coverage_risk", "page_text"}:
            continue
        if label_providers is not None and region.provider not in label_providers:
            continue
        if sum(character.isalnum() for character in region.text) < 2:
            continue
        box = region.bounding_box
        height = max(1, box.bottom - box.top)
        if side < height * 2:
            continue
        if (
            control.left <= box.left
            and control.top <= box.top
            and control.right >= box.right
            and control.bottom >= box.bottom
        ):
            return True
    return False


def _inside_math_region(
    control: BoundingBox,
    regions: list[TextRegion],
) -> bool:
    center_x, center_y = _center(control)
    return any(
        _math_region(region)
        and region.bounding_box.left <= center_x <= region.bounding_box.right
        and region.bounding_box.top <= center_y <= region.bounding_box.bottom
        for region in regions
    )


def _math_region(region: TextRegion) -> bool:
    structure = region.structure if isinstance(region.structure, dict) else {}
    values = (
        region.kind,
        structure.get("role"),
        structure.get("semantic_class"),
        structure.get("block_type"),
    )
    return any(
        isinstance(value, str)
        and value.strip().casefold().replace("-", "_") in MATH_TYPES
        for value in values
    )


def _inside_text_region(
    control: BoundingBox,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
) -> bool:
    center_x, center_y = _center(control)
    control_height = max(1, control.bottom - control.top)
    for region in regions:
        if region.kind in {"checkbox", "table", "coverage_risk", "page_text"}:
            continue
        if label_providers is not None and region.provider not in label_providers:
            continue
        if not _readable_label(region.text):
            continue
        box = region.bounding_box
        region_height = max(1, box.bottom - box.top)
        if region_height > control_height * 8:
            continue
        if box.left <= center_x <= box.right and box.top <= center_y <= box.bottom:
            control_width = max(1, control.right - control.left)
            leading_slot = center_x - box.left <= control_width
            trailing_slot = box.right - center_x <= control_width
            if leading_slot or trailing_slot:
                continue
            return True
    return False


def _center_inside_reader_text(
    control: BoundingBox,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
) -> bool:
    center_x, center_y = _center(control)
    return any(
        region.kind not in {"checkbox", "table", "coverage_risk", "page_text"}
        and (label_providers is None or region.provider in label_providers)
        and _readable_label(region.text)
        and region.bounding_box.left <= center_x <= region.bounding_box.right
        and region.bounding_box.top <= center_y <= region.bounding_box.bottom
        for region in regions
    )


def _label_regions(
    control: BoundingBox,
    label: TextRegion | None,
    regions: list[TextRegion],
) -> list[TextRegion]:
    if label is None or label.bounding_box.left < control.right - 1:
        return [label] if label is not None else []

    candidates = []
    for region in regions:
        text = region.text.strip()
        box = region.bounding_box
        if region is label or region.kind in {"checkbox", "table"}:
            continue
        if region.provider != label.provider or not text.endswith(":"):
            continue
        if box.right > control.left + 1:
            continue
        gap = control.left - box.right
        height = max(1, box.bottom - box.top)
        if gap > max(12, height):
            continue
        if min(control.bottom, box.bottom) <= max(control.top, box.top):
            continue
        candidates.append((gap, abs(_center(box)[1] - _center(control)[1]), region))
    if not candidates:
        return [label]
    context = min(candidates, key=lambda item: (item[0], item[1], item[2].id))[2]
    return [context, label]


def _semantic_label(regions: list[TextRegion]) -> str:
    parts = []
    for region in regions:
        text = region.text.strip().rstrip(":").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _anchor_label(
    detection: ControlDetection,
    regions: list[TextRegion],
) -> TextRegion | None:
    """A label-anchored mark keeps its anchor; geometric nearness would pick the next field."""
    if (
        detection.source not in {"anchored_mark", "null_mark", "ring_mark"}
        or not detection.label_ids
    ):
        return None
    anchor_id = detection.label_ids[0]
    return next((region for region in regions if region.id == anchor_id), None)


def _nearest_label(
    control: BoundingBox,
    regions: list[TextRegion],
    label_providers: frozenset[str] | None,
) -> TextRegion | None:
    center_x, center_y = _center(control)
    candidates = []
    for region in regions:
        text = region.text.strip()
        if not _readable_label(text):
            continue
        if region.kind in {"checkbox", "table"}:
            continue
        box = region.bounding_box
        _, region_y = _center(box)
        height = max(1, box.bottom - box.top)
        if abs(region_y - center_y) > max(control.bottom - control.top, height):
            continue
        if box.left <= center_x <= box.right and box.top <= center_y <= box.bottom:
            distance = 0.0
        elif control.left < box.left < control.right:
            distance = 0.0
        elif control.left < box.right < control.right:
            distance = 50.0
        elif box.left >= control.right - 1:
            distance = box.left - control.right
        elif box.right <= control.left + 1:
            distance = (control.left - box.right) + 50
        else:
            continue
        if distance > 150:
            continue
        provider_penalty = (
            0 if label_providers is None or region.provider in label_providers else 1000
        )
        readability_penalty = (
            0 if sum(character.isalnum() for character in text) >= 2 else 100
        )
        candidates.append(
            (
                provider_penalty
                + readability_penalty
                + distance
                + abs(region_y - center_y) * 2,
                _area(box),
                region,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2].id))[2]


def _readable_label(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    if normalized in {"x", "0", "区", "口", "□", "☐", "☒", "✓", "✔"}:
        return False
    return any(character.isalnum() for character in normalized)


def _explicit_control_label(region: TextRegion) -> bool:
    role = (region.structure or {}).get("role")
    return region.kind == "form_field" or role in {
        "control",
        "control_label",
        "form_field",
    }


def _descriptive_control_label(region: TextRegion) -> bool:
    return any(
        sum(character.isalnum() for character in token) >= 2
        for token in region.text.split()
    )


def _supported_short_option_label(
    detection: ControlDetection,
    label: TextRegion,
    detections: list[ControlDetection],
    labels: list[TextRegion | None],
) -> bool:
    label_text = _short_option_text(label.text)
    if label_text is None:
        return False
    side = _side(detection.bounding_box)
    center_x, center_y = _center(detection.bounding_box)
    for peer, peer_label in zip(detections, labels, strict=True):
        if peer is detection or peer_label is None:
            continue
        peer_text = _short_option_text(peer_label.text)
        if peer_text is None or peer_text == label_text:
            continue
        peer_side = _side(peer.bounding_box)
        peer_x, peer_y = _center(peer.bounding_box)
        if 0.75 <= peer_side / max(1, side) <= 1.25 and min(
            abs(peer_x - center_x), abs(peer_y - center_y)
        ) <= max(side, peer_side):
            return True
    return False


def _short_option_text(text: str) -> str | None:
    normalized = "".join(
        character.casefold() for character in text if character.isalnum()
    )
    return normalized if len(normalized) == 1 else None


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.left + box.right) / 2, (box.top + box.bottom) / 2)


def _area(box: BoundingBox) -> int:
    return max(0, box.right - box.left) * max(0, box.bottom - box.top)


def _side(box: BoundingBox) -> int:
    return max(box.right - box.left, box.bottom - box.top)
