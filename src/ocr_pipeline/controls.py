"""Evidence-linked checkbox detection without a learned model."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ControlDetection:
    bounding_box: BoundingBox
    state: str
    confidence: float
    ink_ratio: float


class GeometricControlStage:
    """Add structured checkbox regions and flag uncertain controls for review."""

    name = "controls"

    def __init__(
        self,
        *,
        minimum_group_size: int = 6,
        label_provider: str | None = None,
    ) -> None:
        if minimum_group_size < 1:
            raise ValueError("minimum_group_size must be positive")
        self.minimum_group_size = minimum_group_size
        self.label_provider = label_provider

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        detections = detect_controls(image_path)
        controls = [
            _control_region(
                detection,
                page_number,
                index,
                regions,
                self.label_provider,
            )
            for index, detection in enumerate(detections, start=1)
        ]
        coverage_missing = bool(controls) and len(controls) < self.minimum_group_size
        for control in controls:
            control.structure["coverage_status"] = (
                "insufficient_control_group" if coverage_missing else "detected"
            )
            if coverage_missing:
                control.resolution = "unreadable"
        return sorted(
            regions + controls,
            key=lambda region: (
                region.reading_order,
                0 if region.kind == "checkbox" else 1,
                region.id,
            ),
        )


def detect_controls(image_path: Path) -> list[ControlDetection]:
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

    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    faint = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1]
    boxes = _square_boxes(otsu, cv2) + _square_boxes(faint, cv2)
    boxes = _remove_input_grids(_deduplicate(boxes))

    return [
        _classify(gray, box, np)
        for box in sorted(boxes, key=lambda item: (item.top, item.left))
    ]


def _square_boxes(mask: Any, cv2: Any) -> list[BoundingBox]:
    height, width = mask.shape
    min_side = max(4, round(min(height, width) * 0.006))
    max_side = max(18, round(min(height, width) * 0.045))
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return []

    boxes = []
    for index, contour in enumerate(contours):
        left, top, box_width, box_height = cv2.boundingRect(contour)
        side = min(box_width, box_height)
        if side < min_side or max(box_width, box_height) > max_side:
            continue
        if not 0.8 <= box_width / box_height <= 1.25:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        rectangularity = cv2.contourArea(contour) / (box_width * box_height)
        if len(polygon) != 4 or rectangularity < 0.6:
            continue
        if hierarchy[0][index][2] < 0:
            continue
        if _joins_text(mask, left, top, box_width, box_height):
            continue
        boxes.append(
            BoundingBox(
                left=left,
                top=top,
                right=left + box_width,
                bottom=top + box_height,
            )
        )
    return boxes


def _joins_text(
    mask: Any,
    left: int,
    top: int,
    width: int,
    height: int,
) -> bool:
    import numpy as np

    page_width = mask.shape[1]
    if height <= 2:
        return True
    left_strip = mask[top + 1 : top + height - 1, max(0, left - 3) : left]
    right_strip = mask[
        top + 1 : top + height - 1,
        left + width : min(page_width, left + width + 3),
    ]
    occupancy = [
        float(np.mean(strip > 0)) for strip in (left_strip, right_strip) if strip.size
    ]
    return bool(occupancy and max(occupancy) > 0.25)


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


def _classify(gray: Any, box: BoundingBox, np: Any) -> ControlDetection:
    side = min(box.right - box.left, box.bottom - box.top)
    margin = max(2, round(side * 0.25))
    inner = gray[
        box.top + margin : box.bottom - margin,
        box.left + margin : box.right - margin,
    ]
    if not inner.size:
        return ControlDetection(box, "ambiguous", 0.0, 0.0)

    dark_ratio = float(np.mean(inner < 200))
    faint_ratio = float(np.mean(inner < 235))
    ink_ratio = max(dark_ratio, faint_ratio * 0.75)
    if dark_ratio >= 0.2 or faint_ratio >= 0.45:
        return ControlDetection(box, "selected", min(0.99, 0.75 + ink_ratio), ink_ratio)
    if dark_ratio <= 0.05 and faint_ratio <= 0.15:
        return ControlDetection(box, "unselected", 0.98, ink_ratio)
    return ControlDetection(box, "ambiguous", 0.5, ink_ratio)


def _control_region(
    detection: ControlDetection,
    page_number: int,
    index: int,
    regions: list[TextRegion],
    label_provider: str | None,
) -> TextRegion:
    label = _nearest_label(detection.bounding_box, regions, label_provider)
    state = detection.state
    resolution = "resolved"
    if state == "ambiguous" or label is None:
        resolution = "unreadable"
    symbol = {"selected": "[x]", "unselected": "[ ]", "ambiguous": "[?]"}[state]
    label_text = label.text.strip() if label is not None else ""
    return TextRegion(
        id=f"p{page_number}-controls-checkbox-{index}",
        kind="checkbox",
        text=f"{symbol} {label_text}".strip(),
        confidence=detection.confidence,
        bounding_box=detection.bounding_box,
        reading_order=label.reading_order if label is not None else 10**9 + index,
        provider=PROVIDER,
        text_provenance={
            "method": "square_contour_with_line_cleanup",
            "label_evidence_ids": [label.id] if label is not None else [],
        },
        resolution=resolution,
        structure={
            "role": "control",
            "control_type": "checkbox",
            "state": state,
            "label": label_text or None,
            "label_evidence_ids": [label.id] if label is not None else [],
            "ink_ratio": round(detection.ink_ratio, 4),
            "model": MODEL,
        },
    )


def _nearest_label(
    control: BoundingBox,
    regions: list[TextRegion],
    label_provider: str | None,
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
        elif box.left >= control.right - 1:
            distance = box.left - control.right
        elif box.right <= control.left + 1:
            distance = (control.left - box.right) + 50
        else:
            continue
        if distance > 150:
            continue
        provider_penalty = (
            0 if label_provider is None or region.provider == label_provider else 1000
        )
        candidates.append(
            (
                provider_penalty + distance + abs(region_y - center_y) * 2,
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


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.left + box.right) / 2, (box.top + box.bottom) / 2)


def _area(box: BoundingBox) -> int:
    return max(0, box.right - box.left) * max(0, box.bottom - box.top)
