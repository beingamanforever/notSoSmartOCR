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
MARK_MODEL = {
    "id": "opencv-label-anchored-residual-v1",
    "origin": "Open Source Vision Foundation",
    "license": "Apache-2.0",
}


@dataclass(frozen=True)
class ControlDetection:
    bounding_box: BoundingBox
    state: str
    confidence: float
    ink_ratio: float
    source: str = "square"


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
        cv2, np, gray = _load_gray(image_path)
        detections = _square_detections(gray, cv2, np)
        detections = _merge_detections(
            detections,
            _anchored_marks(gray, regions, self.label_provider, detections, cv2, np),
        )
        detections = [
            detection
            for detection in detections
            if not _inside_text_region(
                detection.bounding_box,
                regions,
                self.label_provider,
            )
        ]
        labels = [
            _nearest_label(detection.bounding_box, regions, self.label_provider)
            for detection in detections
        ]
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
                self.label_provider,
                detections,
                gray.shape,
            )
            for index, detection in enumerate(detections, start=1)
        ]
        coverage_missing = bool(controls) and len(controls) < self.minimum_group_size
        for control in controls:
            if control.structure["label"] is None:
                control.structure["coverage_status"] = "unmatched_label"
            else:
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
    cv2, np, gray = _load_gray(image_path)
    return _square_detections(gray, cv2, np)


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
        _classify(gray, box, np)
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
        )
        if _side(detection.bounding_box) < minimum_reliable_side
        else detection
        for detection in detections
    ]


def _anchored_marks(
    gray: Any,
    regions: list[TextRegion],
    label_provider: str | None,
    existing: list[ControlDetection],
    cv2: Any,
    np: Any,
) -> list[ControlDetection]:
    height, width = gray.shape
    detections = []
    for region in regions:
        if region.kind in {"checkbox", "table", "coverage_risk", "page_text"}:
            continue
        if label_provider is not None and region.provider != label_provider:
            continue
        if not _readable_label(region.text):
            continue
        if region.confidence is None or region.confidence < 0.9:
            continue

        box = region.bounding_box
        line_height = max(1, box.bottom - box.top)
        vertical_pad = max(2, line_height // 2)
        top = max(0, box.top - vertical_pad)
        bottom = min(height, box.bottom + vertical_pad)
        slot_width = round(line_height * 1.75)
        slots = [BoundingBox(max(0, box.left - slot_width), top, box.left, bottom)]
        slots.append(
            BoundingBox(
                box.right,
                top,
                min(width, box.right + slot_width),
                bottom,
            )
        )
        for slot in slots:
            for detection in _marks_in_slot(gray, slot, line_height, cv2, np):
                if _overlaps_detection(detection.bounding_box, existing + detections):
                    continue
                if _center_inside_reader_text(
                    detection.bounding_box,
                    regions,
                    label_provider,
                ):
                    continue
                if _inside_text_region(detection.bounding_box, regions, label_provider):
                    continue
                detections.append(detection)
    return detections


def _marks_in_slot(
    gray: Any,
    slot: BoundingBox,
    line_height: int,
    cv2: Any,
    np: Any,
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
    detections = []
    for index in range(1, count):
        left, top, box_width, box_height, area = map(int, stats[index])
        if area < 8 or min(box_width, box_height) < 4:
            continue
        if min(box_width, box_height) < line_height * 0.38:
            continue
        if max(box_width, box_height) > line_height * 1.1:
            continue
        component = np.where(
            labels[top : top + box_height, left : left + box_width] == index,
            255,
            0,
        ).astype("uint8")
        if not _has_mark_shape(component, cv2, np):
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
                "anchored_mark",
            )
        )
    return detections


def _has_mark_shape(component: Any, cv2: Any, np: Any) -> bool:
    height, width = component.shape
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
    max_side = max(18, round(min(height, width) * 0.08))
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
        if not 0.85 <= box_width / box_height <= 1.2:
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
    margin = max(2, round(side * 0.4))
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
    detections: list[ControlDetection],
    page_shape: tuple[int, ...],
) -> TextRegion:
    label = _nearest_label(detection.bounding_box, regions, label_provider)
    label_regions = _label_regions(detection.bounding_box, label, regions)
    selection_supported = _selection_supported(
        detection,
        detections,
        page_shape,
    )
    observed_state = detection.state
    state = (
        "ambiguous"
        if observed_state == "selected" and (label is None or not selection_supported)
        else observed_state
    )
    resolution = "resolved"
    if state == "ambiguous" or label is None:
        resolution = "unreadable"
    symbol = {"selected": "[x]", "unselected": "[ ]", "ambiguous": "[?]"}[state]
    label_text = _semantic_label(label_regions)
    label_ids = [region.id for region in label_regions]
    method = (
        "label_anchored_residual_ink"
        if detection.source == "anchored_mark"
        else "square_contour_with_line_cleanup"
    )
    model = MARK_MODEL if detection.source == "anchored_mark" else MODEL
    return TextRegion(
        id=f"p{page_number}-controls-checkbox-{index}",
        kind="checkbox",
        text=f"{symbol} {label_text}".strip(),
        confidence=detection.confidence,
        bounding_box=detection.bounding_box,
        reading_order=(
            min(region.reading_order for region in label_regions)
            if label_regions
            else 10**9 + index
        ),
        provider=PROVIDER,
        text_provenance={
            "method": method,
            "label_evidence_ids": label_ids,
        },
        resolution=resolution,
        structure={
            "role": "control",
            "control_type": "checkbox",
            "state": state,
            "observed_state": observed_state,
            "selection_supported": selection_supported,
            "label": label_text or None,
            "label_evidence_ids": label_ids,
            "ink_ratio": round(detection.ink_ratio, 4),
            "model": model,
        },
    )


def _selection_supported(
    detection: ControlDetection,
    detections: list[ControlDetection],
    page_shape: tuple[int, ...],
) -> bool:
    if detection.state != "selected":
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


def _inside_text_region(
    control: BoundingBox,
    regions: list[TextRegion],
    label_provider: str | None,
) -> bool:
    center_x, center_y = _center(control)
    control_height = max(1, control.bottom - control.top)
    for region in regions:
        if region.kind in {"checkbox", "table", "coverage_risk", "page_text"}:
            continue
        if label_provider is not None and region.provider != label_provider:
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
    label_provider: str | None,
) -> bool:
    center_x, center_y = _center(control)
    return any(
        region.kind not in {"checkbox", "table", "coverage_risk", "page_text"}
        and (label_provider is None or region.provider == label_provider)
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
            0 if label_provider is None or region.provider == label_provider else 1000
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


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.left + box.right) / 2, (box.top + box.bottom) / 2)


def _area(box: BoundingBox) -> int:
    return max(0, box.right - box.left) * max(0, box.bottom - box.top)


def _side(box: BoundingBox) -> int:
    return max(box.right - box.left, box.bottom - box.top)
