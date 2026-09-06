"""Conservative missing-ink proposals owned by printed form anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "opencv-anchored-ink"
MODEL = {
    "id": "opencv-label-line-residual-v1",
    "origin": "Open Source Vision Foundation",
    "license": "Apache-2.0",
}
EXCLUDED_ANCHOR_KINDS = frozenset(
    {"checkbox", "control", "coverage_risk", "layout_block", "page_text", "table"}
)
NON_TEXT_MASK_KINDS = frozenset(
    {"coverage_risk", "layout_block", "page_text", "table", "table_candidate"}
)


class AnchoredInkProposalStage:
    """Propose unresolved ink only beside readable labels and writing lines."""

    name = "anchored-ink"

    def __init__(
        self,
        *,
        label_provider: str | None = None,
        minimum_anchor_confidence: float = 0.7,
        max_proposals: int = 8,
    ) -> None:
        if not 0 <= minimum_anchor_confidence <= 1:
            raise ValueError("minimum_anchor_confidence must be from 0 to 1")
        if max_proposals < 1:
            raise ValueError("max_proposals must be positive")
        self.label_provider = label_provider
        self.minimum_anchor_confidence = minimum_anchor_confidence
        self.max_proposals = max_proposals

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        cv2, gray = _load_gray(image_path)
        mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]
        proposals: list[TextRegion] = []
        anchors = sorted(
            (region for region in regions if self._is_anchor(region)),
            key=lambda region: (region.reading_order, region.id),
        )
        for anchor in anchors:
            proposal = _proposal(
                mask, anchor, regions, page_number, len(proposals), cv2
            )
            if proposal is None or _duplicates(proposal.bounding_box, proposals):
                continue
            proposals.append(proposal)
            if len(proposals) >= self.max_proposals:
                break
        return [*regions, *proposals]

    def _is_anchor(self, region: TextRegion) -> bool:
        return (
            region.resolution == "resolved"
            and region.kind not in EXCLUDED_ANCHOR_KINDS
            and bool(region.text.strip())
            and "\n" not in region.text
            and _field_anchor(region)
            and region.confidence is not None
            and region.confidence >= self.minimum_anchor_confidence
            and (self.label_provider is None or region.provider == self.label_provider)
            and _valid_box(region.bounding_box)
        )


def _proposal(
    mask: Any,
    anchor: TextRegion,
    regions: list[TextRegion],
    page_number: int,
    index: int,
    cv2: Any,
) -> TextRegion | None:
    height, width = mask.shape
    anchor_box = anchor.bounding_box
    line_height = anchor_box.bottom - anchor_box.top
    search = BoundingBox(
        anchor_box.right,
        max(0, anchor_box.top - line_height * 2),
        min(width, anchor_box.right + max(line_height * 30, width // 2)),
        min(height, anchor_box.bottom + line_height * 2),
    )
    line = _nearest_writing_line(mask, search, anchor_box, line_height, cv2)
    if line is None:
        return None
    line = _owned_line(line, anchor, regions, line_height)
    if line is None:
        return None
    ink_box, component_count, residual_area = _residual_ink_box(
        mask,
        line,
        anchor_box,
        line_height,
        regions,
        cv2,
    )
    if ink_box is None:
        return None

    field_ownership = _field_ownership(anchor, regions, page_number)
    provenance = {
        "method": "label_anchored_rule_removed_residual_ink",
        "page_number": page_number,
        "anchor_evidence_ids": [anchor.id],
        "field_ownership": field_ownership,
        "line_bounding_box": [line.left, line.top, line.right, line.bottom],
        "residual_component_count": component_count,
        "residual_area": residual_area,
        "model": MODEL,
    }
    return TextRegion(
        id=f"p{page_number}-anchored-ink-{index + 1}",
        kind="handwriting",
        text="",
        confidence=None,
        bounding_box=ink_box,
        reading_order=anchor.reading_order,
        provider=PROVIDER,
        text_provenance=provenance,
        resolution="unreadable",
        structure={
            "role": "handwriting_candidate",
            "handwriting_candidate": True,
            "handwriting_candidate_source": "anchored_residual",
            "anchor_evidence_ids": [anchor.id],
            "field_ownership": field_ownership,
            "proposal": provenance,
        },
    )


def _field_ownership(
    anchor: TextRegion,
    regions: list[TextRegion],
    page_number: int,
) -> dict[str, object]:
    for region in regions:
        structure = region.structure or {}
        fields = structure.get("fields")
        if structure.get("role") != "layout_block" or not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            label_ids = field.get("label_evidence_ids")
            if not isinstance(label_ids, list) or anchor.id not in label_ids:
                continue
            value_ids = field.get("value_evidence_ids")
            return {
                "field_id": str(field.get("id") or f"p{page_number}-field-{anchor.id}"),
                "owner_block_id": region.id,
                "label": str(field.get("label") or anchor.text),
                "label_evidence_ids": list(label_ids),
                "value_evidence_ids": list(value_ids)
                if isinstance(value_ids, list)
                else [],
            }
    return {
        "field_id": f"p{page_number}-field-{anchor.id}",
        "owner_block_id": None,
        "label": anchor.text,
        "label_evidence_ids": [anchor.id],
        "value_evidence_ids": [],
    }


def _nearest_writing_line(
    mask: Any,
    search: BoundingBox,
    anchor: BoundingBox,
    line_height: int,
    cv2: Any,
) -> BoundingBox | None:
    if search.right - search.left < line_height * 4:
        return None
    crop = mask[search.top : search.bottom, search.left : search.right]
    kernel_width = max(12, line_height * 4)
    horizontal = cv2.morphologyEx(
        crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    contours = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        0
    ]
    candidates = []
    for contour in contours:
        left, top, width, height = cv2.boundingRect(contour)
        if width < line_height * 4 or height > max(3, line_height // 2):
            continue
        line = BoundingBox(
            search.left + left,
            search.top + top,
            search.left + left + width,
            search.top + top + height,
        )
        if (line.top + line.bottom) / 2 < anchor.top:
            continue
        if line.left - anchor.right > line_height * 3:
            continue
        candidates.append(line)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda line: (
            abs((line.top + line.bottom) / 2 - anchor.bottom),
            -(line.right - line.left),
        ),
    )


def _owned_line(
    line: BoundingBox,
    anchor: TextRegion,
    regions: list[TextRegion],
    line_height: int,
) -> BoundingBox | None:
    right = line.right
    for region in regions:
        if (
            region.id == anchor.id
            or region.kind in NON_TEXT_MASK_KINDS
            or not region.text.strip()
            or not _valid_box(region.bounding_box)
        ):
            continue
        box = region.bounding_box
        same_row = box.top <= anchor.bounding_box.bottom + line_height and (
            box.bottom >= anchor.bounding_box.top - line_height
        )
        if same_row and box.left > anchor.bounding_box.right + line_height:
            right = min(right, box.left)
    if right - line.left < line_height * 4:
        return None
    return BoundingBox(line.left, line.top, right, line.bottom)


def _residual_ink_box(
    mask: Any,
    line: BoundingBox,
    anchor: BoundingBox,
    line_height: int,
    regions: list[TextRegion],
    cv2: Any,
) -> tuple[BoundingBox | None, int, int]:
    page_height, page_width = mask.shape
    field = BoundingBox(
        line.left,
        max(0, anchor.top - line_height, line.top - max(18, line_height * 3)),
        line.right,
        min(page_height, line.bottom + max(3, line_height // 2)),
    )
    if field.right <= field.left or field.bottom <= field.top:
        return None, 0, 0
    crop = mask[field.top : field.bottom, field.left : field.right].copy()
    horizontal = cv2.morphologyEx(
        crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(8, line_height * 3), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(8, line_height * 3)),
        ),
    )
    residual = cv2.bitwise_and(
        crop,
        cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)),
    )
    for region in regions:
        if region.kind in NON_TEXT_MASK_KINDS or not _valid_box(region.bounding_box):
            continue
        overlap = _intersection(field, region.bounding_box)
        if overlap is None:
            continue
        residual[
            overlap.top - field.top : overlap.bottom - field.top,
            overlap.left - field.left : overlap.right - field.left,
        ] = 0

    count, _, stats, _ = cv2.connectedComponentsWithStats(residual)
    components: list[tuple[int, int, int, int, int]] = []
    minimum_height = max(2, round(line_height * 0.18))
    for component in range(1, count):
        left, top, width, height, area = map(int, stats[component])
        if area < max(3, round(line_height * line_height * 0.015)):
            continue
        if height < minimum_height:
            continue
        if height <= 2 and width > height * 8:
            continue
        components.append((left, top, width, height, area))
    if not components:
        return None, 0, 0

    left = min(component[0] for component in components)
    top = min(component[1] for component in components)
    right = max(component[0] + component[2] for component in components)
    bottom = max(component[1] + component[3] for component in components)
    residual_area = sum(component[4] for component in components)
    if right - left < max(5, round(line_height * 0.6)):
        return None, len(components), residual_area
    if bottom - top < max(3, round(line_height * 0.35)):
        return None, len(components), residual_area
    if residual_area < max(8, round(line_height * line_height * 0.08)):
        return None, len(components), residual_area

    padding = max(2, line_height // 4)
    return (
        BoundingBox(
            max(field.left, field.left + left - padding),
            max(field.top, field.top + top - padding),
            min(page_width, field.left + right + padding),
            min(page_height, field.top + bottom + padding),
        ),
        len(components),
        residual_area,
    )


def _intersection(first: BoundingBox, second: BoundingBox) -> BoundingBox | None:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)


def _duplicates(box: BoundingBox, proposals: list[TextRegion]) -> bool:
    for proposal in proposals:
        overlap = _intersection(box, proposal.bounding_box)
        if overlap is None:
            continue
        overlap_area = (overlap.right - overlap.left) * (overlap.bottom - overlap.top)
        box_area = (box.right - box.left) * (box.bottom - box.top)
        proposal_box = proposal.bounding_box
        proposal_area = (proposal_box.right - proposal_box.left) * (
            proposal_box.bottom - proposal_box.top
        )
        if overlap_area / min(box_area, proposal_area) >= 0.5:
            return True
    return False


def _valid_box(box: BoundingBox | None) -> bool:
    return box is not None and box.right > box.left and box.bottom > box.top


def _field_anchor(region: TextRegion) -> bool:
    role = str((region.structure or {}).get("role", "")).casefold()
    return region.text.rstrip().endswith((":", "：")) or role in {
        "field_label",
        "label",
    }


def _load_gray(image_path: Path) -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise ReaderError(
            "anchored_ink_dependency_unavailable",
            "Anchored ink proposals require OpenCV and NumPy",
        ) from error
    try:
        with Image.open(image_path) as image:
            gray = np.asarray(image.convert("L"))
    except (OSError, UnidentifiedImageError) as error:
        raise ReaderError("anchored_ink_image_failed", str(error)) from error
    return cv2, gray
