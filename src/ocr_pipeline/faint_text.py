"""Image-driven recovery for faint or tiny text missed by OCR."""

from __future__ import annotations

import copy
import math
import statistics
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

from .contracts import BoundingBox, TextAlternative, TextRegion
from .providers import LocalReader, ReaderError

RecoveryView = Literal[
    "native",
    "global_high_resolution",
    "selective_crop_high_resolution",
]
RECOVERY_VIEWS = frozenset(
    {"native", "global_high_resolution", "selective_crop_high_resolution"}
)
NON_MASKING_KINDS = frozenset(
    {"coverage_risk", "figure", "page_text", "table", "table_candidate"}
)


@dataclass(frozen=True)
class _Proposal:
    box: BoundingBox
    component_count: int
    ink_pixels: int


@dataclass(frozen=True)
class _View:
    path: Path
    origin: BoundingBox
    scale: int


class FaintTinyTextStage:
    """Find unowned text-like pixels, then reread only evidence-backed views."""

    name = "faint-tiny-text"

    def __init__(
        self,
        reader: LocalReader,
        *,
        view: RecoveryView = "selective_crop_high_resolution",
        scale: int = 3,
        minimum_confidence: float = 0.85,
        max_proposals: int = 12,
    ) -> None:
        if view not in RECOVERY_VIEWS:
            raise ValueError(f"Unsupported faint-text recovery view: {view}")
        if scale < 2:
            raise ValueError("Faint-text scale must be at least 2")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("Faint-text confidence must be from 0 to 1")
        if max_proposals < 1:
            raise ValueError("Faint-text proposal limit must be positive")
        self.reader = reader
        self.view = view
        self.scale = scale
        self.minimum_confidence = minimum_confidence
        self.max_proposals = max_proposals

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        try:
            with Image.open(image_path) as opened:
                page = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ReaderError(
                "faint_tiny_image_failed",
                "The faint-text page image could not be opened",
            ) from error

        try:
            proposals = _find_proposals(page, regions, self.max_proposals)
            if len(proposals) > self.max_proposals:
                raise ReaderError(
                    "faint_tiny_proposal_limit_exceeded",
                    f"Found {len(proposals)} faint-text candidates; refusing to "
                    f"reread more than {self.max_proposals}",
                )
            if not proposals:
                return regions
            candidates = self._read(page, image_path, page_number, proposals)
        finally:
            page.close()

        recovered = _resolve_candidates(
            candidates,
            proposals,
            regions,
            page_number,
            self.reader.name,
            self.view,
            self.scale,
            self.minimum_confidence,
        )
        return _insert_recovered_regions(regions, recovered)

    def _read(
        self,
        page: Image.Image,
        image_path: Path,
        page_number: int,
        proposals: list[_Proposal],
    ) -> list[TextRegion]:
        if self.view == "native":
            view = _View(
                image_path,
                BoundingBox(0, 0, page.width, page.height),
                1,
            )
            return _read_views(self.reader, [view], page_number, page.size)

        with tempfile.TemporaryDirectory(prefix="ocr-faint-tiny-") as directory:
            root = Path(directory)
            if self.view == "global_high_resolution":
                path = root / "global.png"
                _save_resized(page, path, self.scale)
                views = [
                    _View(
                        path,
                        BoundingBox(0, 0, page.width, page.height),
                        self.scale,
                    )
                ]
            else:
                views = []
                for index, proposal in enumerate(proposals, start=1):
                    crop = _padded_box(proposal.box, page.size)
                    path = root / f"crop-{index}.png"
                    cropped = page.crop(_box_tuple(crop))
                    try:
                        _save_resized(cropped, path, self.scale)
                    finally:
                        cropped.close()
                    views.append(_View(path, crop, self.scale))
            return _read_views(self.reader, views, page_number, page.size)


def _find_proposals(
    page: Image.Image,
    regions: Sequence[TextRegion],
    proposal_limit: int | None = None,
) -> list[_Proposal]:
    cv2, np = _vision_dependencies()
    gray = np.asarray(page.convert("L"))
    height, width = gray.shape
    if min(width, height) < 11:
        return []

    reference_height = _reference_height(regions, page.size)
    block_size = _odd(min(min(width, height), max(11, reference_height * 3)))
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5,
    )
    padding = max(1, round(reference_height / 4))
    known_boxes = [
        region.bounding_box for region in regions if _masks_pixels(region, page.size)
    ]
    for box in known_boxes:
        left = max(0, box.left - padding)
        top = max(0, box.top - padding)
        right = min(width, box.right + padding)
        bottom = min(height, box.bottom + padding)
        mask[top:bottom, left:right] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    maximum_height = max(3, min(round(height * 0.05), reference_height * 2))
    components = []
    accepted_labels = np.zeros(count, dtype=bool)
    for label in range(1, count):
        left, top, item_width, item_height, area = map(int, stats[label])
        if not _text_like_component(item_width, item_height, area, maximum_height):
            continue
        components.append((label, left, top, item_width, item_height, area))
        accepted_labels[label] = True
    if len(components) < 2:
        return []
    component_mask = np.where(accepted_labels[labels], 255, 0).astype(mask.dtype)

    median_height = max(2, round(statistics.median(item[4] for item in components)))
    grouped = cv2.dilate(
        component_mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(2, round(median_height * 1.25)), max(1, round(median_height / 4))),
        ),
    )
    contours = cv2.findContours(
        grouped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    proposals = []
    ordered_contours = sorted(
        contours,
        key=lambda contour: tuple(cv2.boundingRect(contour)[1::-1]),
    )
    for contour in ordered_contours:
        left, top, item_width, item_height = cv2.boundingRect(contour)
        members = [
            component
            for component in components
            if left <= component[1] + component[3] / 2 <= left + item_width
            and top <= component[2] + component[4] / 2 <= top + item_height
        ]
        if len(members) < 2:
            continue
        box = _component_box(members, page.size, median_height)
        ink_pixels = sum(item[5] for item in members)
        box_area = (box.right - box.left) * (box.bottom - box.top)
        if box.right - box.left < box.bottom - box.top:
            continue
        if not 0.02 <= ink_pixels / box_area <= 0.8:
            continue
        if any(_overlap_fraction(box, known) >= 0.1 for known in known_boxes):
            continue
        proposal = _Proposal(box, len(members), ink_pixels)
        if any(_boxes_duplicate(box, existing.box) for existing in proposals):
            continue
        proposals.append(proposal)
        if proposal_limit is not None and len(proposals) > proposal_limit:
            return proposals
    return proposals


def _read_views(
    reader: LocalReader,
    views: list[_View],
    page_number: int,
    page_size: tuple[int, int],
) -> list[TextRegion]:
    read_batch = getattr(reader, "read_batch", None)
    if len(views) > 1 and callable(read_batch) and getattr(reader, "batch_size", 1) > 1:
        outputs = read_batch(
            [view.path for view in views],
            [page_number] * len(views),
        )
        if not isinstance(outputs, list) or len(outputs) != len(views):
            raise ReaderError(
                "invalid_faint_tiny_output",
                "Faint-text batch reader returned the wrong number of results",
            )
    else:
        outputs = [reader.read(view.path, page_number) for view in views]

    candidates = []
    for view, output in zip(views, outputs, strict=True):
        if isinstance(output, ReaderError):
            raise output
        if not isinstance(output, list) or not all(
            isinstance(region, TextRegion) for region in output
        ):
            raise ReaderError(
                "invalid_faint_tiny_output",
                "Faint-text reader returned invalid regions",
            )
        for region in output:
            translated = _translate(region, view, page_size)
            if translated is not None:
                candidates.append(translated)
    return candidates


def _resolve_candidates(
    candidates: list[TextRegion],
    proposals: list[_Proposal],
    existing: list[TextRegion],
    page_number: int,
    reader_name: str,
    view: RecoveryView,
    scale: int,
    minimum_confidence: float,
) -> list[TextRegion]:
    recovered = []
    owned_existing = [region for region in existing if _owns_text_pixels(region)]
    for proposal_number, proposal in enumerate(proposals, start=1):
        matching = [
            candidate
            for candidate in candidates
            if _candidate_has_text(candidate)
            and _overlap_by_smaller(candidate.bounding_box, proposal.box) >= 0.3
        ]
        matching.sort(
            key=lambda region: (
                region.bounding_box.top,
                region.bounding_box.left,
                region.id,
            )
        )
        added = 0
        for candidate in matching:
            duplicates_existing = any(
                _boxes_duplicate(candidate.bounding_box, region.bounding_box)
                for region in owned_existing
            )
            duplicates_recovery = any(
                _boxes_duplicate(candidate.bounding_box, region.bounding_box)
                for region in recovered
            )
            if duplicates_existing or duplicates_recovery:
                continue
            provenance = _provenance(
                proposal,
                page_number,
                reader_name,
                view,
                scale,
                candidate,
            )
            resolved = (
                candidate.resolution == "resolved"
                and candidate.confidence is not None
                and candidate.confidence >= minimum_confidence
            )
            alternatives = _candidate_alternatives(candidate, provenance, resolved)
            recovered.append(
                TextRegion(
                    id=(f"p{page_number}-faint-tiny-{proposal_number}-{added + 1}"),
                    kind=candidate.kind,
                    text=candidate.text.strip() if resolved else "",
                    confidence=candidate.confidence if resolved else None,
                    bounding_box=candidate.bounding_box,
                    reading_order=max(
                        (region.reading_order for region in existing), default=0
                    )
                    + proposal_number,
                    provider=candidate.provider,
                    text_provenance=provenance,
                    resolution="resolved" if resolved else "unreadable",
                    alternatives=alternatives,
                    structure={
                        "role": "tiny_text_candidate",
                        "recovery_view": view,
                        "proposal": provenance["proposal"],
                    },
                )
            )
            added += 1
        if added:
            continue
        provenance = _provenance(
            proposal,
            page_number,
            reader_name,
            view,
            scale,
            None,
        )
        recovered.append(
            TextRegion(
                id=f"p{page_number}-faint-tiny-{proposal_number}-1",
                kind="text",
                text="",
                confidence=None,
                bounding_box=proposal.box,
                reading_order=max(
                    (region.reading_order for region in existing), default=0
                )
                + proposal_number,
                provider=reader_name,
                text_provenance=provenance,
                resolution="unreadable",
                structure={
                    "role": "tiny_text_candidate",
                    "recovery_view": view,
                    "proposal": provenance["proposal"],
                },
            )
        )
    return recovered


def _insert_recovered_regions(
    existing: list[TextRegion],
    recovered: list[TextRegion],
) -> list[TextRegion]:
    ordered = [
        region
        for _, region in sorted(
            enumerate(existing),
            key=lambda item: (item[1].reading_order, item[0]),
        )
    ]
    tail_order = max((region.reading_order for region in existing), default=0)
    for region in sorted(recovered, key=_geometric_key):
        position = next(
            (
                index
                for index, current in enumerate(ordered)
                if _geometric_key(region) < _geometric_key(current)
            ),
            len(ordered),
        )
        if position < len(ordered):
            region.reading_order = ordered[position].reading_order
        else:
            tail_order += 1
            region.reading_order = tail_order
        ordered.insert(position, region)
    return ordered


def _candidate_has_text(candidate: TextRegion) -> bool:
    return bool(
        candidate.text.strip()
        or any(alternative.text.strip() for alternative in candidate.alternatives)
    )


def _candidate_alternatives(
    candidate: TextRegion,
    recovery_provenance: dict[str, object],
    resolved: bool,
) -> list[TextAlternative]:
    readings = list(candidate.alternatives)
    if not resolved and candidate.text.strip():
        readings.insert(
            0,
            TextAlternative(
                text=candidate.text.strip(),
                confidence=candidate.confidence,
                provider=candidate.provider,
                text_provenance=copy.deepcopy(candidate.text_provenance),
            ),
        )

    alternatives = []
    seen = set()
    for reading in readings:
        text = reading.text.strip()
        key = (" ".join(text.casefold().split()), reading.provider)
        if not text or key in seen:
            continue
        seen.add(key)
        provenance = copy.deepcopy(reading.text_provenance) or {}
        provenance["faint_tiny_recovery"] = {
            key: copy.deepcopy(value)
            for key, value in recovery_provenance.items()
            if key != "source_text_provenance"
        }
        alternatives.append(
            TextAlternative(
                text=text,
                confidence=reading.confidence,
                provider=reading.provider,
                text_provenance=provenance,
            )
        )
    return alternatives


def _provenance(
    proposal: _Proposal,
    page_number: int,
    reader_name: str,
    view: RecoveryView,
    scale: int,
    candidate: TextRegion | None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "method": "unowned_pixel_proposal_and_high_resolution_reread",
        "page_number": page_number,
        "view": view,
        "scale": 1 if view == "native" else scale,
        "reader": reader_name,
        "proposal": {
            "bounding_box": asdict(proposal.box),
            "component_count": proposal.component_count,
            "ink_pixels": proposal.ink_pixels,
        },
    }
    if candidate is not None:
        value["source_region_id"] = candidate.id
        value["source_provider"] = candidate.provider
        value["source_text_provenance"] = copy.deepcopy(candidate.text_provenance)
    return value


def _translate(
    region: TextRegion,
    view: _View,
    page_size: tuple[int, int],
) -> TextRegion | None:
    box = region.bounding_box
    width, height = page_size
    translated = BoundingBox(
        left=max(0, view.origin.left + math.floor(box.left / view.scale)),
        top=max(0, view.origin.top + math.floor(box.top / view.scale)),
        right=min(width, view.origin.left + math.ceil(box.right / view.scale)),
        bottom=min(height, view.origin.top + math.ceil(box.bottom / view.scale)),
    )
    if not _valid_box(translated, page_size):
        return None
    provenance = dict(region.text_provenance or {})
    provenance["faint_tiny_view"] = {
        "origin": asdict(view.origin),
        "scale": view.scale,
    }
    alternatives = []
    for alternative in region.alternatives:
        alternative_provenance = copy.deepcopy(alternative.text_provenance) or {}
        alternative_provenance["faint_tiny_view"] = {
            "origin": asdict(view.origin),
            "scale": view.scale,
        }
        alternatives.append(
            TextAlternative(
                text=alternative.text,
                confidence=alternative.confidence,
                provider=alternative.provider,
                text_provenance=alternative_provenance,
            )
        )
    return TextRegion(
        id=region.id,
        kind=region.kind,
        text=region.text,
        confidence=region.confidence,
        bounding_box=translated,
        reading_order=region.reading_order,
        provider=region.provider,
        text_provenance=provenance,
        resolution=region.resolution,
        alternatives=alternatives,
        structure=copy.deepcopy(region.structure),
    )


def _reference_height(
    regions: Sequence[TextRegion],
    page_size: tuple[int, int],
) -> int:
    heights = [
        region.bounding_box.bottom - region.bounding_box.top
        for region in regions
        if region.text.strip() and _valid_box(region.bounding_box, page_size)
    ]
    if heights:
        return max(3, round(statistics.median(heights)))
    return max(3, round(min(page_size) * 0.02))


def _text_like_component(
    width: int,
    height: int,
    area: int,
    maximum_height: int,
) -> bool:
    if width < 1 or height < 2 or height > maximum_height or area < 2:
        return False
    if width > height * 4:
        return False
    fill = area / (width * height)
    return 0.05 <= fill <= 0.9


def _component_box(
    components: Sequence[tuple[int, int, int, int, int, int]],
    page_size: tuple[int, int],
    median_height: int,
) -> BoundingBox:
    padding = max(1, round(median_height / 3))
    width, height = page_size
    return BoundingBox(
        max(0, min(item[1] for item in components) - padding),
        max(0, min(item[2] for item in components) - padding),
        min(width, max(item[1] + item[3] for item in components) + padding),
        min(height, max(item[2] + item[4] for item in components) + padding),
    )


def _masks_pixels(region: TextRegion, page_size: tuple[int, int]) -> bool:
    return bool(
        _owns_text_pixels(region) and _valid_box(region.bounding_box, page_size)
    )


def _owns_text_pixels(region: TextRegion) -> bool:
    return bool(
        region.resolution == "resolved"
        and region.kind not in NON_MASKING_KINDS
        and region.text.strip()
    )


def _geometric_key(region: TextRegion) -> tuple[int, int, int, int, str]:
    box = region.bounding_box
    return box.top, box.left, box.bottom, box.right, region.id


def _padded_box(box: BoundingBox, page_size: tuple[int, int]) -> BoundingBox:
    padding = max(2, round((box.bottom - box.top) * 0.75))
    width, height = page_size
    return BoundingBox(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(width, box.right + padding),
        min(height, box.bottom + padding),
    )


def _save_resized(image: Image.Image, path: Path, scale: int) -> None:
    resized = image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.LANCZOS
    )
    try:
        resized.save(path, format="PNG")
    finally:
        resized.close()


def _valid_box(box: BoundingBox, page_size: tuple[int, int]) -> bool:
    width, height = page_size
    return bool(
        0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height
    )


def _overlap_fraction(first: BoundingBox, second: BoundingBox) -> float:
    intersection = _intersection(first, second)
    if intersection == 0:
        return 0.0
    first_area = (first.right - first.left) * (first.bottom - first.top)
    return intersection / first_area


def _overlap_by_smaller(first: BoundingBox, second: BoundingBox) -> float:
    intersection = _intersection(first, second)
    if intersection == 0:
        return 0.0
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    return intersection / min(first_area, second_area)


def _boxes_duplicate(first: BoundingBox, second: BoundingBox) -> bool:
    return _overlap_by_smaller(first, second) >= 0.5


def _intersection(first: BoundingBox, second: BoundingBox) -> int:
    width = max(0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    return width * height


def _odd(value: int) -> int:
    return value if value % 2 else value - 1


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.left, box.top, box.right, box.bottom


def _vision_dependencies() -> tuple[object, object]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise ReaderError(
            "faint_tiny_dependency_missing",
            "Faint-text recovery requires OpenCV and NumPy",
        ) from error
    return cv2, np
