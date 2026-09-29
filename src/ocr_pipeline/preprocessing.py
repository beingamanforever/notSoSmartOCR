"""Evidence-preserving OCR views for framed document screenshots."""

from __future__ import annotations

import re
import statistics
import tempfile
import threading
from collections import Counter, deque
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from math import ceil, floor
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageStat, UnidentifiedImageError

from .contracts import BoundingBox, TextRegion
from .providers import LocalReader, ReaderError, TesseractReader

MAX_LOCATOR_SIZE = 512
DARK_PIXEL = 40
DARK_RATIO = 0.85
AGREEMENT_OVERLAP = 0.75
MIN_PAGE_AREA_RATIO = 0.12
MIN_PAGE_HEIGHT_RATIO = 0.5
MIN_PAGE_WIDTH_RATIO = 0.25
MIN_PAGE_FILL_RATIO = 0.75
MIN_PAGE_MARGIN_RATIO = 0.01
MIN_PAGE_RING_CONTRAST = 25
MIN_PAGE_LIKE_EXTERIOR_BRIGHT_RATIO = 0.9
MIN_PARTIAL_PAGE_HEIGHT_RATIO = 0.08


class PageFrameReader:
    """Run the complete OCR pipeline on one confidently isolated page canvas."""

    def __init__(
        self,
        reader: LocalReader,
        *,
        locator: Callable[[Path], BoundingBox | None] | None = None,
    ) -> None:
        self.reader = reader
        self.name = f"{reader.name}-page-frame"
        self.locator = locator
        self.executable = getattr(reader, "executable", None)
        self.batch_size = 1
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        crop, partial_page = self._locate(image_path)
        assessment: dict[str, object] = {
            "page_number": page_number,
            "status": "not_routed" if crop is None else "frame_isolated",
            "partial_page_visible": partial_page,
            "recovery_status": "not_routed" if crop is None else "pending",
        }
        if crop is not None:
            assessment["crop"] = asdict(crop)
        self._save_assessment(page_number, assessment)
        if crop is None:
            return self.reader.read(image_path, page_number)
        with _cropped_page(image_path, crop, "ocr-page-frame-") as crop_path:
            regions = self.reader.read(crop_path, page_number)
        self._update_assessment(
            page_number,
            cropped_reader_review=_reader_needs_review(self.reader, page_number),
            cropped_reader=_reader_page_assessment(self.reader, page_number),
        )
        return regions

    @contextmanager
    def stage_view(self, image_path: Path, page_number: int) -> Iterator[Path]:
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        crop = _assessment_box(assessment.get("crop"))
        if crop is None:
            with _reader_stage_view(self.reader, image_path, page_number) as path:
                yield path
            return
        with _cropped_page(image_path, crop, "ocr-page-stage-") as crop_path:
            with _reader_stage_view(self.reader, crop_path, page_number) as path:
                yield path

    def restore_regions(
        self,
        regions: list[TextRegion],
        page_number: int,
    ) -> list[TextRegion]:
        restore = getattr(self.reader, "restore_regions", None)
        restored = restore(regions, page_number) if callable(restore) else regions
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        crop = _assessment_box(assessment.get("crop"))
        if crop is None:
            return restored
        return [_translate_page_region(region, crop) for region in restored]

    def recover_regions(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        crop = _assessment_box(assessment.get("crop"))
        if crop is None:
            return regions

        full_page = self.reader.read(image_path, page_number)
        restore = getattr(self.reader, "restore_regions", None)
        if callable(restore):
            full_page = restore(full_page, page_number)
        with Image.open(image_path) as source:
            page_size = source.size

        recovered, recovery = _recover_outside_frame(
            full_page,
            regions,
            crop,
            page_number,
            page_size,
            self.reader.name,
        )
        self._update_assessment(
            page_number,
            recovery_status=("recovered" if recovered else "no_outside_evidence"),
            full_page_reader_review=_reader_needs_review(self.reader, page_number),
            **recovery,
        )
        return _page_reading_order([*regions, *recovered])

    def record_recovery_failure(self, page_number: int, error: ReaderError) -> None:
        self._update_assessment(
            page_number,
            recovery_status="failed",
            recovery_failure={"code": error.code, "message": str(error)},
        )

    def page_needs_review(self, page_number: int) -> bool:
        nested_review = getattr(self.reader, "page_needs_review", None)
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        return bool(
            assessment.get("status") == "frame_isolated"
            or assessment.get("partial_page_visible")
            or assessment.get("cropped_reader_review")
            or assessment.get("full_page_reader_review")
            or (callable(nested_review) and nested_review(page_number))
        )

    def coverage_assessment(self, page_count: int) -> dict[str, object]:
        with self._lock:
            pages = [
                dict(
                    self._assessments.get(
                        page_number,
                        {"page_number": page_number, "status": "not_run"},
                    )
                )
                for page_number in range(1, page_count + 1)
            ]
        isolated = sum(page.get("status") == "frame_isolated" for page in pages)
        partial = sum(bool(page.get("partial_page_visible")) for page in pages)
        nested = getattr(self.reader, "coverage_assessment", None)
        nested_assessment = nested(page_count) if callable(nested) else None
        recovered = sum(int(page.get("recovered_regions", 0)) for page in pages)
        recovery_failures = sum(
            page.get("recovery_status") == "failed" for page in pages
        )
        status = "review_recommended" if isolated else "not_assessed"
        message = (
            f"Isolated {isolated} document canvas(es), recovered {recovered} "
            f"outside-frame region(s), and observed {recovery_failures} recovery "
            f"failure(s); {partial} contained a visible partial neighboring page."
            if isolated
            else "No single embedded document canvas was isolated."
        )
        result: dict[str, object] = {
            "status": status,
            "message": message,
            "pages": pages,
        }
        if isinstance(nested_assessment, dict):
            result["nested_reader"] = nested_assessment
        return result

    def _locate(self, image_path: Path) -> tuple[BoundingBox | None, bool]:
        if self.locator is not None:
            return self.locator(image_path), False
        return _locate_document_frame(image_path)

    def _save_assessment(self, page_number: int, value: dict[str, object]) -> None:
        with self._lock:
            self._assessments[page_number] = value

    def _update_assessment(self, page_number: int, **values: object) -> None:
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
            assessment.update(values)
            self._assessments[page_number] = assessment


class RoutedTesseractReader:
    """Use a locally adaptive OCR view only inside a detected document frame."""

    name = "tesseract-routed"

    def __init__(
        self,
        *,
        baseline: LocalReader | None = None,
        enhanced: LocalReader | None = None,
        locator: Callable[[Path], BoundingBox | None] | None = None,
    ) -> None:
        self.baseline = baseline or TesseractReader(page_segmentation_mode=3)
        self.enhanced = enhanced or TesseractReader(
            page_segmentation_mode=3,
            thresholding_method=2,
        )
        self.locator = locator or locate_dark_frame
        self.executable = getattr(self.baseline, "executable", None)
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        crop = self.locator(image_path)
        if crop is None:
            regions = self.baseline.read(image_path, page_number)
            self._save_assessment(
                page_number,
                {
                    "page_number": page_number,
                    "status": "not_assessed",
                    "selected_view": "baseline",
                    "reason": "No framed document canvas was detected",
                },
            )
            return regions

        try:
            with Image.open(image_path) as source:
                with tempfile.TemporaryDirectory(prefix="ocr-view-") as directory:
                    crop_path = Path(directory) / "document.png"
                    source.crop((crop.left, crop.top, crop.right, crop.bottom)).save(
                        crop_path, format="PNG"
                    )
                    baseline = self.baseline.read(crop_path, page_number)
                    enhanced = self.enhanced.read(crop_path, page_number)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError("preprocess_failed", str(error)) from error

        selected, assessment = _select_view(baseline, enhanced)
        assessment.update(
            {
                "page_number": page_number,
                "crop": asdict(crop),
            }
        )
        self._save_assessment(page_number, assessment)
        return _translate_regions(selected, crop, assessment["selected_view"])

    def coverage_assessment(self, page_count: int) -> dict[str, object]:
        with self._lock:
            pages = [
                dict(self._assessments.get(number, {}))
                for number in range(1, page_count + 1)
            ]
        reviewed = [page for page in pages if page.get("status") != "not_assessed"]
        if not reviewed:
            return {
                "status": "not_assessed",
                "message": "Extraction completeness was not independently assessed.",
                "pages": pages,
            }
        selected = sum(page.get("selected_view") == "enhanced" for page in reviewed)
        return {
            "status": "review_recommended",
            "message": (
                f"A framed document view was assessed on {len(reviewed)} page(s); "
                f"the enhanced view was selected on {selected}. Review remains "
                "recommended because both views use the same OCR engine."
            ),
            "pages": pages,
        }

    def _save_assessment(self, page_number: int, value: dict[str, object]) -> None:
        with self._lock:
            self._assessments[page_number] = value


def _locate_document_frame(image_path: Path) -> tuple[BoundingBox | None, bool]:
    dark_frame = locate_dark_frame(image_path)
    if dark_frame is not None:
        return dark_frame, False
    try:
        with Image.open(image_path) as source:
            width, height = source.size
            if width < 80 or height < 80:
                return None, False
            grayscale = source.convert("L")
            scale = min(1.0, MAX_LOCATOR_SIZE / max(width, height))
            reduced = grayscale.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BOX,
            )
    except (OSError, UnidentifiedImageError):
        return None, False

    small_width, small_height = reduced.size
    pixels = list(reduced.getdata())
    components = _bright_components(pixels, small_width, small_height)
    candidates = [
        component
        for component in components
        if _complete_page_candidate(component, pixels, small_width, small_height)
    ]
    if not candidates:
        return None, False
    selected = max(candidates, key=lambda item: item["area"])
    if any(
        candidate is not selected and _similar_page(candidate, selected)
        for candidate in candidates
    ):
        return None, False

    exterior = [
        pixels[y * small_width + x]
        for y in range(small_height)
        for x in range(small_width)
        if not (
            selected["left"] <= x <= selected["right"]
            and selected["top"] <= y <= selected["bottom"]
        )
    ]
    # A printed border can isolate bright paper just like a viewer canvas.
    if exterior and statistics.mean(value >= 230 for value in exterior) >= (
        MIN_PAGE_LIKE_EXTERIOR_BRIGHT_RATIO
    ):
        return None, False

    partial_page = any(
        _partial_page_candidate(
            component,
            selected,
            pixels,
            small_width,
            small_height,
        )
        for component in components
        if component is not selected
    )
    scale_x = width / small_width
    scale_y = height / small_height
    box = BoundingBox(
        left=max(0, floor(int(selected["left"]) * scale_x) - 1),
        top=max(0, floor(int(selected["top"]) * scale_y) - 1),
        right=min(width, ceil((int(selected["right"]) + 1) * scale_x) + 1),
        bottom=min(height, ceil((int(selected["bottom"]) + 1) * scale_y) + 1),
    )
    return box, partial_page


def _bright_components(
    pixels: list[int],
    width: int,
    height: int,
) -> list[dict[str, int]]:
    bright = bytearray(value >= 230 for value in pixels)
    visited = bytearray(width * height)
    components: list[dict[str, int]] = []
    for start, is_bright in enumerate(bright):
        if not is_bright or visited[start]:
            continue
        visited[start] = 1
        pending = deque([start])
        left = right = start % width
        top = bottom = start // width
        count = 0
        while pending:
            index = pending.popleft()
            x = index % width
            y = index // width
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
            count += 1
            neighbors = []
            if x:
                neighbors.append(index - 1)
            if x + 1 < width:
                neighbors.append(index + 1)
            if y:
                neighbors.append(index - width)
            if y + 1 < height:
                neighbors.append(index + width)
            for neighbor in neighbors:
                if bright[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    pending.append(neighbor)
        box_area = (right - left + 1) * (bottom - top + 1)
        components.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "pixels": count,
                "area": box_area,
            }
        )
    return components


def _complete_page_candidate(
    component: dict[str, int],
    pixels: list[int],
    width: int,
    height: int,
) -> bool:
    component_width = component["right"] - component["left"] + 1
    component_height = component["bottom"] - component["top"] + 1
    margin_x = width * MIN_PAGE_MARGIN_RATIO
    margin_y = height * MIN_PAGE_MARGIN_RATIO
    return bool(
        component["area"] >= width * height * MIN_PAGE_AREA_RATIO
        and component_width >= width * MIN_PAGE_WIDTH_RATIO
        and component_height >= height * MIN_PAGE_HEIGHT_RATIO
        and component["pixels"] / component["area"] >= MIN_PAGE_FILL_RATIO
        and component["left"] >= margin_x
        and component["right"] <= width - margin_x
        and component["top"] >= margin_y
        and component["bottom"] <= height - margin_y
        and _ring_contrast(component, pixels, width, height) >= MIN_PAGE_RING_CONTRAST
    )


def _partial_page_candidate(
    component: dict[str, int],
    selected: dict[str, int],
    pixels: list[int],
    width: int,
    height: int,
) -> bool:
    component_width = component["right"] - component["left"] + 1
    component_height = component["bottom"] - component["top"] + 1
    selected_width = selected["right"] - selected["left"] + 1
    horizontal_overlap = max(
        0,
        min(component["right"], selected["right"])
        - max(component["left"], selected["left"])
        + 1,
    )
    touches_edge = component["top"] <= 1 or component["bottom"] >= height - 2
    return bool(
        touches_edge
        and component_height >= height * MIN_PARTIAL_PAGE_HEIGHT_RATIO
        and 0.8 <= component_width / selected_width <= 1.2
        and horizontal_overlap / min(component_width, selected_width) >= 0.8
        and component["pixels"] / component["area"] >= MIN_PAGE_FILL_RATIO
        and _ring_contrast(component, pixels, width, height) >= MIN_PAGE_RING_CONTRAST
    )


def _ring_contrast(
    component: dict[str, int],
    pixels: list[int],
    width: int,
    height: int,
) -> float:
    left = component["left"]
    top = component["top"]
    right = component["right"]
    bottom = component["bottom"]
    ring = []
    if top:
        ring.extend(pixels[(top - 1) * width + left : (top - 1) * width + right + 1])
    if bottom + 1 < height:
        ring.extend(
            pixels[(bottom + 1) * width + left : (bottom + 1) * width + right + 1]
        )
    if left:
        ring.extend(pixels[y * width + left - 1] for y in range(top, bottom + 1))
    if right + 1 < width:
        ring.extend(pixels[y * width + right + 1] for y in range(top, bottom + 1))
    if not ring:
        return 0.0
    inside = [
        pixels[y * width + x]
        for y in range(top, bottom + 1)
        for x in range(left, right + 1)
    ]
    return statistics.mean(inside) - statistics.mean(ring)


def _similar_page(left: dict[str, int], right: dict[str, int]) -> bool:
    left_width = left["right"] - left["left"] + 1
    right_width = right["right"] - right["left"] + 1
    left_height = left["bottom"] - left["top"] + 1
    right_height = right["bottom"] - right["top"] + 1
    return bool(
        0.8 <= left_width / right_width <= 1.2
        and 0.8 <= left_height / right_height <= 1.2
    )


@contextmanager
def _cropped_page(
    image_path: Path,
    crop: BoundingBox,
    prefix: str,
) -> Iterator[Path]:
    try:
        with Image.open(image_path) as source:
            with tempfile.TemporaryDirectory(prefix=prefix) as directory:
                path = Path(directory) / "page.png"
                source.crop((crop.left, crop.top, crop.right, crop.bottom)).save(
                    path,
                    format="PNG",
                )
                yield path
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ReaderError("page_frame_failed", str(error)) from error


@contextmanager
def _reader_stage_view(
    reader: LocalReader,
    image_path: Path,
    page_number: int,
) -> Iterator[Path]:
    stage_view = getattr(reader, "stage_view", None)
    if callable(stage_view):
        with stage_view(image_path, page_number) as path:
            yield path
        return
    yield image_path


def _assessment_box(value: object) -> BoundingBox | None:
    if not isinstance(value, dict):
        return None
    keys = ("left", "top", "right", "bottom")
    if not all(isinstance(value.get(key), int) for key in keys):
        return None
    return BoundingBox(*(int(value[key]) for key in keys))


def _translate_page_region(region: TextRegion, crop: BoundingBox) -> TextRegion:
    provenance = dict(region.text_provenance or {})
    provenance["source_crop"] = asdict(crop)
    return replace(
        region,
        bounding_box=_translate_page_box(region.bounding_box, crop),
        text_provenance=provenance,
        alternatives=list(region.alternatives),
        structure=_translate_page_structure(region.structure, crop),
    )


def _translate_page_structure(value: Any, crop: BoundingBox) -> Any:
    if isinstance(value, list):
        return [_translate_page_structure(item, crop) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"left", "top", "right", "bottom"} and all(
        isinstance(value[key], int) and not isinstance(value[key], bool)
        for key in value
    ):
        translated = _translate_page_box(BoundingBox(**value), crop)
        return asdict(translated)
    return {key: _translate_page_structure(item, crop) for key, item in value.items()}


def _translate_page_box(box: BoundingBox, crop: BoundingBox) -> BoundingBox:
    return BoundingBox(
        left=box.left + crop.left,
        top=box.top + crop.top,
        right=box.right + crop.left,
        bottom=box.bottom + crop.top,
    )


def _recover_outside_frame(
    candidates: list[TextRegion],
    canonical: list[TextRegion],
    crop: BoundingBox,
    page_number: int,
    page_size: tuple[int, int],
    reader_name: str,
) -> tuple[list[TextRegion], dict[str, int]]:
    recovered: list[TextRegion] = []
    inside_frame = 0
    invalid = 0
    duplicates = 0
    for candidate in candidates:
        if not _valid_page_region(candidate, page_size):
            invalid += 1
            continue
        if _boxes_intersect(candidate.bounding_box, crop):
            inside_frame += 1
            continue
        if any(
            _same_region(candidate, existing) for existing in [*canonical, *recovered]
        ):
            duplicates += 1
            continue
        provenance = dict(candidate.text_provenance or {})
        provenance["page_frame_recovery"] = {
            "method": "residual_full_original",
            "source_region_id": candidate.id,
            "source_reader": reader_name,
            "isolated_frame": asdict(crop),
        }
        recovered.append(
            replace(
                candidate,
                id=f"p{page_number}-frame-recovery-{len(recovered) + 1}",
                text_provenance=provenance,
                alternatives=list(candidate.alternatives),
                structure=deepcopy(candidate.structure),
            )
        )
    return recovered, {
        "full_page_regions": len(candidates),
        "inside_frame_regions": inside_frame,
        "invalid_full_page_regions": invalid,
        "duplicate_full_page_regions": duplicates,
        "recovered_regions": len(recovered),
        "preserved_canonical_regions": len(canonical),
    }


def _valid_page_region(region: TextRegion, page_size: tuple[int, int]) -> bool:
    width, height = page_size
    box = region.bounding_box
    return bool(
        region.resolution == "resolved"
        and region.text.strip()
        and 0 <= box.left < box.right <= width
        and 0 <= box.top < box.bottom <= height
    )


def _boxes_intersect(first: BoundingBox, second: BoundingBox) -> bool:
    return bool(
        min(first.right, second.right) > max(first.left, second.left)
        and min(first.bottom, second.bottom) > max(first.top, second.top)
    )


def _same_region(first: TextRegion, second: TextRegion) -> bool:
    return bool(
        " ".join(first.text.casefold().split())
        == " ".join(second.text.casefold().split())
        and _overlap_ratio(first.bounding_box, second.bounding_box) >= AGREEMENT_OVERLAP
    )


def _overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    first_area = _box_area(first)
    second_area = _box_area(second)
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller else 0.0


def _box_area(box: BoundingBox) -> int:
    return max(0, box.right - box.left) * max(0, box.bottom - box.top)


def _page_reading_order(regions: list[TextRegion]) -> list[TextRegion]:
    ordered = sorted(
        regions,
        key=lambda region: (
            region.bounding_box.top,
            region.bounding_box.left,
            region.reading_order,
            region.id,
        ),
    )
    return [
        replace(region, reading_order=order) for order, region in enumerate(ordered, 1)
    ]


def locate_dark_frame(image_path: Path) -> BoundingBox | None:
    """Locate a bright document canvas between persistent dark side bands."""
    try:
        with Image.open(image_path) as source:
            width, height = source.size
            if width < 80 or height < 80:
                return None
            grayscale = source.convert("L")
            scale = min(1.0, MAX_LOCATOR_SIZE / max(width, height))
            small_width = max(1, round(width * scale))
            small_height = max(1, round(height * scale))
            reduced = grayscale.resize(
                (small_width, small_height), Image.Resampling.BOX
            )
    except (OSError, UnidentifiedImageError):
        return None

    pixels = list(reduced.getdata())
    probe_top = small_height // 4
    probe_height = small_height - probe_top
    dark_columns = []
    for x in range(small_width):
        dark = sum(
            pixels[(y * small_width) + x] <= DARK_PIXEL
            for y in range(probe_top, small_height)
        )
        if dark / probe_height >= DARK_RATIO:
            dark_columns.append(x)

    bands = _runs(dark_columns)
    minimum_band = max(2, round(small_width * 0.01))
    left_bands = [
        band
        for band in bands
        if band[1] - band[0] + 1 >= minimum_band and band[1] <= small_width * 0.25
    ]
    right_bands = [
        band
        for band in bands
        if band[1] - band[0] + 1 >= minimum_band and band[0] >= small_width * 0.75
    ]
    pairs = [
        (left, right)
        for left in left_bands
        for right in right_bands
        if right[0] - left[1] >= small_width * 0.5
    ]
    if not pairs:
        return None
    left, right = max(
        pairs,
        key=lambda pair: (pair[0][1] - pair[0][0]) + (pair[1][1] - pair[1][0]),
    )

    framed_rows = []
    for y in range(small_height):
        row = pixels[y * small_width : (y + 1) * small_width]
        left_ratio = _dark_ratio(row[left[0] : left[1] + 1])
        right_ratio = _dark_ratio(row[right[0] : right[1] + 1])
        if left_ratio >= DARK_RATIO and right_ratio >= DARK_RATIO:
            framed_rows.append(y)
    row_runs = _runs(framed_rows)
    if not row_runs:
        return None
    top, bottom = max(row_runs, key=lambda run: run[1] - run[0])
    if bottom - top + 1 < small_height * 0.35:
        return None
    if bottom >= small_height * 0.75:
        bottom = small_height - 1

    inner_left = left[1] + 1
    inner_right = right[0]
    canvas = reduced.crop((inner_left, top, inner_right, bottom + 1))
    if ImageStat.Stat(canvas).mean[0] < 150:
        return None

    scale_x = width / small_width
    scale_y = height / small_height
    box = BoundingBox(
        left=round(inner_left * scale_x),
        top=round(top * scale_y),
        right=round(inner_right * scale_x),
        bottom=min(height, round((bottom + 1) * scale_y)),
    )
    if box.right - box.left < width * 0.5 or box.bottom - box.top < height * 0.35:
        return None
    return box


def _select_view(
    baseline: list[TextRegion], enhanced: list[TextRegion]
) -> tuple[list[TextRegion], dict[str, object]]:
    baseline_tokens = _token_counts(baseline)
    enhanced_tokens = _token_counts(enhanced)
    baseline_count = sum(baseline_tokens.values())
    enhanced_count = sum(enhanced_tokens.values())
    retained = sum((baseline_tokens & enhanced_tokens).values())
    baseline_recall = retained / baseline_count if baseline_count else 0.0
    added = sum((enhanced_tokens - baseline_tokens).values())
    removed = sum((baseline_tokens - enhanced_tokens).values())
    baseline_confidence = _mean_confidence(baseline)
    enhanced_confidence = _mean_confidence(enhanced)
    selects_enhanced = (
        baseline_count > 0
        and baseline_recall >= 0.98
        and added >= 2
        and enhanced_count <= baseline_count * 1.25
        and enhanced_confidence >= baseline_confidence - 0.01
    )
    selected_view = "enhanced" if selects_enhanced else "baseline"
    status = "recovered" if selects_enhanced else "uncertain"
    return (
        enhanced if selects_enhanced else baseline,
        {
            "status": status,
            "selected_view": selected_view,
            "baseline_tokens": baseline_count,
            "enhanced_tokens": enhanced_count,
            "baseline_token_recall": round(baseline_recall, 6),
            "added_tokens": added,
            "removed_tokens": removed,
            "baseline_mean_confidence": round(baseline_confidence, 6),
            "enhanced_mean_confidence": round(enhanced_confidence, 6),
        },
    )


def _translate_regions(
    regions: list[TextRegion], crop: BoundingBox, selected_view: object
) -> list[TextRegion]:
    translated = []
    for region in regions:
        provenance = dict(region.text_provenance or {})
        provenance.update(
            {
                "source_crop": asdict(crop),
                "selected_view": selected_view,
            }
        )
        box = region.bounding_box
        translated.append(
            TextRegion(
                id=region.id,
                kind=region.kind,
                text=region.text,
                confidence=region.confidence,
                bounding_box=BoundingBox(
                    left=box.left + crop.left,
                    top=box.top + crop.top,
                    right=box.right + crop.left,
                    bottom=box.bottom + crop.top,
                ),
                reading_order=region.reading_order,
                provider=region.provider,
                text_provenance=provenance,
                resolution=region.resolution,
                alternatives=list(region.alternatives),
                structure=region.structure,
            )
        )
    return translated


def _reader_needs_review(reader: LocalReader, page_number: int) -> bool:
    check = getattr(reader, "page_needs_review", None)
    return bool(callable(check) and check(page_number))


def _reader_page_assessment(
    reader: LocalReader,
    page_number: int,
) -> dict[str, object] | None:
    coverage = getattr(reader, "coverage_assessment", None)
    if not callable(coverage):
        return None
    assessment = coverage(page_number)
    if not isinstance(assessment, dict):
        return None
    pages = assessment.get("pages")
    if not isinstance(pages, list) or len(pages) < page_number:
        return None
    page = pages[page_number - 1]
    return deepcopy(page) if isinstance(page, dict) else None


def _token_counts(regions: list[TextRegion]) -> Counter[str]:
    return Counter(
        token
        for region in regions
        for token in re.findall(r"[a-z0-9%$]+", region.text.casefold())
    )


def _mean_confidence(regions: list[TextRegion]) -> float:
    values = [region.confidence for region in regions if region.confidence is not None]
    return sum(values) / len(values) if values else 0.0


def _dark_ratio(values: list[int]) -> float:
    return sum(value <= DARK_PIXEL for value in values) / len(values) if values else 0.0


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs
