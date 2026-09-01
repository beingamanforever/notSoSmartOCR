"""Evidence-preserving OCR views for framed document screenshots."""

from __future__ import annotations

import re
import statistics
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

from .contracts import BoundingBox, TextAlternative, TextRegion
from .providers import LocalReader, ReaderError, TesseractReader

MAX_LOCATOR_SIZE = 512
DARK_PIXEL = 40
DARK_RATIO = 0.85
TILE_COUNT = 3
MATCH_OVERLAP = 0.5
AGREEMENT_OVERLAP = 0.75
AGREEMENT_CONFIDENCE = 0.85
MIN_AREA_RATIO = 0.35


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


class TiledReader:
    """Rerun tiny text in overlapping tiles without replacing baseline evidence."""

    def __init__(
        self,
        reader: LocalReader,
        *,
        maximum_median_height: int = 10,
        tile_count: int = TILE_COUNT,
        overlap_fraction: float = 0.06,
    ) -> None:
        if maximum_median_height < 1:
            raise ValueError("maximum_median_height must be positive")
        if tile_count < 2:
            raise ValueError("tile_count must be at least 2")
        if not 0 <= overlap_fraction < 0.25:
            raise ValueError("overlap_fraction must be from 0 to 0.25")
        self.reader = reader
        self.name = f"{reader.name}-tiled"
        self.maximum_median_height = maximum_median_height
        self.tile_count = tile_count
        self.overlap_fraction = overlap_fraction
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        baseline = self.reader.read(image_path, page_number)
        heights = [
            region.bounding_box.bottom - region.bounding_box.top for region in baseline
        ]
        median_height = statistics.median(heights) if heights else 0
        if median_height > self.maximum_median_height:
            self._save_assessment(
                page_number,
                {
                    "page_number": page_number,
                    "status": "not_routed",
                    "selected_view": "baseline",
                    "median_region_height": median_height,
                },
            )
            return baseline

        try:
            with Image.open(image_path) as source:
                width, height = source.size
                tiled = self._read_tiles(source, width, height, page_number)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError("preprocess_failed", str(error)) from error

        selected, assessment = _fuse_tiled_view(baseline, tiled)
        assessment.update(
            {
                "page_number": page_number,
                "median_region_height": median_height,
                "tile_count": self.tile_count,
                "overlap_fraction": self.overlap_fraction,
            }
        )
        self._save_assessment(page_number, assessment)
        return selected

    def coverage_assessment(self, page_count: int) -> dict[str, object]:
        with self._lock:
            pages = [
                dict(self._assessments.get(number, {}))
                for number in range(1, page_count + 1)
            ]
        routed = [page for page in pages if page.get("status") != "not_routed"]
        if not routed:
            return {
                "status": "not_assessed",
                "message": "No tiny-text tile route was triggered.",
                "pages": pages,
            }
        preserved = sum(
            int(page.get("preserved_baseline_regions", 0)) for page in routed
        )
        promoted = sum(
            int(page.get("promoted_tile_only_regions", 0)) for page in routed
        )
        conflicts = sum(int(page.get("conflicting_candidates", 0)) for page in routed)
        unresolved = sum(
            int(page.get("unresolved_tile_only_regions", 0)) for page in routed
        )
        return {
            "status": "review_recommended",
            "message": (
                f"Tiny-text fusion assessed {len(routed)} page(s), preserved "
                f"{preserved} baseline region(s), promoted {promoted} agreed tile-only "
                f"region(s), exposed {conflicts} conflict(s), and kept {unresolved} "
                "unsupported region(s) for review."
            ),
            "pages": pages,
        }

    def _read_tiles(
        self,
        source: Image.Image,
        width: int,
        height: int,
        page_number: int,
    ) -> list[TextRegion]:
        overlap = round(height * self.overlap_fraction)
        regions = []
        with tempfile.TemporaryDirectory(prefix="ocr-tiles-") as directory:
            for tile_number in range(self.tile_count):
                core_top = round(tile_number * height / self.tile_count)
                core_bottom = round((tile_number + 1) * height / self.tile_count)
                crop_top = max(0, core_top - overlap)
                crop_bottom = min(height, core_bottom + overlap)
                tile_path = Path(directory) / f"tile-{tile_number + 1}.png"
                source.crop((0, crop_top, width, crop_bottom)).save(
                    tile_path, format="PNG"
                )
                tile_regions = self.reader.read(tile_path, page_number)
                for region in tile_regions:
                    regions.append(
                        _translate_tile_region(
                            region,
                            crop_top,
                            tile_number + 1,
                            len(regions) + 1,
                            page_number,
                            self.name,
                        )
                    )
        return regions

    def _save_assessment(self, page_number: int, value: dict[str, object]) -> None:
        with self._lock:
            self._assessments[page_number] = value


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


def _fuse_tiled_view(
    baseline: list[TextRegion], tiled: list[TextRegion]
) -> tuple[list[TextRegion], dict[str, object]]:
    fused = [
        replace(region, alternatives=list(region.alternatives)) for region in baseline
    ]
    unmatched: list[TextRegion] = []
    exact_matches = 0
    conflicts = 0

    for candidate in tiled:
        match = _best_overlap(candidate, fused[: len(baseline)])
        if match is None:
            unmatched.append(candidate)
            continue

        index, _ = match
        target = fused[index]
        target.alternatives.append(_as_alternative(candidate))
        if _normalized_text(target.text) == _normalized_text(candidate.text):
            exact_matches += 1
            continue
        conflicts += 1

    tile_only, promoted, unresolved = _resolve_tile_only(unmatched)
    fused.extend(tile_only)

    baseline_tokens = _token_counts(baseline)
    tiled_tokens = _token_counts(tiled)
    baseline_count = sum(baseline_tokens.values())
    tiled_count = sum(tiled_tokens.values())
    resolved_tile_only = [
        region for region in tile_only if region.resolution == "resolved"
    ]
    added = sum(_token_counts(resolved_tile_only).values())
    baseline_confidence = _mean_confidence(baseline)
    tiled_confidence = _mean_confidence(tiled)
    status = "recovered" if promoted else "uncertain"
    return (
        fused,
        {
            "status": status,
            "selected_view": "fused",
            "baseline_regions": len(baseline),
            "preserved_baseline_regions": len(baseline),
            "tiled_candidates": len(tiled),
            "exact_overlap_candidates": exact_matches,
            "conflicting_candidates": conflicts,
            "tile_only_candidates": len(unmatched),
            "promoted_tile_only_regions": promoted,
            "unresolved_tile_only_regions": unresolved,
            "output_regions": len(fused),
            "baseline_tokens": baseline_count,
            "tiled_tokens": tiled_count,
            "baseline_token_recall": 1.0,
            "added_tokens": added,
            "removed_tokens": 0,
            "baseline_mean_confidence": round(baseline_confidence, 6),
            "tiled_mean_confidence": round(tiled_confidence, 6),
        },
    )


def _resolve_tile_only(
    candidates: list[TextRegion],
) -> tuple[list[TextRegion], int, int]:
    groups: list[list[TextRegion]] = []
    for candidate in candidates:
        group = next(
            (
                existing
                for existing in groups
                if all(
                    _boxes_match(
                        candidate.bounding_box,
                        item.bounding_box,
                        AGREEMENT_OVERLAP,
                    )
                    for item in existing
                )
            ),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)

    resolved = []
    promoted = 0
    unresolved = 0
    for group in groups:
        representative = max(group, key=lambda region: region.confidence or 0.0)
        alternatives = [
            _as_alternative(region) for region in group if region is not representative
        ]
        texts = {_normalized_text(region.text) for region in group}
        views = {(region.text_provenance or {}).get("tile_number") for region in group}
        confident = all(
            (region.confidence or 0.0) >= AGREEMENT_CONFIDENCE for region in group
        )
        agrees = len(group) >= 2 and len(views) >= 2 and len(texts) == 1 and confident
        if agrees:
            resolution = "resolved"
            promoted += 1
        elif len(texts) > 1:
            resolution = "conflicting"
            unresolved += 1
        else:
            resolution = "unreadable"
            unresolved += 1
        resolved.append(
            replace(
                representative,
                resolution=resolution,
                alternatives=alternatives,
                structure={
                    "role": "tiny_text_candidate",
                    "support_views": len(views),
                },
            )
        )
    return resolved, promoted, unresolved


def _best_overlap(
    candidate: TextRegion, regions: list[TextRegion]
) -> tuple[int, float] | None:
    matches = [
        (index, _overlap_ratio(candidate.bounding_box, region.bounding_box))
        for index, region in enumerate(regions)
        if candidate.kind == region.kind
        and _boxes_match(candidate.bounding_box, region.bounding_box, MATCH_OVERLAP)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[1])


def _boxes_match(first: BoundingBox, second: BoundingBox, overlap: float) -> bool:
    first_area = _box_area(first)
    second_area = _box_area(second)
    larger = max(first_area, second_area)
    area_ratio = min(first_area, second_area) / larger if larger else 0.0
    return area_ratio >= MIN_AREA_RATIO and _overlap_ratio(first, second) >= overlap


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


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9%$]+", text.casefold()))


def _as_alternative(region: TextRegion) -> TextAlternative:
    return TextAlternative(
        text=region.text,
        confidence=region.confidence,
        provider=region.provider,
        text_provenance=dict(region.text_provenance or {}),
    )


def _translate_tile_region(
    region: TextRegion,
    crop_top: int,
    tile_number: int,
    order: int,
    page_number: int,
    stage_name: str,
) -> TextRegion:
    provenance = dict(region.text_provenance or {})
    provenance.update(
        {
            "stage": stage_name,
            "tile_number": tile_number,
            "tile_top": crop_top,
        }
    )
    box = region.bounding_box
    return TextRegion(
        id=f"p{page_number}-tiny-tile-{order}",
        kind=region.kind,
        text=region.text,
        confidence=region.confidence,
        bounding_box=BoundingBox(
            left=box.left,
            top=box.top + crop_top,
            right=box.right,
            bottom=box.bottom + crop_top,
        ),
        reading_order=order,
        provider=region.provider,
        text_provenance=provenance,
        resolution=region.resolution,
        alternatives=list(region.alternatives),
        structure=region.structure,
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
