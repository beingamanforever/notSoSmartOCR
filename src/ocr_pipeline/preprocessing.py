"""Evidence-preserving OCR views for framed document screenshots."""

from __future__ import annotations

import re
import statistics
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from math import ceil, floor
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
WIDE_BAND_SCALE = 3
CONFIRMATION_RECALL = 0.98
CONFIRMATION_SIMILARITY = 0.98


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
        with tempfile.TemporaryDirectory(prefix="ocr-tiles-") as directory:
            tiles: list[tuple[Path, int, int]] = []
            for tile_number in range(self.tile_count):
                core_top = round(tile_number * height / self.tile_count)
                core_bottom = round((tile_number + 1) * height / self.tile_count)
                crop_top = max(0, core_top - overlap)
                crop_bottom = min(height, core_bottom + overlap)
                tile_path = Path(directory) / f"tile-{tile_number + 1}.png"
                source.crop((0, crop_top, width, crop_bottom)).save(
                    tile_path, format="PNG"
                )
                tiles.append((tile_path, crop_top, tile_number + 1))

            read_batch = getattr(self.reader, "read_batch", None)
            if callable(read_batch) and getattr(self.reader, "batch_size", 1) > 1:
                results = read_batch(
                    [path for path, _, _ in tiles],
                    [page_number] * len(tiles),
                )
                if not isinstance(results, list) or len(results) != len(tiles):
                    raise ReaderError(
                        "invalid_batch_output",
                        "Tile reader returned the wrong number of results",
                    )
            else:
                results = [
                    self.reader.read(tile_path, page_number)
                    for tile_path, _, _ in tiles
                ]

            regions = []
            for (tile_path, crop_top, tile_number), tile_regions in zip(
                tiles,
                results,
                strict=True,
            ):
                if isinstance(tile_regions, ReaderError):
                    raise tile_regions
                if not isinstance(tile_regions, list) or not all(
                    isinstance(region, TextRegion) for region in tile_regions
                ):
                    raise ReaderError(
                        "invalid_batch_output",
                        f"Tile reader returned an invalid result for {tile_path.name}",
                    )
                for region in tile_regions:
                    regions.append(
                        _translate_tile_region(
                            region,
                            crop_top,
                            tile_number,
                            len(regions) + 1,
                            page_number,
                            self.name,
                        )
                    )
            return regions

    def _save_assessment(self, page_number: int, value: dict[str, object]) -> None:
        with self._lock:
            self._assessments[page_number] = value


class WideBandFallbackReader:
    """Reread only shallow, page-wide, low-confidence text bands."""

    def __init__(
        self,
        reader: LocalReader,
        fallback_reader: LocalReader,
        *,
        confirmation_reader: LocalReader | None = None,
        low_confidence: float = 0.9,
        minimum_fallback_confidence: float = 0.8,
        minimum_width_fraction: float = 0.65,
        maximum_height_fraction: float = 0.08,
        crop_padding: int = 8,
    ) -> None:
        if not 0 <= low_confidence <= 1:
            raise ValueError("low_confidence must be from 0 to 1")
        if not 0 <= minimum_fallback_confidence <= 1:
            raise ValueError("minimum_fallback_confidence must be from 0 to 1")
        if not 0 < minimum_width_fraction <= 1:
            raise ValueError("minimum_width_fraction must be from 0 to 1")
        if not 0 < maximum_height_fraction < 1:
            raise ValueError("maximum_height_fraction must be from 0 to 1")
        if crop_padding < 0:
            raise ValueError("crop_padding must not be negative")
        self.reader = reader
        self.fallback_reader = fallback_reader
        self.confirmation_reader = confirmation_reader
        self.name = f"{reader.name}-wide-band-fallback"
        self.low_confidence = low_confidence
        self.minimum_fallback_confidence = minimum_fallback_confidence
        self.minimum_width_fraction = minimum_width_fraction
        self.maximum_height_fraction = maximum_height_fraction
        self.crop_padding = crop_padding
        self._assessments: dict[int, dict[str, object]] = {}
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        baseline = self.reader.read(image_path, page_number)
        baseline_needs_review = _reader_needs_review(self.reader, page_number)
        try:
            with Image.open(image_path) as source:
                width, height = source.size
                groups = _wide_band_groups(
                    baseline,
                    width,
                    height,
                    low_confidence=self.low_confidence,
                    minimum_width_fraction=self.minimum_width_fraction,
                    maximum_height_fraction=self.maximum_height_fraction,
                    minimum_gap=self.crop_padding,
                )
                qualifying_count = sum(len(group) for group in groups)
                if not groups:
                    self._save_assessment(
                        page_number,
                        {
                            "page_number": page_number,
                            "status": "not_routed",
                            "selected_view": "baseline",
                            "qualifying_regions": 0,
                            "band_count": 0,
                            "replaced_bands": 0,
                            "nested_reader_review": baseline_needs_review,
                            "bands": [],
                        },
                    )
                    return baseline
                selected_groups: dict[str, list[TextRegion]] = {}
                alternative_regions: dict[str, TextRegion] = {}
                band_assessments = []
                with tempfile.TemporaryDirectory(prefix="ocr-wide-bands-") as directory:
                    for band_number, group in enumerate(groups, start=1):
                        crop = _padded_box(
                            [region.bounding_box for region in group],
                            width,
                            height,
                            self.crop_padding,
                        )
                        crop_path = Path(directory) / f"band-{band_number}.png"
                        cropped = source.crop(
                            (crop.left, crop.top, crop.right, crop.bottom)
                        )
                        cropped.resize(
                            (
                                (crop.right - crop.left) * WIDE_BAND_SCALE,
                                (crop.bottom - crop.top) * WIDE_BAND_SCALE,
                            ),
                            Image.Resampling.LANCZOS,
                        ).save(crop_path, format="PNG")
                        try:
                            fallback = self.fallback_reader.read(crop_path, page_number)
                        except ReaderError as error:
                            band_assessments.append(
                                {
                                    "band_number": band_number,
                                    "source_region_ids": [
                                        region.id for region in group
                                    ],
                                    "crop": asdict(crop),
                                    "selected_view": "baseline",
                                    "reason": "fallback_failed",
                                    "failure_code": error.code,
                                }
                            )
                            continue

                        fallback_needs_review = _reader_needs_review(
                            self.fallback_reader,
                            page_number,
                        )

                        usable_fallback = [
                            region for region in fallback if _band_tokens(region.text)
                        ]
                        translated = _translate_fallback_regions(
                            usable_fallback,
                            crop,
                            page_number,
                            band_number,
                            self.name,
                        )
                        selected, assessment = _assess_band_replacement(
                            group,
                            translated,
                            self.minimum_fallback_confidence,
                        )
                        if fallback_needs_review:
                            selected = False
                            assessment["selected_view"] = "baseline"
                            assessment["reason"] = "fallback_requires_review"
                            assessment["selection_reason"] = None
                        confirmation_regions: list[TextRegion] = []
                        if (
                            not selected
                            and not fallback_needs_review
                            and self.confirmation_reader is not None
                            and assessment["confirmation_candidate"]
                        ):
                            try:
                                confirmation = self.confirmation_reader.read(
                                    crop_path,
                                    page_number,
                                )
                            except ReaderError as error:
                                assessment.update(
                                    {
                                        "confirmation_status": "failed",
                                        "confirmation_failure_code": error.code,
                                    }
                                )
                            else:
                                usable_confirmation = [
                                    region
                                    for region in confirmation
                                    if _band_tokens(region.text)
                                ]
                                confirmation_regions = _translate_fallback_regions(
                                    usable_confirmation,
                                    crop,
                                    page_number,
                                    band_number,
                                    self.name,
                                    selected_view="confirmation",
                                )
                                confirmed, confirmation_assessment = (
                                    _assess_band_confirmation(
                                        translated,
                                        confirmation_regions,
                                        self.minimum_fallback_confidence,
                                    )
                                )
                                confirmation_review = _reader_needs_review(
                                    self.confirmation_reader,
                                    page_number,
                                )
                                if confirmation_review:
                                    confirmed = False
                                    confirmation_assessment["confirmation_status"] = (
                                        "requires_review"
                                    )
                                assessment.update(confirmation_assessment)
                                assessment["confirmation_review"] = confirmation_review
                                if confirmed:
                                    selected = True
                                    assessment.update(
                                        {
                                            "selected_view": "fallback",
                                            "reason": "evidence_confirmed",
                                            "selection_reason": (
                                                "independent_view_confirmation"
                                            ),
                                        }
                                    )
                        assessment.update(
                            {
                                "band_number": band_number,
                                "source_region_ids": [region.id for region in group],
                                "crop": asdict(crop),
                                "ignored_fallback_regions": len(fallback)
                                - len(usable_fallback),
                                "fallback_review": fallback_needs_review,
                            }
                        )
                        band_assessments.append(assessment)
                        if selected:
                            selected_regions = _attach_originals(
                                translated,
                                group,
                            )
                            if confirmation_regions:
                                selected_regions = _attach_fallbacks(
                                    selected_regions,
                                    confirmation_regions,
                                    source="confirmation",
                                )
                            selected_groups[group[0].id] = selected_regions
                        elif translated:
                            evidence_regions = _attach_fallbacks(group, translated)
                            if confirmation_regions:
                                evidence_regions = _attach_fallbacks(
                                    evidence_regions,
                                    confirmation_regions,
                                    source="confirmation",
                                )
                            alternative_regions.update(
                                {region.id: region for region in evidence_regions}
                            )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ReaderError("preprocess_failed", str(error)) from error

        replaced_ids = {
            region.id
            for group in groups
            if group[0].id in selected_groups
            for region in group
        }
        replaced_bands = len(selected_groups)
        assessment = {
            "page_number": page_number,
            "status": "recovered" if replaced_bands else "uncertain",
            "selected_view": "fused" if replaced_bands else "baseline",
            "qualifying_regions": qualifying_count,
            "band_count": len(groups),
            "replaced_bands": replaced_bands,
            "nested_reader_review": baseline_needs_review
            or any(
                bool(band.get("fallback_review"))
                or bool(band.get("confirmation_review"))
                for band in band_assessments
            ),
            "bands": band_assessments,
        }
        self._save_assessment(page_number, assessment)
        if not replaced_bands and not alternative_regions:
            return baseline

        output = []
        for region in baseline:
            if region.id in selected_groups:
                output.extend(selected_groups[region.id])
            elif region.id in alternative_regions:
                output.append(alternative_regions[region.id])
            elif region.id not in replaced_ids:
                output.append(region)
        output = [
            replace(region, reading_order=order)
            for order, region in enumerate(output, 1)
        ]
        return output

    def page_needs_review(self, page_number: int) -> bool:
        with self._lock:
            assessment = dict(self._assessments.get(page_number, {}))
        return (
            bool(assessment.get("nested_reader_review"))
            or assessment.get("status") == "uncertain"
        )

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
                "message": "No low-confidence wide text band was routed.",
                "pages": pages,
            }
        replaced = sum(int(page.get("replaced_bands", 0)) for page in routed)
        bands = sum(int(page.get("band_count", 0)) for page in routed)
        return {
            "status": "review_recommended",
            "message": (
                f"Wide-band fallback assessed {bands} band(s) on {len(routed)} "
                f"page(s) and accepted {replaced} evidence-backed replacement(s)."
            ),
            "pages": pages,
        }

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


def _wide_band_groups(
    regions: list[TextRegion],
    width: int,
    height: int,
    *,
    low_confidence: float,
    minimum_width_fraction: float,
    maximum_height_fraction: float,
    minimum_gap: int,
) -> list[list[TextRegion]]:
    lines: list[list[TextRegion]] = []
    candidates = [
        region
        for region in regions
        if _is_band_candidate(region, height, maximum_height_fraction)
    ]
    for region in sorted(
        candidates,
        key=lambda item: (
            item.bounding_box.top,
            item.bounding_box.left,
            item.reading_order,
        ),
    ):
        line = next(
            (
                current
                for current in reversed(lines)
                if _same_text_line(current, region)
            ),
            None,
        )
        if line is None:
            lines.append([region])
        else:
            line.append(region)

    selected = [
        line
        for line in lines
        if (
            _box_union([region.bounding_box for region in line]).right
            - _box_union([region.bounding_box for region in line]).left
        )
        / width
        >= minimum_width_fraction
        and _mean_confidence(line) < low_confidence
    ]
    selected_ids = {region.id for line in selected for region in line}
    groups: list[list[TextRegion]] = []
    current: list[TextRegion] = []
    for line in lines:
        if not any(region.id in selected_ids for region in line):
            current = []
            continue
        line = sorted(line, key=lambda region: (region.bounding_box.left, region.id))
        if not current or not _adjacent_text_lines(current, line, minimum_gap):
            groups.append(line)
            current = groups[-1]
        else:
            current.extend(line)
    return [
        sorted(group, key=lambda region: (region.reading_order, region.id))
        for group in groups
    ]


def _is_band_candidate(
    region: TextRegion,
    height: int,
    maximum_height_fraction: float,
) -> bool:
    box = region.bounding_box
    return (
        bool(_band_tokens(region.text))
        and region.resolution in {"resolved", "conflicting"}
        and region.confidence is not None
        and box.right > box.left
        and box.bottom > box.top
        and (box.bottom - box.top) / height <= maximum_height_fraction
    )


def _same_text_line(line: list[TextRegion], region: TextRegion) -> bool:
    box = _box_union([item.bounding_box for item in line])
    other = region.bounding_box
    overlap = max(0, min(box.bottom, other.bottom) - max(box.top, other.top))
    shorter = min(box.bottom - box.top, other.bottom - other.top)
    center_gap = abs((box.top + box.bottom) - (other.top + other.bottom)) / 2
    return overlap >= shorter * 0.5 or center_gap <= max(2, shorter * 0.35)


def _adjacent_text_lines(
    current: list[TextRegion],
    following: list[TextRegion],
    minimum_gap: int,
) -> bool:
    current_box = _box_union([region.bounding_box for region in current])
    following_box = _box_union([region.bounding_box for region in following])
    vertical_gap = max(0, following_box.top - current_box.bottom)
    maximum_gap = max(
        minimum_gap,
        current_box.bottom - current_box.top,
        following_box.bottom - following_box.top,
    )
    overlap = max(
        0,
        min(current_box.right, following_box.right)
        - max(current_box.left, following_box.left),
    )
    narrower = min(
        current_box.right - current_box.left,
        following_box.right - following_box.left,
    )
    return vertical_gap <= maximum_gap and overlap >= narrower * 0.5


def _padded_box(
    boxes: list[BoundingBox],
    width: int,
    height: int,
    padding: int,
) -> BoundingBox:
    box = _box_union(boxes)
    return BoundingBox(
        left=max(0, box.left - padding),
        top=max(0, box.top - padding),
        right=min(width, box.right + padding),
        bottom=min(height, box.bottom + padding),
    )


def _box_union(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        left=min(box.left for box in boxes),
        top=min(box.top for box in boxes),
        right=max(box.right for box in boxes),
        bottom=max(box.bottom for box in boxes),
    )


def _translate_fallback_regions(
    regions: list[TextRegion],
    crop: BoundingBox,
    page_number: int,
    band_number: int,
    stage_name: str,
    *,
    selected_view: str = "fallback",
) -> list[TextRegion]:
    translated = []
    for index, region in enumerate(regions, start=1):
        box = region.bounding_box
        provenance = dict(region.text_provenance or {})
        provenance.update(
            {
                "stage": stage_name,
                "source_crop": asdict(crop),
                "upscale_factor": WIDE_BAND_SCALE,
                "selected_view": selected_view,
            }
        )
        translated.append(
            replace(
                region,
                id=(f"p{page_number}-wide-band-{band_number}-{selected_view}-{index}"),
                bounding_box=BoundingBox(
                    left=crop.left + floor(box.left / WIDE_BAND_SCALE),
                    top=crop.top + floor(box.top / WIDE_BAND_SCALE),
                    right=crop.left + ceil(box.right / WIDE_BAND_SCALE),
                    bottom=crop.top + ceil(box.bottom / WIDE_BAND_SCALE),
                ),
                text_provenance=provenance,
                alternatives=list(region.alternatives),
            )
        )
    return translated


def _assess_band_replacement(
    baseline: list[TextRegion],
    fallback: list[TextRegion],
    minimum_fallback_confidence: float,
) -> tuple[bool, dict[str, object]]:
    baseline_tokens = _band_token_counts(baseline)
    fallback_tokens = _band_token_counts(fallback)
    baseline_count = sum(baseline_tokens.values())
    fallback_count = sum(fallback_tokens.values())
    added = sum((fallback_tokens - baseline_tokens).values())
    removed = sum((baseline_tokens - fallback_tokens).values())
    retained = sum((fallback_tokens & baseline_tokens).values())
    baseline_token_recall = retained / baseline_count if baseline_count else 0.0
    baseline_characters = sum(
        len("".join(_band_tokens(region.text))) for region in baseline
    )
    fallback_characters = sum(
        len("".join(_band_tokens(region.text))) for region in fallback
    )
    coverage_ratio = (
        fallback_characters / baseline_characters if baseline_characters else 0.0
    )
    baseline_confidence = _mean_confidence(baseline)
    fallback_confidence = _mean_confidence(fallback)
    baseline_text = " ".join(_band_tokens(" ".join(region.text for region in baseline)))
    fallback_text = " ".join(_band_tokens(" ".join(region.text for region in fallback)))
    text_similarity = SequenceMatcher(None, baseline_text, fallback_text).ratio()
    valid_text = bool(fallback) and all(
        _valid_fallback_region(region) for region in fallback
    )
    repeated_text_risk = _has_repeated_text_risk(fallback)
    plausible_density = _has_plausible_text_density(fallback)
    correction_token_count = fallback_count >= baseline_count and fallback_count <= max(
        baseline_count + 2, ceil(baseline_count * 1.25)
    )
    correction_coverage = 0.9 <= coverage_ratio <= 1.35
    better_confidence = (
        fallback_confidence >= minimum_fallback_confidence
        and fallback_confidence >= baseline_confidence + 0.05
    )
    correction = (
        correction_token_count
        and correction_coverage
        and text_similarity >= 0.6
        and better_confidence
    )
    missing_text_candidate = (
        baseline_confidence <= 0.55
        and fallback_confidence >= minimum_fallback_confidence
        and fallback_confidence >= baseline_confidence + 0.15
        and fallback_count >= ceil(baseline_count * 1.25)
        and fallback_count <= baseline_count * 3
        and 1.1 <= coverage_ratio <= 3.0
    )
    missing_text_recovery = (
        missing_text_candidate and baseline_token_recall >= MATCH_OVERLAP
    )
    confirmation_candidate = (
        missing_text_candidate
        and baseline_token_recall < MATCH_OVERLAP
        and valid_text
        and plausible_density
        and not repeated_text_risk
    )
    selected = (
        valid_text
        and plausible_density
        and (correction or missing_text_recovery)
        and not repeated_text_risk
    )
    selection_reason = (
        "missing_text_recovery" if missing_text_recovery else "correction"
    )
    return selected, {
        "selected_view": "fallback" if selected else "baseline",
        "reason": "evidence_improved" if selected else "fallback_not_better",
        "selection_reason": selection_reason if selected else None,
        "valid_text": valid_text,
        "plausible_density": plausible_density,
        "baseline_tokens": baseline_count,
        "fallback_tokens": fallback_count,
        "added_tokens": added,
        "removed_tokens": removed,
        "baseline_token_recall": round(baseline_token_recall, 6),
        "coverage_ratio": round(coverage_ratio, 6),
        "text_similarity": round(text_similarity, 6),
        "baseline_mean_confidence": round(baseline_confidence, 6),
        "fallback_mean_confidence": round(fallback_confidence, 6),
        "repeated_text_risk": repeated_text_risk,
        "confirmation_candidate": confirmation_candidate,
    }


def _assess_band_confirmation(
    fallback: list[TextRegion],
    confirmation: list[TextRegion],
    minimum_confidence: float,
) -> tuple[bool, dict[str, object]]:
    fallback_tokens = _band_token_counts(fallback)
    confirmation_tokens = _band_token_counts(confirmation)
    shared = sum((fallback_tokens & confirmation_tokens).values())
    fallback_count = sum(fallback_tokens.values())
    confirmation_count = sum(confirmation_tokens.values())
    fallback_recall = shared / fallback_count if fallback_count else 0.0
    confirmation_recall = shared / confirmation_count if confirmation_count else 0.0
    fallback_text = " ".join(_band_tokens(" ".join(r.text for r in fallback)))
    confirmation_text = " ".join(_band_tokens(" ".join(r.text for r in confirmation)))
    similarity = SequenceMatcher(None, fallback_text, confirmation_text).ratio()
    valid_text = bool(confirmation) and all(
        _valid_fallback_region(region) for region in confirmation
    )
    plausible_density = _has_plausible_text_density(confirmation)
    repeated_text_risk = _has_repeated_text_risk(confirmation)
    literal_agreement = _literal_tokens(fallback) == _literal_tokens(confirmation)
    confirmed = (
        valid_text
        and plausible_density
        and not repeated_text_risk
        and _mean_confidence(confirmation) >= minimum_confidence
        and fallback_recall >= CONFIRMATION_RECALL
        and confirmation_recall >= CONFIRMATION_RECALL
        and similarity >= CONFIRMATION_SIMILARITY
        and literal_agreement
    )
    return confirmed, {
        "confirmation_status": "agreed" if confirmed else "disagreed",
        "confirmation_regions": len(confirmation),
        "confirmation_mean_confidence": round(_mean_confidence(confirmation), 6),
        "confirmation_fallback_recall": round(fallback_recall, 6),
        "confirmation_token_recall": round(confirmation_recall, 6),
        "confirmation_text_similarity": round(similarity, 6),
        "confirmation_valid_text": valid_text,
        "confirmation_plausible_density": plausible_density,
        "confirmation_repeated_text_risk": repeated_text_risk,
        "confirmation_literal_agreement": literal_agreement,
    }


def _has_repeated_text_risk(regions: list[TextRegion]) -> bool:
    normalized_regions = [
        " ".join(_band_tokens(region.text)) for region in regions if region.text.strip()
    ]
    if any(
        current == previous and len(current.split()) >= 2
        for previous, current in zip(
            normalized_regions,
            normalized_regions[1:],
            strict=False,
        )
    ):
        return True
    tokens = " ".join(normalized_regions).split()
    for size in range(2, min(8, len(tokens) // 2) + 1):
        for start in range(len(tokens) - (size * 2) + 1):
            if (
                tokens[start : start + size]
                == tokens[start + size : start + (size * 2)]
            ):
                return True
    return False


def _valid_fallback_region(region: TextRegion) -> bool:
    provenance = region.text_provenance or {}
    crop = provenance.get("source_crop")
    box = region.bounding_box
    return (
        bool(_band_tokens(region.text))
        and region.confidence is not None
        and 0 <= region.confidence <= 1
        and isinstance(crop, dict)
        and box.left >= crop.get("left", 0)
        and box.top >= crop.get("top", 0)
        and box.right <= crop.get("right", -1)
        and box.bottom <= crop.get("bottom", -1)
        and _box_area(box) > 0
    )


def _has_plausible_text_density(regions: list[TextRegion]) -> bool:
    for region in regions:
        box = region.bounding_box
        height = box.bottom - box.top
        if height <= 0:
            return False
        characters = len("".join(_band_tokens(region.text)))
        width_in_text_heights = (box.right - box.left) / height
        if characters > max(4, ceil(width_in_text_heights * 4)):
            return False
    return True


def _attach_originals(
    fallback: list[TextRegion], baseline: list[TextRegion]
) -> list[TextRegion]:
    alternatives: list[list[TextAlternative]] = [
        list(region.alternatives) for region in fallback
    ]
    conflicts = [region.resolution == "conflicting" for region in fallback]
    for original in baseline:
        target = max(
            range(len(fallback)),
            key=lambda index: _overlap_ratio(
                fallback[index].bounding_box,
                original.bounding_box,
            ),
        )
        provenance = dict(original.text_provenance or {})
        provenance.update(
            {
                "original_region_id": original.id,
                "original_bounding_box": asdict(original.bounding_box),
            }
        )
        alternatives[target].append(
            TextAlternative(
                text=original.text,
                confidence=original.confidence,
                provider=original.provider,
                text_provenance=provenance,
            )
        )
        alternatives[target].extend(original.alternatives)
        conflicts[target] = (
            conflicts[target]
            or original.resolution == "conflicting"
            or any(
                _normalized_text(alternative.text) != _normalized_text(original.text)
                for alternative in original.alternatives
            )
        )
    return [
        replace(
            region,
            resolution="conflicting" if conflicts[index] else region.resolution,
            alternatives=alternatives[index],
        )
        for index, region in enumerate(fallback)
    ]


def _attach_fallbacks(
    baseline: list[TextRegion],
    fallback: list[TextRegion],
    *,
    source: str = "fallback",
) -> list[TextRegion]:
    alternatives = [list(region.alternatives) for region in baseline]
    for candidate in fallback:
        target = max(
            range(len(baseline)),
            key=lambda index: _overlap_ratio(
                baseline[index].bounding_box,
                candidate.bounding_box,
            ),
        )
        provenance = dict(candidate.text_provenance or {})
        provenance.update(
            {
                f"{source}_region_id": candidate.id,
                f"{source}_bounding_box": asdict(candidate.bounding_box),
            }
        )
        alternatives[target].append(
            TextAlternative(
                text=candidate.text,
                confidence=candidate.confidence,
                provider=candidate.provider,
                text_provenance=provenance,
            )
        )
    return [
        replace(
            region,
            resolution=region.resolution,
            alternatives=alternatives[index],
        )
        for index, region in enumerate(baseline)
    ]


def _reader_needs_review(reader: LocalReader, page_number: int) -> bool:
    check = getattr(reader, "page_needs_review", None)
    return bool(callable(check) and check(page_number))


def _band_token_counts(regions: list[TextRegion]) -> Counter[str]:
    return Counter(token for region in regions for token in _band_tokens(region.text))


def _literal_tokens(regions: list[TextRegion]) -> list[str]:
    return [
        token
        for region in regions
        for token in _band_tokens(region.text)
        if token in {"%", "$"} or any(character.isdigit() for character in token)
    ]


def _band_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+|[%$]", text.casefold())


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
