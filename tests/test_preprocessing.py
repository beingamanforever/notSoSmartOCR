from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.orientation import OrientationReader
from ocr_pipeline.preprocessing import (
    PageFrameReader,
    RoutedTesseractReader,
    TiledReader,
    WideBandFallbackReader,
    _assess_band_confirmation,
    locate_dark_frame,
    locate_document_frame,
)
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.rendering import render_evidence


class ControlledView:
    name = "controlled-view"

    def __init__(self, text: str, confidence: float = 0.9) -> None:
        self.text = text
        self.confidence = confidence
        self.sizes = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            self.sizes.append(image.size)
        return [
            TextRegion(
                id=f"p{page_number}-word-1",
                kind="word",
                text=self.text,
                confidence=self.confidence,
                bounding_box=BoundingBox(1, 2, 50, 20),
                reading_order=1,
                provider=self.name,
                text_provenance={"method": "controlled"},
            )
        ]


class ControlledTileView:
    name = "controlled-tile-view"

    def __init__(self, *, region_height: int = 5) -> None:
        self.region_height = region_height
        self.calls = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.calls.append(image_path.name)
        tile_text = {
            "tile-1.png": "one two three four",
            "tile-2.png": "five six seven eight",
            "tile-3.png": "nine ten eleven twelve",
        }
        text = tile_text.get(
            image_path.name,
            "one two three four five six seven eight nine ten",
        )
        top = 1 if image_path.name == "tile-1.png" else 10
        return [
            TextRegion(
                id=f"p{page_number}-word-1",
                kind="word",
                text=text,
                confidence=0.9,
                bounding_box=BoundingBox(1, top, 90, top + self.region_height),
                reading_order=1,
                provider=self.name,
                text_provenance={"method": "controlled"},
            )
        ]


class ScriptedTileView:
    name = "scripted-tile-view"

    def __init__(
        self,
        outputs: dict[str, list[tuple[str, BoundingBox, float]]],
    ) -> None:
        self.outputs = outputs
        self.calls = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.calls.append(image_path.name)
        return [
            TextRegion(
                id=f"p{page_number}-{image_path.stem}-{index}",
                kind="word",
                text=text,
                confidence=confidence,
                bounding_box=box,
                reading_order=index,
                provider=self.name,
                text_provenance={"method": "scripted"},
            )
            for index, (text, box, confidence) in enumerate(
                self.outputs.get(image_path.name, []), start=1
            )
        ]


class ScriptedBandView:
    name = "scripted-band-view"

    def __init__(
        self,
        outputs: dict[str, list[tuple[str, BoundingBox, float]]],
        *,
        fail_on_crop: bool = False,
    ) -> None:
        self.outputs = outputs
        self.fail_on_crop = fail_on_crop
        self.calls: list[str] = []
        self.sizes: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            size = image.size
        with self._lock:
            self.calls.append(image_path.name)
            self.sizes.append(size)
        if self.fail_on_crop and image_path.name.startswith("band-"):
            raise ReaderError("reader_failed", "controlled fallback failure")
        return [
            TextRegion(
                id=f"p{page_number}-{image_path.stem}-{index}",
                kind="text",
                text=text,
                confidence=confidence,
                bounding_box=box,
                reading_order=index,
                provider=self.name,
                text_provenance={"method": "scripted-band"},
            )
            for index, (text, box, confidence) in enumerate(
                self.outputs.get(image_path.name, []), start=1
            )
        ]


class DelayedBandView(ScriptedBandView):
    def __init__(self, delay: float) -> None:
        super().__init__({})
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return super().read(image_path, page_number)
        finally:
            with self._lock:
                self.active -= 1


class RecordingStage:
    name = "recording"

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        with Image.open(image_path) as image:
            self.sizes.append(image.size)
        regions[0].structure = {
            "role": "table",
            "cells": [
                {
                    "bbox": {"left": 10, "top": 20, "right": 50, "bottom": 40},
                    "span_bboxes": [{"left": 12, "top": 22, "right": 20, "bottom": 30}],
                    "resolution": "resolved",
                }
            ],
        }
        return regions


class ScriptedFrameView:
    name = "scripted-frame-view"

    def __init__(self, *, fail_full_page: bool = False) -> None:
        self.fail_full_page = fail_full_page
        self.sizes: list[tuple[int, int]] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            size = image.size
        self.sizes.append(size)
        if size == (200, 100):
            return [
                TextRegion(
                    id=f"p{page_number}-body",
                    kind="text",
                    text="High quality body",
                    confidence=0.99,
                    bounding_box=BoundingBox(10, 10, 110, 30),
                    reading_order=1,
                    provider=self.name,
                    text_provenance={"method": "cropped-frame"},
                )
            ]
        if self.fail_full_page:
            raise ReaderError("full_page_failed", "controlled full-page failure")
        values = [
            ("header", "Header navigation", BoundingBox(10, 10, 150, 30), "resolved"),
            ("body", "Lower quality body", BoundingBox(110, 110, 210, 130), "resolved"),
            ("footer", "Bottom footer", BoundingBox(10, 250, 150, 270), "resolved"),
            (
                "footer-copy",
                "Bottom footer",
                BoundingBox(10, 250, 150, 270),
                "resolved",
            ),
            ("empty", "", BoundingBox(10, 40, 30, 50), "resolved"),
            ("uncertain", "Guess", BoundingBox(10, 50, 80, 65), "unreadable"),
            ("outside-page", "Invalid", BoundingBox(390, 280, 410, 310), "resolved"),
        ]
        return [
            TextRegion(
                id=f"p{page_number}-{region_id}",
                kind="text",
                text=text,
                confidence=0.9,
                bounding_box=box,
                reading_order=index,
                provider=self.name,
                text_provenance={"method": "full-page"},
                resolution=resolution,
            )
            for index, (region_id, text, box, resolution) in enumerate(values, 1)
        ]


def test_dark_frame_locator_finds_document_canvas(tmp_path: Path) -> None:
    source = tmp_path / "screenshot.png"
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 30, 19, 119), fill="black")
    draw.rectangle((180, 30, 199, 119), fill="black")
    image.save(source)

    assert locate_dark_frame(source) == BoundingBox(20, 30, 180, 120)


def test_dark_frame_locator_leaves_full_page_documents_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)

    assert locate_dark_frame(source) is None


def test_document_frame_locator_ignores_browser_chrome_and_flags_partial_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "viewer.png"
    image = Image.new("RGB", (600, 500), (135, 135, 135))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 599, 30), fill="white")
    draw.rectangle((200, 60, 399, 359), fill="white")
    draw.rectangle((200, 380, 399, 499), fill="white")
    draw.text((240, 150), "Attention Is All You Need", fill="black")
    image.save(source)

    assert locate_document_frame(source) == BoundingBox(199, 58, 401, 361)
    reader = PageFrameReader(ControlledView("page"))
    reader.read(source, 1)
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    assert assessment["pages"][0]["partial_page_visible"] is True


def test_document_frame_locator_does_not_crop_a_printed_form_border() -> None:
    source = Path(__file__).parents[1] / "artifacts" / "demo" / "scanned_form.png"
    controlled = ControlledView("Full form")

    assert locate_document_frame(source) is None
    result = process_document(source, PageFrameReader(controlled))

    assert controlled.sizes == [(754, 1000)]
    assert result.pages[0].text.value == "Full form"


def test_page_frame_wraps_orientation_and_all_stages_then_restores_nested_boxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "viewer.png"
    Image.new("RGB", (300, 240), (130, 130, 130)).save(source)
    crop = BoundingBox(40, 30, 240, 180)
    controlled = ControlledView("Attention", 0.99)
    oriented = OrientationReader(
        controlled,
        osd_detector=lambda _: {"angle": 90, "confidence": 20.0},
        defer_restore=True,
    )
    reader = PageFrameReader(oriented, locator=lambda _: crop)
    stage = RecordingStage()

    result = process_document(source, reader, stages=[stage])

    assert controlled.sizes == [
        (150, 200),
        (200, 150),
        (240, 300),
        (300, 240),
    ]
    assert stage.sizes == [(150, 200)]
    page = result.pages[0]
    assert (page.width, page.height) == (300, 240)
    region = next(
        item for item in page.regions if "source_crop" in item.text_provenance
    )
    assert region.bounding_box == BoundingBox(220, 31, 238, 80)
    cell = region.structure["cells"][0]
    assert cell["bbox"] == {"left": 200, "top": 40, "right": 220, "bottom": 80}
    assert cell["span_bboxes"] == [{"left": 210, "top": 42, "right": 218, "bottom": 50}]
    assert region.text_provenance["source_crop"] == {
        "left": 40,
        "top": 30,
        "right": 240,
        "bottom": 180,
    }


def test_page_frame_recovers_only_valid_outside_evidence_end_to_end(
    tmp_path: Path,
) -> None:
    source = tmp_path / "screenshot.png"
    Image.new("RGB", (400, 300), "white").save(source)
    crop = BoundingBox(100, 100, 300, 200)
    scripted = ScriptedFrameView()
    reader = PageFrameReader(scripted, locator=lambda _: crop)
    stage = RecordingStage()

    result = process_document(source, reader, stages=[stage])

    page = result.pages[0]
    assert page.route == "review"
    assert page.text.value == "Header navigation High quality body Bottom footer"
    assert [region.text for region in page.regions] == [
        "Header navigation",
        "High quality body",
        "Bottom footer",
    ]
    assert [region.reading_order for region in page.regions] == [1, 2, 3]
    assert scripted.sizes == [(200, 100), (400, 300)]
    assert stage.sizes == [(200, 100)]

    body = page.regions[1]
    assert body.id == "p1-body"
    assert body.confidence == 0.99
    assert body.bounding_box == BoundingBox(110, 110, 210, 130)
    assert body.text_provenance == {
        "method": "cropped-frame",
        "source_crop": {
            "left": 100,
            "top": 100,
            "right": 300,
            "bottom": 200,
        },
    }
    assert body.structure["cells"][0]["bbox"] == {
        "left": 110,
        "top": 120,
        "right": 150,
        "bottom": 140,
    }

    header = page.regions[0]
    assert header.id == "p1-frame-recovery-1"
    assert header.text_provenance == {
        "method": "full-page",
        "page_frame_recovery": {
            "method": "residual_full_original",
            "source_region_id": "p1-header",
            "source_reader": "scripted-frame-view",
            "isolated_frame": {
                "left": 100,
                "top": 100,
                "right": 300,
                "bottom": 200,
            },
        },
    }
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    recovered = assessment["pages"][0]
    assert recovered["recovery_status"] == "recovered"
    assert recovered["full_page_regions"] == 7
    assert recovered["inside_frame_regions"] == 1
    assert recovered["invalid_full_page_regions"] == 3
    assert recovered["duplicate_full_page_regions"] == 1
    assert recovered["recovered_regions"] == 2
    assert recovered["preserved_canonical_regions"] == 1


def test_page_frame_recovery_failure_keeps_cropped_canonical_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "screenshot.png"
    Image.new("RGB", (400, 300), "white").save(source)
    scripted = ScriptedFrameView(fail_full_page=True)
    reader = PageFrameReader(
        scripted,
        locator=lambda _: BoundingBox(100, 100, 300, 200),
    )

    result = process_document(source, reader)

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "High quality body"
    assert result.pages[0].regions[0].bounding_box == BoundingBox(110, 110, 210, 130)
    assert result.pages[0].failure_ids == ["failure-1"]
    assert result.failures[0].code == "full_page_failed"
    assessment = reader.coverage_assessment(1)["pages"][0]
    assert assessment["recovery_status"] == "failed"
    assert assessment["recovery_failure"] == {
        "code": "full_page_failed",
        "message": "controlled full-page failure",
    }


def test_page_frame_does_not_rerun_full_page_without_an_isolated_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(source)
    controlled = ControlledView("Full page")
    reader = PageFrameReader(controlled, locator=lambda _: None)

    result = process_document(source, reader)

    assert controlled.sizes == [(400, 300)]
    assert result.pages[0].text.value == "Full page"
    assert result.pages[0].route == "accept_local"
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "not_assessed"
    assert assessment["pages"][0]["recovery_status"] == "not_routed"


def test_routed_reader_selects_supported_enhanced_view_and_translates_boxes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "screenshot.png"
    Image.new("RGB", (200, 120), "white").save(source)
    baseline = ControlledView("one two three four five six seven eight nine ten")
    enhanced = ControlledView(
        "one two three four five six seven eight nine ten eleven twelve", 0.91
    )
    crop = BoundingBox(20, 30, 180, 120)
    reader = RoutedTesseractReader(
        baseline=baseline,
        enhanced=enhanced,
        locator=lambda _: crop,
    )

    regions = reader.read(source, 1)

    assert baseline.sizes == enhanced.sizes == [(160, 90)]
    assert regions[0].text.endswith("eleven twelve")
    assert regions[0].bounding_box == BoundingBox(21, 32, 70, 50)
    assert regions[0].text_provenance == {
        "method": "controlled",
        "source_crop": {
            "left": 20,
            "top": 30,
            "right": 180,
            "bottom": 120,
        },
        "selected_view": "enhanced",
    }
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    assert assessment["pages"][0]["status"] == "recovered"
    assert assessment["pages"][0]["added_tokens"] == 2


def test_routed_reader_rejects_conflicting_enhanced_view(tmp_path: Path) -> None:
    source = tmp_path / "screenshot.png"
    Image.new("RGB", (200, 120), "white").save(source)
    baseline = ControlledView("one two three four five six seven eight nine ten")
    enhanced = ControlledView("different content entirely")
    reader = RoutedTesseractReader(
        baseline=baseline,
        enhanced=enhanced,
        locator=lambda _: BoundingBox(20, 30, 180, 120),
    )

    regions = reader.read(source, 1)

    assert regions[0].text == baseline.text
    assessment = reader.coverage_assessment(1)
    assert assessment["pages"][0]["status"] == "uncertain"
    assert assessment["pages"][0]["selected_view"] == "baseline"


def test_tiled_reader_preserves_baseline_and_exposes_uncertain_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 120), "white").save(source)
    controlled = ScriptedTileView(
        {
            "page.png": [
                ("Alpha", BoundingBox(5, 10, 45, 20), 0.92),
                ("Beta", BoundingBox(5, 50, 45, 60), 0.91),
            ],
            "tile-1.png": [("Alpha", BoundingBox(5, 10, 45, 20), 0.93)],
            "tile-2.png": [("Bela", BoundingBox(5, 17, 45, 27), 0.90)],
            "tile-3.png": [("Ghost", BoundingBox(5, 25, 45, 35), 0.94)],
        }
    )
    reader = TiledReader(controlled)

    regions = reader.read(source, 1)

    assert controlled.calls == [
        "page.png",
        "tile-1.png",
        "tile-2.png",
        "tile-3.png",
    ]
    assert [region.id for region in regions[:2]] == ["p1-page-1", "p1-page-2"]
    assert [region.text for region in regions] == ["Alpha", "Beta", "Ghost"]
    assert regions[0].resolution == "resolved"
    assert [alternative.text for alternative in regions[0].alternatives] == ["Alpha"]
    assert regions[1].resolution == "resolved"
    assert [alternative.text for alternative in regions[1].alternatives] == ["Bela"]
    assert regions[2].id == "p1-tiny-tile-3"
    assert regions[2].resolution == "unreadable"
    assert regions[2].text_provenance == {
        "method": "scripted",
        "stage": "scripted-tile-view-tiled",
        "tile_number": 3,
        "tile_top": 73,
    }
    assert render_evidence(regions).value == "Alpha Beta"
    assert render_evidence(regions).evidence_ids == ["p1-page-1", "p1-page-2"]
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    page = assessment["pages"][0]
    assert page["ran"] is True
    assert page["tile_reader"] == "scripted-tile-view"
    assert page["tile_views_run"] == 3
    assert page["tiled_candidates"] == 3
    assert page["selected_view"] == "fused"
    assert page["baseline_regions"] == 2
    assert page["preserved_baseline_regions"] == 2
    assert page["exact_overlap_candidates"] == 1
    assert page["conflicting_candidates"] == 1
    assert page["unresolved_tile_only_regions"] == 1
    assert page["removed_tokens"] == 0


def test_tiled_reader_keeps_same_engine_tile_agreement_unresolved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 120), "white").save(source)
    controlled = ScriptedTileView(
        {
            "page.png": [("Base", BoundingBox(5, 5, 45, 10), 0.95)],
            "tile-1.png": [("fine print", BoundingBox(5, 36, 45, 44), 0.92)],
            "tile-2.png": [("fine print", BoundingBox(5, 3, 45, 11), 0.93)],
        }
    )
    reader = TiledReader(controlled)

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == ["Base", "fine print"]
    assert regions[0].id == "p1-page-1"
    assert regions[1].id == "p1-tiny-tile-2"
    assert regions[1].resolution == "unreadable"
    assert regions[1].structure == {
        "role": "tiny_text_candidate",
        "support_views": 2,
    }
    assert [alternative.text for alternative in regions[1].alternatives] == [
        "fine print"
    ]
    assert render_evidence(regions).value == "Base"
    assert render_evidence(regions).evidence_ids == ["p1-page-1"]
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["promoted_tile_only_regions"] == 0
    assert page["unresolved_tile_only_regions"] == 1
    assert page["added_tokens"] == 0


def test_tiled_reader_batches_independent_tiles(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 120), "white").save(source)

    class BatchTileView(ScriptedTileView):
        batch_size = 3

        def __init__(self) -> None:
            super().__init__(
                {
                    "page.png": [("Base", BoundingBox(5, 5, 45, 10), 0.95)],
                    "tile-1.png": [],
                    "tile-2.png": [],
                    "tile-3.png": [],
                }
            )
            self.batches: list[list[str]] = []

        def read_batch(
            self, paths: list[Path], page_numbers: list[int]
        ) -> list[list[TextRegion]]:
            self.batches.append([path.name for path in paths])
            assert page_numbers == [1, 1, 1]
            return [self.read(path, 1) for path in paths]

    controlled = BatchTileView()

    regions = TiledReader(controlled).read(source, 1)

    assert [region.text for region in regions] == ["Base"]
    assert controlled.batches == [["tile-1.png", "tile-2.png", "tile-3.png"]]


def test_tiled_reader_skips_normal_size_text(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 120), "white").save(source)
    controlled = ControlledTileView(region_height=18)
    reader = TiledReader(controlled)

    regions = reader.read(source, 1)

    assert controlled.calls == ["page.png"]
    assert regions[0].text.endswith("nine ten")
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "not_assessed"
    page = assessment["pages"][0]
    assert page["status"] == "not_routed"
    assert page["ran"] is True
    assert page["tile_views_run"] == 0
    assert page["tiled_candidates"] == 0


def test_wide_band_reader_groups_only_adjacent_qualifying_bands_and_upscales(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 140), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("Heading", BoundingBox(20, 8, 100, 20), 0.98),
                ("ambulates wih walker", BoundingBox(20, 30, 170, 38), 0.65),
                ("folow up tomorow", BoundingBox(22, 40, 175, 48), 0.68),
                ("Section marker", BoundingBox(20, 49, 100, 52), 0.98),
                ("isolated weak line", BoundingBox(18, 52, 172, 60), 0.7),
            ]
        }
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                ("ambulates with walker", BoundingBox(24, 24, 450, 48), 0.95),
                ("follow up tomorrow", BoundingBox(30, 54, 459, 78), 0.96),
            ],
            "band-2.png": [
                ("isolated weak line", BoundingBox(24, 24, 486, 48), 0.94),
            ],
        }
    )
    reader = WideBandFallbackReader(primary, fallback)

    regions = reader.read(source, 1)

    assert primary.calls == ["page.png"]
    assert dict(zip(fallback.calls, fallback.sizes, strict=True)) == {
        "band-1.png": (513, 102),
        "band-2.png": (510, 72),
    }
    assert [region.text for region in regions] == [
        "Heading",
        "ambulates with walker",
        "follow up tomorrow",
        "Section marker",
        "isolated weak line",
    ]
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "review_recommended"
    page = assessment["pages"][0]
    assert page["status"] == "recovered"
    assert page["qualifying_regions"] == 3
    assert page["band_count"] == 2
    assert page["replaced_bands"] == 2


def test_wide_band_reader_bounds_ordered_fallbacks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 320), "white").save(source)
    baseline = [
        (
            f"weak band {index}",
            BoundingBox(10, 10 + index * 30, 190, 18 + index * 30),
            0.6,
        )
        for index in range(10)
    ]
    fallback = DelayedBandView(0.03)
    reader = WideBandFallbackReader(ScriptedBandView({"page.png": baseline}), fallback)

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == [text for text, _, _ in baseline]
    assert len(fallback.calls) == 8
    assert fallback.max_active == 1
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["fallback_reader_runs"] == 8
    assert page["band_count"] == 10
    assert page["assessed_band_count"] == 8
    assert page["omitted_bands"] == 2
    assert [band["band_number"] for band in page["bands"]] == list(range(1, 9))
    assert "band_call_limit_reached" in page["review_reasons"]


def test_wide_band_reader_translates_boxes_and_keeps_original_alternatives(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    original = [
        ("ambulates wih walker", BoundingBox(20, 30, 170, 38), 0.65),
        ("folow up tomorow", BoundingBox(22, 40, 175, 48), 0.68),
    ]
    primary = ScriptedBandView({"page.png": original})
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                ("ambulates with walker", BoundingBox(24, 24, 450, 48), 0.95),
                ("follow up tomorrow", BoundingBox(30, 54, 459, 78), 0.96),
            ]
        }
    )
    reader = WideBandFallbackReader(primary, fallback)

    regions = reader.read(source, 1)

    assert [region.bounding_box for region in regions] == [
        BoundingBox(20, 30, 162, 38),
        BoundingBox(22, 40, 165, 48),
    ]
    assert [alternative.text for alternative in regions[0].alternatives] == [
        original[0][0]
    ]
    assert [alternative.text for alternative in regions[1].alternatives] == [
        original[1][0]
    ]
    assert regions[0].alternatives[0].text_provenance == {
        "method": "scripted-band",
        "original_region_id": "p1-page-1",
        "original_bounding_box": {
            "left": 20,
            "top": 30,
            "right": 170,
            "bottom": 38,
        },
    }
    assert regions[0].text_provenance == {
        "method": "scripted-band",
        "stage": "scripted-band-view-wide-band-fallback",
        "source_crop": {"left": 12, "top": 22, "right": 183, "bottom": 56},
        "upscale_factor": 3,
        "selected_view": "fallback",
    }


def test_wide_band_reader_preserves_tiled_conflict_and_routes_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    tiled_view = ScriptedTileView(
        {
            "page.png": [
                ("ambulates with walker", BoundingBox(10, 20, 190, 28), 0.6),
            ],
            "tile-1.png": [
                ("ambulates with waller", BoundingBox(10, 20, 190, 28), 0.9),
            ],
        }
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                ("ambulates with walker", BoundingBox(24, 24, 564, 48), 0.95),
            ],
        }
    )
    reader = WideBandFallbackReader(TiledReader(tiled_view), fallback)

    result = process_document(source, reader)

    region = result.pages[0].regions[0]
    assert region.text == "ambulates with walker"
    assert [alternative.text for alternative in region.alternatives] == [
        "ambulates with walker",
        "ambulates with waller",
    ]
    assert region.alternatives[1].text_provenance == {
        "method": "scripted",
        "stage": "scripted-tile-view-tiled",
        "tile_number": 1,
        "tile_top": 0,
    }
    assert region.resolution == "conflicting"
    assert result.pages[0].route == "review"
    assert reader.coverage_assessment(1)["pages"][0]["replaced_bands"] == 1


def test_wide_band_reader_does_not_route_ordinary_or_high_confidence_regions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("strong wide line", BoundingBox(10, 20, 190, 28), 0.95),
                ("weak narrow note", BoundingBox(10, 40, 80, 48), 0.6),
                ("weak tall block", BoundingBox(10, 60, 190, 90), 0.6),
            ]
        }
    )
    fallback = ScriptedBandView({})
    reader = WideBandFallbackReader(primary, fallback)

    regions = reader.read(source, 1)

    assert fallback.calls == []
    assert [region.text for region in regions] == [
        "strong wide line",
        "weak narrow note",
        "weak tall block",
    ]
    assessment = reader.coverage_assessment(1)
    assert assessment["status"] == "not_assessed"
    assert assessment["pages"][0]["status"] == "not_routed"


def test_narrow_decimal_reread_preserves_canonical_text_for_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 160), "white").save(source)
    primary = ScriptedBandView(
        {"page.png": [("53", BoundingBox(190, 40, 215, 50), 0.45)]}
    )
    fallback = ScriptedBandView(
        {"band-1.png": [("5.3", BoundingBox(24, 24, 99, 54), 0.99)]}
    )
    reader = WideBandFallbackReader(primary, fallback)

    result = process_document(source, reader)

    [region] = result.pages[0].regions
    assert region.id == "p1-page-1"
    assert region.text == "53"
    assert region.confidence == 0.45
    assert region.bounding_box == BoundingBox(190, 40, 215, 50)
    assert result.pages[0].route == "review"
    assert [alternative.text for alternative in region.alternatives] == ["5.3"]
    assert region.alternatives[0].text_provenance == {
        "method": "scripted-band",
        "stage": "scripted-band-view-wide-band-fallback",
        "source_crop": {"left": 182, "top": 32, "right": 223, "bottom": 58},
        "upscale_factor": 3,
        "selected_view": "fallback",
        "fallback_region_id": "p1-wide-band-1-fallback-1",
        "fallback_bounding_box": {
            "left": 190,
            "top": 40,
            "right": 215,
            "bottom": 50,
        },
    }
    page = reader.coverage_assessment(1)["pages"][0]
    assert fallback.calls == ["band-1.png"]
    assert page["fallback_reader_runs"] == 1
    assert page["replaced_bands"] == 0
    assert page["bands"][0]["review_only"] is True
    assert page["bands"][0]["reason"] == "precision_sensitive_review"


def test_wide_band_reader_groups_low_confidence_words_into_one_band(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (300, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("faint", BoundingBox(20, 12, 70, 20), 0.7),
                ("instruction", BoundingBox(76, 12, 156, 20), 0.72),
                ("line", BoundingBox(164, 12, 220, 20), 0.74),
            ]
        }
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                ("faint instruction line", BoundingBox(24, 24, 624, 54), 0.95),
            ]
        }
    )

    reader = WideBandFallbackReader(primary, fallback)
    regions = reader.read(source, 1)

    assert fallback.calls == ["band-1.png"]
    assert [region.text for region in regions] == ["faint instruction line"]
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["qualifying_regions"] == 3
    assert page["replaced_bands"] == 1


def test_wide_band_reader_preserves_baseline_when_fallback_fails(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    baseline = [
        ("weak but present text", BoundingBox(10, 20, 190, 28), 0.6),
    ]
    primary = ScriptedBandView({"page.png": baseline})
    fallback = ScriptedBandView({}, fail_on_crop=True)
    reader = WideBandFallbackReader(primary, fallback)

    result = process_document(source, reader)
    regions = result.pages[0].regions

    assert [region.text for region in regions] == [baseline[0][0]]
    assert result.pages[0].route == "review"
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["status"] == "uncertain"
    assert page["replaced_bands"] == 0
    assert page["bands"][0]["reason"] == "fallback_failed"
    assert page["bands"][0]["failure_code"] == "reader_failed"


def test_wide_band_reader_rejects_deletion_and_repeated_text(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("retain every baseline token here", BoundingBox(10, 20, 190, 28), 0.6),
                ("second adjacent baseline line", BoundingBox(10, 30, 190, 38), 0.62),
            ]
        }
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                ("repeated text repeated text", BoundingBox(24, 24, 510, 48), 0.99),
            ]
        }
    )
    reader = WideBandFallbackReader(primary, fallback)

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == [
        "retain every baseline token here",
        "second adjacent baseline line",
    ]
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selected_view"] == "baseline"
    assert band["removed_tokens"] > 0
    assert band["repeated_text_risk"] is True


def test_wide_band_reader_recovers_missing_text_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (1205, 781), "white").save(source)
    weak_tokens = [f"b{index:02d}" for index in range(31)]
    baseline = [
        (" ".join(weak_tokens[:16]), BoundingBox(80, 30, 1100, 50), 0.42),
        (" ".join(weak_tokens[16:]), BoundingBox(80, 54, 1100, 74), 0.446),
    ]
    baseline.extend(
        (f"strong-{index}", BoundingBox(20, 100, 200, 112), 0.96) for index in range(84)
    )
    recovered = []
    recovered_tokens = [*weak_tokens, *(f"r{index:02d}" for index in range(37))]
    for index, token in enumerate(recovered_tokens):
        row, column = divmod(index, 34)
        recovered.append(
            (
                token,
                BoundingBox(
                    24 + (column * 88),
                    24 + (row * 60),
                    104 + (column * 88),
                    44 + (row * 60),
                ),
                0.86,
            )
        )
    recovered.append(("---", BoundingBox(3020, 24, 3070, 44), 0.99))
    primary = ScriptedBandView({"page.png": baseline})
    fallback = ScriptedBandView({"band-1.png": recovered})
    reader = WideBandFallbackReader(primary, fallback)

    result = process_document(source, reader)

    assert result.status == "success"
    assert len(result.pages[0].regions) == 152
    assert all(region.text != "---" for region in result.pages[0].regions)
    assessment = reader.coverage_assessment(1)["pages"][0]
    assert assessment["ran"] is True
    assert assessment["fallback_reader"] == "scripted-band-view"
    assert assessment["fallback_reader_runs"] == 1
    assert assessment["fallback_candidates"] == 69
    assert assessment["replaced_bands"] == 1
    band = assessment["bands"][0]
    assert band["selection_reason"] == "missing_text_recovery"
    assert band["baseline_tokens"] == 31
    assert band["fallback_tokens"] == 68
    assert band["baseline_token_recall"] == 1.0
    assert band["baseline_mean_confidence"] == 0.433
    assert band["fallback_mean_confidence"] == 0.86
    assert band["ignored_fallback_regions"] == 1
    assert band["valid_text"] is True
    assert band["plausible_density"] is True
    assert band["repeated_text_risk"] is False


def test_wide_band_reader_rejects_weak_missing_text_fallback(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("b00 b01 b02", BoundingBox(10, 20, 190, 28), 0.4),
            ]
        }
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                (
                    f"r{index:02d}",
                    BoundingBox(24 + index * 50, 24, 64 + index * 50, 44),
                    0.79,
                )
                for index in range(6)
            ]
        }
    )
    confirmation = ScriptedBandView({})
    reader = WideBandFallbackReader(
        primary,
        fallback,
        confirmation_reader=confirmation,
    )

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == ["b00 b01 b02"]
    assert confirmation.calls == []
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selected_view"] == "baseline"
    assert band["fallback_mean_confidence"] == 0.79


def test_wide_band_reader_keeps_unrelated_fallback_as_review_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (200, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("canonical clinical text", BoundingBox(10, 20, 190, 28), 0.4),
            ]
        }
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                ("unrelated", BoundingBox(24, 24, 160, 44), 0.96),
                ("hallucinated", BoundingBox(170, 24, 330, 44), 0.97),
                ("content", BoundingBox(340, 24, 460, 44), 0.98),
            ]
        }
    )
    reader = WideBandFallbackReader(primary, fallback)

    result = process_document(source, reader)

    region = result.pages[0].regions[0]
    assert region.text == "canonical clinical text"
    assert region.resolution == "resolved"
    assert result.pages[0].route == "review"
    assert {alternative.text for alternative in region.alternatives} == {
        "unrelated",
        "hallucinated",
        "content",
    }
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selected_view"] == "baseline"
    assert band["baseline_token_recall"] == 0.0


def test_wide_band_reader_keeps_same_engine_confirmation_for_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (300, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("alpha beta gamma delta", BoundingBox(10, 20, 290, 28), 0.4),
            ]
        }
    )
    recovered = "one two three four five six seven eight"
    fallback = ScriptedBandView(
        {"band-1.png": [(recovered, BoundingBox(24, 24, 840, 48), 0.87)]}
    )
    confirmation = ScriptedBandView(
        {"band-1.png": [(recovered, BoundingBox(24, 24, 840, 48), 0.86)]}
    )
    reader = WideBandFallbackReader(
        primary,
        fallback,
        confirmation_reader=confirmation,
    )

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == ["alpha beta gamma delta"]
    assert confirmation.calls == ["band-1.png"]
    alternatives = regions[0].alternatives
    assert {alternative.text for alternative in alternatives} == {
        recovered,
    }
    assert {
        alternative.text_provenance["selected_view"] for alternative in alternatives
    } == {
        "fallback",
        "confirmation",
    }
    support = next(
        alternative
        for alternative in alternatives
        if alternative.text_provenance["selected_view"] == "confirmation"
    )
    assert support.text_provenance["confirmation_region_id"]
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selected_view"] == "baseline"
    assert band["reason"] == "same_engine_agreement_review"
    assert band["selection_reason"] is None
    assert band["confirmation_status"] == "agreed"
    assert band["confirmation_fallback_recall"] == 1.0
    assert band["confirmation_token_recall"] == 1.0
    assert band["confirmation_text_similarity"] == 1.0
    assert regions[0].id != support.text_provenance["confirmation_region_id"]
    assert reader.page_needs_review(1) is True


def test_wide_band_reader_rejects_disagreeing_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (300, 120), "white").save(source)
    primary_text = "alpha beta gamma delta"
    fallback_text = "one two three four five six seven eight"
    confirmation_text = "nine ten eleven twelve thirteen fourteen fifteen sixteen"
    primary = ScriptedBandView(
        {"page.png": [(primary_text, BoundingBox(10, 20, 290, 28), 0.4)]}
    )
    fallback = ScriptedBandView(
        {"band-1.png": [(fallback_text, BoundingBox(24, 24, 840, 48), 0.87)]}
    )
    confirmation = ScriptedBandView(
        {
            "band-1.png": [
                (confirmation_text, BoundingBox(24, 24, 840, 48), 0.86),
            ]
        }
    )
    reader = WideBandFallbackReader(
        primary,
        fallback,
        confirmation_reader=confirmation,
    )

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == [primary_text]
    assert confirmation.calls == ["band-1.png"]
    assert {alternative.text for alternative in regions[0].alternatives} == {
        fallback_text,
        confirmation_text,
    }
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selected_view"] == "baseline"
    assert band["confirmation_status"] == "disagreed"
    assert band["confirmation_fallback_recall"] == 0.0


def test_wide_band_reader_skips_confirmation_when_fallback_is_selected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (300, 120), "white").save(source)
    primary = ScriptedBandView(
        {
            "page.png": [
                ("alpha beta gamma delta", BoundingBox(10, 20, 290, 28), 0.4),
            ]
        }
    )
    fallback_text = "alphx betx gammx deltx one"
    fallback = ScriptedBandView(
        {"band-1.png": [(fallback_text, BoundingBox(24, 24, 840, 48), 0.87)]}
    )
    confirmation = ScriptedBandView(
        {
            "band-1.png": [
                ("unrelated confirmation text", BoundingBox(24, 24, 840, 48), 0.9)
            ]
        }
    )
    reader = WideBandFallbackReader(
        primary,
        fallback,
        confirmation_reader=confirmation,
    )

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == [fallback_text]
    assert confirmation.calls == []
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selection_reason"] == "correction"


def test_wide_band_reader_rejects_confirmed_text_with_literal_disagreement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (300, 120), "white").save(source)
    baseline_text = " ".join(f"base{chr(97 + index)}" for index in range(20))
    shared_tokens = [f"t{index:02d}" for index in range(49)]
    fallback_lines = [
        " ".join(shared_tokens[:25]),
        " ".join([*shared_tokens[25:], "warfarin10mg"]),
    ]
    confirmation_lines = [
        fallback_lines[0],
        " ".join([*shared_tokens[25:], "warfarin1mg"]),
    ]
    primary = ScriptedBandView(
        {"page.png": [(baseline_text, BoundingBox(10, 20, 290, 28), 0.4)]}
    )
    fallback = ScriptedBandView(
        {
            "band-1.png": [
                (fallback_lines[0], BoundingBox(24, 3, 840, 27), 0.87),
                (fallback_lines[1], BoundingBox(24, 36, 840, 60), 0.87),
            ]
        }
    )
    confirmation = ScriptedBandView(
        {
            "band-1.png": [
                (confirmation_lines[0], BoundingBox(24, 3, 840, 27), 0.86),
                (confirmation_lines[1], BoundingBox(24, 36, 840, 60), 0.86),
            ]
        }
    )
    reader = WideBandFallbackReader(
        primary,
        fallback,
        confirmation_reader=confirmation,
    )

    regions = reader.read(source, 1)

    assert [region.text for region in regions] == [baseline_text]
    assert confirmation.calls == ["band-1.png"]
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["confirmation_status"] == "disagreed"
    assert band["confirmation_fallback_recall"] == 0.98
    assert band["confirmation_token_recall"] == 0.98
    assert band["confirmation_literal_agreement"] is False


def test_wide_band_confirmation_rejects_reordered_literals() -> None:
    shared = [f"token{index:03d}" for index in range(200)]
    fallback_text = " ".join([*shared, "warfarin10mg", "heparin1mg"])
    confirmation_text = " ".join([*shared, "heparin1mg", "warfarin10mg"])
    crop = {"left": 0, "top": 0, "right": 10000, "bottom": 20}

    def region(text: str) -> TextRegion:
        return TextRegion(
            id="region",
            kind="text",
            text=text,
            confidence=0.9,
            bounding_box=BoundingBox(0, 0, 10000, 20),
            reading_order=1,
            provider="test",
            text_provenance={"source_crop": crop},
        )

    confirmed, assessment = _assess_band_confirmation(
        [region(fallback_text)],
        [region(confirmation_text)],
        0.8,
    )

    assert assessment["confirmation_fallback_recall"] == 1.0
    assert assessment["confirmation_token_recall"] == 1.0
    assert assessment["confirmation_text_similarity"] >= 0.98
    assert assessment["confirmation_literal_agreement"] is False
    assert confirmed is False
