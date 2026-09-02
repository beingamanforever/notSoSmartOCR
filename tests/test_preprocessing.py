from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.preprocessing import (
    RoutedTesseractReader,
    TiledReader,
    WideBandFallbackReader,
    _assess_band_confirmation,
    locate_dark_frame,
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

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.calls.append(image_path.name)
        with Image.open(image_path) as image:
            self.sizes.append(image.size)
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
    assert page["selected_view"] == "fused"
    assert page["baseline_regions"] == 2
    assert page["preserved_baseline_regions"] == 2
    assert page["exact_overlap_candidates"] == 1
    assert page["conflicting_candidates"] == 1
    assert page["unresolved_tile_only_regions"] == 1
    assert page["removed_tokens"] == 0


def test_tiled_reader_promotes_only_multi_view_tile_agreement(tmp_path: Path) -> None:
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
    assert regions[1].resolution == "resolved"
    assert regions[1].structure == {
        "role": "tiny_text_candidate",
        "support_views": 2,
    }
    assert [alternative.text for alternative in regions[1].alternatives] == [
        "fine print"
    ]
    assert render_evidence(regions).value == "Base fine print"
    assert render_evidence(regions).evidence_ids == [
        "p1-page-1",
        "p1-tiny-tile-2",
    ]
    page = reader.coverage_assessment(1)["pages"][0]
    assert page["promoted_tile_only_regions"] == 1
    assert page["unresolved_tile_only_regions"] == 0
    assert page["added_tokens"] == 2


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
    assert assessment["pages"][0]["status"] == "not_routed"


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
    assert fallback.calls == ["band-1.png", "band-2.png"]
    assert fallback.sizes == [(513, 102), (510, 72)]
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


def test_wide_band_reader_accepts_independently_confirmed_recovery(
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

    assert [region.text for region in regions] == [recovered]
    assert confirmation.calls == ["band-1.png"]
    alternatives = regions[0].alternatives
    assert {alternative.text for alternative in alternatives} == {
        "alpha beta gamma delta",
        recovered,
    }
    support = next(
        alternative for alternative in alternatives if alternative.text == recovered
    )
    assert support.text_provenance["selected_view"] == "confirmation"
    assert support.text_provenance["confirmation_region_id"]
    band = reader.coverage_assessment(1)["pages"][0]["bands"][0]
    assert band["selection_reason"] == "independent_view_confirmation"
    assert band["confirmation_status"] == "agreed"
    assert band["confirmation_fallback_recall"] == 1.0
    assert band["confirmation_token_recall"] == 1.0
    assert band["confirmation_text_similarity"] == 1.0
    assert regions[0].id != support.text_provenance["confirmation_region_id"]


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
