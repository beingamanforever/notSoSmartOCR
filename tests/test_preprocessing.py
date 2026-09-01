from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.preprocessing import (
    RoutedTesseractReader,
    TiledReader,
    locate_dark_frame,
)
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
