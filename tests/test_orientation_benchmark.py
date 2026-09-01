from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.providers import ReaderError

from experiments import orientation_benchmark


class FakeReader:
    name = "fake"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            assert image.size == (20, 40)
        return [
            TextRegion(
                id="second",
                kind="text",
                text="world",
                confidence=0.8,
                bounding_box=BoundingBox(1, 20, 10, 30),
                reading_order=2,
                provider=self.name,
            ),
            TextRegion(
                id="first",
                kind="text",
                text="hello",
                confidence=0.9,
                bounding_box=BoundingBox(1, 1, 10, 10),
                reading_order=1,
                provider=self.name,
            ),
        ]


class SelectiveReader:
    name = "selective"

    def __init__(self, failures: set[int] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[int] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        angle = int(image_path.stem.rsplit("-", maxsplit=1)[1])
        self.calls.append(angle)
        if angle in self.failures:
            raise ReaderError("view_failed", f"angle {angle} failed")
        confidence = {0: 0.4, 90: 0.6, 180: 0.95, 270: 0.5}[angle]
        return [
            TextRegion(
                id=f"angle-{angle}",
                kind="text",
                text=f"text at {angle}",
                confidence=confidence,
                bounding_box=BoundingBox(0, 0, 10, 10),
                reading_order=1,
                provider=self.name,
            )
        ]


def test_rotated_view_returns_text_in_original_page_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "black").save(image_path)

    regions = orientation_benchmark.RotatedViewReader(FakeReader(), 90).read(
        image_path,
        1,
    )

    assert len(regions) == 1
    assert regions[0].text == "hello world"
    assert regions[0].bounding_box == BoundingBox(0, 0, 40, 20)
    assert regions[0].provider == "fake-rotate-90"


def test_zero_rotation_normalizes_exif_before_reader(tmp_path: Path) -> None:
    image_path = tmp_path / "tagged.jpg"
    image = Image.new("RGB", (40, 20), "black")
    exif = image.getexif()
    exif[274] = 6
    image.save(image_path, exif=exif)

    regions = orientation_benchmark.RotatedViewReader(FakeReader(), 0).read(
        image_path,
        1,
    )

    assert regions[0].bounding_box == BoundingBox(0, 0, 20, 40)
    assert regions[0].text_provenance == {
        "exif_orientation": 6,
        "exif_transposed": True,
    }


def test_rectify_document_extracts_page_quadrilateral() -> None:
    image = Image.new("RGB", (400, 400), "black")
    ImageDraw.Draw(image).polygon(
        [(100, 40), (330, 80), (360, 350), (40, 320)],
        fill="white",
    )

    rectified = orientation_benchmark.rectify_document(image)

    assert rectified.width > 250
    assert rectified.height > 250
    assert rectified.getpixel((rectified.width // 2, rectified.height // 2))[0] > 240


def test_rectify_document_keeps_content_outside_inner_box() -> None:
    image = Image.new("RGB", (600, 600), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(-80, 260), (250, -80), (680, 170), (680, 430), (350, 680), (-80, 500)],
        fill="white",
    )
    draw.rectangle((120, 140, 500, 450), outline="black", width=10)
    draw.rectangle((260, 35, 340, 85), fill="red")
    draw.rectangle((260, 515, 340, 565), fill="blue")

    rectified = np.asarray(orientation_benchmark.rectify_document(image))

    red_pixels = (rectified[:, :, 0] > 180) & (rectified[:, :, 1] < 100)
    blue_pixels = (rectified[:, :, 2] > 180) & (rectified[:, :, 1] < 100)
    assert red_pixels.sum() > 100
    assert blue_pixels.sum() > 100


def test_rectification_padding_preserves_content_beyond_detected_edge() -> None:
    image = Image.new("RGB", (400, 400), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 50, 300, 350), fill="white")
    draw.rectangle((95, 180, 98, 220), fill="red")

    unpadded = np.asarray(
        orientation_benchmark.rectify_document(image, padding_fraction=0)
    )
    padded = np.asarray(orientation_benchmark.rectify_document(image))

    unpadded_red = (unpadded[:, :, 0] > 180) & (unpadded[:, :, 1] < 100)
    padded_red = (padded[:, :, 0] > 180) & (padded[:, :, 1] < 100)
    assert unpadded_red.sum() == 0
    assert padded_red.sum() > 20


def test_corner_order_keeps_four_unique_points_for_diamond() -> None:
    corners = np.array(
        [[200, 10], [390, 200], [200, 390], [10, 200]],
        dtype=np.float32,
    )

    ordered = orientation_benchmark._order_corners(corners)

    assert len({tuple(point) for point in ordered}) == 4


def test_corner_order_is_cyclic_for_perspective_trapezoid() -> None:
    corners = np.array(
        [[100, 100], [150, 80], [300, 400], [200, 350]],
        dtype=np.float32,
    )

    ordered = orientation_benchmark._order_corners(corners)

    assert ordered.tolist() == corners.tolist()


def test_orientation_score_prefers_confident_literal_text() -> None:
    strong = [
        TextRegion(
            id="strong",
            kind="text",
            text="literal text",
            confidence=0.95,
            bounding_box=BoundingBox(0, 0, 10, 10),
            reading_order=1,
            provider="fake",
        )
    ]
    weak = [
        TextRegion(
            id="weak",
            kind="text",
            text="ssssssss guessed text",
            confidence=0.6,
            bounding_box=BoundingBox(0, 0, 10, 10),
            reading_order=1,
            provider="fake",
        )
    ]

    assert (
        orientation_benchmark.orientation_score(strong)["rank"]
        > (orientation_benchmark.orientation_score(weak)["rank"])
    )


def test_orientation_score_prefers_complete_page_over_tiny_fragment() -> None:
    fragment = [
        TextRegion(
            id="fragment",
            kind="word",
            text="A",
            confidence=0.99,
            bounding_box=BoundingBox(0, 0, 10, 10),
            reading_order=1,
            provider="fake",
        )
    ]
    complete = [
        TextRegion(
            id=f"word-{index}",
            kind="word",
            text="clear",
            confidence=0.98,
            bounding_box=BoundingBox(0, index, 10, index + 1),
            reading_order=index,
            provider="fake",
        )
        for index in range(1, 11)
    ]

    assert (
        orientation_benchmark.orientation_score(complete)["rank"]
        > (orientation_benchmark.orientation_score(fragment)["rank"])
    )


def test_auto_orientation_selects_best_surviving_view(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "black").save(image_path)
    reader = orientation_benchmark.AutoOrientationReader(SelectiveReader(failures={90}))

    regions = reader.read(image_path, 1)

    assert regions[0].text == "text at 180"
    assert regions[0].bounding_box == BoundingBox(0, 0, 40, 20)
    assert reader.selections[0]["angle"] == 180
    assert reader.selections[0]["view_failures"] == {
        "90": {"code": "view_failed", "message": "angle 90 failed"}
    }


def test_auto_orientation_uses_confident_osd_once(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "black").save(image_path)
    base_reader = SelectiveReader()
    monkeypatch.setattr(
        orientation_benchmark,
        "detect_tesseract_orientation",
        lambda path, executable: {
            "angle": 180,
            "rotate_clockwise": 180,
            "confidence": 20.0,
            "script": "Latin",
            "script_confidence": 12.0,
        },
    )
    reader = orientation_benchmark.AutoOrientationReader(
        base_reader,
        osd_executable="tesseract",
    )

    regions = reader.read(image_path, 1)

    assert regions[0].text == "text at 180"
    assert base_reader.calls == [180]
    assert reader.selections[0]["selector"] == "tesseract_osd"
    assert reader.selections[0]["angle"] == 180


def test_auto_orientation_direct_osd_skips_word_reader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "black").save(image_path)

    class DirectReader(SelectiveReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            raise AssertionError("direct OSD must skip the word reader")

        def read_with_merge_level(
            self,
            image_path: Path,
            page_number: int,
            merge_level: str,
        ) -> list[TextRegion]:
            assert image_path.stem == "page-180"
            assert merge_level == "paragraph"
            return [
                TextRegion(
                    id="final",
                    kind="text",
                    text="final text",
                    confidence=0.9,
                    bounding_box=BoundingBox(0, 0, 10, 10),
                    reading_order=1,
                    provider=self.name,
                )
            ]

    monkeypatch.setattr(
        orientation_benchmark,
        "detect_tesseract_orientation",
        lambda path, executable: {
            "angle": 180,
            "rotate_clockwise": 180,
            "confidence": 3.0,
            "script": "Latin",
            "script_confidence": 1.0,
        },
    )
    reader = orientation_benchmark.AutoOrientationReader(
        DirectReader(),
        osd_executable="tesseract",
        osd_min_confidence=0,
        final_merge_level="paragraph",
        direct_osd=True,
    )

    regions = reader.read(image_path, 1)

    assert regions[0].text == "final text"
    assert reader.selections[0]["selector"] == "tesseract_osd_direct"
    assert reader.selections[0]["view_scores"] == {}


def test_auto_orientation_records_all_view_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "black").save(image_path)
    reader = orientation_benchmark.AutoOrientationReader(
        SelectiveReader(failures=set(orientation_benchmark.ROTATIONS))
    )

    with pytest.raises(ReaderError) as raised:
        reader.read(image_path, 1)

    assert raised.value.code == "orientation_views_failed"
    assert reader.selections[0]["angle"] is None
    assert set(reader.selections[0]["view_failures"]) == {
        "0",
        "90",
        "180",
        "270",
    }


def test_auto_orientation_records_empty_final_rerun(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "black").save(image_path)

    class EmptyFinalReader(SelectiveReader):
        def read_with_merge_level(
            self,
            image_path: Path,
            page_number: int,
            merge_level: str,
        ) -> list[TextRegion]:
            assert merge_level == "paragraph"
            return []

    reader = orientation_benchmark.AutoOrientationReader(
        EmptyFinalReader(),
        final_merge_level="paragraph",
    )

    with pytest.raises(ReaderError) as raised:
        reader.read(image_path, 1)

    assert raised.value.code == "final_empty_output"
    assert reader.selections[0]["failure"]["code"] == "final_empty_output"
    summary = orientation_benchmark.orientation_attempt_summary(
        reader.selections,
        [{"status": "failed"}],
    )
    assert summary["selection_failures"] == 1


def test_orientation_attempt_summary_counts_recovered_view_failure() -> None:
    selections = [
        {
            "angle": 180,
            "selected_score": {},
            "view_scores": {"0": {}, "180": {}, "270": {}},
            "view_failures": {"90": {"code": "view_failed"}},
        }
    ]

    summary = orientation_benchmark.orientation_attempt_summary(
        selections,
        [{"status": "success"}],
    )

    assert summary == {
        "pages": 1,
        "view_attempts": 4,
        "view_failures": 1,
        "affected_pages": 1,
        "recovered_pages": 1,
        "selection_failures": 0,
        "osd_attempts": 0,
        "osd_failures": 0,
        "osd_selected_pages": 0,
        "osd_direct_pages": 0,
        "osd_fallback_pages": 0,
    }


def test_tesseract_osd_parses_clockwise_correction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        orientation_benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "Orientation in degrees: 270\n"
                "Rotate: 90\n"
                "Orientation confidence: 18.5\n"
                "Script: Latin\n"
                "Script confidence: 9.4\n"
            ),
            stderr="warning",
        ),
    )

    result = orientation_benchmark.detect_tesseract_orientation(tmp_path / "page.png")

    assert result == {
        "angle": 270,
        "rotate_clockwise": 90,
        "confidence": 18.5,
        "script": "Latin",
        "script_confidence": 9.4,
    }


def test_rectification_failure_is_explicit(tmp_path: Path) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (40, 20), "black").save(image_path)

    with pytest.raises(ReaderError) as raised:
        orientation_benchmark.RotatedViewReader(
            FakeReader(),
            0,
            rectify=True,
        ).read(image_path, 1)

    assert getattr(raised.value, "code", None) == "rectification_failed"


def test_orientation_cli_uses_exemplars_and_all_integer_rotations(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class FakeNemotron:
        name = "nemotron-ocr-v2"

        def __init__(self, *, language: str, merge_level: str) -> None:
            assert language == "en"
            assert merge_level == "paragraph"
            self.merge_level = merge_level

    def fake_run_benchmark(dataset, root, workers, reader, **options):
        calls.append((dataset, workers, reader.name, options))
        return {"summary": {"cases": 1}}

    monkeypatch.setattr(orientation_benchmark, "NemotronOCRV2Reader", FakeNemotron)
    monkeypatch.setattr(orientation_benchmark, "run_benchmark", fake_run_benchmark)
    output = tmp_path / "orientation.json"

    assert orientation_benchmark.main([str(tmp_path), str(output)]) == 0

    assert [call[2] for call in calls] == [
        "nemotron-ocr-v2-rotate-0",
        "nemotron-ocr-v2-rotate-90",
        "nemotron-ocr-v2-rotate-180",
        "nemotron-ocr-v2-rotate-270",
    ]
    assert all(call[3]["clinocr_role"] == "exemplar" for call in calls)
    assert all(call[3]["selected_subsets"] == {"rotated"} for call in calls)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cold_start_in_angle"] == 0
    assert payload["rectify"] is False
    assert payload["rectify_padding"] is None
    assert payload["auto_select"] is False
    assert payload["merge_level"] == "paragraph"
    assert payload["final_merge_level"] is None
    assert payload["orientation_selector"] == "nemotron"
    assert payload["osd_direct"] is False
    assert list(payload["runs"]) == ["0", "90", "180", "270"]


def test_orientation_cli_auto_selects_and_serializes_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (40, 20), "black").save(source)

    class FakeNemotron(SelectiveReader):
        name = "nemotron-ocr-v2"

        def __init__(self, *, language: str, merge_level: str) -> None:
            assert language == "en"
            assert merge_level == "word"
            super().__init__(failures={90})

        def read_with_merge_level(
            self,
            image_path: Path,
            page_number: int,
            merge_level: str,
        ) -> list[TextRegion]:
            assert merge_level == "paragraph"
            return self.read(image_path, page_number)

    def fake_run_benchmark(dataset, root, workers, reader, **options):
        regions = reader.read(source, 1)
        assert regions[0].text == "text at 180"
        return {"cases": [{"status": "success"}], "summary": {"cases": 1}}

    monkeypatch.setattr(orientation_benchmark, "NemotronOCRV2Reader", FakeNemotron)
    monkeypatch.setattr(orientation_benchmark, "run_benchmark", fake_run_benchmark)
    output = tmp_path / "orientation.json"

    assert (
        orientation_benchmark.main([str(tmp_path), str(output), "--auto-select"]) == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["merge_level"] == "word"
    assert payload["final_merge_level"] == "paragraph"
    assert payload["rectify_padding"] is None
    assert payload["cold_start_in_angle"] is None
    automatic_run = payload["runs"]["auto"]
    assert automatic_run["cases"][0]["orientation_selection"]["angle"] == 180
    assert (
        automatic_run["cases"][0]["orientation_selection"]["final_merge_level"]
        == "paragraph"
    )
    assert automatic_run["orientation_summary"] == {
        "pages": 1,
        "view_attempts": 4,
        "view_failures": 1,
        "affected_pages": 1,
        "recovered_pages": 1,
        "selection_failures": 0,
        "osd_attempts": 0,
        "osd_failures": 0,
        "osd_selected_pages": 0,
        "osd_direct_pages": 0,
        "osd_fallback_pages": 0,
    }
