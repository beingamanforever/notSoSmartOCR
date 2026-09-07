from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.preprocessing import RestoredViewReader
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.restoration import DocResRestorer


def region(
    identifier: str,
    text: str,
    confidence: float,
    box: tuple[int, int, int, int],
    provider: str = "controlled",
) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind="word",
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(*box),
        reading_order=int(identifier.rsplit("-", 1)[-1]),
        provider=provider,
        text_provenance={"method": "controlled"},
    )


class ControlledRestoreView:
    """Reads faint text on the source page and more text on the restored page."""

    name = "controlled-restore-view"

    def __init__(self, baseline_confidence: float = 0.3) -> None:
        self.baseline_confidence = baseline_confidence
        self.calls: list[str] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.calls.append(image_path.name)
        if "restored" in image_path.name:
            return [
                region("p1-word-1", "Allergies:", 0.95, (10, 10, 90, 30)),
                region("p1-word-2", "Gender:", 0.94, (10, 40, 80, 60)),
            ]
        return [
            region(
                "p1-word-1", "Allergies:", self.baseline_confidence, (10, 10, 90, 30)
            )
        ]


class ControlledRestorer:
    def __init__(self, *, usable: bool = True, failing: bool = False) -> None:
        self.usable = usable
        self.failing = failing
        self.restored: list[Path] = []

    def available(self) -> bool:
        return self.usable

    def restore(self, image_path: Path, destination: Path) -> Path:
        if self.failing:
            raise ReaderError("restoration_write_failed", "no")
        self.restored.append(image_path)
        shutil.copyfile(image_path, destination)
        return destination


def page(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("L", (200, 120), 255).save(path)
    return path


def test_restored_view_preserves_baseline_and_promotes_recovered_regions(
    tmp_path,
) -> None:
    view = ControlledRestoreView()
    restorer = ControlledRestorer()
    reader = RestoredViewReader(view, restorer)

    regions = reader.read(page(tmp_path), 1)

    texts = [item.text for item in regions]
    assert "Allergies:" in texts
    baseline = next(item for item in regions if item.text == "Allergies:")
    assert baseline.confidence == 0.3, "baseline transcription must survive untouched"
    assert baseline.alternatives, "the restored reading is attached as an alternative"

    # A single restored view gives no second observation, so a restored-only region is
    # surfaced as evidence for review and never promoted to canonical text on its own.
    recovered = next(item for item in regions if item.text == "Gender:")
    assert recovered.resolution != "resolved"

    assessment = reader.coverage_assessment(1)["pages"][0]
    assert assessment["status"] == "uncertain"
    assert assessment["restoration_model"]["license"] == "MIT"
    assert assessment["unresolved_restored_regions"] == 1
    assert assessment["added_tokens"] == 0, "nothing is promoted without corroboration"


def test_restored_view_skips_legible_pages(tmp_path) -> None:
    view = ControlledRestoreView(baseline_confidence=0.96)
    restorer = ControlledRestorer()
    reader = RestoredViewReader(view, restorer)

    regions = reader.read(page(tmp_path), 1)

    assert [item.text for item in regions] == ["Allergies:"]
    assert restorer.restored == [], "a legible page must not pay for restoration"
    assert reader.coverage_assessment(1)["pages"][0]["reason"] == "baseline_legible"


def test_restored_view_skips_when_the_restorer_is_unavailable(tmp_path) -> None:
    restorer = ControlledRestorer(usable=False)
    reader = RestoredViewReader(ControlledRestoreView(), restorer)

    regions = reader.read(page(tmp_path), 1)

    assert [item.text for item in regions] == ["Allergies:"]
    assert reader.coverage_assessment(1)["pages"][0]["reason"] == "restorer_unavailable"


def test_restored_view_falls_back_to_baseline_when_restoration_fails(tmp_path) -> None:
    reader = RestoredViewReader(
        ControlledRestoreView(), ControlledRestorer(failing=True)
    )

    regions = reader.read(page(tmp_path), 1)

    assert [item.text for item in regions] == ["Allergies:"]
    assessment = reader.coverage_assessment(1)["pages"][0]
    assert assessment["status"] == "failed"
    assert assessment["review_reasons"] == ["restoration_failed"]


def test_docres_restorer_reports_unavailable_without_a_checkout(tmp_path) -> None:
    assert not DocResRestorer(tmp_path / "docres.pkl", tmp_path).available()


def test_large_pages_are_tiled_at_source_resolution_without_seams() -> None:
    """A clinical scan must not be squashed into a square before restoration."""
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    restorer = DocResRestorer(Path("unused.pkl"), Path("unused"))
    # An identity model: whatever the tiler feeds in comes back as the RGB channels, so a
    # correct reassembly reproduces the source exactly and any seam or offset shows up.
    restorer._model = (None, None, None)
    restorer._predict = lambda patch, torch, model, np: patch[:, :, :3].copy()

    rows, columns = numpy.mgrid[0:2200, 0:1700]
    image = numpy.stack(
        [
            (rows % 251).astype("uint8"),
            (columns % 241).astype("uint8"),
            rows.astype("uint8"),
        ],
        axis=-1,
    )
    combined = numpy.concatenate((image, image), -1)

    restored = restorer._restore_tiles(combined, numpy)

    assert restored.shape == image.shape, "restoration must keep the source resolution"
    assert numpy.array_equal(restored, image), "tiles must reassemble without seams"


def test_every_decorating_reader_reports_its_coverage_assessment(tmp_path) -> None:
    """Asking only the outermost reader dropped the tile and restoration assessments."""
    from ocr_pipeline.demo import _coverage_assessment

    class Inner:
        name = "inner"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return []

        def coverage_assessment(self, page_count: int) -> dict[str, object]:
            return {"status": "review_recommended", "message": "inner", "pages": []}

    class Outer:
        name = "outer"

        def __init__(self, reader) -> None:
            self.reader = reader

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return []

        def coverage_assessment(self, page_count: int) -> dict[str, object]:
            return {"status": "review_recommended", "message": "outer", "pages": []}

    assessment = _coverage_assessment(Outer(Inner()), 1)

    assert [item["reader"] for item in assessment["readers"]] == ["outer", "inner"]
    assert assessment["message"] == "outer", "the outermost stays the headline"
