from __future__ import annotations

from pathlib import Path

from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.risk import EvidenceRiskStage


def test_small_text_evidence_routes_review_without_changing_text(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path)
    regions = [_region(index, 0.85, 8) for index in range(12)]

    result = process_document(
        source,
        FixedReader(regions),
        stages=(EvidenceRiskStage(text_provider="fixed"),),
    )

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == " ".join(
        f"evidence {index}" for index in range(12)
    )
    risk = result.pages[0].regions[-1]
    assert risk.kind == "coverage_risk"
    assert risk.resolution == "unreadable"
    assert risk.structure["reasons"] == ["small_text_evidence"]


def test_large_low_confidence_regions_route_handwriting_like_page(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path)
    regions = [_region(index, 0.95, 14) for index in range(10)]
    regions.extend([_region(10, 0.7, 42), _region(11, 0.75, 50)])

    result = process_document(
        source,
        FixedReader(regions),
        stages=(EvidenceRiskStage(text_provider="fixed"),),
    )

    risk = result.pages[0].regions[-1]
    assert result.pages[0].route == "review"
    assert risk.structure["reasons"] == ["large_low_confidence_regions"]
    assert risk.structure["metrics"]["large_low_confidence_regions"] == 2


def test_strong_evidence_is_unchanged(tmp_path: Path) -> None:
    source = _image(tmp_path)
    regions = [_region(index, 0.95, 14) for index in range(12)]

    result = process_document(
        source,
        FixedReader(regions),
        stages=(EvidenceRiskStage(text_provider="fixed"),),
    )

    assert result.pages[0].route == "accept_local"
    assert result.pages[0].regions == regions


def test_uncertain_table_text_routes_review(tmp_path: Path) -> None:
    source = _image(tmp_path)
    regions = [_region(index, 0.91, 14) for index in range(12)]
    for region in regions:
        region.structure = {"role": "table_source", "parent_id": "table-1"}
    regions.append(
        TextRegion(
            id="table-1",
            kind="table",
            text="",
            confidence=0.98,
            bounding_box=BoundingBox(10, 300, 390, 500),
            reading_order=13,
            provider="table-specialist",
        )
    )

    result = process_document(
        source,
        FixedReader(regions),
        stages=(EvidenceRiskStage(text_provider="fixed"),),
    )

    assert result.pages[0].route == "review"
    risk = result.pages[0].regions[-1]
    assert risk.structure["reasons"] == ["table_text_uncertainty"]


def test_non_primary_specialist_regions_do_not_distort_signal(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path)
    regions = [_region(index, 0.95, 14) for index in range(12)]
    regions.append(
        TextRegion(
            id="table-word",
            kind="word",
            text="weak challenger",
            confidence=0.1,
            bounding_box=BoundingBox(0, 0, 20, 4),
            reading_order=13,
            provider="table-challenger",
        )
    )

    result = process_document(
        source,
        FixedReader(regions),
        stages=(EvidenceRiskStage(text_provider="fixed"),),
    )

    assert result.pages[0].route == "accept_local"
    assert all(region.kind != "coverage_risk" for region in result.pages[0].regions)


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(path)
    return path


def _region(index: int, confidence: float, height: int) -> TextRegion:
    top = 10 + (index * 20)
    return TextRegion(
        id=f"region-{index}",
        kind="text",
        text=f"evidence {index}",
        confidence=confidence,
        bounding_box=BoundingBox(10, top, 200, top + height),
        reading_order=index + 1,
        provider="fixed",
    )


class FixedReader:
    name = "fixed"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions
