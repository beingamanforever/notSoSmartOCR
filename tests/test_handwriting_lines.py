from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.handwriting_lines import (
    CANDIDATE_SOURCE,
    DocTRLineDetector,
    HandwritingLineStage,
    _group_lines,
)
from ocr_pipeline.providers import ReaderError


class ScriptedDetector:
    name = "scripted-lines"

    def __init__(self, lines: list[BoundingBox]) -> None:
        self.lines = lines
        self.calls: list[Path] = []

    @property
    def provenance(self) -> dict[str, object]:
        return {"id": "scripted"}

    def detect(self, image_path: Path) -> list[BoundingBox]:
        self.calls.append(image_path)
        return list(self.lines)


def _region(
    identifier: str,
    text: str,
    box: tuple[int, int, int, int],
    confidence: float | None,
    *,
    kind: str = "text",
) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind=kind,
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(*box),
        reading_order=int(identifier.rsplit("-", 1)[-1]),
        provider="reader",
        text_provenance={},
        resolution="resolved",
        structure={},
    )


def _page(count: int, confidence: float) -> list[TextRegion]:
    return [
        _region(
            f"r-{index}",
            f"line {index}",
            (10, 20 * index, 200, 20 * index + 18),
            confidence,
        )
        for index in range(1, count + 1)
    ]


def test_stage_proposes_lines_when_most_of_the_page_reads_poorly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (240, 240), "white").save(source)
    detector = ScriptedDetector(
        [BoundingBox(8, 18, 210, 40), BoundingBox(8, 38, 210, 60)]
    )

    result = HandwritingLineStage(detector).apply(source, 3, _page(10, 0.5))

    proposals = [region for region in result if region.kind == "handwriting"]
    assert len(proposals) == 2
    assert [region.id for region in proposals] == [
        "p3-handwriting-line-1",
        "p3-handwriting-line-2",
    ]
    for region in proposals:
        assert region.text == ""
        assert region.confidence is None
        assert region.resolution == "unreadable"
        assert region.structure["handwriting_candidate"] is True
        assert region.structure["handwriting_candidate_source"] == CANDIDATE_SOURCE
    # proposals must not disturb the reading order of existing evidence
    assert proposals[0].reading_order > max(r.reading_order for r in _page(10, 0.5))


def test_stage_leaves_a_confidently_read_page_untouched(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (240, 240), "white").save(source)
    detector = ScriptedDetector([BoundingBox(8, 18, 210, 40)])

    regions = _page(10, 0.95)
    assert HandwritingLineStage(detector).apply(source, 1, regions) == regions
    assert detector.calls == []


def test_stage_fires_on_a_single_poorly_read_field(tmp_path: Path) -> None:
    """A mixed printed form must still reach the specialist for its few written fields."""
    source = tmp_path / "page.png"
    Image.new("RGB", (240, 240), "white").save(source)
    detector = ScriptedDetector([BoundingBox(8, 18, 210, 40)])

    regions = _page(9, 0.98)
    regions.append(_region("r-50", "welking", (10, 20, 200, 38), 0.63))

    result = HandwritingLineStage(detector).apply(source, 1, regions)

    assert sum(region.kind == "handwriting" for region in result) == 1


def test_stage_skips_evidence_owned_by_the_table_or_formula_specialists(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (240, 240), "white").save(source)
    detector = ScriptedDetector([BoundingBox(8, 18, 210, 40)])

    for owner in ("table_source", "formula"):
        region = _region("r-50", "cell", (10, 20, 200, 38), 0.4)
        region.structure = {"role": owner}
        assert HandwritingLineStage(detector).apply(source, 1, [region]) == [region]
    assert detector.calls == []


def test_stage_only_proposes_lines_over_poorly_read_evidence(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 400), "white").save(source)
    regions = _page(9, 0.4)
    regions.append(_region("r-99", "clean", (10, 300, 200, 318), 0.99))
    # one line sits over low-confidence text, the other over the confident region
    detector = ScriptedDetector(
        [BoundingBox(8, 18, 210, 40), BoundingBox(8, 298, 210, 320)]
    )

    result = HandwritingLineStage(detector).apply(source, 1, regions)

    proposals = [region for region in result if region.kind == "handwriting"]
    assert [region.bounding_box.top for region in proposals] == [18]


def test_stage_bounds_the_number_of_proposed_lines(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (240, 900), "white").save(source)
    detector = ScriptedDetector(
        [BoundingBox(8, 20 * index, 210, 20 * index + 18) for index in range(1, 40)]
    )

    result = HandwritingLineStage(detector, max_lines=5).apply(
        source, 1, _page(20, 0.3)
    )

    assert sum(region.kind == "handwriting" for region in result) == 5


def test_line_grouping_merges_words_and_keeps_separate_rows_apart() -> None:
    words = [
        BoundingBox(10, 10, 40, 30),
        BoundingBox(45, 12, 80, 32),
        BoundingBox(10, 60, 40, 80),
    ]

    assert _group_lines(words) == [
        BoundingBox(10, 10, 80, 32),
        BoundingBox(10, 60, 40, 80),
    ]


def test_detector_reports_an_unreadable_page_as_a_reader_error(tmp_path: Path) -> None:
    detector = DocTRLineDetector()
    detector._predictor = lambda images: (_ for _ in ()).throw(ValueError("bad page"))

    with pytest.raises(ReaderError) as raised:
        detector.detect(tmp_path / "missing.png")
    assert raised.value.code == "line_detection_failed"


def test_stage_rejects_invalid_configuration() -> None:
    detector = ScriptedDetector([])
    with pytest.raises(ValueError, match="confidence_threshold"):
        HandwritingLineStage(detector, confidence_threshold=1.5)
    with pytest.raises(ValueError, match="max_lines"):
        HandwritingLineStage(detector, max_lines=0)


def test_dense_printed_table_text_does_not_become_handwriting(tmp_path: Path) -> None:
    """Low confidence is not evidence of handwriting: dense table print reads poorly too.

    Table geometry is known even when its structure parse was rejected, so the stage
    must exclude by area rather than by role.
    """
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 400), "white").save(source)
    table = _region("t-1", "", (20, 20, 380, 300), None, kind="table_candidate")
    table.resolution = "unreadable"
    cells = [
        _region(
            f"c-{index}",
            "COVIDD-99 RRN,,LNP-S",
            (30, 30 + index * 20, 370, 46 + index * 20),
            0.55,
        )
        for index in range(1, 10)
    ]
    detector = ScriptedDetector([BoundingBox(28, 28, 372, 60)])

    result = HandwritingLineStage(detector).apply(source, 1, [table, *cells])

    assert sum(region.kind == "handwriting" for region in result) == 0
    assert detector.calls == []
