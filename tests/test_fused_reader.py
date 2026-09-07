"""Falcon's text on Nemotron's geometry: each reader covers the other's weakness."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.fused_reader import FusedReader, fuse
from ocr_pipeline.providers import ReaderError


def _region(
    id_, text, box, *, confidence=0.9, provider="falcon-perception", kind="text"
):
    return TextRegion(
        id=id_,
        kind=kind,
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(*box),
        reading_order=1,
        provider=provider,
        text_provenance={"method": "falcon_perception_layout_ocr"},
    )


def _word(text, box, confidence):
    return _region(
        f"w-{text}", text, box, confidence=confidence, provider="nemotron-ocr-v2"
    )


def test_region_takes_recognition_confidence_and_word_boxes_from_the_word_reader() -> (
    None
):
    regions = [_region("r1", "Deposits market share", (0, 0, 200, 40), confidence=0.51)]
    words = [
        _word("Deposits", (5, 10, 60, 30), 0.99),
        _word("market", (65, 10, 120, 30), 0.80),
        _word("share", (125, 10, 180, 30), 0.70),
    ]

    (fused,) = fuse(regions, words, 1)

    # The layout detector's 0.51 said nothing about the characters; this is recognition.
    assert fused.confidence == pytest.approx((0.99 + 0.80 + 0.70) / 3)
    assert fused.text_provenance["confidence_meaning"] == (
        "recognition, from the word reader"
    )
    assert fused.text_provenance["recognition_confidence_min"] == pytest.approx(0.70)
    assert fused.text_provenance["geometry_provider"] == "nemotron-ocr-v2"
    # Falcon's text is untouched; only geometry and confidence come from the other reader.
    assert fused.text == "Deposits market share"
    boxes = fused.structure["word_evidence"]
    assert [item["text"] for item in boxes] == ["Deposits", "market", "share"]
    assert boxes[0]["bbox"] == {"left": 5, "top": 10, "right": 60, "bottom": 30}


def test_words_outside_every_region_are_recovered_as_lines_with_real_geometry() -> None:
    """The JPMorgan page was detected as one table, losing every bullet beside it."""
    regions = [_region("r1", "<the table>", (0, 0, 100, 100), kind="table")]
    words = [
        # Claimed by the table region and absent from its text: under-read repair.
        _word("Serve", (10, 10, 40, 30), 0.95),
        _word("84M", (45, 10, 80, 30), 0.93),
        # Outside every region: uncovered-word recovery.
        _word("On-ground", (200, 10, 260, 30), 0.91),
        _word("presence", (265, 12, 330, 32), 0.90),
        _word("in", (200, 60, 215, 80), 0.88),
        _word("177", (220, 60, 250, 80), 0.87),
    ]

    fused = fuse(regions, words, 1)

    assert fused[0].text == "<the table>"
    uncovered = [
        region
        for region in fused
        if (region.text_provenance or {}).get("method") == "word_reader_uncovered"
    ]
    assert [region.text for region in uncovered] == [
        "On-ground presence",
        "in 177",
    ]
    first = uncovered[0]
    assert first.confidence == pytest.approx((0.91 + 0.90) / 2)
    # The union of the words it was built from, never an invented box.
    box = first.bounding_box
    assert (box.left, box.top, box.right, box.bottom) == (200, 10, 330, 32)
    repaired = [
        region
        for region in fused
        if (region.text_provenance or {}).get("method") == "region_underread_repair"
    ]
    assert [region.text for region in repaired] == ["Serve 84M"]


def test_a_region_with_no_word_under_it_keeps_its_own_box_and_says_so() -> None:
    regions = [_region("r1", "Ambulance", (0, 0, 50, 20), confidence=0.42)]

    (fused,) = fuse(regions, [], 1)

    assert fused.confidence == 0.42
    assert fused.bounding_box == BoundingBox(0, 0, 50, 20)
    assert fused.text_provenance["geometry"] == (
        "no recognised word fell inside this region"
    )
    assert "word_evidence" not in (fused.structure or {})


def test_a_word_is_claimed_by_one_region_only() -> None:
    regions = [
        _region("outer", "outer", (0, 0, 200, 200)),
        _region("inner", "inner", (10, 10, 50, 50)),
    ]
    # "outer" is also the region's text, so the claim is a read word, not an under-read.
    words = [_word("outer", (20, 20, 40, 40), 0.9)]

    outer, inner = fuse(regions, words, 1)

    assert outer.text_provenance["word_count"] == 1
    assert inner.text_provenance["geometry"] == (
        "no recognised word fell inside this region"
    )


def test_losing_the_geometry_reader_never_loses_the_text() -> None:
    class Text:
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [_region("r1", "Patient Name: Hugh Brown", (0, 0, 100, 20))]

    class Broken:
        def read_with_merge_level(
            self, image_path: Path, page_number: int, merge_level: str
        ) -> list[TextRegion]:
            raise ReaderError("nemotron_predict_failed", "out of memory")

    regions = FusedReader(Text(), Broken()).read(Path("page.png"), 1)

    assert [region.text for region in regions] == ["Patient Name: Hugh Brown"]


def test_invented_text_is_caught_but_unreadable_orientation_is_not() -> None:
    """Both regions have zero recognised words; only one of them is a hallucination."""
    # The barcode strip on 04_homecare_referral: 3898 characters of invented Italian
    # in a thin band, where the word reader recognised nothing at all.
    barcode = _region(
        "bar", "I ROMANI IN UN MILANO " + "(1967) " * 553, (0, 0, 600, 40)
    )
    # The arXiv sidebar on academic_paper: real text, rotated 90 degrees, which the word
    # reader cannot read in that orientation. 37 characters in a tall narrow strip.
    sidebar = _region(
        "side", "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", (11, 291, 44, 782)
    )
    # Words printed elsewhere on the page calibrate the character size. The small-print
    # instruction line sets the tightest packing, as it does on a real form.
    words = [
        _word("Abstract", (100, 100, 180, 120), 0.98),
        _word("dominant", (100, 130, 180, 150), 0.97),
        _word("pre-populated", (100, 160, 152, 168), 0.90),
    ]

    fused_bar, fused_side, *_ = fuse([barcode, sidebar], words, 1)

    assert fused_bar.resolution == "unreadable"
    assert "invented" in fused_bar.text_provenance["read_terminated"]
    # Kept as evidence of what the model produced, but the body renders nothing for it.
    assert "I ROMANI" in fused_bar.text

    assert fused_side.resolution == "resolved"
    assert "read_terminated" not in fused_side.text_provenance
    assert fused_side.text == "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"


def test_dense_small_print_a_page_does_carry_is_not_called_invented() -> None:
    """The wellness form's instruction line is 400 characters in an 802x24 band."""
    instructions = _region(
        "top",
        "This assessment form is pre-populated with existing data associated with this "
        "patient and is displayed in bold font. " * 3,
        (9, 5, 811, 29),
    )
    words = [
        _word("PROVIDER", (100, 100, 180, 120), 0.98),
        # Small print elsewhere on the page shows how tightly it sets type.
        _word("pre-populated", (100, 160, 152, 168), 0.90),
    ]

    fused, *_ = fuse([instructions], words, 1)

    assert fused.resolution == "resolved"
    assert "read_terminated" not in fused.text_provenance


def test_claimed_words_absent_from_the_region_text_come_back_as_child_regions() -> None:
    """financial_table: the table region claims the bullet column's words but its table
    text never includes them. The words have real boxes, so they come back as lines."""
    region = _region(
        "r1", "Deposits\t$187\nBranches\t2,641", (0, 0, 600, 100), kind="table"
    )
    words = [
        _word("Deposits", (10, 10, 80, 25), 0.99),
        _word("$187", (100, 10, 140, 25), 0.98),
        _word("Branches", (10, 40, 80, 55), 0.97),
        _word("2,641", (100, 40, 140, 55), 0.96),
        # The bullet column: inside the region box, absent from its text.
        _word("Serve", (300, 10, 340, 25), 0.95),
        _word("84M", (345, 10, 380, 25), 0.94),
        _word("consumers", (385, 10, 460, 25), 0.93),
    ]

    fused = fuse([region], words, 1)

    parent = fused[0]
    assert parent.text == "Deposits\t$187\nBranches\t2,641"
    flags = [item["in_region_text"] for item in parent.structure["word_evidence"]]
    assert flags == [True, True, True, True, False, False, False]
    assert parent.text_provenance["underread_recovered_words"] == 3

    (child,) = [item for item in fused if item.id.startswith("r1-underread-")]
    assert child.text == "Serve 84M consumers"
    assert child.text_provenance["method"] == "region_underread_repair"
    assert child.text_provenance["parent_region_id"] == "r1"
    assert child.confidence == pytest.approx((0.95 + 0.94 + 0.93) / 3)
    box = child.bounding_box
    assert (box.left, box.top, box.right, box.bottom) == (300, 10, 460, 25)
    assert parent.text_provenance["underread_child_ids"] == [child.id]


def test_a_word_the_text_contains_once_but_the_page_prints_twice_is_recovered() -> None:
    region = _region("r1", "Total revenue", (0, 0, 300, 100))
    words = [
        _word("Total", (10, 10, 50, 25), 0.99),
        _word("revenue", (55, 10, 110, 25), 0.98),
        _word("Total", (10, 60, 50, 75), 0.97),
    ]

    fused = fuse([region], words, 1)

    flags = [item["in_region_text"] for item in fused[0].structure["word_evidence"]]
    assert flags == [True, True, False]
    (child,) = [item for item in fused if "underread" in item.id]
    assert child.text == "Total"


def test_a_fully_read_region_emits_no_repair_children() -> None:
    region = _region("r1", "Patient Name: Hugh Brown", (0, 0, 300, 40))
    words = [
        _word("Patient", (10, 10, 60, 30), 0.99),
        _word("Name:", (65, 10, 110, 30), 0.98),
        _word("Hugh", (115, 10, 150, 30), 0.97),
        _word("Brown", (155, 10, 200, 30), 0.96),
    ]

    fused = fuse([region], words, 1)

    assert len(fused) == 1
    assert "underread_recovered_words" not in fused[0].text_provenance
    assert all(
        item["in_region_text"] for item in fused[0].structure["word_evidence"]
    )


def test_a_glyph_only_word_is_never_recovered_as_missing_text() -> None:
    region = _region("r1", "Colorado", (0, 0, 300, 40))
    words = [
        _word("Colorado", (10, 10, 80, 30), 0.99),
        _word("//", (100, 10, 120, 30), 0.60),
    ]

    fused = fuse([region], words, 1)

    assert len(fused) == 1
    assert fused[0].structure["word_evidence"][1]["in_region_text"] is True


def test_no_words_anywhere_means_no_capacity_claim() -> None:
    """With nothing recognised on the page there is no measured character size."""
    region = _region("r1", "x" * 5000, (0, 0, 10, 10))

    (fused,) = fuse([region], [], 1)

    assert fused.resolution == "resolved"
    assert "read_terminated" not in fused.text_provenance
