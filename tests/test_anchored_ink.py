from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.anchored_ink import AnchoredInkProposalStage, MODEL, PROVIDER
from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.pipeline import process_document


class FixedReader:
    name = "base"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions


def test_visible_ink_beside_label_becomes_unresolved_evidence(tmp_path: Path) -> None:
    source = tmp_path / "filled-line.png"
    image = Image.new("L", (360, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((80, 62, 330, 62), fill="black", width=1)
    draw.line((105, 48, 115, 36, 124, 54, 134, 38), fill="black", width=3)
    draw.line((142, 52, 153, 35, 165, 53, 178, 39), fill="black", width=3)
    image.save(source)
    anchor = _region("re-label", "RE:", BoundingBox(20, 45, 60, 60))

    result = process_document(
        source,
        FixedReader([anchor]),
        stages=[AnchoredInkProposalStage(label_provider="base")],
    )

    page = result.pages[0]
    proposal = next(region for region in page.regions if region.kind == "handwriting")
    assert page.route == "review"
    assert page.text.value == "RE:"
    assert proposal.text == ""
    assert proposal.confidence is None
    assert proposal.resolution == "unreadable"
    assert proposal.provider == PROVIDER
    assert proposal.bounding_box.left <= 105
    assert proposal.bounding_box.right >= 178
    assert proposal.structure == {
        "role": "handwriting_candidate",
        "handwriting_candidate": True,
        "handwriting_candidate_source": "anchored_residual",
        "anchor_evidence_ids": ["re-label"],
        "proposal": proposal.text_provenance,
    }
    assert proposal.text_provenance["method"] == (
        "label_anchored_rule_removed_residual_ink"
    )
    assert proposal.text_provenance["anchor_evidence_ids"] == ["re-label"]
    assert proposal.text_provenance["residual_component_count"] > 0
    assert proposal.text_provenance["residual_area"] > 0
    assert proposal.text_provenance["model"] == MODEL


def test_blank_writing_line_does_not_create_proposal(tmp_path: Path) -> None:
    source = tmp_path / "blank-line.png"
    image = Image.new("L", (360, 120), "white")
    ImageDraw.Draw(image).line((80, 62, 330, 62), fill="black", width=1)
    image.save(source)
    anchor = _region("re-label", "RE:", BoundingBox(20, 45, 60, 60))

    result = process_document(
        source,
        FixedReader([anchor]),
        stages=[AnchoredInkProposalStage(label_provider="base")],
    )

    page = result.pages[0]
    assert page.route == "accept_local"
    assert page.text.value == "RE:"
    assert page.regions == [anchor]


def test_previous_row_line_is_not_owned_by_later_label(tmp_path: Path) -> None:
    source = tmp_path / "stacked-fields.png"
    image = Image.new("L", (360, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.line((80, 45, 330, 45), fill="black", width=1)
    draw.line((105, 31, 125, 39), fill="black", width=3)
    draw.line((80, 92, 330, 92), fill="black", width=1)
    image.save(source)
    anchor = _region("date-label", "Date:", BoundingBox(20, 58, 60, 75))

    result = process_document(
        source,
        FixedReader([anchor]),
        stages=[AnchoredInkProposalStage(label_provider="base")],
    )

    assert result.pages[0].regions == [anchor]


def _region(region_id: str, text: str, box: BoundingBox) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="text",
        text=text,
        confidence=0.98,
        bounding_box=box,
        reading_order=1,
        provider="base",
    )
