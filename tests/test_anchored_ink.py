from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.anchored_ink import AnchoredInkProposalStage, MODEL, PROVIDER
from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.evidence_layout import EvidenceLayoutStage
from ocr_pipeline.handwriting import HandwritingStage
from ocr_pipeline.pipeline import process_document


class FixedReader:
    name = "base"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions


class HandwritingReader:
    name = "handwriting-reader"
    max_batch_items = 16
    provenance = {"id": "fixture-handwriting", "revision": "test"}

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def transcribe_batch(self, images: list[Image.Image]) -> list[str]:
        self.calls += 1
        return self.outputs


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
        "field_ownership": {
            "field_id": "p1-field-re-label",
            "owner_block_id": None,
            "label": "RE:",
            "label_evidence_ids": ["re-label"],
            "value_evidence_ids": [],
        },
        "proposal": proposal.text_provenance,
    }
    assert proposal.text_provenance["method"] == (
        "label_anchored_rule_removed_residual_ink"
    )
    assert proposal.text_provenance["anchor_evidence_ids"] == ["re-label"]
    assert (
        proposal.text_provenance["field_ownership"]
        == (proposal.structure["field_ownership"])
    )
    assert proposal.text_provenance["residual_component_count"] > 0
    assert proposal.text_provenance["residual_area"] > 0
    assert proposal.text_provenance["model"] == MODEL


def test_structured_field_proposal_is_reread_but_not_auto_adopted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "filled-line.png"
    image = Image.new("L", (360, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((80, 62, 330, 62), fill="black", width=1)
    draw.line((105, 48, 115, 36, 124, 54, 134, 38), fill="black", width=3)
    image.save(source)
    specialist = HandwritingReader(["Take 5 mg", "Take 5 mg"])

    result = process_document(
        source,
        FixedReader([_region("dose-label", "Dose:", BoundingBox(20, 45, 60, 60))]),
        stages=[
            EvidenceLayoutStage(),
            AnchoredInkProposalStage(label_provider="base"),
            HandwritingStage(specialist, text_provider="base"),
        ],
    )

    proposal = next(
        region for region in result.pages[0].regions if region.kind == "handwriting"
    )
    ownership = proposal.structure["field_ownership"]
    assert specialist.calls == 1
    assert ownership["field_id"] == "p1-layout-1-field-1"
    assert ownership["owner_block_id"] == "p1-layout-1"
    assert ownership["label_evidence_ids"] == ["dose-label"]
    assert proposal.text == ""
    assert proposal.resolution == "unreadable"
    assert [(item.text, item.provider) for item in proposal.alternatives] == [
        ("Take 5 mg", "handwriting-reader")
    ]
    assert proposal.alternatives[0].text_provenance["field_ownership"] == ownership
    assert proposal.structure["handwriting_attempt"]["outcome"] == ("candidate_pending")
    assert proposal.structure["handwriting_attempt"]["reason"] == (
        "awaiting_independent_validation"
    )


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
