from __future__ import annotations

from pathlib import Path

from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.handwriting import HandwritingStage
from ocr_pipeline.handwriting_classifier import HandwritingClassifierStage
from ocr_pipeline.pipeline import process_document


class PageReader:
    name = "base"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return self.regions


class Scores:
    name = "classifier"
    provenance = {"id": "fixture", "revision": "test"}

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.sizes: list[tuple[int, int]] = []

    def score_batch(self, images: list[Image.Image]) -> list[float]:
        self.sizes = [image.size for image in images]
        return self.values


class CropReader:
    name = "handwriting-reader"
    max_batch_items = 32
    provenance = {"id": "fixture", "revision": "test"}

    def __init__(self, values: list[str]) -> None:
        self.values = values
        self.sizes: list[tuple[int, int]] = []

    def transcribe_batch(self, images: list[Image.Image]) -> list[str]:
        self.sizes = [image.size for image in images]
        return self.values


def test_dual_view_agreement_marks_existing_region_without_rewriting(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    regions = [_region("field", "Jardiance", 0.42, BoundingBox(20, 20, 60, 36), 7)]
    classifier = Scores([0.97, 0.94])
    stage = HandwritingClassifierStage(
        classifier,
        text_provider="base",
        context_padding=10,
    )

    result = process_document(image_path, PageReader(regions), stages=[stage])

    proposed = result.pages[0].regions[0]
    assert classifier.sizes == [(40, 16), (60, 36)]
    assert proposed.id == "field"
    assert proposed.text == "Jardiance"
    assert proposed.reading_order == 7
    assert proposed.provider == "base"
    assert proposed.structure == {
        "handwriting_candidate": True,
        "handwriting_candidate_source": "classifier",
        "handwriting_classifier": {
            "method": "dual_crop_classifier_agreement",
            "page_number": 1,
            "threshold": 0.9,
            "scores": {"tight": 0.97, "context": 0.94},
            "crops": {
                "tight": {"bounding_box": [20, 20, 60, 36]},
                "context": {"bounding_box": [10, 10, 70, 46]},
            },
            "model": classifier.provenance,
            "decision": "candidate",
        },
    }
    assert result.pages[0].route == "accept_local"


def test_printed_tall_region_is_not_marked_without_classifier_agreement(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    printed = _region("printed", "TOTAL", 0.4, BoundingBox(5, 5, 40, 55), 1)
    classifier = Scores([0.02, 0.01])

    result = process_document(
        image_path,
        PageReader([printed]),
        stages=[HandwritingClassifierStage(classifier, text_provider="base")],
    )

    assert classifier.sizes
    assert result.pages[0].regions[0].structure is None
    assert result.pages[0].route == "accept_local"


def test_allows_table_text_but_excludes_semantic_and_control_regions(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (220, 180), "white").save(image_path)
    regions = [
        _region(
            "header",
            "Patient Name",
            0.2,
            BoundingBox(5, 5, 55, 20),
            1,
            structure={"role": "header"},
        ),
        _region(
            "footer",
            "Page 1",
            0.2,
            BoundingBox(5, 150, 45, 165),
            2,
            structure={"role": "footer"},
        ),
        _region("table-text", "120/80", 0.2, BoundingBox(70, 30, 120, 48), 3),
        _region(
            "table",
            "",
            None,
            BoundingBox(60, 20, 160, 90),
            4,
            kind="table",
        ),
        _region("control-text", "Yes", 0.2, BoundingBox(20, 95, 60, 112), 5),
        _region(
            "control",
            "[x]",
            0.9,
            BoundingBox(15, 90, 40, 118),
            6,
            kind="checkbox",
        ),
        _region("eligible", "Amlodipine", 0.2, BoundingBox(130, 120, 190, 138), 7),
    ]
    classifier = Scores([0.95, 0.95, 0.95, 0.95])

    result = process_document(
        image_path,
        PageReader(regions),
        stages=[HandwritingClassifierStage(classifier, text_provider="base")],
    )

    by_id = {region.id: region for region in result.pages[0].regions}
    assert classifier.sizes and len(classifier.sizes) == 4
    assert by_id["table-text"].structure["handwriting_candidate"] is True
    assert by_id["eligible"].structure["handwriting_candidate"] is True
    for region_id in ("header", "footer", "control-text"):
        assert by_id[region_id].structure is None or not by_id[region_id].structure.get(
            "handwriting_candidate"
        )


def test_table_cell_candidate_flows_to_specialist_as_add_only_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "form.png"
    Image.new("RGB", (180, 100), "white").save(image_path)
    cell = _region(
        "medication-cell",
        "Jardlance",
        0.2,
        BoundingBox(40, 30, 110, 48),
        1,
        structure={"role": "table_source", "parent_id": "medication-grid"},
    )
    cell.alternatives.append(
        TextAlternative(
            text="Jardiance",
            confidence=0.8,
            provider="tesseract",
            text_provenance={"method": "fixture"},
        )
    )
    table = _region(
        "medication-grid",
        "",
        None,
        BoundingBox(20, 15, 160, 80),
        2,
        kind="table",
    )
    classifier = Scores([0.98, 0.96])
    crop_reader = CropReader(["Jardiance", "Jardiance"])

    result = process_document(
        image_path,
        PageReader([cell, table]),
        stages=[
            HandwritingClassifierStage(classifier, text_provider="base"),
            HandwritingStage(crop_reader, text_provider="base"),
        ],
    )

    page = result.pages[0]
    processed = page.regions[0]
    assert classifier.sizes == [(70, 18), (94, 42)]
    assert crop_reader.sizes == [(70, 18), (94, 42)]
    assert page.route == "review"
    assert processed.text == "Jardlance"
    assert processed.provider == "base"
    assert processed.structure["role"] == "table_source"
    assert processed.structure["parent_id"] == "medication-grid"
    assert processed.structure["handwriting_candidate_source"] == "classifier"
    assert processed.structure["handwriting_classifier"]["decision"] == "candidate"
    assert processed.structure["handwriting_review"]["reason"] == (
        "specialist_candidate"
    )
    assert [(item.text, item.provider) for item in processed.alternatives] == [
        ("Jardiance", "tesseract"),
        ("Jardiance", "handwriting-reader"),
    ]
    assert processed.alternatives[-1].text_provenance["view"] == "agreed"
    assert page.regions[1] == table


def test_view_disagreement_records_review_evidence_without_routing(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (80, 60), "white").save(image_path)
    region = _region("field", "weak", 0.2, BoundingBox(10, 10, 45, 25), 1)
    classifier = Scores([0.96, 0.4])

    result = process_document(
        image_path,
        PageReader([region]),
        stages=[HandwritingClassifierStage(classifier, text_provider="base")],
    )

    proposed = result.pages[0].regions[0]
    assert "handwriting_candidate" not in proposed.structure
    assert proposed.structure["handwriting_classifier"]["decision"] == (
        "view_disagreement"
    )
    assert proposed.structure["handwriting_classifier"]["review"] == {
        "required": False,
        "reason": "view_disagreement",
    }
    assert proposed.text == "weak"
    assert proposed.reading_order == 1
    assert result.pages[0].route == "accept_local"


def test_strict_prefilter_rejects_invalid_candidates_without_model_call(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 100), "white").save(image_path)
    missing_box = _region("box", "text", 0.2, BoundingBox(1, 1, 2, 2), 7)
    missing_box.bounding_box = None  # type: ignore[assignment]
    regions = [
        _region("empty", " ", 0.2, BoundingBox(1, 1, 20, 12), 1),
        _region("confidence", "text", None, BoundingBox(1, 15, 20, 25), 2),
        _region("strong", "text", 0.75, BoundingBox(1, 28, 20, 38), 3),
        _region(
            "provider",
            "text",
            0.2,
            BoundingBox(1, 41, 20, 51),
            4,
            provider="other",
        ),
        _region("multiline", "one\ntwo", 0.2, BoundingBox(1, 54, 20, 64), 5),
        _region("long", "x" * 65, 0.2, BoundingBox(1, 67, 100, 77), 6),
        missing_box,
    ]
    classifier = Scores([])

    result = process_document(
        image_path,
        PageReader(regions),
        stages=[HandwritingClassifierStage(classifier, text_provider="base")],
    )

    assert classifier.sizes == []
    page = result.pages[0]
    semantic_regions = page.regions[: len(regions)]
    assert [region.text for region in semantic_regions] == [
        region.text for region in regions
    ]
    assert [region.reading_order for region in semantic_regions] == list(range(1, 8))
    assert all(region.structure is None for region in semantic_regions)
    assert page.text.value == " ".join(region.text for region in regions)
    assert page.text.evidence_ids == [region.id for region in regions]
    assert page.route == "review"
    assert page.regions[-1].structure == {
        "role": "coverage_risk",
        "reasons": [
            "empty_content",
            "tail_repetition",
            "repeated_suffix",
            "invalid_bbox",
        ],
        "region_risks": [
            {"region_id": "empty", "reasons": ["empty_content"]},
            {
                "region_id": "long",
                "reasons": ["tail_repetition", "repeated_suffix"],
            },
            {"region_id": "box", "reasons": ["invalid_bbox"]},
        ],
    }


def _region(
    region_id: str,
    text: str,
    confidence: float | None,
    box: BoundingBox,
    order: int,
    *,
    kind: str = "text",
    provider: str = "base",
    structure: dict[str, object] | None = None,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind=kind,
        text=text,
        confidence=confidence,
        bounding_box=box,
        reading_order=order,
        provider=provider,
        structure=structure,
    )
