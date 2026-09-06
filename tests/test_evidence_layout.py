from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, PageResult, TextAlternative, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.evidence_layout import EvidenceLayoutStage
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import TesseractReader
from ocr_pipeline.rendering import render_evidence, render_page_markdown


def _word(
    identifier: str,
    text: str,
    box: tuple[int, int, int, int],
    order: int,
    *,
    resolution: str = "resolved",
) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind="text",
        text=text,
        confidence=0.9,
        bounding_box=BoundingBox(*box),
        reading_order=order,
        provider="nemotron-ocr-v2",
        text_provenance={"merge_level": "word"},
        resolution=resolution,  # type: ignore[arg-type]
        alternatives=(
            [TextAlternative("unsupported", 0.8, "challenger")]
            if resolution != "resolved"
            else []
        ),
    )


def _atomic(
    identifier: str,
    text: str,
    kind: str,
    box: tuple[int, int, int, int],
    order: int,
) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind=kind,
        text=text,
        confidence=0.9,
        bounding_box=BoundingBox(*box),
        reading_order=order,
        provider="nemotron-ocr-v2",
    )


def _layout(regions: list[TextRegion]) -> list[TextRegion]:
    return EvidenceLayoutStage().apply(Path("unused.png"), 1, regions)


def _blocks(regions: list[TextRegion]) -> list[TextRegion]:
    return [region for region in regions if region.kind == "layout_block"]


def _field_ids(block: TextRegion) -> list[list[str]]:
    return [field["evidence_ids"] for field in block.structure["fields"]]  # type: ignore[index]


def test_coalesces_wrapped_lines_into_one_evidence_owned_paragraph() -> None:
    result = _layout(
        [
            _word("a", "Clinical", (10, 10, 55, 20), 1),
            _word("b", "summary", (60, 10, 110, 20), 2),
            _word("c", "continues", (10, 24, 65, 34), 3),
            _word("d", "here.", (70, 24, 100, 34), 4),
        ]
    )

    [block] = _blocks(result)
    assert block.structure == {
        "role": "layout_block",
        "block_type": "paragraph",
        "presentation_rank": 1,
        "child_evidence_ids": ["a", "b", "c", "d"],
        "source_reading_orders": [
            {"evidence_id": "a", "reading_order": 1},
            {"evidence_id": "b", "reading_order": 2},
            {"evidence_id": "c", "reading_order": 3},
            {"evidence_id": "d", "reading_order": 4},
        ],
        "lines": [
            {"evidence_ids": ["a", "b"]},
            {"evidence_ids": ["c", "d"]},
        ],
    }
    assert block.text == "Clinical summary continues here."
    assert block.bounding_box == BoundingBox(10, 10, 110, 34)
    assert all(
        (region.structure or {}).get("layout_owner_id") == block.id
        for region in result[:4]
    )


def test_orientation_residual_is_atomic_to_spatial_line_grouping() -> None:
    margin = _word(
        "margin",
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023",
        (10, 20, 40, 700),
        1,
    )
    margin.text_provenance = {
        "merge_level": "word",
        "orientation_residual": {
            "method": "side_margin_crop_rotation",
            "source_view_angle": 90,
            "selected_view_angle": 0,
            "source_margin": "left",
        },
    }
    body = _word("body", "Attention Is All You Need", (120, 100, 300, 120), 2)

    result = _layout([margin, body])
    blocks = _blocks(result)

    assert [block.text for block in blocks] == [
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023",
        "Attention Is All You Need",
    ]
    assert [block.structure["child_evidence_ids"] for block in blocks] == [  # type: ignore[index]
        ["margin"],
        ["body"],
    ]
    assert blocks[0].structure["block_type"] == "aside_text"  # type: ignore[index]
    assert (result[0].text_provenance or {})["orientation_residual"] == {
        "method": "side_margin_crop_rotation",
        "source_view_angle": 90,
        "selected_view_angle": 0,
        "source_margin": "left",
    }


def test_orientation_residual_words_share_one_aside_block() -> None:
    provenance = {
        "merge_level": "word",
        "orientation_residual": {
            "method": "side_margin_crop_rotation",
            "source_view_angle": 270,
            "selected_view_angle": 0,
            "source_margin": "left",
            "source_crop": [0, 0, 80, 700],
        },
    }
    first = _word("margin-1", "arXiv:1706.03762v7", (10, 20, 30, 240), 10)
    second = _word("margin-2", "[cs.CL] 2 Aug 2023", (10, 245, 30, 460), 11)
    first.text_provenance = provenance
    second.text_provenance = provenance

    result = _layout([first, second])
    [block] = _blocks(result)

    assert block.text == "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"
    assert block.structure["block_type"] == "aside_text"  # type: ignore[index]
    assert block.structure["child_evidence_ids"] == [  # type: ignore[index]
        "margin-1",
        "margin-2",
    ]


def test_wide_aligned_content_becomes_a_form_row_with_spatial_segments() -> None:
    result = _layout(
        [
            _word("label", "Patient", (10, 10, 55, 22), 1),
            _word("value", "Lee", (60, 10, 82, 22), 2),
            _word("date-label", "Date", (220, 10, 250, 22), 3),
            _word("date", "03/18/2025", (255, 10, 330, 22), 4),
        ]
    )

    [block] = _blocks(result)
    assert block.structure["block_type"] == "form_row"  # type: ignore[index]
    assert block.structure["segments"] == [  # type: ignore[index]
        {"evidence_ids": ["label", "value"]},
        {"evidence_ids": ["date-label", "date"]},
    ]
    assert _field_ids(block) == [
        ["label", "value"],
        ["date-label", "date"],
    ]
    assert block.text == "Patient Lee | Date 03/18/2025"
    assert block.bounding_box == BoundingBox(10, 10, 330, 22)
    assert not any(region.kind == "table" for region in result)


def test_printed_formula_tokens_do_not_become_form_rows_or_headings() -> None:
    result = _layout(
        [
            _word("integral", "∫", (10, 10, 24, 32), 1),
            _word("fraction", "x²", (42, 10, 62, 32), 2),
            _word("equals", "=", (82, 10, 94, 32), 3),
            _word("root", "√2ax+x²", (112, 10, 182, 32), 4),
        ]
    )

    [block] = _blocks(result)
    assert block.structure["block_type"] == "formula"  # type: ignore[index]
    assert block.structure["formula_recognition"] == "heuristic"  # type: ignore[index]
    assert block.text == "∫ x² = √2ax+x²"
    assert block.confidence is None


def test_formula_page_groups_fraction_fragments_before_form_rows() -> None:
    result = _layout(
        [
            _word("integral", "∫ x² = y", (10, 10, 180, 30), 1),
            _word("denominator", "2ax x", (60, 24, 110, 40), 2),
            _atomic("second", "y = 2", "formula", (10, 70, 100, 90), 3),
            _atomic("third", "z = 3", "formula", (10, 110, 100, 130), 4),
            _word("lower", "∫ d x", (10, 160, 100, 180), 5),
            _word("one", "1", (70, 210, 80, 220), 6),
            _word("a-left", "a", (10, 225, 20, 235), 7),
            _word("a-right", "a", (72, 225, 82, 235), 8),
        ]
    )
    owners = {
        evidence_id: block
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["integral"].id == owners["denominator"].id
    assert owners["one"].id == owners["a-left"].id == owners["a-right"].id
    assert owners["a-left"].structure["block_type"] == "formula"


def test_operator_text_is_only_a_formula_candidate() -> None:
    [block] = _blocks(_layout([_word("text", "to = be", (10, 10, 80, 30), 1)]))

    assert block.structure["block_type"] == "formula"  # type: ignore[index]
    assert block.structure["formula_recognition"] == "heuristic"  # type: ignore[index]


def test_connected_formula_lines_become_one_multiline_owner() -> None:
    result = _layout(
        [
            _word("first", "∫ x²", (10, 10, 100, 30), 1),
            _word("continuation", "= x³ / 3", (20, 50, 110, 70), 2),
            _word("separate", "3) √ y²", (15, 90, 95, 110), 3),
        ]
    )
    owners = {
        evidence_id: block.id
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["first"] == owners["continuation"]
    assert owners["continuation"] != owners["separate"]


def test_historical_formula_artifact_keeps_page_header_out_of_equations() -> None:
    image_path = Path(__file__).parents[1] / "artifacts/demo/formula_scan.png"
    regions = [
        _word("page-number", "122", (69, 69, 88, 83), 1),
        _word("running-head", "EXAMPLES ON THE", (202, 69, 307, 83), 2),
        _word("and", "And", (84, 101, 109, 114), 3),
        _word("integral", "∫", (112, 96, 132, 139), 4),
        _word("fraction", "da / √(2ax + x²)", (145, 96, 218, 124), 5),
        _word("equals", "=", (226, 103, 238, 116), 6),
        _word("result", "log {x + a + √(2ax + x²)}", (245, 99, 425, 122), 7),
        _word("middle", "∫ x² = y", (100, 300, 250, 340), 8),
        _word("lower-expression", "(Qae−1)^−h", (163, 653, 258, 671), 9),
        _word("last-left", "we", (146, 670, 159, 676), 10),
        _word("last-right", "nN", (265, 657, 350, 690), 11),
    ]

    result = EvidenceLayoutStage().apply(image_path, 1, regions)
    blocks = _blocks(result)
    owners = {
        evidence_id: block
        for block in blocks
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["page-number"].id == owners["running-head"].id
    assert owners["page-number"].structure["block_type"] != "formula"
    assert {
        owners[item].id for item in ("and", "integral", "fraction", "equals", "result")
    } == {owners["and"].id}
    assert owners["and"].structure["block_type"] == "formula"
    assert {owners[item].id for item in ("lower-expression",)} == {
        owners["lower-expression"].id
    }
    assert owners["lower-expression"].structure["block_type"] == "formula"
    assert owners["last-left"].id == owners["last-right"].id
    assert owners["last-left"].id != owners["lower-expression"].id
    assert owners["last-left"].structure["block_type"] == "form_row"
    assert owners["lower-expression"].structure["formula_grouping"]["method"] == (  # type: ignore[index]
        "source_ink_band"
    )
    for identifier in ("lower-expression",):
        source = next(region for region in result if region.id == identifier)
        assert source.structure["layout_owner_type"] == "formula"  # type: ignore[index]


def test_formula_page_keeps_clinical_label_value_rows_as_forms() -> None:
    regions = [
        _word("integral", "∫", (10, 10, 20, 20), 1),
        _word("x", "x²", (30, 10, 45, 20), 2),
        _word("equals", "=", (55, 10, 65, 20), 3),
        _word("y", "y", (75, 10, 85, 20), 4),
        _word("a", "a", (10, 40, 20, 50), 5),
        _word("plus", "+", (30, 40, 40, 50), 6),
        _word("b", "b", (50, 40, 60, 50), 7),
        _word("root", "√", (10, 70, 20, 80), 8),
        _word("z", "z²", (30, 70, 45, 80), 9),
        _word("minus", "−", (55, 70, 65, 80), 10),
        _word("two", "2", (75, 70, 85, 80), 11),
        _word("patient-label", "Patient", (10, 110, 70, 120), 12),
        _word("patient-value", "Lee", (150, 110, 175, 120), 13),
    ]

    blocks = _blocks(_layout(regions))
    owners = {
        evidence_id: block.structure["block_type"]
        for block in blocks
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["patient-label"] == "form_row"
    assert owners["patient-value"] == "form_row"
    assert {owners["integral"], owners["plus"], owners["root"]} == {"formula"}


def test_ruled_formula_crop_keeps_nearby_date_row_as_form(tmp_path: Path) -> None:
    image_path = tmp_path / "formula-and-form.png"
    image = Image.new("RGB", (500, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "x = 1", fill="black")
    draw.line((10, 43, 490, 43), fill="black", width=2)
    draw.text((260, 47), "Date: 03/18/2025", fill="black")
    image.save(image_path)
    regions = [
        _atomic("formula", "x = 1", "formula", (20, 20, 120, 39), 1),
        _word("date-label", "Date:", (260, 47, 300, 60), 2),
        _word("date", "03/18/2025", (306, 47, 390, 60), 3),
    ]

    result = EvidenceLayoutStage().apply(image_path, 1, regions)
    owners = {
        evidence_id: block
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["formula"].structure["block_type"] == "formula"
    assert owners["date-label"].id == owners["date"].id
    assert owners["date"].structure["block_type"] == "form_row"
    assert owners["formula"].id != owners["date"].id


def test_formula_grouping_keeps_same_baseline_form_row_distinct(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "formula-and-id.png"
    image = Image.new("RGB", (300, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "x = 1", fill="black")
    draw.line((20, 30, 200, 30), fill="black")
    draw.text((105, 20), "ID", fill="black")
    draw.text((210, 20), "42", fill="black")
    image.save(image_path)
    regions = [
        _atomic("formula", "x = 1", "formula", (20, 20, 100, 40), 1),
        _word("id-label", "ID", (105, 20, 125, 40), 2),
        _word("id-value", "42", (210, 20, 230, 40), 3),
    ]

    result = EvidenceLayoutStage().apply(image_path, 1, regions)
    owners = {
        evidence_id: block
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["formula"].structure["block_type"] == "formula"
    assert owners["id-label"].id == owners["id-value"].id
    assert owners["id-value"].structure["block_type"] == "form_row"
    assert owners["formula"].id != owners["id-value"].id


@pytest.mark.parametrize(
    "prose",
    ("This is ordinary prose", "See it", "And so", "evidence-based note"),
)
def test_formula_page_keeps_nearby_ordinary_prose_out_of_formulas(
    prose: str,
) -> None:
    result = _layout(
        [
            _atomic("first", "x = 1", "formula", (10, 10, 100, 30), 1),
            _atomic("second", "y = 2", "formula", (10, 40, 100, 60), 2),
            _atomic("third", "z = 3", "formula", (10, 70, 100, 90), 3),
            _word(
                "prose",
                prose,
                (10, 100, 180, 120),
                4,
            ),
        ]
    )
    owners = {
        evidence_id: block
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["prose"].structure["block_type"] != "formula"
    assert owners["prose"].id not in {
        owners["first"].id,
        owners["second"].id,
        owners["third"].id,
    }


def test_formula_page_keeps_variable_fragment_with_its_expression() -> None:
    result = _layout(
        [
            _atomic("first", "x = 1", "formula", (10, 10, 100, 30), 1),
            _atomic("second", "y = 2", "formula", (10, 40, 100, 60), 2),
            _atomic("third", "z = 3", "formula", (10, 70, 100, 90), 3),
            _word("d", "d", (10, 100, 20, 120), 4),
            _word("x", "x", (24, 100, 34, 120), 5),
        ]
    )
    owners = {
        evidence_id: block
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["d"].id == owners["x"].id
    assert owners["d"].structure["block_type"] == "formula"


def test_formula_page_keeps_distant_context_fragments_out_of_formulas() -> None:
    result = _layout(
        [
            _atomic("first", "x = 1", "formula", (10, 10, 100, 30), 1),
            _atomic("second", "y = 2", "formula", (10, 40, 100, 60), 2),
            _atomic("third", "z = 3", "formula", (10, 70, 100, 90), 3),
            _word("fragment-a", "A B", (10, 300, 60, 320), 4),
            _word("fragment-b", "Q1", (10, 325, 60, 345), 5),
        ]
    )
    owners = {
        evidence_id: block
        for block in _blocks(result)
        for evidence_id in block.structure["child_evidence_ids"]
    }

    assert owners["fragment-a"].structure["block_type"] != "formula"
    assert owners["fragment-b"].structure["block_type"] != "formula"


def test_financial_glossary_does_not_trigger_formula_page_typing() -> None:
    result = _layout(
        [
            _word(
                "assets",
                "AUM=Assets under management ETF=Exchange-traded funds",
                (10, 10, 360, 20),
                1,
            ),
            _word(
                "units",
                "M=Millions B=Billions",
                (10, 40, 180, 50),
                2,
            ),
            _word(
                "markets",
                "ECM=Equity capital market EOP=End of period",
                (10, 70, 320, 80),
                3,
            ),
            _word(
                "footnote",
                "For footnoted information, refer to pages 58-59.",
                (10, 100, 360, 110),
                4,
            ),
            _word(
                "share",
                "Market share 80%, 80% = 122% «= 12.4%",
                (10, 130, 260, 140),
                5,
            ),
        ]
    )

    assert all(
        block.structure["block_type"] != "formula"  # type: ignore[index]
        for block in _blocks(result)
    )


def test_historical_formula_artifact_is_typed_from_real_ocr_output() -> None:
    executable = shutil.which("tesseract")
    if executable is None:
        pytest.skip("Tesseract is unavailable")
    image_path = Path(__file__).parents[1] / "artifacts/demo/formula_scan.png"

    result = process_document(
        image_path,
        TesseractReader(executable=executable, page_segmentation_mode=3),
        stages=[EvidenceLayoutStage()],
    )
    blocks = _blocks(result.pages[0].regions)

    assert blocks[0].structure["block_type"] == "header"  # type: ignore[index]
    assert (
        sum(
            block.structure["block_type"] == "formula"  # type: ignore[index]
            for block in blocks[1:]
        )
        >= 3
    )


def test_pairs_a_geometrically_separate_label_with_its_value() -> None:
    result = _layout(
        [
            _word("label", "Estimated GFR:", (10, 10, 100, 22), 1),
            _word("value", "53", (165, 10, 180, 22), 2),
            _word("date-label", "Lab Date:", (260, 10, 320, 22), 3),
            _word("date", "03/18/2025", (325, 10, 400, 22), 4),
        ]
    )

    [block] = _blocks(result)
    assert block.structure["segments"] == [  # type: ignore[index]
        {"evidence_ids": ["label"]},
        {"evidence_ids": ["value"]},
        {"evidence_ids": ["date-label", "date"]},
    ]
    assert _field_ids(block) == [
        ["label", "value"],
        ["date-label", "date"],
    ]
    assert block.text == "Estimated GFR: 53 | Lab Date: 03/18/2025"


def test_splits_multiple_fields_inside_one_spatial_segment() -> None:
    result = _layout(
        [
            _word("date-label", "Date of Service:", (10, 10, 100, 22), 1),
            _word("date", "03/18/2025", (105, 10, 180, 22), 2),
            _word("age-label", "Age:", (300, 10, 330, 22), 3),
            _word("age", "85", (335, 10, 350, 22), 4),
            _word("gender-label", "Gender:", (365, 10, 420, 22), 5),
        ]
    )

    [block] = _blocks(result)
    assert block.structure["segments"] == [  # type: ignore[index]
        {"evidence_ids": ["date-label", "date"]},
        {"evidence_ids": ["age-label", "age", "gender-label"]},
    ]
    assert _field_ids(block) == [
        ["date-label", "date"],
        ["age-label", "age"],
        ["gender-label"],
    ]
    assert block.text == "Date of Service: 03/18/2025 | Age: 85 | Gender:"


def test_form_field_preserves_single_character_value_and_source_evidence() -> None:
    result = _layout(
        [
            _word("gender-label", "Gender:", (10, 10, 62, 22), 1),
            _word("gender-value", "F", (70, 10, 78, 22), 2),
        ]
    )

    [block] = _blocks(result)
    [field] = block.structure["fields"]  # type: ignore[index]
    assert field == {
        "id": "p1-layout-1-field-1",
        "label": "Gender:",
        "raw_value": "F",
        "evidence_ids": ["gender-label", "gender-value"],
        "label_evidence_ids": ["gender-label"],
        "value_evidence_ids": ["gender-value"],
        "normalization_history": [],
        "state": "present",
        "source_geometry": {
            "evidence_ids": ["gender-label", "gender-value"],
            "bounding_box": {"left": 10, "top": 10, "right": 78, "bottom": 22},
        },
        "answer_geometry": {
            "evidence_ids": ["gender-value"],
            "bounding_box": {"left": 70, "top": 10, "right": 78, "bottom": 22},
        },
        "recognition_evidence": [
            {
                "evidence_id": "gender-label",
                "raw_text": "Gender:",
                "provider": "nemotron-ocr-v2",
                "confidence": 0.9,
                "resolution": "resolved",
                "alternatives": [],
            },
            {
                "evidence_id": "gender-value",
                "raw_text": "F",
                "provider": "nemotron-ocr-v2",
                "confidence": 0.9,
                "resolution": "resolved",
                "alternatives": [],
            },
        ],
        "association_status": "linked",
        "association_confidence": None,
    }
    assert block.text == "Gender: F"


def test_label_only_form_field_is_not_assumed_blank() -> None:
    result = _layout([_word("gender-label", "Gender:", (10, 10, 62, 22), 1)])

    [block] = _blocks(result)
    [field] = block.structure["fields"]  # type: ignore[index]
    assert field["state"] == "not_located"
    assert field["raw_value"] is None
    assert field["value_evidence_ids"] == []
    assert field["answer_geometry"] is None
    assert field["association_status"] == "unmatched"
    assert field["state"] != "blank_verified"


def test_conflicting_form_value_retains_each_reading_without_confidence_merge() -> None:
    value = _word(
        "gender-value",
        "F",
        (70, 10, 78, 22),
        2,
        resolution="conflicting",
    )
    value.alternatives = [TextAlternative("P", None, "manual-review")]

    result = _layout([_word("gender-label", "Gender:", (10, 10, 62, 22), 1), value])

    [block] = _blocks(result)
    [field] = block.structure["fields"]  # type: ignore[index]
    assert field["state"] == "conflicting_readings"
    assert field["raw_value"] == "F"
    assert field["association_confidence"] is None
    assert field["recognition_evidence"][1]["alternatives"] == [
        {"raw_text": "P", "provider": "manual-review", "confidence": None}
    ]


def test_compound_form_labels_stay_with_their_values() -> None:
    result = _layout(
        [
            _word("fax", "FAX", (10, 10, 38, 24), 1),
            _word("fax-label", "NUMBER:", (42, 10, 100, 24), 2),
            _word("fax-value", "(336) 335-7392", (105, 10, 205, 24), 3),
            _word("phone", "PHONE", (250, 10, 300, 24), 4),
            _word("phone-label", "NUMBER:", (304, 10, 362, 24), 5),
            _word("phone-value", "(336) 335-7363", (367, 10, 467, 24), 6),
        ]
    )

    [block] = _blocks(result)
    assert block.structure["block_type"] == "form_row"  # type: ignore[index]
    assert _field_ids(block) == [
        ["fax", "fax-label", "fax-value"],
        ["phone", "phone-label", "phone-value"],
    ]
    assert block.text == ("FAX NUMBER: (336) 335-7392 | PHONE NUMBER: (336) 335-7363")


def test_tall_word_boxes_do_not_transitively_merge_adjacent_lines() -> None:
    result = _layout(
        [
            _word("line-1-a", "Confidential", (10, 10, 100, 24), 1),
            _word("line-1-tall", "notice", (105, 8, 150, 35), 2),
            _word("line-2-a", "Do not", (10, 29, 58, 43), 3),
            _word("line-2-b", "distribute", (63, 29, 130, 43), 4),
        ]
    )

    blocks = _blocks(result)
    assert [block.structure["lines"] for block in blocks] == [  # type: ignore[index]
        [{"evidence_ids": ["line-1-a", "line-1-tall"]}],
        [{"evidence_ids": ["line-2-a", "line-2-b"]}],
    ]
    assert [block.text for block in blocks] == [
        "Confidential notice",
        "Do not distribute",
    ]


def test_tables_controls_and_their_linked_sources_are_not_owned() -> None:
    table_word = _word("table-word", "Dose", (10, 10, 40, 20), 1)
    control_label = _word("control-label", "Fall risk", (10, 40, 60, 50), 2)
    free = _word("free", "Notes", (10, 70, 45, 80), 3)
    table = TextRegion(
        id="table",
        kind="table",
        text="Dose",
        confidence=0.9,
        bounding_box=BoundingBox(5, 5, 100, 30),
        reading_order=4,
        provider="table-transformer",
        structure={
            "role": "table",
            "row_count": 1,
            "column_count": 1,
            "cells": [{"text": "Dose", "evidence_ids": ["table-word"]}],
        },
    )
    control = TextRegion(
        id="control",
        kind="checkbox",
        text="[x] Fall risk",
        confidence=0.9,
        bounding_box=BoundingBox(5, 35, 65, 55),
        reading_order=5,
        provider="geometry",
        structure={"role": "control", "label_evidence_ids": ["control-label"]},
    )

    result = _layout([table_word, control_label, free, table, control])

    [block] = _blocks(result)
    assert block.structure["child_evidence_ids"] == ["free"]  # type: ignore[index]
    assert table_word.structure is None
    assert control_label.structure is None
    assert table.structure["role"] == "table"  # type: ignore[index]
    assert control.structure["role"] == "control"  # type: ignore[index]


def test_control_label_keeps_the_controls_presentation_rank() -> None:
    before = _word("before", "Before", (10, 10, 50, 20), 1)
    label = _word("label", "Fall risk", (30, 40, 90, 50), 30)
    after = _word("after", "After", (10, 70, 45, 80), 31)
    control = TextRegion(
        id="control",
        kind="checkbox",
        text="[x] Fall risk",
        confidence=0.9,
        bounding_box=BoundingBox(10, 35, 25, 55),
        reading_order=2,
        provider="geometry",
        structure={"role": "control", "label_evidence_ids": ["label"]},
    )

    result = _layout([before, label, after, control])

    label_result = next(region for region in result if region.id == "label")
    control_result = next(region for region in result if region.id == "control")
    assert (
        label_result.structure["presentation_rank"]
        == (  # type: ignore[index]
            control_result.structure["presentation_rank"]  # type: ignore[index]
        )
    )
    assert render_evidence(result).value == "Before Fall risk [x] After"


def test_every_eligible_source_has_exactly_one_owner_and_unmatched_evidence_remains() -> (
    None
):
    source = _word("source", "Body", (10, 10, 45, 20), 1)
    unmatched = TextRegion(
        id="figure",
        kind="figure",
        text="Figure 1",
        confidence=0.9,
        bounding_box=BoundingBox(10, 50, 100, 90),
        reading_order=2,
        provider="canonical",
    )
    result = _layout([source, unmatched])
    [block] = _blocks(result)

    assert (result[0].structure or {})["layout_owner_id"] == block.id
    assert result[1].id == unmatched.id
    assert result[1].reading_order == unmatched.reading_order
    assert (result[1].structure or {})["presentation_rank"] == 2
    assert unmatched.structure is None
    assert block.structure["child_evidence_ids"] == ["source"]  # type: ignore[index]


def test_rendering_uses_only_resolved_children_and_preserves_raw_alternatives() -> None:
    resolved = _word("resolved", "Medication", (10, 10, 75, 20), 1)
    disputed = _word(
        "disputed",
        "Metforrnin",
        (80, 10, 140, 20),
        2,
        resolution="conflicting",
    )
    result = _layout([resolved, disputed])
    evidence = render_evidence(result)
    dictionaries = [
        {
            "id": region.id,
            "kind": region.kind,
            "text": region.text,
            "confidence": region.confidence,
            "bounding_box": {
                "left": region.bounding_box.left,
                "top": region.bounding_box.top,
                "right": region.bounding_box.right,
                "bottom": region.bounding_box.bottom,
            },
            "reading_order": region.reading_order,
            "provider": region.provider,
            "resolution": region.resolution,
            "alternatives": [
                alternative.__dict__ for alternative in region.alternatives
            ],
            "structure": region.structure,
        }
        for region in result
    ]

    markdown = render_page_markdown(dictionaries, evidence.evidence_ids)

    assert markdown == "Medication"
    assert "Metforrnin" not in markdown
    assert "unsupported" not in markdown
    assert result[1].text == "Metforrnin"
    assert result[1].alternatives[0].text == "unsupported"


def test_demo_and_download_share_the_same_owned_form_row() -> None:
    class FormReader:
        name = "form-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                _word("label", "Patient", (10, 10, 55, 22), 1),
                _word("value", "Lee", (60, 10, 82, 22), 2),
                _word("date-label", "Date", (220, 10, 250, 22), 3),
                _word("date", "03/18/2025", (255, 10, 330, 22), 4),
            ]

    image = io.BytesIO()
    Image.new("RGB", (360, 80), "white").save(image, format="PNG")
    app = create_app(FormReader(), stages=(EvidenceLayoutStage(),))

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("form.png", image.getvalue(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        html = client.get("/").text

    assert response.status_code == 200
    page = payload["result"]["pages"][0]
    block = next(
        region for region in page["regions"] if region["kind"] == "layout_block"
    )
    assert block["text"] == "Patient Lee | Date 03/18/2025"
    assert page["text"]["value"] == "Patient Lee | Date 03/18/2025"
    assert "Patient Lee | Date 03/18/2025" in markdown
    assert "? renderStructuredPresentation(page, presentation)" in html
    assert ": renderCanonicalPage(page);" in html
    assert "function canonicalLayoutText(region, sourceIndex)" in html


def test_author_cards_keep_names_affiliations_and_emails_together(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "authors.png"
    Image.new("RGB", (500, 180), "white").save(image_path)
    regions = [
        _word("name-a", "Ashish Vaswani", (40, 20, 130, 34), 1),
        _word("name-b", "Noam Shazeer", (260, 20, 345, 34), 2),
        _word("org-a", "Google Brain", (45, 39, 125, 53), 3),
        _word("org-b", "Google Brain", (265, 39, 345, 53), 4),
        _word("mail-a", "a@example.com", (38, 58, 132, 72), 5),
        _word("mail-b", "n@example.com", (255, 58, 350, 72), 6),
        _word("solo-name", "Illia Polosukhin", (185, 100, 300, 114), 7),
        _word("solo-mail", "i@example.com", (195, 119, 290, 133), 8),
    ]

    result = EvidenceLayoutStage().apply(image_path, 1, regions)
    blocks = [
        region
        for region in result
        if (region.structure or {}).get("role") == "layout_block"
    ]

    assert [block.text for block in blocks] == [
        "Ashish Vaswani Google Brain a@example.com",
        "Noam Shazeer Google Brain n@example.com",
        "Illia Polosukhin i@example.com",
    ]
    assert [block.structure["child_evidence_ids"] for block in blocks] == [
        ["name-a", "org-a", "mail-a"],
        ["name-b", "org-b", "mail-b"],
        ["solo-name", "solo-mail"],
    ]
    assert [block.structure["block_type"] for block in blocks] == [
        "author",
        "author",
        "author",
    ]


def test_financial_artifact_keeps_sidebar_evidence_out_of_metric_rows() -> None:
    image_path = Path(__file__).parents[1] / "artifacts/demo/financial_table.png"
    regions = [
        _word("metric-1", "Average deposits ($B)", (198, 179, 330, 192), 1),
        _word("value-1", "$1,064", (642, 179, 683, 192), 2),
        _word("aside-1", "Serve 84M U.S. consumers", (732, 179, 963, 207), 3),
        _word("metric-2", "Deposits market share", (199, 211, 327, 224), 4),
        _word("value-2", "11.3%", (647, 211, 683, 224), 5),
        _word("aside-2", "7M active digital customers", (732, 221, 985, 250), 6),
        _word("metric-3", "Primary market share", (199, 256, 330, 269), 7),
        _word("value-3", "9.7%", (650, 256, 683, 269), 8),
        _word("aside-3", "Primary bank relationships", (732, 256, 1013, 293), 9),
    ]

    blocks = _blocks(EvidenceLayoutStage().apply(image_path, 1, regions))

    for block in blocks:
        child_ids = set(block.structure["child_evidence_ids"])
        assert not (
            any(
                identifier.startswith("metric-") or identifier.startswith("value-")
                for identifier in child_ids
            )
            and any(identifier.startswith("aside-") for identifier in child_ids)
        )
    assert {
        identifier
        for block in blocks
        for identifier in block.structure["child_evidence_ids"]
    } == {region.id for region in regions}


def test_inferred_sidebar_separator_does_not_cross_unrelated_footer(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sidebar.png"
    Image.new("RGB", (600, 500), "white").save(image_path)
    regions = []
    order = 1
    for prefix, top_values in (("upper", (20, 50, 80)), ("lower", (300, 330, 360))):
        for row, top in enumerate(top_values):
            regions.extend(
                [
                    _word(
                        f"{prefix}-metric-{row}",
                        "Metric",
                        (20, top, 90, top + 10),
                        order,
                    ),
                    _word(
                        f"{prefix}-value-{row}",
                        "42",
                        (250, top, 280, top + 10),
                        order + 1,
                    ),
                    _word(
                        f"{prefix}-aside-{row}",
                        "Supporting context",
                        (520, top, 590, top + 10),
                        order + 2,
                    ),
                ]
            )
            order += 3
    regions.extend(
        [
            _word("footer-label", "Notes:", (20, 180, 80, 190), order),
            _word(
                "footer-value",
                "No known allergies",
                (430, 180, 550, 190),
                order + 1,
            ),
        ]
    )

    blocks = _blocks(EvidenceLayoutStage().apply(image_path, 1, regions))
    footer_block = next(
        block
        for block in blocks
        if "footer-label" in block.structure["child_evidence_ids"]
    )

    assert footer_block.structure["child_evidence_ids"] == [
        "footer-label",
        "footer-value",
    ]
    assert all(
        not start <= 185 <= end
        for boundary in footer_block.structure["pre_cut_boundaries"]
        if boundary["orientation"] == "vertical"
        for start, end in [boundary["span"]]
    )


def test_financial_artifact_bounds_inferred_sidebar_to_supported_real_ocr_spans() -> (
    None
):
    executable = shutil.which("tesseract")
    if executable is None:
        pytest.skip("Tesseract is unavailable")
    image_path = Path(__file__).parents[1] / "artifacts/demo/financial_table.png"

    result = process_document(
        image_path,
        TesseractReader(executable=executable, page_segmentation_mode=3),
        stages=[EvidenceLayoutStage()],
    )
    blocks = _blocks(result.pages[0].regions)
    sources = {region.id: region for region in result.pages[0].regions}
    boundaries = [
        boundary
        for boundary in blocks[0].structure["pre_cut_boundaries"]  # type: ignore[index]
        if boundary["orientation"] == "vertical"
    ]
    for boundary in boundaries:
        position = boundary["position"]
        start, end = boundary["span"]
        assert not any(
            start < (block.bounding_box.top + block.bounding_box.bottom) / 2 < end
            and any(
                (
                    sources[identifier].bounding_box.left
                    + sources[identifier].bounding_box.right
                )
                / 2
                <= position
                for identifier in block.structure["child_evidence_ids"]
            )
            and any(
                (
                    sources[identifier].bounding_box.left
                    + sources[identifier].bounding_box.right
                )
                / 2
                > position
                for identifier in block.structure["child_evidence_ids"]
            )
            for block in blocks
        )

    assert any(
        block.bounding_box.top >= 1360 and block.bounding_box.right > 900
        for block in blocks
    )


def test_demo_interleaves_layout_blocks_and_table_with_one_presentation_rank() -> None:
    class MixedReader:
        name = "mixed-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                _atomic("title", "Report", "title", (40, 20, 180, 40), 1),
                _word("body", "Body copy", (40, 140, 180, 160), 2),
                TextRegion(
                    id="table",
                    kind="table",
                    text="Metric Value Revenue 42",
                    confidence=0.9,
                    bounding_box=BoundingBox(40, 70, 300, 120),
                    reading_order=99,
                    provider="table-transformer",
                    structure={
                        "role": "table",
                        "row_count": 2,
                        "column_count": 2,
                        "cells": [
                            {"row_nums": [0], "column_nums": [0], "text": "Metric"},
                            {"row_nums": [0], "column_nums": [1], "text": "Value"},
                            {
                                "row_nums": [1],
                                "column_nums": [0],
                                "text": "Revenue",
                            },
                            {"row_nums": [1], "column_nums": [1], "text": "42"},
                        ],
                    },
                ),
            ]

    class MixedPresentationReader:
        name = "falcon-review"

        def read_page(
            self,
            image_path: Path,
            page: PageResult,
        ) -> list[TextRegion]:
            del image_path
            blocks = {
                (region.structure or {}).get("block_type"): region
                for region in page.regions
                if (region.structure or {}).get("role") == "layout_block"
            }
            return [
                TextRegion(
                    id="title-presentation",
                    kind="title",
                    text="Report",
                    confidence=None,
                    bounding_box=blocks["title"].bounding_box,
                    reading_order=50,
                    provider=self.name,
                    structure={"category": "title"},
                    text_provenance={"source_region_id": blocks["title"].id},
                ),
                TextRegion(
                    id="body-presentation",
                    kind="page_text",
                    text="Body copy",
                    confidence=None,
                    bounding_box=blocks["line"].bounding_box,
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": blocks["line"].id},
                ),
            ]

    content = io.BytesIO()
    Image.new("RGB", (360, 200), "white").save(content, format="PNG")
    app = create_app(
        MixedReader(),
        stages=(EvidenceLayoutStage(),),
        presentation_reader=MixedPresentationReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("mixed.png", content.getvalue(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        html = client.get("/").text

    assert response.status_code == 200
    page = payload["result"]["pages"][0]
    top_level = [
        region
        for region in page["regions"]
        if region.get("structure", {}).get("presentation_rank")
    ]
    assert [
        (region["id"], region["structure"]["presentation_rank"]) for region in top_level
    ] == [
        ("table", 2),
        ("p1-layout-1", 1),
        ("p1-layout-2", 3),
    ]
    assert (
        next(region for region in page["regions"] if region["id"] == "table")[
            "reading_order"
        ]
        == 99
    )
    assert page["text"]["value"] == "Report Metric Value Revenue 42 Body copy"
    assert all(
        block["rendering"]["status"] == "selected"
        for block in payload["presentation"]["pages"][0]["blocks"]
    )
    assert markdown.index("### Report") < markdown.index("| Metric | Value |")
    assert markdown.index("| Metric | Value |") < markdown.index("Body copy")
    assert "function presentationOrder(region" in html


def test_demo_uses_a_vertical_rule_as_a_hard_column_pre_cut() -> None:
    class TwoColumnReader:
        name = "two-column-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                _atomic("header", "Page heading", "page_header", (40, 20, 180, 40), 1),
                _atomic("left-1", "Left one", "caption", (40, 100, 110, 120), 2),
                _atomic("right-1", "Right one", "caption", (360, 100, 445, 120), 3),
                _atomic("left-2", "Left two", "caption", (40, 140, 110, 160), 4),
                _atomic("right-2", "Right two", "caption", (360, 140, 445, 160), 5),
                _atomic("footer", "Page footer", "page_footer", (40, 430, 160, 450), 6),
            ]

    image = Image.new("RGB", (600, 500), "white")
    ImageDraw.Draw(image).line((300, 70, 300, 400), fill="black", width=3)
    content = io.BytesIO()
    image.save(content, format="PNG")
    app = create_app(TwoColumnReader(), stages=(EvidenceLayoutStage(),))

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("columns.png", content.getvalue(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    assert response.status_code == 200
    page = payload["result"]["pages"][0]
    blocks = [region for region in page["regions"] if region["kind"] == "layout_block"]
    assert [block["text"] for block in blocks] == [
        "Page heading",
        "Left one",
        "Left two",
        "Right one",
        "Right two",
        "Page footer",
    ]
    assert page["text"]["value"] == (
        "Page heading Left one Left two Right one Right two Page footer"
    )
    assert "Left one\n\nLeft two\n\nRight one\n\nRight two" in markdown
    assert [block["reading_order"] for block in blocks] == list(range(1, 7))
    assert [
        block["structure"]["source_reading_orders"][0]["reading_order"]
        for block in blocks
    ] == [1, 2, 4, 3, 5, 6]
    boundaries = blocks[0]["structure"]["pre_cut_boundaries"]
    assert all(
        block["structure"]["pre_cut_boundaries"] == boundaries for block in blocks
    )
    assert len(boundaries) == 1
    assert boundaries[0]["orientation"] == "vertical"
    assert abs(boundaries[0]["position"] - 300) <= 3
    assert boundaries[0]["span"][0] < 100
    assert boundaries[0]["span"][1] > 160


def test_horizontal_rule_splits_otherwise_adjacent_paragraph_lines(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sections.png"
    image = Image.new("RGB", (500, 200), "white")
    ImageDraw.Draw(image).line((20, 29, 480, 29), fill="black", width=2)
    image.save(image_path)

    result = EvidenceLayoutStage().apply(
        image_path,
        1,
        [
            _word("above", "Above section", (40, 10, 180, 24), 1),
            _word("below", "Below section", (40, 34, 180, 48), 2),
        ],
    )

    assert [block.text for block in _blocks(result)] == [
        "Above section",
        "Below section",
    ]


def test_text_underline_does_not_split_a_rendered_paragraph() -> None:
    class UnderlinedTextReader:
        name = "underlined-text-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                _word("first", "First sentence", (40, 10, 200, 24), 1),
                _word("second", "continues here", (40, 34, 170, 48), 2),
            ]

    image = Image.new("RGB", (500, 200), "white")
    ImageDraw.Draw(image).line((40, 27, 200, 27), fill="black", width=2)
    content = io.BytesIO()
    image.save(content, format="PNG")
    app = create_app(UnderlinedTextReader(), stages=(EvidenceLayoutStage(),))

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("underline.png", content.getvalue(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    assert response.status_code == 200
    page = payload["result"]["pages"][0]
    [block] = [region for region in page["regions"] if region["kind"] == "layout_block"]
    assert block["text"] == "First sentence continues here"
    assert "pre_cut_boundaries" not in block["structure"]
    assert "First sentence continues here" in markdown
