from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, PageResult, TextAlternative, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.evidence_layout import EvidenceLayoutStage
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
    assert block.structure["fields"] == [  # type: ignore[index]
        {"evidence_ids": ["label", "value"]},
        {"evidence_ids": ["date-label", "date"]},
    ]
    assert block.text == "Patient Lee | Date 03/18/2025"
    assert block.bounding_box == BoundingBox(10, 10, 330, 22)
    assert not any(region.kind == "table" for region in result)


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
    assert block.structure["fields"] == [  # type: ignore[index]
        {"evidence_ids": ["label", "value"]},
        {"evidence_ids": ["date-label", "date"]},
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
    assert block.structure["fields"] == [  # type: ignore[index]
        {"evidence_ids": ["date-label", "date"]},
        {"evidence_ids": ["age-label", "age"]},
        {"evidence_ids": ["gender-label"]},
    ]
    assert block.text == "Date of Service: 03/18/2025 | Age: 85 | Gender:"


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
