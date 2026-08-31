from __future__ import annotations

import pytest

from ocr_pipeline.cascade import (
    apply_region_patches,
    build_patch_request,
    identify_risky_regions,
    regions_have_fewer_risks,
)
from ocr_pipeline.contracts import (
    BoundingBox,
    DocumentResult,
    EvidenceText,
    PageResult,
    TextRegion,
)


def test_risky_region_patch_updates_only_authorized_text() -> None:
    document = _document()
    risks = identify_risky_regions(document)
    assert risks == {
        "risky": ["empty_content", "duplicate_reading_order"],
        "table": ["duplicate_reading_order", "non_rectangular_html_table"],
    }

    request = build_patch_request(document, risks)
    assert request["authorized_region_ids"] == ["risky", "table"]

    result = apply_region_patches(
        document,
        request,
        [{"id": "risky", "text": "recovered", "provider": "local"}],
    )

    assert document.pages[0].regions[1].text == ""
    assert result.pages[0].regions[0].text == "stable"
    assert result.pages[0].regions[1].text == "recovered"
    assert result.pages[0].regions[1].kind == "word"
    assert result.pages[0].text.value.startswith("stable recovered")
    assert regions_have_fewer_risks(document, result, ["risky"])
    assert result is not document


def test_empty_non_text_visual_region_is_not_risky() -> None:
    document = _document()
    visual = document.pages[0].regions[1]
    visual.kind = "header-image"
    visual.reading_order = 3
    document.pages[0].regions[2].reading_order = 2

    assert "risky" not in identify_risky_regions(document)


@pytest.mark.parametrize(
    "text",
    [
        "Accession No. Accession No. Accession No.",
        "Patient ID Name Date Status Result " * 4,
    ],
)
def test_repeated_token_phrase_is_risky(text: str) -> None:
    document = _document()
    document.pages[0].regions[0].text = text

    assert "repeated_text" in identify_risky_regions(document)["stable"]


def test_normal_prose_single_word_and_table_repetition_are_not_risky() -> None:
    document = _document()
    stable, repeated_word, table = document.pages[0].regions
    stable.text = "The patient returned for a routine follow up visit today."
    repeated_word.text = "yes yes yes yes yes yes"
    table.text = "row value row value row value"
    table.reading_order = 3

    assert identify_risky_regions(document) == {}


def test_risk_reasons_cover_geometry_order_and_malformed_tables() -> None:
    document = _document()
    document.pages[0].regions[1].bounding_box.right = 101
    document.pages[0].regions[1].reading_order = 9
    document.pages[0].regions[2].text = "<table><tr><td>broken</table>"

    risks = identify_risky_regions(document)

    assert risks["risky"] == [
        "empty_content",
        "invalid_bbox",
        "out_of_range_reading_order",
    ]
    assert risks["table"] == ["malformed_html_table"]


@pytest.mark.parametrize(
    "invalid_span",
    ["colspan='0'", "rowspan='invalid'"],
)
def test_rowspan_does_not_hide_later_invalid_span(invalid_span: str) -> None:
    document = _document()
    document.pages[0].regions[2].reading_order = 3
    document.pages[0].regions[2].text = (
        "<table><tr><td rowspan='2'>a</td><td>b</td></tr>"
        f"<tr><td {invalid_span}>c</td></tr></table>"
    )

    assert identify_risky_regions(document)["table"] == ["malformed_html_table"]


def test_valid_rowspan_leaves_rectangularity_indeterminate() -> None:
    document = _document()
    document.pages[0].regions[2].reading_order = 3
    document.pages[0].regions[2].text = (
        "<table><tr><td rowspan='2'>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
    )

    assert identify_risky_regions(document) == {"risky": ["empty_content"]}


@pytest.mark.parametrize(
    "patch",
    [
        {"id": "stable", "text": "unauthorized"},
        {"id": "risky", "text": "replacement", "kind": "table"},
        {
            "id": "risky",
            "text": "replacement",
            "bounding_box": {"left": 0, "top": 0, "right": 99, "bottom": 10},
        },
    ],
)
def test_patch_rejects_unauthorized_or_protected_changes(
    patch: dict[str, object],
) -> None:
    document = _document()
    request = build_patch_request(document)

    with pytest.raises(ValueError):
        apply_region_patches(document, request, [patch])

    assert document.pages[0].regions[1].text == ""


def _document() -> DocumentResult:
    regions = [
        TextRegion(
            id="stable",
            kind="word",
            text="stable",
            confidence=0.99,
            bounding_box=BoundingBox(0, 0, 10, 10),
            reading_order=1,
            provider="local",
        ),
        TextRegion(
            id="risky",
            kind="word",
            text="",
            confidence=0.2,
            bounding_box=BoundingBox(10, 0, 20, 10),
            reading_order=2,
            provider="local",
        ),
        TextRegion(
            id="table",
            kind="table",
            text="<table><tr><td>a</td></tr><tr><td>b</td><td>c</td></tr></table>",
            confidence=0.8,
            bounding_box=BoundingBox(0, 20, 80, 80),
            reading_order=2,
            provider="local",
        ),
    ]
    return DocumentResult(
        document_id="doc",
        source={"name": "doc.png", "kind": "image"},
        status="success",
        pages=[
            PageResult(
                page_number=1,
                width=100,
                height=100,
                reader="local",
                route="review",
                text=EvidenceText("stable", [region.id for region in regions]),
                regions=regions,
            )
        ],
    )
