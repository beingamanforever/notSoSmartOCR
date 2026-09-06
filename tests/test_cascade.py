from __future__ import annotations

import pytest

from ocr_pipeline.cascade import (
    add_runtime_risk_evidence,
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


def test_invalid_date_is_flagged_but_not_rewritten() -> None:
    document = _document(
        TextRegion(
            id="date",
            kind="text",
            text="DOB: 02/31/2024",
            confidence=0.9,
            bounding_box=BoundingBox(0, 0, 20, 20),
            reading_order=1,
            provider="local",
        )
    )

    assert identify_risky_regions(document) == {"date": ["invalid_calendar_date"]}


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


def test_repeated_character_corruption_is_risky() -> None:
    document = _document()
    document.pages[0].regions[0].text = "te eeessssmennt pppatttt"

    assert "character_repetition" in identify_risky_regions(document)["stable"]


def test_tail_repetition_is_flagged_without_changing_literal_text() -> None:
    document = _document()
    text = "stable prefix " + "AB12|" * 8
    document.pages[0].regions[0].text = text

    risks = identify_risky_regions(document)

    assert "tail_repetition" in risks["stable"]
    assert document.pages[0].regions[0].text == text


def test_tail_repetition_requires_eight_nonblank_units() -> None:
    document = _document()
    document.pages[0].regions[0].text = "stable prefix " + "AB12|" * 7

    assert "tail_repetition" not in identify_risky_regions(document).get("stable", [])


def test_long_repeated_suffix_is_flagged_without_changing_literal_text() -> None:
    document = _document()
    suffix = "Medication dosage remains unchanged."
    text = f"stable prefix {suffix}{suffix.upper()}"
    document.pages[0].regions[0].text = text

    risks = identify_risky_regions(document)

    assert "repeated_suffix" in risks["stable"]
    assert "tail_repetition" not in risks["stable"]
    assert document.pages[0].regions[0].text == text


def test_short_or_nonadjacent_suffix_is_not_flagged() -> None:
    document = _document()
    document.pages[0].regions[0].text = "prefix short suffix short suffix"

    assert "repeated_suffix" not in identify_risky_regions(document).get("stable", [])


@pytest.mark.parametrize("orders", [(7,), (1, 1)])
def test_offline_order_diagnostics_do_not_drive_runtime_review(
    orders: tuple[int, ...],
) -> None:
    regions = [
        TextRegion(
            id=f"region-{index}",
            kind="text",
            text=f"literal {index}",
            confidence=0.9,
            bounding_box=BoundingBox(index * 10, 0, (index + 1) * 10, 10),
            reading_order=order,
            provider="local",
        )
        for index, order in enumerate(orders)
    ]
    document = _document(*regions)
    before = len(document.pages[0].regions)

    offline_reasons = {
        reason
        for reasons in identify_risky_regions(document).values()
        for reason in reasons
    }

    assert offline_reasons & {
        "duplicate_reading_order",
        "out_of_range_reading_order",
    }
    assert add_runtime_risk_evidence(document) == set()
    assert len(document.pages[0].regions) == before


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
    document.pages[0].regions[
        2
    ].text = (
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


def _document(*custom_regions: TextRegion) -> DocumentResult:
    regions = list(custom_regions) or [
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
