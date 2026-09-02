from ocr_pipeline.rendering import render_page_markdown


def _region(
    region_id: str,
    text: str,
    order: int,
    box: dict[str, int],
    *,
    merge_level: str = "word",
) -> dict[str, object]:
    return {
        "id": region_id,
        "kind": "text",
        "text": text,
        "reading_order": order,
        "resolution": "resolved",
        "bounding_box": box,
        "text_provenance": {"merge_level": merge_level},
    }


def test_markdown_groups_word_level_text_regions_on_the_same_row() -> None:
    regions = [
        _region(
            "one",
            "Patient",
            1,
            {"left": 10, "top": 20, "right": 55, "bottom": 32},
        ),
        _region(
            "two",
            "Name",
            2,
            {"left": 59, "top": 20, "right": 90, "bottom": 32},
        ),
        _region(
            "paragraph",
            "Preserve this paragraph.",
            3,
            {"left": 94, "top": 20, "right": 220, "bottom": 32},
            merge_level="paragraph",
        ),
    ]

    markdown = render_page_markdown(regions, ["one", "two", "paragraph"])

    assert markdown == "Patient Name\n\nPreserve this paragraph."


def test_markdown_keeps_word_level_text_in_separate_columns() -> None:
    regions = [
        _region(
            "left-one",
            "Patient",
            1,
            {"left": 10, "top": 20, "right": 55, "bottom": 32},
        ),
        _region(
            "left-two",
            "Name",
            2,
            {"left": 59, "top": 20, "right": 90, "bottom": 32},
        ),
        _region(
            "right-one",
            "Member",
            3,
            {"left": 210, "top": 20, "right": 260, "bottom": 32},
        ),
        _region(
            "right-two",
            "Phone",
            4,
            {"left": 264, "top": 20, "right": 300, "bottom": 32},
        ),
    ]

    markdown = render_page_markdown(
        regions,
        ["left-one", "left-two", "right-one", "right-two"],
    )

    assert markdown == "Patient Name\n\nMember Phone"
