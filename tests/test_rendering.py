from dataclasses import asdict

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.rendering import render_evidence, render_page_markdown


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


def test_resolved_formula_uses_owner_text_in_plain_and_markdown_rendering() -> None:
    child = TextRegion(
        id="child",
        kind="word",
        text="stale child text",
        confidence=0.9,
        bounding_box=BoundingBox(10, 10, 100, 30),
        reading_order=1,
        provider="reader",
        structure={"layout_owner_id": "formula"},
    )
    formula = TextRegion(
        id="formula",
        kind="layout_block",
        text=r"\int_0^1 x^2\,dx = \frac{1}{3}",
        confidence=0.9,
        bounding_box=BoundingBox(10, 10, 200, 40),
        reading_order=2,
        provider="canonical",
        structure={
            "role": "layout_block",
            "block_type": "formula",
            "lines": [{"evidence_ids": ["child"]}],
        },
    )
    regions = [child, formula]

    evidence = render_evidence(regions)

    assert evidence.value == formula.text
    assert (
        render_page_markdown(
            [asdict(region) for region in regions], evidence.evidence_ids
        )
        == f"$$\n{formula.text}\n$$"
    )


def test_heuristic_formula_text_is_not_wrapped_as_verified_math() -> None:
    formula = TextRegion(
        id="candidate",
        kind="layout_block",
        text="to = be",
        confidence=None,
        bounding_box=BoundingBox(10, 10, 80, 30),
        reading_order=1,
        provider="evidence-spatial-layout",
        structure={
            "role": "layout_block",
            "block_type": "formula",
            "formula_recognition": "heuristic",
        },
    )

    evidence = render_evidence([formula])

    assert evidence.value == "to = be"
    assert render_page_markdown([asdict(formula)], evidence.evidence_ids) == "to = be"


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


def test_markdown_leaves_unreadable_control_state_blank_without_changing_evidence() -> (
    None
):
    control = {
        "id": "foley",
        "kind": "checkbox",
        "text": "[?] Foley (specify type and specific orders)",
        "reading_order": 1,
        "resolution": "unreadable",
        "alternatives": [{"text": "[x] Foley (specify type and specific orders)"}],
        "structure": {
            "role": "control",
            "label": "Foley (specify type and specific orders)",
        },
    }

    markdown = render_page_markdown([control], [])

    assert markdown == "- [?] Foley (specify type and specific orders)"
    assert control["text"] == "[?] Foley (specify type and specific orders)"
    assert control["resolution"] == "unreadable"


def test_markdown_omits_unlabeled_unreadable_control() -> None:
    control = {
        "id": "unknown",
        "kind": "checkbox",
        "text": "[?]",
        "reading_order": 1,
        "resolution": "unreadable",
    }

    assert render_page_markdown([control], []) == ""


def test_markdown_never_suppresses_structural_table_linked_by_a_control() -> None:
    table = {
        "id": "table",
        "kind": "table",
        "text": "",
        "reading_order": 1,
        "resolution": "resolved",
        "structure": {
            "role": "table",
            "row_count": 2,
            "column_count": 2,
            "cells": [
                {"row_nums": [0], "column_nums": [0], "text": "Task"},
                {"row_nums": [0], "column_nums": [1], "text": "Wed"},
                {"row_nums": [1], "column_nums": [0], "text": "Bed Bath"},
                {"row_nums": [1], "column_nums": [1], "text": "x"},
            ],
        },
    }
    control = {
        "id": "control",
        "kind": "checkbox",
        "text": "[x] Bed Bath (Wed)",
        "reading_order": 2,
        "resolution": "resolved",
        "structure": {
            "role": "control",
            "label_evidence_ids": ["table"],
        },
    }

    markdown = render_page_markdown([table, control], ["table"])

    assert "| Task | Wed |" in markdown
    assert "- [x] Bed Bath (Wed)" in markdown


def test_markdown_renders_spanning_table_as_safe_html_once() -> None:
    table = {
        "id": "table",
        "kind": "table",
        "text": "literal fallback",
        "reading_order": 1,
        "resolution": "resolved",
        "structure": {
            "role": "table",
            "row_count": 2,
            "column_count": 2,
            "cells": [
                {
                    "row_nums": [0, 1],
                    "column_nums": [0],
                    "text": "Group <script>alert(1)</script>",
                    "projected_row_header": True,
                },
                {"row_nums": [0], "column_nums": [1], "text": "A & B"},
                {"row_nums": [1], "column_nums": [1], "text": ""},
            ],
        },
    }

    markdown = render_page_markdown([table], ["table"])

    assert markdown == (
        "<table>\n"
        "<tbody>\n"
        "<tr>\n"
        '<th rowspan="2">Group &lt;script&gt;alert(1)&lt;/script&gt;</th>\n'
        "<td>A &amp; B</td>\n"
        "</tr>\n"
        "<tr>\n"
        "<td></td>\n"
        "</tr>\n"
        "</tbody>\n"
        "</table>"
    )
    assert markdown.count("Group") == 1
    assert "<script>" not in markdown


def test_complete_non_spanning_table_keeps_explicit_blank_in_markdown() -> None:
    table = {
        "id": "table",
        "kind": "table",
        "text": "fallback",
        "reading_order": 1,
        "resolution": "resolved",
        "structure": {
            "role": "table",
            "row_count": 1,
            "column_count": 2,
            "cells": [
                {"row_nums": [0], "column_nums": [0], "text": "Name"},
                {"row_nums": [0], "column_nums": [1], "text": ""},
            ],
        },
    }

    assert render_page_markdown([table], ["table"]) == ("| Name |  |\n| --- | --- |")


def test_invalid_table_topology_keeps_literal_evidence_without_diagnostic_copy() -> (
    None
):
    table = {
        "id": "table",
        "kind": "table",
        "text": "<table><tr><td>literal evidence</td></tr></table>",
        "reading_order": 1,
        "resolution": "resolved",
        "structure": {
            "role": "table",
            "row_count": 1,
            "column_count": 2,
            "cells": [{"row_nums": [0], "column_nums": [0], "text": "A"}],
        },
    }

    markdown = render_page_markdown([table], ["table"])

    assert markdown == "    <table><tr><td>literal evidence</td></tr></table>"
    assert "structure problem" not in markdown.casefold()
    assert "| A |" not in markdown


def test_unresolved_table_cells_render_blank_without_mutating_evidence() -> None:
    cells = [
        {"row_nums": [0], "column_nums": [0], "text": "Medication"},
        {
            "row_nums": [0],
            "column_nums": [1],
            "text": "Metforrnin",
            "resolution": "conflicting",
            "alternatives": [{"text": "Metformin"}],
        },
        {"row_nums": [1], "column_nums": [0], "text": "Dose"},
        {
            "row_nums": [1],
            "column_nums": [1],
            "text": "",
            "resolution": "unreadable",
        },
    ]
    table = {
        "id": "table",
        "kind": "table",
        "text": "",
        "reading_order": 1,
        "resolution": "resolved",
        "structure": {
            "role": "table",
            "row_count": 2,
            "column_count": 2,
            "cells": cells,
        },
    }

    markdown = render_page_markdown([table], ["table"])

    assert markdown == "| Medication |  |\n| --- | --- |\n| Dose |  |"
    assert "Metforrnin" not in markdown
    assert "Metformin" not in markdown
    assert "unreadable" not in markdown.casefold()
    assert "conflicting" not in markdown.casefold()
    assert cells[1]["text"] == "Metforrnin"
    assert cells[1]["alternatives"] == [{"text": "Metformin"}]
