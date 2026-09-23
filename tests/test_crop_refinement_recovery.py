"""Exercise crop validation through the real refinement request boundary."""

import copy
import json

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox
from ocr_pipeline.markdown_refinement import refine_page
from ocr_pipeline.rendering import render_page_markdown
from ocr_pipeline.tableformer_structure import TableFormerStructure
from ocr_pipeline.tables import TableCell


@pytest.mark.parametrize(
    "malformed_response",
    [
        {"bbox": {"l": 0, "t": 0, "r": 300, "b": 200}, "start_row_offset_idx": "oops"},
        {"bbox": {"l": 0, "t": 0, "r": float("inf"), "b": 200}},
    ],
    ids=["malformed-row-offset", "infinite-bounding-box"],
)
def test_one_invalid_crop_does_not_discard_valid_recognition(
    tmp_path, monkeypatch, malformed_response
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "page.png"
    Image.new("RGB", (300, 300), "white").save(image)
    page = {
        "page_number": 1,
        "width": 300,
        "height": 300,
        "regions": [
            {
                "id": "hand",
                "kind": "handwriting",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 300, "bottom": 100},
            },
            {
                "id": "table",
                "kind": "table",
                "text": "original table",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 100, "right": 300, "bottom": 300},
            },
        ],
    }
    original = copy.deepcopy(page)

    broken = False

    class Structure:
        def predict(self, panel):
            if broken:
                return [TableCell(BoundingBox(150, 0, 300, 200), (0,), (1,))]
            return [
                TableCell(BoundingBox(0, 0, 150, 200), (0,), (0,)),
                TableCell(BoundingBox(150, 0, 300, 200), (0,), (1,)),
            ]

        def model_provenance(self):
            return {"id": "fixture"}

    duplicate = False

    def transport(request, timeout):
        payload = json.loads(request.data)
        assert "crop-1" in json.dumps(payload)
        assert payload["messages"][0]["role"] == "system"
        if not broken:
            assert (
                "even when it differs from these predictions"
                in payload["messages"][1]["content"][0]["text"]
            )
        content = {
            "crops": [
                {"region_id": "crop-1", "markdown": "dose 5 mg"},
                {
                    "region_id": "crop-1" if duplicate else "crop-2",
                    "markdown": "<table><tr><td>wrong grid</td></tr></table>",
                },
            ]
        }
        return (
            200,
            {},
            json.dumps(
                {
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {"content": json.dumps(content)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    result = refine_page(
        image, page, "base output", transport=transport, table_reader=Structure()
    )
    assert result["status"] == "partial"
    assert "dose 5 mg" in result["content"]["markdown"]
    assert "wrong grid" not in result["content"]["markdown"]
    assert result["accepted_crop_ids"] == ["crop-1"]
    assert result["rejected_crop_ids"] == ["crop-2"]
    assert result["validation_errors"]
    assert result["cost"] == 0.001
    assert page == original

    duplicate = True
    rejected = refine_page(
        image, page, "base output", transport=transport, table_reader=Structure()
    )
    assert rejected["status"] == "invalid"
    assert rejected["accepted_crop_ids"] == []
    assert rejected["content"]["markdown"] == "base output"

    duplicate = False
    broken = True
    partial = refine_page(
        image, page, "base output", transport=transport, table_reader=Structure()
    )
    assert partial["status"] == "partial"
    assert partial["accepted_crop_ids"] == ["crop-1"]
    assert "dose 5 mg" in partial["content"]["markdown"]
    assert "crop-2" in partial["table_structure_errors"]
    assert "crop-2" in partial["invalid_table_grids"]

    malformed = refine_page(
        image,
        page,
        "base output",
        transport=transport,
        table_reader=TableFormerStructure(predictor=lambda panel: [malformed_response]),
    )
    assert malformed["status"] == "partial"
    assert malformed["accepted_crop_ids"] == ["crop-1"]
    assert malformed["rejected_crop_ids"] == ["crop-2"]
    assert "dose 5 mg" in malformed["content"]["markdown"]
    assert "original table" in malformed["content"]["markdown"]
    assert "wrong grid" not in malformed["content"]["markdown"]
    assert "crop-2" in malformed["table_structure_errors"]
    assert any(
        "crop-2 structure prediction failed" in error
        for error in malformed["validation_errors"]
    )
    assert malformed["table_grids"] == {}
    assert malformed["canonical_unchanged"] is True
    assert page == original


def test_matching_table_grid_replaces_text_in_canonical_physical_cells(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "table.png"
    Image.new("RGB", (200, 200), "white").save(image)
    cells = [
        {
            "id": f"cell-{index}",
            "row_nums": [index // 2],
            "column_nums": [index % 2],
            "text": value,
            "bounding_box": {
                "left": 100 * (index % 2),
                "top": 100 * (index // 2),
                "right": 100 * (index % 2 + 1),
                "bottom": 100 * (index // 2 + 1),
            },
            "resolution": "unreadable",
            "decision": "falcon_agreement",
            "evidence_ids": [f"falcon-word-{index}"],
        }
        for index, value in enumerate(("old A", "old B", "old C", "old D"))
    ]
    page = {
        "page_number": 1,
        "width": 200,
        "height": 200,
        "regions": [
            {
                "id": "table",
                "kind": "table",
                "text": "old A old B old C old D",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 200},
                "structure": {
                    "role": "table",
                    "row_count": 2,
                    "column_count": 2,
                    "header_row_count": 0,
                    "cells": cells,
                },
            }
        ],
    }

    class Structure:
        def predict(self, panel):
            return (
                TableCell(BoundingBox(0, 0, 100, 100), (0,), (0,)),
                TableCell(BoundingBox(100, 0, 200, 100), (0,), (1,)),
                TableCell(BoundingBox(0, 100, 100, 200), (1,), (0,)),
                TableCell(BoundingBox(100, 100, 200, 200), (1,), (1,)),
            )

        def model_provenance(self):
            return {"id": "fixture"}

    def transport(request, timeout):
        payload = json.loads(request.data)
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": (
                        "<table><tr><th>old C</th><th>old D</th></tr>"
                        "<tr><td>old A</td><td>old B</td></tr></table>"
                        "\nhandwritten note"
                    ),
                }
            ]
        }
        return (
            200,
            {},
            json.dumps(
                {
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {"content": json.dumps(answer)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    result = refine_page(
        image, page, "original", transport=transport, table_reader=Structure()
    )
    assert result["status"] == "generated"
    assert result["accepted_crop_ids"] == ["crop-1"]
    assert "old C" in result["content"]["markdown"]
    assert "handwritten note" not in result["content"]["markdown"]
    refined = result["refined_regions"][0]
    assert refined["kind"] == "table"
    assert [cell["text"] for cell in refined["structure"]["cells"]] == [
        "old C",
        "old D",
        "old A",
        "old B",
    ]
    assert all(
        "confidence" not in cell
        and "confidence_kind" not in cell
        and "decision" not in cell
        and cell["evidence_ids"] == [f"falcon-word-{index}"]
        for cell in refined["structure"]["cells"]
        for index in [int(cell["id"].split("-")[-1])]
    )
    assert refined["alternatives"][0]["decision_state"] == "superseded"
    assert refined["structure"]["header_row_count"] == 0
    assert (
        refined["text_provenance"]["physical_cell_alignment"]
        == "falcon_tableformer_qwen_consensus"
    )
    assert (
        json.loads(json.dumps(result["refined_regions"])) == result["refined_regions"]
    )
    assert page["regions"][0]["structure"]["cells"] == cells

    ambiguous = copy.deepcopy(page)
    ambiguous["regions"][0]["structure"]["cells"].pop()
    rejected = refine_page(
        image, ambiguous, "original", transport=transport, table_reader=Structure()
    )
    assert rejected["status"] == "invalid"
    assert rejected["content"]["markdown"] == "original"
    assert (
        rejected["refined_regions"][0]["structure"]["cells"]
        == ambiguous["regions"][0]["structure"]["cells"]
    )


@pytest.mark.parametrize(
    ("word_provider", "word_provenance"),
    [
        ("nemotron-ocr-v2", {}),
        (
            "falcon-on-nemotron",
            {"method": "word_reader_uncovered", "geometry_provider": "nemotron-ocr-v2"},
        ),
    ],
)
def test_form_refinement_preserves_separately_resolved_word_ownership(
    tmp_path, monkeypatch, word_provider, word_provenance
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "form.png"
    Image.new("RGB", (300, 200), "white").save(image)
    page = {
        "page_number": 1,
        "width": 300,
        "height": 200,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 150},
                "text_provenance": {"layout_category": "form"},
            },
            {
                "id": "word",
                "kind": "text",
                "text": "Dose",
                "resolution": "resolved",
                "provider": word_provider,
                "text_provenance": word_provenance,
                "bounding_box": {"left": 20, "top": 20, "right": 65, "bottom": 40},
            },
            {
                "id": "other-word",
                "kind": "text",
                "text": "Nurse",
                "resolution": "resolved",
                "provider": "nemotron-ocr-v2",
                "bounding_box": {"left": 220, "top": 20, "right": 275, "bottom": 40},
            },
        ],
    }
    original = copy.deepcopy(page)
    base = render_page_markdown(page["regions"], [r["id"] for r in page["regions"]])
    answer_text = "Dose: 5 mg"

    def transport(request, timeout):
        payload = json.loads(request.data)
        return (
            200,
            {},
            json.dumps(
                {
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "crops": [
                                            {
                                                "region_id": "crop-1",
                                                "markdown": answer_text,
                                            }
                                        ]
                                    }
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    result = refine_page(image, page, base, transport=transport)

    assert result["status"] == "invalid"
    assert result["accepted_crop_ids"] == []
    assert result["rejected_crop_ids"] == ["crop-1"]
    assert result["content"]["markdown"] == base
    assert (
        render_page_markdown(
            result["refined_regions"], [r["id"] for r in result["refined_regions"]]
        )
        == base
    )
    assert "Dose" in base and "Nurse" in base
    assert "Dose: 5 mg" not in base
    assert result["refined_regions"][0]["resolution"] == "unreadable"
    assert result["refined_regions"][0]["alternatives"][0]["text"] == "Dose: 5 mg"
    assert (
        result["refined_regions"][0]["alternatives"][0]["decision_state"] == "pending"
    )
    assert result["refined_regions"][0]["alternatives"][0]["text_provenance"][
        "overlapping_region_ids"
    ] == ["word"]
    assert page == original

    disjoint = copy.deepcopy(page)
    disjoint["regions"][1]["bounding_box"] = {
        "left": 201,
        "top": 20,
        "right": 219,
        "bottom": 40,
    }
    disjoint_result = refine_page(image, disjoint, base, transport=transport)
    assert disjoint_result["status"] == "generated"
    assert disjoint_result["accepted_crop_ids"] == ["crop-1"]

    answer_text = "☑ None of the Above\n☐ Z1. None of the Above"
    conflicting = refine_page(image, disjoint, base, transport=transport)
    assert conflicting["status"] == "invalid"
    assert conflicting["accepted_crop_ids"] == []
    assert conflicting["content"]["markdown"] == base
    assert conflicting["refined_regions"][0]["alternatives"][0]["text"] == answer_text
    assert "unaligned form controls" in conflicting["validation_errors"][0]

    answer_text = "☑ A1. Yes\n☐ B1. Yes"
    distinct = refine_page(image, disjoint, base, transport=transport)
    assert distinct["status"] == "invalid"
    assert distinct["content"]["markdown"] == base
    assert distinct["refined_regions"][0]["alternatives"][0]["text"] == answer_text

    answer_text = "Question A: ☐ No\n☑ Yes\nQuestion B: ☑ No\n☐ Yes"
    repeated = refine_page(image, disjoint, base, transport=transport)
    assert repeated["status"] == "invalid"
    assert repeated["content"]["markdown"] == base
    assert repeated["refined_regions"][0]["alternatives"][0]["text"] == answer_text
    assert repeated["refined_regions"][0]["alternatives"][0]["text_provenance"][
        "unaligned_controls"
    ]
