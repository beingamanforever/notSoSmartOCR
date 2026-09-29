"""Exercise crop validation through the real refinement request boundary."""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, EvidenceText, PageResult, TextRegion
from ocr_pipeline.cross_page_tables import (
    CrossPageTableStage,
    TableContinuationPrediction,
)
from ocr_pipeline.demo import create_app
from ocr_pipeline.markdown_refinement import _owned_form_rows, refine_page
from ocr_pipeline.rendering import render_page_markdown


def test_cross_page_tables_use_refined_cells_once(tmp_path):
    source = tmp_path / "pages.tiff"
    pages = [Image.new("RGB", (100, 100), "white") for _ in range(2)]
    pages[0].save(source, save_all=True, append_images=pages[1:], format="TIFF")

    def table(page_number, value):
        values = ("Medication", "Dose", value, "10 mg")
        cells = [
            {
                "id": f"p{page_number}-cell-{index + 1}",
                "row_nums": [index // 2],
                "column_nums": [index % 2],
                "text": text,
                "resolution": "resolved",
                "column_header": index < 2,
            }
            for index, text in enumerate(values)
        ]
        return TextRegion(
            id=f"p{page_number}-table",
            kind="table",
            text="\n".join(values),
            confidence=0.99,
            bounding_box=BoundingBox(5, 5, 95, 95),
            reading_order=1,
            provider="fixture",
            structure={
                "role": "table",
                "row_count": 2,
                "column_count": 2,
                "cells": cells,
            },
        )

    class Reader:
        name = "table-fixture"

        def read(self, image_path: Path, page_number: int):
            return [table(page_number, ("Aspirin", "Metformin")[page_number - 1])]

    class Classifier:
        name = "table-continuation-fixture"
        calls = 0

        def classify(self, *args):
            self.calls += 1
            return TableContinuationPrediction(0.93, {"checkpoint": "fixture"})

    def refine(document, page_paths):
        assert len(page_paths) == 2
        document.pages[1].regions[0].structure["cells"][2]["text"] = "Refined Metformin"

    classifier = Classifier()
    app = create_app(
        Reader(),
        crop_refiner=refine,
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("pages.tiff", source.read_bytes(), "image/tiff")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert classifier.calls == 1
    assert [run["stage"] for run in payload["stage_execution"]] == [
        "qwen-crop-refinement",
        "qwen-crop-refinement",
        "cross-page-tables",
    ]
    assert "Refined Metformin" in [
        cell["text"] for cell in payload["result"]["table_continuations"][0]["cells"]
    ]


def test_form_source_adds_physically_owned_controls_without_replacing_text(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "form.png"
    Image.new("RGB", (200, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 20, "top": 10, "right": 180, "bottom": 90},
                "structure": {"role": "form_source"},
            },
            {
                "id": "existing",
                "kind": "checkbox",
                "text": "Existing option",
                "resolution": "resolved",
                "bounding_box": {"left": 28, "top": 18, "right": 92, "bottom": 34},
                "structure": {"label": "Existing option", "state": "unselected"},
            },
        ],
    }

    duplicate_crop = False

    def transport(request, timeout):
        payload = json.loads(request.data)
        assert "form_source" in json.dumps(payload["messages"][1])
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "Existing option",
                            "group_label": "Choices",
                            "state": "unselected",
                            "bbox_2d": [50, 100, 450, 300],
                        },
                        {
                            "label": "Recovered option",
                            "group_label": "Choices",
                            "state": "selected",
                            "bbox_2d": [50, 500, 700, 725],
                        },
                    ],
                }
            ]
        }
        if duplicate_crop:
            answer["crops"].append(copy.deepcopy(answer["crops"][0]))
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
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    result = refine_page(image, page, "base output", transport=transport)

    recovered = [
        region
        for region in result["refined_regions"]
        if (region.get("text_provenance") or {}).get("method")
        == "qwen_form_option_extraction"
    ]
    assert result["status"] == "generated"
    assert result["accepted_crop_ids"] == []
    assert result["accepted_control_crop_ids"] == ["crop-1"]
    assert result["form_controls_added"] == 1
    assert len(recovered) == 1
    assert recovered[0]["text"] == "Recovered option"
    assert recovered[0]["structure"]["state"] == "selected"
    assert recovered[0]["bounding_box"] == {
        "left": 28,
        "top": 50,
        "right": 132,
        "bottom": 68,
    }
    assert "[x] Recovered option" in result["content"]["markdown"]
    assert "[form options extracted]" not in result["content"]["markdown"]

    duplicate_crop = True
    malformed = refine_page(image, page, "base output", transport=transport)
    assert malformed["status"] == "invalid"
    assert malformed["form_controls_added"] == 0
    assert malformed["accepted_control_crop_ids"] == []
    assert malformed["refined_regions"] is None


def test_owned_form_row_uses_single_row_evidence_for_nonoverlapping_controls(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "form.png"
    Image.new("RGB", (200, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
                "structure": {"role": "form_source"},
            },
            {
                "id": "row",
                "kind": "layout_block",
                "text": "□ Existing option □ Recovered option",
                "resolution": "resolved",
                "bounding_box": {"left": 10, "top": 10, "right": 190, "bottom": 50},
                "structure": {
                    "role": "layout_block",
                    "block_type": "form_row",
                    "child_evidence_ids": ["row-evidence"],
                    "fields": [{"evidence_ids": ["row-evidence"]}],
                },
            },
            {
                "id": "row-evidence",
                "kind": "text",
                "text": "□ Existing option □ Recovered option",
                "resolution": "resolved",
                "bounding_box": {
                    "left": 10,
                    "top": 20,
                    "right": 190,
                    "bottom": 50,
                },
                "structure": {"layout_owner_id": "row"},
            },
            {
                "id": "signature",
                "kind": "signature",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 10, "top": 60, "right": 100, "bottom": 90},
            },
        ],
    }

    def transport(request, timeout):
        payload = json.loads(request.data)
        prompt = json.dumps(payload["messages"][1])
        schema_name = payload["response_format"]["json_schema"]["name"]
        if schema_name == "page_crops":
            assert "page_context" not in prompt
            assert "owned_form_row" not in prompt
            assert "form_source" in prompt
            crops = [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "Uncovered option",
                            "group_label": "Other",
                            "state": "unselected",
                            "bbox_2d": [100, 700, 600, 900],
                        }
                    ],
                },
                {"region_id": "crop-2", "markdown": "[signature]"},
            ]
        else:
            assert schema_name == "owned_form_rows"
            assert "page_context" in prompt
            assert "owned_form_row" in prompt
            assert (
                sum(
                    item.get("type") == "image_url"
                    for item in payload["messages"][1]["content"]
                )
                == 2
            )
            crops = [
                {
                    "region_id": "crop-3",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "Existing option",
                            "group_label": "Choices",
                            "state": "unselected",
                            "bbox_2d": [56, 125, 389, 500],
                        },
                        {
                            "label": "Recovered option",
                            "group_label": "Choices",
                            "state": "selected",
                            "bbox_2d": [500, 125, 950, 500],
                        },
                    ],
                }
            ]
        answer = {"crops": crops}
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
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                }
            ).encode(),
        )

    result = refine_page(image, page, "base output", transport=transport)

    recovered = [
        region
        for region in result["refined_regions"]
        if region.get("text") == "Recovered option"
        and (region.get("text_provenance") or {}).get("method")
        == "qwen_form_option_extraction"
    ]
    row = next(region for region in result["refined_regions"] if region["id"] == "row")
    assert result["status"] == "generated"
    assert result["form_controls_added"] == 3
    assert result["accepted_control_crop_ids"] == ["crop-1", "crop-3"]
    assert row["kind"] == "layout_block"
    assert row["text"] == "[ ] Existing option [x] Recovered option"
    assert recovered[0]["structure"]["layout_owner_id"] == "row"
    assert recovered[0]["structure"]["label_evidence_ids"] == ["row-evidence"]
    added = [
        region
        for region in result["refined_regions"]
        if (region.get("text_provenance") or {}).get("method")
        == "qwen_form_option_extraction"
        and region.get("structure", {}).get("layout_owner_id") == "row"
    ]
    assert all(
        region["structure"]["label_evidence_ids"] == ["row-evidence"]
        for region in added
    )
    assert {region["text"] for region in added} == {
        "Existing option",
        "Recovered option",
    }
    assert result["content"]["markdown"].count("[ ] Existing option") == 1
    assert result["content"]["markdown"].count("[x] Recovered option") == 1
    assert result["content"]["markdown"].count("Uncovered option") == 1


def test_owned_form_row_failure_keeps_completed_primary_request(tmp_path, monkeypatch):
    import ocr_pipeline.markdown_refinement as refinement

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.setattr(refinement.time, "sleep", lambda seconds: None)
    image = tmp_path / "form.png"
    Image.new("RGB", (200, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
                "structure": {"role": "form_source"},
            },
            {
                "id": "row",
                "kind": "layout_block",
                "text": "□ First □ Second",
                "resolution": "resolved",
                "bounding_box": {"left": 10, "top": 10, "right": 190, "bottom": 50},
                "structure": {"role": "layout_block", "block_type": "form_row"},
            },
            {
                "id": "signature",
                "kind": "signature",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 10, "top": 60, "right": 100, "bottom": 90},
            },
        ],
    }

    def transport(request, timeout):
        payload = json.loads(request.data)
        if payload["response_format"]["json_schema"]["name"] == "owned_form_rows":
            return 500, {}, b'{"error":{"message":"row reader unavailable"}}'
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "Uncovered option",
                            "group_label": "Other",
                            "state": "unselected",
                            "bbox_2d": [100, 700, 600, 900],
                        }
                    ],
                },
                {"region_id": "crop-2", "markdown": "[signature]"},
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
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "cost": 0.001,
                    },
                }
            ).encode(),
        )

    result = refine_page(image, page, "base output", transport=transport)

    assert result["status"] == "partial"
    assert result["cost"] == 0.001
    # The row request's HTTP 500 is retried before it is given up.
    assert result["attempts"] == 1 + refinement.QWEN_ATTEMPTS
    assert result["accepted_crop_ids"] == ["crop-2"]
    assert result["rejected_crop_ids"] == ["crop-1", "crop-3"]
    assert result["accepted_control_crop_ids"] == ["crop-1"]
    assert result["request_errors"][0]["request"] == "owned_form_rows"
    assert "[signature]" in result["content"]["markdown"]


@pytest.mark.parametrize("option_box", ([0, 600, 100, 900], [0, 0, 1000, 1000]))
def test_owned_form_row_keeps_ambiguous_label_ownership_out_of_canonical_text(
    tmp_path, monkeypatch, option_box
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "form.png"
    Image.new("RGB", (200, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
                "structure": {"role": "form_source"},
            },
            {
                "id": "row",
                "kind": "layout_block",
                "text": "□ Option A □ Option B",
                "resolution": "resolved",
                "bounding_box": {"left": 0, "top": 10, "right": 200, "bottom": 50},
                "structure": {
                    "role": "layout_block",
                    "block_type": "form_row",
                    "child_evidence_ids": ["label-a", "label-b"],
                    "fields": [
                        {"evidence_ids": ["label-a"]},
                        {"evidence_ids": ["label-b"]},
                    ],
                },
            },
            {
                "id": "label-a",
                "kind": "text",
                "text": "Option A",
                "resolution": "resolved",
                "bounding_box": {"left": 10, "top": 15, "right": 80, "bottom": 30},
                "structure": {"layout_owner_id": "row"},
            },
            {
                "id": "label-b",
                "kind": "text",
                "text": "Option B",
                "resolution": "resolved",
                "bounding_box": {
                    "left": 120,
                    "top": 15,
                    "right": 190,
                    "bottom": 30,
                },
                "structure": {"layout_owner_id": "row"},
            },
        ],
    }

    def transport(request, timeout):
        payload = json.loads(request.data)
        schema_name = payload["response_format"]["json_schema"]["name"]
        crops = (
            [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [],
                }
            ]
            if schema_name == "page_crops"
            else [
                {
                    "region_id": "crop-2",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "Option A",
                            "group_label": "Choices",
                            "state": "selected",
                            "bbox_2d": option_box,
                        }
                    ],
                }
            ]
        )
        return (
            200,
            {},
            json.dumps(
                {
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {"content": json.dumps({"crops": crops})},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                }
            ).encode(),
        )

    result = refine_page(image, page, "base output", transport=transport)
    unresolved = [
        region
        for region in result["refined_regions"]
        if (region.get("text_provenance") or {}).get("method")
        == "qwen_form_option_extraction"
    ]

    assert result["status"] == "partial"
    assert result["form_controls_added"] == 0
    assert result["form_controls_unresolved"] == 1
    assert result["accepted_control_crop_ids"] == []
    assert unresolved[0]["resolution"] == "unreadable"
    assert unresolved[0]["structure"]["label_evidence_ids"] == []
    assert "[x] Option A" not in result["content"]["markdown"]
    assert result["content"]["markdown"].count("Option A") == 1


def test_owned_form_rows_keep_side_by_side_columns():
    form = {
        "id": "form",
        "kind": "figure",
        "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
        "structure": {"role": "form_source"},
    }
    rows = [
        {
            "id": side,
            "kind": "layout_block",
            "text": "□ First □ Second",
            "reading_order": index,
            "bounding_box": {
                "left": left,
                "top": 10,
                "right": right,
                "bottom": 40,
            },
            "structure": {
                "role": "layout_block",
                "block_type": "form_row",
                "child_evidence_ids": [f"{side}-1", f"{side}-2"],
            },
        }
        for index, (side, left, right) in enumerate(
            (("left", 0, 95), ("right", 105, 200)), start=1
        )
    ]

    assert [row["id"] for row in _owned_form_rows([form, *rows])] == [
        "left",
        "right",
    ]


def test_form_source_without_structured_controls_stays_unresolved(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "form.png"
    Image.new("RGB", (100, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 100,
        "height": 100,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 100, "bottom": 100},
                "structure": {"role": "form_source"},
            },
            {
                "id": "box",
                "kind": "checkbox",
                "text": "Yes",
                "resolution": "resolved",
                "bounding_box": {"left": 10, "top": 10, "right": 40, "bottom": 20},
                "structure": {"state": "unselected", "label": "Yes"},
            },
        ],
    }

    def transport(request, timeout):
        payload = json.loads(request.data)
        content = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [],
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
                            "message": {"content": json.dumps(content)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "cost": 0.0,
                    },
                }
            ).encode(),
        )

    result = refine_page(image, page, "base output", transport=transport)

    assert result["status"] == "invalid"
    assert result["form_controls_added"] == 0
    assert result["accepted_control_crop_ids"] == []
    assert result["refined_regions"] is None
    assert "no structured form controls" in result["validation_errors"][0]


def test_checkbox_label_refinement_preserves_control_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "checkbox.png"
    Image.new("RGB", (200, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "reply",
                "kind": "checkbox",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 20, "top": 20, "right": 180, "bottom": 80},
                "structure": {"label": "", "state": "unselected"},
            }
        ],
    }

    def transport(request, timeout):
        payload = json.loads(request.data)
        content = {"crops": [{"region_id": "crop-1", "markdown": "☐ Please Reply"}]}
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
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                }
            ).encode(),
        )

    result = refine_page(image, page, "", transport=transport)
    refined = result["refined_regions"][0]

    assert refined["kind"] == "checkbox"
    assert refined["text"] == "Please Reply"
    assert refined["structure"] == {
        "label": "Please Reply",
        "state": "unselected",
    }
    assert "[ ] Please Reply" in result["content"]["markdown"]


def test_qwen_text_refinement_does_not_render_as_a_control():
    region = {
        "id": "note",
        "kind": "handwriting",
        "text": "Recovered note",
        "reading_order": 1,
        "resolution": "resolved",
        "text_provenance": {"method": "qwen_crop_refinement"},
    }

    assert render_page_markdown([region], ["note"]) == "Recovered note"


@pytest.mark.parametrize("block_type", ["form_row", "spatial_row"])
@pytest.mark.parametrize(
    ("state", "marker"),
    [("selected", "[x]"), ("unselected", "[ ]")],
)
def test_refined_control_replaces_its_stale_owned_label(block_type, state, marker):
    row = {
        "id": "row",
        "kind": "layout_block",
        "text": "Old label",
        "reading_order": 1,
        "resolution": "resolved",
        "structure": {
            "role": "layout_block",
            "block_type": block_type,
            "fields": [{"evidence_ids": ["label"]}],
        },
    }
    label = {
        "id": "label",
        "kind": "text",
        "text": "Old label",
        "reading_order": 2,
        "resolution": "resolved",
        "bounding_box": {"left": 20, "top": 20, "right": 100, "bottom": 40},
        "structure": {"layout_owner_id": "row"},
    }
    control = {
        "id": "control",
        "kind": "checkbox",
        "text": "Corrected label",
        "reading_order": 3,
        "resolution": "resolved",
        "bounding_box": {"left": 5, "top": 20, "right": 15, "bottom": 40},
        "text_provenance": {"method": "qwen_crop_refinement"},
        "structure": {
            "state": state,
            "label": "Corrected label",
            "label_evidence_ids": ["label"],
        },
    }

    assert (
        render_page_markdown([row, label, control], ["row", "label"])
        == f"{marker} Corrected label"
    )


def test_invalid_falcon_table_does_not_discard_valid_recognition(tmp_path, monkeypatch):
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
                "structure": {
                    "role": "table",
                    "row_count": 1,
                    "column_count": 2,
                    "cells": [
                        {
                            "id": "cell-1",
                            "row_nums": [0],
                            "column_nums": [0],
                            "text": "original table",
                        }
                    ],
                },
            },
        ],
    }
    original = copy.deepcopy(page)

    def transport(request, timeout):
        payload = json.loads(request.data)
        assert "crop-1" in json.dumps(payload)
        assert payload["messages"][0]["role"] == "system"
        content = {
            "crops": [
                {"region_id": "crop-1", "markdown": "dose 5 mg"},
                {
                    "region_id": "crop-2",
                    "markdown": "[table cells extracted]",
                    "table_cells": [{"cell_id": "r1c1", "text": "wrong grid"}],
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

    result = refine_page(image, page, "base output", transport=transport)
    assert result["status"] == "partial"
    assert "dose 5 mg" in result["content"]["markdown"]
    assert "[?] dose 5 mg" not in result["content"]["markdown"]
    assert "wrong grid" not in result["content"]["markdown"]
    assert result["accepted_crop_ids"] == ["crop-1"]
    assert result["rejected_crop_ids"] == ["crop-2"]
    assert result["validation_errors"]
    assert result["cost"] == 0.001
    assert "crop-2" in result["table_structure_errors"]
    assert "crop-2" in result["invalid_table_grids"]
    assert result["table_grids"] == {}
    assert any(
        "crop-2 canonical table structure is invalid" in error
        for error in result["validation_errors"]
    )
    assert result["canonical_unchanged"] is True
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
        for index, value in enumerate(("", "old B", "old C", "old D"))
    ]
    cells[0].update(
        {
            "source": "fixture-layout",
            "confidence": 0.42,
            "confidence_kind": "decoder_token",
            "decision": "source_reader",
            "supporters": ["layout-proposal-1"],
            "text_provenance": {"method": "source_table_cell"},
        }
    )
    cells[1]["source"] = "falcon"
    cells[1]["text_provenance"] = {"method": "falcon_table_cell"}
    page = {
        "page_number": 1,
        "width": 200,
        "height": 200,
        "regions": [
            {
                "id": "table",
                "kind": "table",
                "text": "old B old C old D",
                "provider": "fixture-layout",
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

    def transport(request, timeout):
        payload = json.loads(request.data)
        prompt = payload["messages"][1]["content"][0]["text"]
        assert "source reader's canonical table topology" in prompt
        assert "Falcon" not in prompt
        assert "TableFormer" not in prompt
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[table cells extracted]",
                    "table_cells": [
                        {"cell_id": "r1c1", "text": ""},
                        {"cell_id": "r1c2", "text": "new B"},
                        {"cell_id": "r2c1", "text": "new C"},
                        {"cell_id": "r2c2", "text": "new D"},
                    ],
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

    result = refine_page(image, page, "original", transport=transport)
    assert result["status"] == "generated"
    assert result["accepted_crop_ids"] == ["crop-1"]
    assert result["table_structure_source"] == "canonical_table_topology"
    assert result["table_grids"]["crop-1"]["topology_provenance"] == {
        "method": "canonical_table_topology",
        "provider": "fixture-layout",
    }
    assert "new A" not in result["content"]["markdown"]
    assert "[table cells extracted]" not in result["content"]["markdown"]
    refined = result["refined_regions"][0]
    assert refined["kind"] == "table"
    assert [cell["text"] for cell in refined["structure"]["cells"]] == [
        "",
        "new B",
        "new C",
        "new D",
    ]
    assert all(
        "confidence" not in cell
        and "confidence_kind" not in cell
        and "decision" not in cell
        and cell["evidence_ids"] == [f"falcon-word-{index}"]
        for cell in refined["structure"]["cells"][1:]
        for index in [int(cell["id"].split("-")[-1])]
    )
    fallback = refined["structure"]["cells"][0]
    assert all(fallback[key] == value for key, value in cells[0].items())
    assert fallback["cell_id"] == "r1c1"
    assert refined["resolution"] == "unreadable"
    assert refined["alternatives"][0]["decision_state"] == "superseded"
    assert refined["structure"]["header_row_count"] == 0
    assert [cell["cell_id"] for cell in refined["structure"]["cells"]] == [
        "r1c1",
        "r1c2",
        "r2c1",
        "r2c2",
    ]
    assert (
        refined["text_provenance"]["table_cell_alignment"]
        == "canonical_row_column_topology"
    )
    assert refined["text_provenance"]["topology_provider"] == "fixture-layout"
    assert refined["structure"]["cells"][0]["source"] == "fixture-layout"
    assert refined["structure"]["cells"][0]["text_provenance"] == {
        "method": "source_table_cell"
    }
    assert refined["structure"]["cells"][1]["alternatives"][-1] == {
        "text": "old B",
        "confidence": None,
        "provider": "falcon",
        "text_provenance": {"method": "falcon_table_cell"},
        "decision_state": "superseded",
    }
    assert refined["structure"]["cells"][2]["alternatives"][-1]["provider"] == (
        "fixture-layout"
    )
    assert refined["structure"]["cells"][1]["text_provenance"]["source_provenance"] == {
        "method": "falcon_table_cell"
    }
    assert (
        refined["structure"]["cells"][1]["text_provenance"]["table_cell_alignment"]
        == "canonical_row_column_topology"
    )
    assert (
        refined["structure"]["cells"][1]["text_provenance"]["topology_provider"]
        == "fixture-layout"
    )
    assert refined["structure"]["cells"][0]["bounding_box"] == {
        "left": 0,
        "top": 0,
        "right": 100,
        "bottom": 100,
    }
    assert (
        json.loads(json.dumps(result["refined_regions"])) == result["refined_regions"]
    )
    assert page["regions"][0]["structure"]["cells"] == cells

    class Classifier:
        name = "must-not-run"
        calls = 0

        def classify(self, *args):
            self.calls += 1
            return TableContinuationPrediction(1.0, {})

    unresolved_table = TextRegion(
        id="first-table",
        kind="table",
        text=refined["text"],
        confidence=None,
        bounding_box=BoundingBox(0, 0, 200, 200),
        reading_order=1,
        provider="fixture-layout",
        resolution=refined["resolution"],
        structure=refined["structure"],
    )
    resolved_table = copy.deepcopy(unresolved_table)
    resolved_table.id = "second-table"
    resolved_table.resolution = "resolved"
    pages = [
        PageResult(
            page_number=page_number,
            width=200,
            height=200,
            reader="fixture-layout",
            route="fixture",
            text=EvidenceText("", []),
            regions=[table],
        )
        for page_number, table in enumerate((unresolved_table, resolved_table), start=1)
    ]
    classifier = Classifier()
    assert CrossPageTableStage(classifier).apply([image, image], pages) == []
    assert classifier.calls == 0

    incomplete_cells = [
        {"cell_id": "r1c1", "text": "new A"},
        {"cell_id": "r1c2", "text": "new B"},
        {"cell_id": "r2c1", "text": "new C"},
    ]

    def incomplete_transport(request, timeout):
        payload = json.loads(request.data)
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[table cells extracted]",
                    "table_cells": incomplete_cells,
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
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode(),
        )

    rejected = refine_page(
        image,
        page,
        "original",
        transport=incomplete_transport,
    )
    assert rejected["status"] == "invalid"
    assert rejected["content"]["markdown"] == "original"
    assert "missing IDs r2c2" in rejected["validation_errors"][0]

    incomplete_cells[:] = [
        {"cell_id": "r1c1", "text": "first"},
        {"cell_id": "r1c1", "text": "second"},
        {"cell_id": "r1c2", "text": "new B"},
        {"cell_id": "r2c1", "text": "new C"},
        {"cell_id": "r2c2", "text": "new D"},
    ]
    duplicate = refine_page(
        image,
        page,
        "original",
        transport=incomplete_transport,
    )
    assert duplicate["status"] == "invalid"
    assert duplicate["content"]["markdown"] == "original"
    assert "duplicate table cell ID r1c1" in duplicate["validation_errors"][0]


def test_table_cell_ids_preserve_merged_spans(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "merged-table.png"
    Image.new("RGB", (200, 200), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 200,
        "regions": [
            {
                "id": "table",
                "kind": "table",
                "text": "old table",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 200},
                "structure": {
                    "role": "table",
                    "row_count": 2,
                    "column_count": 2,
                    "header_row_count": 0,
                    "cells": [
                        {
                            "id": "merged-heading",
                            "row_nums": [0],
                            "column_nums": [0, 1],
                            "text": "old heading",
                            "column_header": True,
                        },
                        {
                            "id": "left",
                            "row_nums": [1],
                            "column_nums": [0],
                            "text": "old left",
                        },
                        {
                            "id": "right",
                            "row_nums": [1],
                            "column_nums": [1],
                            "text": "old right",
                        },
                    ],
                },
            }
        ],
    }

    def transport(request, timeout):
        payload = json.loads(request.data)
        prompt = payload["messages"][1]["content"][0]["text"]
        assert '"cell_id": "r1c1"' in prompt
        assert '"columns": [0, 1]' in prompt
        answer = {
            "crops": [
                {
                    "region_id": "crop-1",
                    "markdown": "[table cells extracted]",
                    "table_cells": [
                        {"cell_id": "r1c1", "text": "Merged heading"},
                        {"cell_id": "r2c1", "text": "Left"},
                        {"cell_id": "r2c2", "text": "Right"},
                    ],
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
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode(),
        )

    result = refine_page(image, page, "original", transport=transport)

    assert result["status"] == "generated"
    structure = result["refined_regions"][0]["structure"]
    assert structure["row_count"] == 2
    assert structure["column_count"] == 2
    assert structure["cells"][0]["column_nums"] == [0, 1]
    assert 'colspan="2"' in result["content"]["markdown"]
    assert result["content"]["markdown"].count("Merged heading") == 1


def _answering(crops):
    def transport(request, timeout):
        payload = json.loads(request.data)
        body = {
            "model": payload["model"],
            "choices": [
                {
                    "message": {"content": json.dumps({"crops": crops})},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001},
        }
        return 200, {}, json.dumps(body).encode()

    return transport


def test_refilled_grid_of_a_truncated_table_stays_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "table.png"
    Image.new("RGB", (200, 100), "white").save(image)
    cells = [
        {
            "id": f"cell-{index}",
            "row_nums": [0],
            "column_nums": [index],
            "text": f"old {index}",
            "bounding_box": {
                "left": 100 * index,
                "top": 0,
                "right": 100 * index + 100,
                "bottom": 100,
            },
            "resolution": "resolved",
        }
        for index in range(2)
    ]
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "table",
                "kind": "table",
                "text": "old 0 old 1",
                "provider": "falcon-perception",
                # The decoder spent its budget: rows after these are missing.
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
                "structure": {
                    "role": "table",
                    "row_count": 1,
                    "column_count": 2,
                    "cells": cells,
                },
            }
        ],
    }
    transport = _answering(
        [
            {
                "region_id": "crop-1",
                "markdown": "[table cells extracted]",
                "table_cells": [
                    {"cell_id": "r1c1", "text": "new 0"},
                    {"cell_id": "r1c2", "text": "new 1"},
                ],
            }
        ]
    )

    refined = refine_page(image, page, "original", transport=transport)[
        "refined_regions"
    ][0]

    assert [cell["text"] for cell in refined["structure"]["cells"]] == [
        "new 0",
        "new 1",
    ]
    assert refined["resolution"] == "unreadable"


def test_refined_title_keeps_its_kind_and_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "title.png"
    Image.new("RGB", (200, 60), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 60,
        "regions": [
            {
                "id": "title",
                "kind": "title",
                "text": "Discharge Sumary",
                "provider": "falcon-perception",
                "resolution": "unreadable",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 60},
                "structure": {"presentation_rank": 1, "layout_owner_id": "block-1"},
            }
        ],
    }
    transport = _answering([{"region_id": "crop-1", "markdown": "Discharge Summary"}])

    result = refine_page(image, page, "original", transport=transport)
    refined = result["refined_regions"][0]

    assert refined["kind"] == "title"
    assert refined["text"] == "Discharge Summary"
    assert refined["structure"] == {
        "presentation_rank": 1,
        "layout_owner_id": "block-1",
    }


def test_a_form_without_boxes_or_marks_is_not_read_for_controls(tmp_path, monkeypatch):
    """Qwen returned each filled field of a plain header form as a checked control."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    image = tmp_path / "form.png"
    Image.new("RGB", (200, 100), "white").save(image)
    page = {
        "page_number": 1,
        "width": 200,
        "height": 100,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "resolved",
                "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
                "structure": {"role": "form_source"},
            },
            {
                "id": "field",
                "kind": "text",
                "text": "Phone Number 573-339-6130",
                "confidence": 0.95,
                "resolution": "resolved",
                "bounding_box": {"left": 10, "top": 10, "right": 190, "bottom": 40},
            },
        ],
    }

    def transport(request, timeout):
        raise AssertionError("no crop needs Qwen")

    result = refine_page(image, page, "base output", transport=transport)

    assert result.get("form_controls_added", 0) == 0


def test_qwen_marks_a_pen_ticked_box_the_detector_calls_unselected():
    from ocr_pipeline.markdown_refinement import _form_control_regions

    form = {
        "id": "form",
        "kind": "figure",
        "bounding_box": {"left": 0, "top": 0, "right": 200, "bottom": 100},
        "structure": {"role": "form_source"},
    }

    def detected(state):
        return {
            "id": "box",
            "kind": "checkbox",
            "text": "Hypertension",
            "bounding_box": {"left": 0, "top": 0, "right": 100, "bottom": 50},
            "text_provenance": {"method": "learned_checkbox_detection"},
            "structure": {"state": state, "label": "Hypertension"},
        }

    crop = {
        "region_id": "crop-1",
        "box": dict(form["bounding_box"]),
        "source_ids": ["form"],
    }

    def merge(heron_state, qwen_state):
        control = {
            "label": "Hypertension",
            "state": qwen_state,
            "bbox_2d": [0, 0, 500, 500],
        }
        records = [{"region_id": "crop-1", "controls": [control]}]
        box = detected(heron_state)
        return _form_control_regions(
            [crop], records, {"form": form, "box": box}, [form, box], "qwen", 0
        )

    assert merge("unselected", "selected") == ([], [], {"box"})
    # A detected mark is never cleared by the reader.
    controls, errors, marked = merge("selected", "unselected")
    assert (controls, marked) == ([], set())
    assert errors == ["crop-1 disagrees with an existing physical control"]


def test_a_glyph_sized_crop_is_upscaled_without_padding():
    import base64
    import io

    from ocr_pipeline.markdown_refinement import QWEN_MIN_SIDE, _image_content

    content = _image_content(Image.new("RGB", (8, 12), "black"))
    data = content["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(data))) as sent:
        assert min(sent.size) >= QWEN_MIN_SIDE
        assert sent.width * 12 == sent.height * 8
        assert sent.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_a_reader_option_off_the_printed_box_is_the_same_control():
    from ocr_pipeline.markdown_refinement import _same_form_control

    printed = {
        "text": "Please Reply",
        "structure": {"label": "Please Reply"},
        "bounding_box": {"left": 1980, "top": 2556, "right": 2519, "bottom": 2730},
    }
    read = {
        "label": "Please Reply",
        "bounding_box": {"left": 1877, "top": 2535, "right": 2102, "bottom": 2712},
    }
    other = {"label": "Please Comment", "bounding_box": read["bounding_box"]}

    assert _same_form_control(read, printed)
    assert not _same_form_control(other, printed)


@pytest.mark.parametrize("angle", [0, 90])
def test_thin_form_control_maps_sent_pixels_back_to_source(
    tmp_path, monkeypatch, angle
):
    import base64
    import io

    from PIL import ImageDraw, ImageOps

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    source = Image.new("RGB", (220, 40), "white")
    ImageDraw.Draw(source).rectangle((20, 12, 99, 17), fill="black")
    path = tmp_path / "thin-form.png"
    source.save(path)
    page = {
        "page_number": 1,
        "width": 220,
        "height": 40,
        "regions": [
            {
                "id": "form",
                "kind": "figure",
                "text": "",
                "resolution": "unreadable",
                "bounding_box": {"left": 10, "top": 10, "right": 210, "bottom": 22},
                "structure": {"role": "form_source"},
            },
            {
                "id": "known",
                "kind": "checkbox",
                "text": "Known option",
                "resolution": "resolved",
                "bounding_box": {"left": 180, "top": 10, "right": 190, "bottom": 22},
                "structure": {"state": "unselected"},
            },
        ],
    }
    original = copy.deepcopy(page)

    def transport(request, timeout):
        body = json.loads(request.data)
        images = [
            part["image_url"]["url"]
            for part in body["messages"][1]["content"]
            if part["type"] == "image_url"
        ]
        assert len(images) == 1
        with Image.open(
            io.BytesIO(base64.b64decode(images[0].split(",", 1)[1]))
        ) as sent:
            # Locate the actual synthetic mark in the submitted image, as the reader does.
            box = ImageOps.invert(sent.convert("RGB")).getbbox()
            normalized = [
                round(value * 1000 / side)
                for value, side in zip(box, (sent.width, sent.height) * 2)
            ]
        return _answering(
            [
                {
                    "region_id": "crop-1",
                    "markdown": "[form options extracted]",
                    "controls": [
                        {
                            "label": "New option",
                            "group_label": "",
                            "state": "selected",
                            "bbox_2d": normalized,
                        }
                    ],
                }
            ]
        )(request, timeout)

    result = refine_page(
        path, page, "original", rotation_degrees=angle, transport=transport
    )
    assert result["status"] == "generated"
    control = next(r for r in result["refined_regions"] if r["text"] == "New option")
    assert control["bounding_box"] == {
        "left": 20,
        "top": 12,
        "right": 100,
        "bottom": 18,
    }
    assert control["structure"]["state"] == "selected"
    assert page == original
