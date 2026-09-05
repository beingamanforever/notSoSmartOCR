from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, EvidenceText, PageResult, TextRegion
from ocr_pipeline.evidence_layout import EvidenceLayoutStage
from ocr_pipeline.falcon_presentation import FalconPresentationReader
from ocr_pipeline.providers import ReaderError


class RecordingFalcon:
    name = "falcon-ocr"
    generation = {"category": "plain", "max_new_tokens": 100}
    provenance = {"id": "tiiuae/Falcon-OCR", "identity_verified": True}

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[tuple[int, int]], list[str]]] = []

    def transcribe_crops(
        self, images: list[Image.Image], categories: list[str]
    ) -> list[str]:
        self.calls.append(([image.size for image in images], list(categories)))
        if self.outputs is not None:
            return self.outputs
        return [f"raw {category}" for category in categories]


def _region(
    identifier: str,
    kind: str,
    box: BoundingBox,
    order: int,
    *,
    structure: dict[str, object] | None = None,
    resolution: Literal["resolved", "unreadable", "conflicting"] = "resolved",
) -> TextRegion:
    return TextRegion(
        id=identifier,
        kind=kind,
        text=f"canonical {identifier}",
        confidence=0.8,
        bounding_box=box,
        reading_order=order,
        provider="canonical",
        resolution=resolution,
        structure=structure,
    )


def _table_structure() -> dict[str, object]:
    return {
        "role": "table",
        "row_count": 1,
        "column_count": 1,
        "cells": [{"row_nums": [0], "column_nums": [0], "text": "value"}],
    }


def _segmented_table_structure(
    *, spanning_rows: tuple[int, ...] | None = None
) -> dict[str, object]:
    cells = []
    for row, (top, bottom) in enumerate(((10, 30), (35, 55), (80, 100), (105, 125))):
        if spanning_rows and row in spanning_rows:
            continue
        for column, (left, right) in enumerate(((20, 90), (100, 180))):
            cells.append(
                {
                    "id": f"cell-{row}-{column}",
                    "bbox": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    },
                    "row_nums": [row],
                    "column_nums": [column],
                    "text": f"{row},{column}",
                    "evidence_ids": [f"evidence-{row}-{column}"],
                }
            )
    if spanning_rows:
        cells.extend(
            {
                "id": f"span-{column}",
                "bbox": {
                    "left": left,
                    "top": 10,
                    "right": right,
                    "bottom": 55,
                },
                "row_nums": list(spanning_rows),
                "column_nums": [column],
                "text": "span",
                "evidence_ids": [f"span-evidence-{column}"],
            }
            for column, (left, right) in enumerate(((20, 90), (100, 180)))
        )
    return {
        "role": "table",
        "row_count": 4,
        "column_count": 2,
        "cells": cells,
    }


def test_routes_only_unresolved_generic_text(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    model = RecordingFalcon(["recovered unresolved text"])
    page = _page(
        [
            _region("text", "text", BoundingBox(5, 5, 50, 20), 1),
            _region(
                "unresolved-text",
                "text",
                BoundingBox(5, 22, 55, 36),
                2,
                resolution="unreadable",
            ),
            TextRegion(
                id="empty-table",
                kind="table",
                text="",
                confidence=0.8,
                bounding_box=BoundingBox(5, 25, 90, 70),
                reading_order=3,
                provider="table-transformer",
                structure={
                    "role": "table",
                    "row_count": 0,
                    "column_count": 0,
                    "cells": [],
                },
            ),
        ]
    )

    output = FalconPresentationReader(model).read_page(image_path, page)

    assert [region.text for region in output] == ["recovered unresolved text"]
    assert model.calls == [([(50, 14)], ["text"])]


def _page(regions: list[TextRegion]) -> PageResult:
    return PageResult(
        page_number=1,
        width=200,
        height=160,
        reader="canonical",
        route="review",
        text=EvidenceText("canonical page", [region.id for region in regions]),
        regions=regions,
    )


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (200, 160), "white").save(path)
    return path


def test_batches_supported_canonical_regions_once_with_exact_categories(
    tmp_path: Path,
) -> None:
    kinds = [
        ("text", None),
        ("table", None),
        ("formula", "formula"),
        ("equation", "formula"),
        ("title", "title"),
        ("heading", "section-header"),
        ("caption", "caption"),
        ("footnote", "footnote"),
        ("list_item", "list-item"),
        ("page_header", "page-header"),
        ("page_footer", "page-footer"),
        ("handwriting", None),
        ("paragraph", None),
        ("text_block", None),
    ]
    regions = [
        _region(
            f"r{index}",
            kind,
            BoundingBox(index, index, index + 10, index + 8),
            index + 20,
            structure=_table_structure() if kind == "table" else None,
        )
        for index, (kind, _) in enumerate(kinds)
    ]
    page = _page(regions)
    before = copy.deepcopy(page)
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon).read_page(_image(tmp_path), page)  # type: ignore[arg-type]

    expected_categories = [category for _, category in kinds if category is not None]
    assert falcon.calls == [([(10, 8)] * len(expected_categories), expected_categories)]
    assert [region.text for region in output] == [
        f"raw {category}" for category in expected_categories
    ]
    assert [region.bounding_box for region in output] == [
        region.bounding_box
        for region, (_, category) in zip(regions, kinds, strict=True)
        if category is not None
    ]
    assert [region.reading_order for region in output] == [
        region.reading_order
        for region, (_, category) in zip(regions, kinds, strict=True)
        if category is not None
    ]
    assert page == before


def test_uses_structural_semantics_but_skips_words_controls_risks_and_sources(
    tmp_path: Path,
) -> None:
    valid = _region(
        "explicit-block",
        "text",
        BoundingBox(0, 0, 30, 20),
        3,
        structure={"semantic_class": "text block"},
    )
    skipped = [
        _region("word", "word", BoundingBox(0, 21, 30, 30), 4),
        _region("risk", "coverage_risk", BoundingBox(0, 41, 30, 50), 6),
        _region("control", "control", BoundingBox(0, 51, 30, 60), 7),
        _region("checkbox", "checkbox", BoundingBox(0, 61, 30, 70), 8),
        _region(
            "cell",
            "text",
            BoundingBox(0, 71, 30, 80),
            9,
            structure={"role": "table_cell", "semantic_class": "text_block"},
        ),
        _region(
            "source",
            "table",
            BoundingBox(0, 81, 30, 90),
            10,
            structure={"role": "table-source"},
        ),
        _region(
            "candidate",
            "table",
            BoundingBox(0, 91, 30, 100),
            11,
            structure={"role": "table candidate"},
        ),
        _region(
            "handwriting-candidate",
            "handwriting",
            BoundingBox(0, 101, 30, 110),
            12,
            structure={"role": "handwriting_candidate"},
        ),
    ]
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon).read_page(
        _image(tmp_path), _page([valid, *skipped])
    )

    assert output == []
    assert falcon.calls == []


def test_structural_special_category_overrides_generic_text_kind(
    tmp_path: Path,
) -> None:
    formula = _region(
        "formula",
        "text",
        BoundingBox(10, 10, 100, 40),
        1,
        structure={"semantic_class": "formula"},
    )
    falcon = RecordingFalcon([r"x^2 + y^2"])

    [output] = FalconPresentationReader(falcon).read_page(
        _image(tmp_path), _page([formula])
    )

    assert falcon.calls == [([(90, 30)], ["formula"])]
    assert output.text_provenance["category"] == "formula"  # type: ignore[index]


def test_crops_layout_blocks_and_never_their_owned_word_sources(
    tmp_path: Path,
) -> None:
    word = _region("word", "text", BoundingBox(10, 10, 40, 20), 1)
    word.text_provenance = {"merge_level": "word"}
    word.structure = {"layout_owner_id": "block"}
    block = _region(
        "block",
        "layout_block",
        BoundingBox(10, 10, 100, 40),
        1,
        structure={
            "role": "layout_block",
            "block_type": "paragraph",
            "child_evidence_ids": ["word"],
        },
        resolution="conflicting",
    )
    falcon = RecordingFalcon(["model transcription"])

    [result] = FalconPresentationReader(falcon).read_page(
        _image(tmp_path), _page([word, block])
    )

    assert falcon.calls == [([(90, 30)], ["text"])]
    assert result.text_provenance["source_region_id"] == "block"  # type: ignore[index]
    assert result.text_provenance["source_evidence_ids"] == ["word"]  # type: ignore[index]


def test_never_routes_handwriting_or_its_layout_owner_as_generic_text(
    tmp_path: Path,
) -> None:
    handwriting = _region(
        "handwriting",
        "handwriting",
        BoundingBox(10, 10, 80, 30),
        1,
        structure={"role": "handwriting_candidate"},
        resolution="unreadable",
    )
    flagged_text = _region(
        "flagged-text",
        "text",
        BoundingBox(10, 35, 80, 55),
        2,
        structure={"is_handwritten": True},
        resolution="conflicting",
    )
    owner = _region(
        "owner",
        "layout_block",
        BoundingBox(10, 10, 100, 60),
        1,
        structure={
            "role": "layout_block",
            "block_type": "paragraph",
            "child_evidence_ids": ["handwriting", "flagged-text"],
        },
        resolution="unreadable",
    )
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon).read_page(
        _image(tmp_path), _page([handwriting, flagged_text, owner])
    )

    assert output == []
    assert falcon.calls == []


def test_routes_only_table_with_unresolved_cell_evidence(tmp_path: Path) -> None:
    resolved = _region(
        "resolved-table",
        "table",
        BoundingBox(10, 10, 90, 60),
        1,
        structure=_table_structure(),
    )
    unresolved_structure = _table_structure()
    cells = unresolved_structure["cells"]
    assert isinstance(cells, list)
    cells[0]["resolution"] = "unreadable"
    unresolved = _region(
        "unresolved-table",
        "table",
        BoundingBox(100, 10, 190, 60),
        2,
        structure=unresolved_structure,
    )
    falcon = RecordingFalcon(["<table><tr><td>value</td></tr></table>"])

    [output] = FalconPresentationReader(falcon).read_page(
        _image(tmp_path), _page([resolved, unresolved])
    )

    assert falcon.calls == [([(90, 50)], ["table"])]
    assert output.text_provenance["source_region_id"] == "unresolved-table"  # type: ignore[index]


def test_skips_canonical_form_rows_before_falcon_generation(tmp_path: Path) -> None:
    form_row = _region(
        "form-row",
        "layout_block",
        BoundingBox(10, 10, 190, 40),
        1,
        structure={
            "role": "layout_block",
            "block_type": "form_row",
            "child_evidence_ids": ["label", "value"],
        },
    )
    paragraph = _region(
        "paragraph",
        "layout_block",
        BoundingBox(10, 50, 190, 80),
        2,
        structure={"role": "layout_block", "block_type": "paragraph"},
    )
    page = _page([form_row, paragraph])
    before = copy.deepcopy(page)
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon).read_page(_image(tmp_path), page)

    assert output == []
    assert falcon.calls == []
    assert page == before


def test_local_ruled_form_divider_preserves_rows_and_skips_falcon(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "ruled-form.png"
    image = Image.new("RGB", (600, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 580, 140), outline="black", width=2)
    draw.line((300, 20, 300, 140), fill="black", width=2)
    draw.line((20, 80, 580, 80), fill="black", width=2)
    image.save(image_path)
    rows = [
        _region("patient-label", "text", BoundingBox(40, 40, 100, 54), 1),
        _region("patient", "text", BoundingBox(108, 40, 260, 54), 2),
        _region("date-label", "text", BoundingBox(340, 40, 380, 54), 3),
        _region("date", "text", BoundingBox(388, 40, 520, 54), 4),
        _region("plan-label", "text", BoundingBox(40, 100, 85, 114), 5),
        _region("plan", "text", BoundingBox(93, 100, 260, 114), 6),
        _region("id-label", "text", BoundingBox(340, 100, 365, 114), 7),
        _region("id", "text", BoundingBox(373, 100, 430, 114), 8),
    ]
    for region, text in zip(
        rows,
        ("Patient:", "Lee", "Date:", "03/18/2025", "Plan:", "Anthem", "ID:", "1234"),
        strict=True,
    ):
        region.text = text
        region.text_provenance = {"merge_level": "word"}

    laid_out = EvidenceLayoutStage().apply(image_path, 1, rows)
    blocks = [region for region in laid_out if region.kind == "layout_block"]

    assert [block.structure["block_type"] for block in blocks] == [  # type: ignore[index]
        "form_row",
        "form_row",
    ]
    assert [block.text for block in blocks] == [
        "Patient: Lee | Date: 03/18/2025",
        "Plan: Anthem | ID: 1234",
    ]
    assert [block.structure["fields"] for block in blocks] == [  # type: ignore[index]
        [
            {"evidence_ids": ["patient-label", "patient"]},
            {"evidence_ids": ["date-label", "date"]},
        ],
        [
            {"evidence_ids": ["plan-label", "plan"]},
            {"evidence_ids": ["id-label", "id"]},
        ],
    ]
    assert all(
        boundary["orientation"] != "vertical"
        for block in blocks
        for boundary in (block.structure or {}).get("pre_cut_boundaries", [])
    )

    page = PageResult(
        page_number=1,
        width=600,
        height=180,
        reader="canonical",
        route="review",
        text=EvidenceText("canonical page", [region.id for region in laid_out]),
        regions=laid_out,
    )
    falcon = RecordingFalcon()

    assert FalconPresentationReader(falcon).read_page(image_path, page) == []
    assert falcon.calls == []


def test_deduplicates_exact_regions_skips_invalid_boxes_and_caps_batch(
    tmp_path: Path,
) -> None:
    duplicate = _region("same", "title", BoundingBox(1, 1, 21, 11), 1)
    regions = [
        duplicate,
        copy.deepcopy(duplicate),
        _region("outside", "title", BoundingBox(0, 0, 201, 10), 2),
        _region("inverted", "title", BoundingBox(20, 10, 10, 20), 3),
        _region("second", "caption", BoundingBox(30, 20, 60, 40), 4),
        _region("capped", "footnote", BoundingBox(70, 20, 100, 40), 5),
    ]
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon, max_crops=2).read_page(
        _image(tmp_path), _page(regions)
    )

    assert falcon.calls == [([(20, 10), (30, 20)], ["title", "caption"])]
    assert [region.id for region in output] == [
        "same-falcon-presentation",
        "second-falcon-presentation",
    ]


def test_preserves_raw_response_and_records_review_only_provenance(
    tmp_path: Path,
) -> None:
    raw = "  <table><tr><td>☑ Exact</td></tr></table>\n"
    source = _region(
        "table",
        "table",
        BoundingBox(10, 20, 90, 70),
        8,
        structure=_table_structure(),
        resolution="conflicting",
    )
    falcon = RecordingFalcon([raw])

    [result] = FalconPresentationReader(falcon).read_page(
        _image(tmp_path), _page([source])
    )

    assert result.text == raw
    assert result.provider == "falcon-presentation"
    assert result.confidence is None
    assert result.structure == {
        "role": "presentation_challenger",
        "review_only": True,
        "category": "table",
        "source_region_id": "table",
        "source_kind": "table",
        "control_glyphs_authoritative": False,
        "output_validation": {
            "raw_response_preserved": True,
            "nonempty": True,
            "exact_repetition_loop": False,
            "termination_observable": False,
            "truncation_observable": False,
        },
    }
    assert result.text_provenance == {
        "method": "falcon_core_category_crop_generation",
        "review_only": True,
        "source_region_id": "table",
        "source_kind": "table",
        "category": "table",
        "model": falcon.provenance,
        "generation": {
            "category": "table",
            "max_new_tokens": 100,
            "effective_max_new_tokens": 100,
        },
        "raw_response": raw,
        "output_validation": result.structure["output_validation"],
    }


def test_segments_tall_table_at_rows_with_full_width_and_provenance(
    tmp_path: Path,
) -> None:
    source = _region(
        "table",
        "table",
        BoundingBox(10, 5, 190, 150),
        8,
        structure=_segmented_table_structure(),
        resolution="unreadable",
    )
    page = _page([source])
    before = copy.deepcopy(page)
    falcon = RecordingFalcon(["first raw", "second raw"])

    output = FalconPresentationReader(falcon, max_table_segment_height=50).read_page(
        _image(tmp_path), page
    )

    assert falcon.calls == [([(180, 45), (180, 45)], ["table", "table"])]
    assert [region.id for region in output] == [
        "table-falcon-presentation-segment-0",
        "table-falcon-presentation-segment-1",
    ]
    assert [region.bounding_box for region in output] == [
        BoundingBox(10, 10, 190, 55),
        BoundingBox(10, 80, 190, 125),
    ]
    expected_segments = [
        {
            "index": 0,
            "row_start": 0,
            "row_end": 2,
            "source_region_id": "table",
            "source_cell_ids": ["cell-0-0", "cell-0-1", "cell-1-0", "cell-1-1"],
            "source_evidence_ids": [
                "evidence-0-0",
                "evidence-0-1",
                "evidence-1-0",
                "evidence-1-1",
            ],
        },
        {
            "index": 1,
            "row_start": 2,
            "row_end": 4,
            "source_region_id": "table",
            "source_cell_ids": ["cell-2-0", "cell-2-1", "cell-3-0", "cell-3-1"],
            "source_evidence_ids": [
                "evidence-2-0",
                "evidence-2-1",
                "evidence-3-0",
                "evidence-3-1",
            ],
        },
    ]
    assert [region.structure["table_segment"] for region in output] == expected_segments
    assert [
        region.text_provenance["table_segment"] for region in output
    ] == expected_segments
    assert [region.text_provenance["raw_response"] for region in output] == [
        "first raw",
        "second raw",
    ]
    assert page == before


def test_row_span_is_never_split_and_indivisible_oversized_table_is_skipped(
    tmp_path: Path,
) -> None:
    source = _region(
        "table",
        "table",
        BoundingBox(10, 5, 190, 150),
        8,
        structure=_segmented_table_structure(spanning_rows=(0, 1)),
        resolution="unreadable",
    )
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon, max_table_segment_height=40).read_page(
        _image(tmp_path), _page([source])
    )

    assert output == []
    assert falcon.calls == []


def test_row_span_stays_whole_when_later_legal_boundary_can_split(
    tmp_path: Path,
) -> None:
    source = _region(
        "table",
        "table",
        BoundingBox(10, 5, 190, 150),
        8,
        structure=_segmented_table_structure(spanning_rows=(0, 1)),
        resolution="unreadable",
    )
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon, max_table_segment_height=50).read_page(
        _image(tmp_path), _page([source])
    )

    assert falcon.calls == [([(180, 45), (180, 45)], ["table", "table"])]
    assert [region.structure["table_segment"]["row_start"] for region in output] == [
        0,
        2,
    ]
    assert [region.structure["table_segment"]["row_end"] for region in output] == [
        2,
        4,
    ]


def test_table_segments_are_atomic_under_crop_limit(tmp_path: Path) -> None:
    regions = [
        _region("title", "title", BoundingBox(0, 0, 9, 4), 1),
        _region(
            "table",
            "table",
            BoundingBox(10, 5, 190, 150),
            2,
            structure=_segmented_table_structure(),
            resolution="unreadable",
        ),
        _region("caption", "caption", BoundingBox(0, 151, 20, 159), 3),
    ]
    falcon = RecordingFalcon()

    output = FalconPresentationReader(
        falcon, max_crops=2, max_table_segment_height=50
    ).read_page(_image(tmp_path), _page(regions))

    assert falcon.calls == [([(9, 4), (20, 8)], ["title", "caption"])]
    assert [region.id for region in output] == [
        "title-falcon-presentation",
        "caption-falcon-presentation",
    ]


def test_invalid_canonical_table_topology_is_skipped(tmp_path: Path) -> None:
    structure = _segmented_table_structure()
    structure["cells"][1]["row_nums"] = [0, 1]
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon).read_page(
        _image(tmp_path),
        _page(
            [
                _region(
                    "table",
                    "table",
                    BoundingBox(10, 5, 190, 150),
                    1,
                    structure=structure,
                )
            ]
        ),
    )

    assert output == []
    assert falcon.calls == []


def test_empty_selection_does_not_call_falcon(tmp_path: Path) -> None:
    falcon = RecordingFalcon()

    output = FalconPresentationReader(falcon).read_page(
        _image(tmp_path),
        _page([_region("word", "word", BoundingBox(0, 0, 10, 10), 1)]),
    )

    assert output == []
    assert falcon.calls == []


def test_complete_canonical_page_skips_image_decode_and_falcon(tmp_path: Path) -> None:
    falcon = RecordingFalcon()
    missing_image = tmp_path / "missing.png"
    page = _page(
        [
            _region("text", "text", BoundingBox(0, 0, 10, 10), 1),
            _region(
                "table",
                "table",
                BoundingBox(0, 10, 20, 20),
                2,
                structure=_table_structure(),
            ),
        ]
    )

    output = FalconPresentationReader(falcon).read_page(missing_image, page)

    assert output == []
    assert falcon.calls == []


@pytest.mark.parametrize("max_crops", [0, -1])
def test_requires_positive_crop_limit(max_crops: int) -> None:
    with pytest.raises(ValueError, match="max_crops"):
        FalconPresentationReader(RecordingFalcon(), max_crops=max_crops)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_height", [0, -1])
def test_requires_positive_table_segment_height(max_height: int) -> None:
    with pytest.raises(ValueError, match="segment height"):
        FalconPresentationReader(  # type: ignore[arg-type]
            RecordingFalcon(), max_table_segment_height=max_height
        )


def test_invalid_image_fails_cleanly(tmp_path: Path) -> None:
    image_path = tmp_path / "bad.png"
    image_path.write_text("not an image")

    with pytest.raises(ReaderError) as raised:
        FalconPresentationReader(RecordingFalcon()).read_page(  # type: ignore[arg-type]
            image_path,
            _page([_region("title", "title", BoundingBox(0, 0, 10, 10), 1)]),
        )

    assert raised.value.code == "falcon_presentation_image_failed"


@pytest.mark.parametrize("outputs", [[], [""], [42]])
def test_invalid_batch_output_fails_without_mutating_page(
    tmp_path: Path, outputs: list[object]
) -> None:
    page = _page([_region("title", "title", BoundingBox(0, 0, 10, 10), 1)])
    before = copy.deepcopy(page)
    falcon = RecordingFalcon()
    falcon.outputs = outputs  # type: ignore[assignment]

    with pytest.raises(ReaderError) as raised:
        FalconPresentationReader(falcon).read_page(_image(tmp_path), page)  # type: ignore[arg-type]

    assert raised.value.code == "falcon_presentation_output_failed"
    assert page == before
