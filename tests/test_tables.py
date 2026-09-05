from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import ocr_pipeline.tables as table_module
from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.tables import (
    DEFAULT_DETECTION_ID,
    TableCell,
    TableChallenger,
    TablePrediction,
    TatrTableExtractor,
    TatrTableStage,
    sauvola_view,
)


def test_table_stage_builds_markdown_and_preserves_nested_evidence(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    regions = [
        _region("label", "Metric", 5, (10, 10, 45, 30), 0.98),
        _region("value", "42", 6, (55, 10, 90, 30), 0.96),
        _region("footer", "Footer", 20, (10, 80, 60, 95), 0.99),
    ]
    stage = TatrTableStage(FixedExtractor())

    output = stage.apply(image_path, 1, regions)

    table = next(region for region in output if region.kind == "table")
    assert table.text == "| Metric | 42 |\n| --- | --- |"
    assert table.bounding_box == BoundingBox(5, 5, 95, 40)
    assert table.reading_order == 5
    assert table.structure == {
        "role": "table",
        "row_count": 1,
        "column_count": 2,
        "cells": [
            {
                "id": "p1-tables-table-1-cell-1",
                "bbox": {"left": 5, "top": 5, "right": 50, "bottom": 40},
                "row_nums": [0],
                "column_nums": [0],
                "text": "Metric",
                "source": "base",
                "confidence": 0.98,
                "resolution": "resolved",
                "alternatives": [],
                "evidence_ids": ["label"],
                "supporters": [{"provider": "base", "evidence_ids": ["label"]}],
                "column_header": True,
                "projected_row_header": False,
                "span_bboxes": [],
                "decision": "primary",
            },
            {
                "id": "p1-tables-table-1-cell-2",
                "bbox": {"left": 50, "top": 5, "right": 95, "bottom": 40},
                "row_nums": [0],
                "column_nums": [1],
                "text": "42",
                "source": "base",
                "confidence": 0.96,
                "resolution": "resolved",
                "alternatives": [],
                "evidence_ids": ["value"],
                "supporters": [{"provider": "base", "evidence_ids": ["value"]}],
                "column_header": True,
                "projected_row_header": False,
                "span_bboxes": [],
                "decision": "primary",
            },
        ],
        "model": {"id": "fake", "origin": "test"},
        "detection_confidence": 0.99,
    }
    assert regions[0].structure == {
        "role": "table_source",
        "parent_id": table.id,
    }
    assert regions[1].structure == {
        "role": "table_source",
        "parent_id": table.id,
    }
    assert regions[2].structure is None


def test_agreed_challengers_replace_low_confidence_primary(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    primary = [_region("value", "4Z", 1, (55, 10, 90, 30), 0.55)]
    stage = TatrTableStage(
        OneCellExtractor(),
        challengers=[
            TableChallenger(
                "tesseract_raw",
                FixedReader([_region("raw", "42", 1, (10, 10, 45, 30), 0.91)]),
            ),
            TableChallenger(
                "sauvola",
                FixedReader([_region("enhanced", "42", 1, (10, 10, 45, 30), 0.96)]),
            ),
        ],
    )

    output = stage.apply(image_path, 1, primary)

    table = next(region for region in output if region.kind == "table")
    cell = table.structure["cells"][0]
    assert cell["text"] == "42"
    assert cell["source"] == "sauvola"
    assert cell["decision"] == "low_primary_confidence"
    assert cell["resolution"] == "resolved"
    assert cell["alternatives"][0]["text"] == "4Z"
    assert cell["evidence_ids"] == [
        "p1-tables-tesseract-raw-t1-source-1",
        "p1-tables-sauvola-t1-source-1",
    ]
    assert cell["supporters"] == [
        {
            "provider": "tesseract_raw",
            "evidence_ids": ["p1-tables-tesseract-raw-t1-source-1"],
        },
        {
            "provider": "sauvola",
            "evidence_ids": ["p1-tables-sauvola-t1-source-1"],
        },
    ]
    sources = [
        region
        for region in output
        if region.structure and region.structure.get("role") == "table_source"
    ]
    assert {region.id for region in sources} == {
        "value",
        "p1-tables-tesseract-raw-t1-source-1",
        "p1-tables-sauvola-t1-source-1",
    }


def test_parallel_challengers_preserve_ordered_table_evidence(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    barrier = threading.Barrier(2)
    primary = [_region("value", "4Z", 1, (55, 10, 90, 30), 0.55)]
    stage = TatrTableStage(
        OneCellExtractor(),
        challengers=[
            TableChallenger(
                "raw",
                BarrierReader(
                    [_region("raw", "42", 1, (10, 10, 45, 30), 0.91)],
                    barrier,
                ),
            ),
            TableChallenger(
                "enhanced",
                BarrierReader(
                    [_region("enhanced", "42", 1, (10, 10, 45, 30), 0.96)],
                    barrier,
                ),
            ),
        ],
        parallel_challengers=True,
    )

    output = stage.apply(image_path, 1, primary)

    table = next(region for region in output if region.kind == "table")
    cell = table.structure["cells"][0]
    assert cell["text"] == "42"
    assert cell["source"] == "enhanced"
    assert [item["provider"] for item in cell["supporters"]] == [
        "raw",
        "enhanced",
    ]


def test_parallel_challengers_reuse_one_crop_per_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _image(tmp_path)
    crop_calls: list[BoundingBox] = []
    prepare_calls: list[tuple[int, int]] = []
    original_crop = table_module._table_crop

    def record_crop(
        image: Image.Image,
        box: BoundingBox,
        padding: tuple[int, int],
    ) -> tuple[Image.Image, tuple[int, int]]:
        crop_calls.append(box)
        return original_crop(image, box, padding)

    def prepare(image: Image.Image) -> Image.Image:
        prepare_calls.append(image.size)
        return image.convert("L")

    monkeypatch.setattr(table_module, "_table_crop", record_crop)
    primary = [
        _region("left", "4Z", 1, (10, 10, 30, 25), 0.55),
        _region("right", "4Z", 2, (70, 10, 90, 25), 0.55),
    ]
    raw = FixedReader([_region("raw", "42", 1, (10, 10, 30, 25), 0.91)])
    enhanced = FixedReader([_region("enhanced", "42", 1, (10, 10, 30, 25), 0.96)])

    output = TatrTableStage(
        TwoTableExtractor(),
        challengers=[
            TableChallenger("raw", raw),
            TableChallenger("enhanced", enhanced, prepare=prepare),
        ],
        parallel_challengers=True,
    ).apply(image_path, 1, primary)

    tables = [region for region in output if region.kind == "table"]
    assert crop_calls == [
        BoundingBox(5, 5, 45, 40),
        BoundingBox(60, 5, 100, 40),
    ]
    assert raw.image_sizes == [(50, 45), (50, 45)]
    assert enhanced.image_sizes == [(50, 45), (50, 45)]
    assert prepare_calls == [(50, 45), (50, 45)]
    assert [
        [
            supporter["provider"]
            for supporter in table.structure["cells"][0]["supporters"]
        ]
        for table in tables
    ] == [["raw", "enhanced"], ["raw", "enhanced"]]


def test_strong_disagreement_is_explicit_and_routes_review(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    stage = TatrTableStage(
        OneCellExtractor(),
        challengers=[
            TableChallenger(
                "raw",
                FixedReader([_region("raw", "43", 1, (10, 10, 45, 30), 0.99)]),
            ),
            TableChallenger(
                "enhanced",
                FixedReader([_region("enh", "43", 1, (10, 10, 45, 30), 0.99)]),
            ),
        ],
    )

    result = process_document(
        image_path,
        FixedReader([_region("base", "42", 1, (55, 10, 90, 30), 0.99)]),
        stages=[stage],
    )

    table = next(region for region in result.pages[0].regions if region.kind == "table")
    cell = table.structure["cells"][0]
    assert table.resolution == "resolved"
    assert cell["resolution"] == "conflicting"
    assert cell["decision"] == "strong_disagreement"
    assert cell["text"] == "42"
    assert [item["text"] for item in cell["alternatives"]] == ["43"]
    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "| 42 |\n| --- |"
    assert result.pages[0].text.evidence_ids == [table.id]


def test_agreed_challengers_restore_only_supported_currency_wrapper(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    raw = [
        _region("raw-currency", "$", 1, (52, 10, 57, 30), 0.37),
        _region("raw-noise", "=.", 2, (58, 10, 64, 30), 0.34),
        _region("raw-value", "7,500,542", 3, (65, 10, 92, 30), 0.79),
    ]
    enhanced = [
        _region("enh-currency", "$", 1, (52, 10, 57, 30), 0.37),
        _region("enh-noise", "=.", 2, (58, 10, 64, 30), 0.34),
        _region("enh-value", "7,500,542", 3, (65, 10, 92, 30), 0.79),
    ]
    stage = TatrTableStage(
        OneCellExtractor(),
        challengers=[
            TableChallenger("raw", FixedReader(raw), scope="page"),
            TableChallenger("enhanced", FixedReader(enhanced), scope="page"),
        ],
    )

    output = stage.apply(
        image_path,
        1,
        [_region("primary", "7,500,542", 1, (65, 10, 92, 30), 0.94)],
    )

    table = next(region for region in output if region.kind == "table")
    cell = table.structure["cells"][0]
    assert cell["text"] == "$ 7,500,542"
    assert cell["decision"] == "supported_wrapper"
    assert cell["resolution"] == "resolved"
    assert [item["provider"] for item in cell["supporters"]] == [
        "base",
        "raw",
        "enhanced",
    ]
    assert table.text == "| $ 7,500,542 |\n| --- |"


def test_currency_or_sign_disagreement_is_not_silently_equivalent(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    stage = TatrTableStage(
        OneCellExtractor(),
        challengers=[
            TableChallenger(
                "raw",
                FixedReader([_region("raw", "$ (123)", 1, (52, 10, 92, 30), 0.99)]),
                scope="page",
            ),
            TableChallenger(
                "enhanced",
                FixedReader([_region("enh", "$ (123)", 1, (52, 10, 92, 30), 0.99)]),
                scope="page",
            ),
        ],
    )

    output = stage.apply(
        image_path,
        1,
        [_region("primary", "123", 1, (55, 10, 90, 30), 0.99)],
    )

    table = next(region for region in output if region.kind == "table")
    cell = table.structure["cells"][0]
    assert cell["text"] == "$ (123)"
    assert cell["decision"] == "supported_wrapper"
    assert cell["resolution"] == "resolved"


def test_span_grid_avoids_overlapping_union_box_row_errors(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    middle = _region("middle", "middle", 1, (12, 24, 35, 36), 0.9)

    output = TatrTableStage(OverlapExtractor()).apply(image_path, 1, [middle])

    table = next(region for region in output if region.kind == "table")
    assert [cell["text"] for cell in table.structure["cells"]] == [
        "",
        "middle",
        "",
    ]


def test_each_source_region_is_assigned_to_only_one_cell(tmp_path: Path) -> None:
    image_path = _image(tmp_path)
    wide = _region("wide", "once", 1, (45, 10, 55, 30), 0.9)

    output = TatrTableStage(FixedExtractor()).apply(image_path, 1, [wide])

    table = next(region for region in output if region.kind == "table")
    assert [cell["text"] for cell in table.structure["cells"]] == ["once", ""]


def test_broad_source_region_is_not_forced_into_one_table_cell(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    broad = _region("broad", "left right", 1, (5, 10, 95, 30), 0.9)

    output = TatrTableStage(FixedExtractor()).apply(image_path, 1, [broad])

    table = next(region for region in output if region.kind == "table")
    assert [cell["text"] for cell in table.structure["cells"]] == ["", ""]
    assert all(cell["resolution"] == "unreadable" for cell in table.structure["cells"])
    assert all(
        cell["decision"] == "no_cell_evidence" for cell in table.structure["cells"]
    )
    assert broad.structure is None
    assert broad in output


def test_table_crops_translate_challengers_without_duplicate_ids(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path)
    challenger = FixedReader([_region("local", "seen", 1, (10, 10, 30, 25), 0.95)])
    regions = [
        _region("left", "left", 1, (10, 10, 30, 25), 0.95),
        _region("right", "right", 2, (70, 10, 90, 25), 0.95),
    ]

    output = TatrTableStage(
        TwoTableExtractor(),
        challengers=[TableChallenger("crop", challenger)],
    ).apply(image_path, 1, regions)

    evidence = [region for region in output if region.provider == "crop"]
    assert [region.id for region in evidence] == [
        "p1-tables-crop-t1-source-1",
        "p1-tables-crop-t2-source-1",
    ]
    assert [region.bounding_box for region in evidence] == [
        BoundingBox(10, 10, 30, 25),
        BoundingBox(65, 10, 85, 25),
    ]
    assert [region.structure["parent_id"] for region in evidence] == [
        "p1-tables-table-1",
        "p1-tables-table-2",
    ]
    assert challenger.image_sizes == [(50, 45), (50, 45)]


def test_tatr_adapter_translates_crop_cells_to_page_coordinates(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path, (200, 120))
    pipeline = FakeTatrPipeline()
    extractor = TatrTableExtractor(
        tmp_path,
        tmp_path / "detection.pth",
        tmp_path / "structure.pth",
        device="cpu",
        pipeline=pipeline,
    )

    predictions = extractor.extract(
        image_path,
        [
            {
                "bbox": [60, 30, 80, 40],
                "text": "x",
                "span_num": 0,
                "line_num": 0,
                "block_num": 0,
            }
        ],
    )

    assert len(predictions) == 1
    assert predictions[0].bounding_box == BoundingBox(50, 20, 150, 100)
    assert predictions[0].cells == (
        TableCell(BoundingBox(55, 25, 100, 60), (0,), (0,), True, False),
        TableCell(BoundingBox(100, 25, 155, 60), (0,), (1,), True, False),
    )
    assert predictions[0].model["origin"] == "Microsoft"
    assert predictions[0].model["license"] == "MIT"
    assert pipeline.detect_tokens[0]["bbox"] == [60, 30, 80, 40]


def test_tatr_adapter_keeps_blank_section_gap_in_one_clinical_form(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path, (2500, 3000))
    row_tops = tuple(range(0, 1300, 130)) + tuple(range(1839, 2244, 101))
    source = [
        _region(
            f"cell-{row}-{column}",
            f"value {row} {column}",
            row * 5 + column + 1,
            (
                119 + column * 440,
                373 + top,
                519 + column * 440,
                423 + top,
            ),
            0.99,
        )
        for row, top in enumerate(row_tops)
        for column in range(5)
    ]
    extractor = TatrTableExtractor(
        tmp_path,
        tmp_path / "detection.pth",
        tmp_path / "structure.pth",
        device="cpu",
        crop_padding=0,
        pipeline=ClinicalFormTatrPipeline(row_tops),
    )

    result = process_document(
        image_path,
        FixedReader(source),
        stages=[TatrTableStage(extractor)],
    )

    tables = [region for region in result.pages[0].regions if region.kind == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.bounding_box == BoundingBox(104, 368, 2426, 2801)
    assert table.structure["row_count"] == 15
    assert table.structure["column_count"] == 5
    assert len(table.structure["cells"]) == 75
    assert "partition" not in table.structure["model"]
    source_ids = [
        evidence_id
        for cell in table.structure["cells"]
        for evidence_id in cell["evidence_ids"]
    ]
    assert source_ids == [region.id for region in source]
    page_sources = [
        region for region in result.pages[0].regions if region.id in source_ids
    ]
    assert {region.structure["parent_id"] for region in page_sources} == {table.id}
    assert result.pages[0].text.evidence_ids == [table.id]


def test_stricter_detection_score_keeps_object_crop_pairing(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (220, 120))
    pipeline = ScoreTatrPipeline()
    extractor = TatrTableExtractor(
        tmp_path,
        tmp_path / "detection.pth",
        tmp_path / "structure.pth",
        device="cpu",
        minimum_detection_score=0.8,
        pipeline=pipeline,
    )

    predictions = extractor.extract(image_path, [])

    assert [prediction.bounding_box for prediction in predictions] == [
        BoundingBox(110, 20, 200, 100)
    ]
    assert pipeline.recognized_sizes == [(100, 90)]


def test_nested_duplicate_table_detection_is_suppressed(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (220, 120))
    pipeline = NestedDuplicateTatrPipeline()
    extractor = TatrTableExtractor(
        tmp_path,
        tmp_path / "detection.pth",
        tmp_path / "structure.pth",
        device="cpu",
        pipeline=pipeline,
    )

    predictions = extractor.extract(image_path, [])

    assert [prediction.bounding_box for prediction in predictions] == [
        BoundingBox(10, 10, 210, 110)
    ]
    assert pipeline.recognized_sizes == [(210, 115)]


def test_ruled_proposal_is_recognized_when_tatr_detects_nothing(
    tmp_path: Path,
) -> None:
    image_path = _ruled_image(tmp_path, [((20, 20, 220, 140), 3, 2)])
    pipeline = RuledTatrPipeline(rows=3, columns=2)
    extractor = _ruled_extractor(tmp_path, pipeline)

    predictions = extractor.extract(image_path, [])

    assert len(predictions) == 1
    assert predictions[0].confidence is None
    assert predictions[0].model["proposal"] == {
        "source": "opencv_ruled_table",
        "detection_confidence_calibrated": False,
        "used_for_detection": True,
        "tatr_row_count": 3,
        "tatr_column_count": 2,
    }
    assert pipeline.recognized_sizes == [(200, 120)]


def test_broad_tatr_detection_is_replaced_by_separate_ruled_tables(
    tmp_path: Path,
) -> None:
    grids = [
        ((20, 20, 220, 90), 2, 2),
        ((20, 120, 220, 190), 2, 2),
    ]
    image_path = _ruled_image(tmp_path, grids)
    pipeline = RuledTatrPipeline(
        rows=2,
        columns=2,
        detection_box=(10, 10, 230, 200),
    )

    predictions = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])

    assert [prediction.bounding_box for prediction in predictions] == [
        BoundingBox(20, 20, 220, 90),
        BoundingBox(20, 120, 220, 190),
    ]
    assert pipeline.recognized_sizes == [(200, 70), (200, 70)]
    assert all(prediction.confidence is None for prediction in predictions)


def test_invalid_broad_detection_is_reparsed_as_horizontal_table_panels(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (240, 220), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 80, 140, 200):
        draw.line((20, y, 220, y), fill="black", width=2)
    image_path = tmp_path / "panels.png"
    image.save(image_path)
    pipeline = PanelTatrPipeline()

    predictions = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])

    assert [prediction.bounding_box for prediction in predictions] == [
        BoundingBox(20, 20, 220, 80),
        BoundingBox(20, 80, 220, 140),
        BoundingBox(20, 140, 220, 200),
    ]
    assert pipeline.recognized_sizes == [
        (200, 180),
        (200, 60),
        (200, 60),
        (200, 60),
    ]
    assert all(
        prediction.model["proposal"]["source"]
        == "opencv_horizontal_panel_decomposition"
        for prediction in predictions
    )
    assert all(
        prediction.model["proposal"]["parent_topology_conflicts"] == 1
        for prediction in predictions
    )


def test_horizontal_panel_recovery_rolls_back_when_a_child_is_invalid(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (240, 220), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 80, 140, 200):
        draw.line((20, y, 220, y), fill="black", width=2)
    image_path = tmp_path / "panels.png"
    image.save(image_path)

    class InvalidChildPipeline(PanelTatrPipeline):
        def recognize(
            self,
            image: Image.Image,
            tokens: list[dict[str, object]],
            **options: object,
        ) -> dict[str, object]:
            if image.height <= 70 and self.recognized_sizes.count(image.size) == 1:
                self.recognized_sizes.append(image.size)
                return {"cells": [[]]}
            return super().recognize(image, tokens, **options)

    predictions = _ruled_extractor(tmp_path, InvalidChildPipeline()).extract(
        image_path,
        [],
    )

    assert [prediction.bounding_box for prediction in predictions] == [
        BoundingBox(20, 20, 220, 200)
    ]


def test_horizontal_panel_recovery_ignores_separator_inside_detector_padding(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (240, 220), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 80, 140, 200):
        draw.line((20, y, 220, y), fill="black", width=2)

    panels = table_module._horizontal_table_panels(
        image,
        BoundingBox(14, 14, 226, 206),
    )

    assert panels == [
        BoundingBox(14, 14, 226, 80),
        BoundingBox(14, 80, 226, 140),
        BoundingBox(14, 140, 226, 200),
    ]


def test_matching_ruled_proposal_does_not_duplicate_tatr_detection(
    tmp_path: Path,
) -> None:
    box = (20, 20, 220, 140)
    image_path = _ruled_image(tmp_path, [(box, 2, 2)])
    pipeline = RuledTatrPipeline(rows=2, columns=2, detection_box=box)

    predictions = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])

    assert len(predictions) == 1
    assert predictions[0].confidence == 0.98
    assert predictions[0].model["proposal"]["used_for_detection"] is False
    assert pipeline.recognized_sizes == [(200, 120)]


def test_underlines_prose_and_page_border_do_not_create_ruled_table_proposals(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (260, 180), "white")
    draw = ImageDraw.Draw(image)
    for row, text in enumerate(("Patient name", "Date of birth", "Address")):
        top = 20 + row * 45
        draw.text((20, top), text, fill="black")
        draw.line((20, top + 20, 220, top + 20), fill="black", width=2)
    image_path = tmp_path / "underlines.png"
    image.save(image_path)
    pipeline = RuledTatrPipeline(rows=2, columns=2)

    predictions = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])

    assert predictions == []
    assert pipeline.recognized_sizes == []

    border = Image.new("RGB", (240, 210), "white")
    draw = ImageDraw.Draw(border)
    for y in (0, 105, 209):
        draw.line((0, y, 239, y), fill="black", width=2)
    for x in (0, 120, 239):
        draw.line((x, 0, x, 209), fill="black", width=2)
    border.save(image_path)

    assert _ruled_extractor(tmp_path, pipeline).extract(image_path, []) == []
    assert pipeline.recognized_sizes == []


def test_ruled_crops_keep_original_rgb_and_independent_token_transforms(
    tmp_path: Path,
) -> None:
    grids = [
        ((20, 20, 220, 90), 2, 2),
        ((20, 120, 220, 190), 2, 2),
    ]
    image_path = _ruled_image(tmp_path, grids, marker=(40, 40, (255, 0, 0)))
    tokens = [
        {"bbox": [30, 30, 50, 45], "text": "upper", "source_id": "u"},
        {"bbox": [30, 130, 50, 145], "text": "lower", "source_id": "l"},
    ]
    original_tokens = [dict(token, bbox=list(token["bbox"])) for token in tokens]
    pipeline = RuledTatrPipeline(rows=2, columns=2)

    _ruled_extractor(tmp_path, pipeline).extract(image_path, tokens)

    assert pipeline.first_pixels == [(255, 0, 0), (255, 255, 255)]
    assert pipeline.recognized_tokens == [
        [{"bbox": [10.0, 10.0, 30.0, 25.0], "text": "upper", "source_id": "u"}],
        [{"bbox": [10.0, 10.0, 30.0, 25.0], "text": "lower", "source_id": "l"}],
    ]
    assert pipeline.token_object_ids[0] != pipeline.token_object_ids[1]
    assert tokens == original_tokens


def test_clean_ruled_grid_repairs_a_missing_tatr_column(tmp_path: Path) -> None:
    box = (20, 20, 220, 180)
    image_path = _ruled_image(tmp_path, [(box, 8, 4)])
    pipeline = RuledTatrPipeline(rows=8, columns=3, detection_box=box)

    prediction = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])[0]

    assert len(prediction.cells) == 32
    assert max(cell.row_nums[0] for cell in prediction.cells) == 7
    assert max(cell.column_nums[0] for cell in prediction.cells) == 3
    assert prediction.confidence == 0.98
    assert prediction.bounding_box == BoundingBox(*box)
    assert prediction.model["detection"]["id"] == DEFAULT_DETECTION_ID
    assert prediction.model["proposal"]["tatr_row_count"] == 8
    assert prediction.model["proposal"]["tatr_column_count"] == 3
    assert prediction.model["proposal"]["used_for_detection"] is False
    assert prediction.model["proposal"]["grid_repair"] == "full_rule_cartesian"


def test_ruled_grid_keeps_distant_tatr_row_count(tmp_path: Path) -> None:
    box = (20, 20, 220, 180)
    image_path = _ruled_image(tmp_path, [(box, 6, 4)])
    pipeline = RuledTatrPipeline(rows=4, columns=4, detection_box=box)

    prediction = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])[0]

    assert len(prediction.cells) == 16
    assert "grid_repair" not in prediction.model["proposal"]
    assert prediction.model["proposal"]["tatr_row_count"] == 4
    assert prediction.model["proposal"]["tatr_column_count"] == 4


def test_spanning_tatr_grid_is_not_replaced_by_ruled_cartesian_cells(
    tmp_path: Path,
) -> None:
    image_path = _ruled_image(tmp_path, [((20, 20, 220, 180), 3, 3)])
    pipeline = RuledTatrPipeline(rows=3, columns=2, spanning=True)

    prediction = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])[0]

    assert len(prediction.cells) == 5
    assert prediction.cells[0].column_nums == (0, 1)
    assert "grid_repair" not in prediction.model["proposal"]


def test_empty_structure_is_not_accepted_for_ruled_proposal(tmp_path: Path) -> None:
    image_path = _ruled_image(tmp_path, [((20, 20, 220, 140), 2, 2)])
    pipeline = RuledTatrPipeline(rows=0, columns=0)

    predictions = _ruled_extractor(tmp_path, pipeline).extract(image_path, [])

    assert predictions == []


def test_near_page_two_by_two_keeps_primary_text_and_adds_diagnostic(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path, (100, 100))
    source = [
        _region("alpha", "Alpha", 1, (2, 2, 20, 20), 0.95),
        _region("beta", "Beta", 2, (52, 2, 70, 20), 0.94),
    ]

    result = process_document(
        image_path,
        FixedReader(source),
        stages=[TatrTableStage(GridExtractor(2, 2))],
    )

    page = result.pages[0]
    assert page.text.value == "Alpha Beta"
    assert page.text.evidence_ids == ["alpha", "beta"]
    assert page.route == "review"
    assert source[0].structure is None
    assert source[1].structure is None
    assert not any(region.kind == "table" for region in page.regions)
    diagnostic = next(
        region for region in page.regions if region.kind == "table_candidate"
    )
    assert diagnostic.text == ""
    assert diagnostic.resolution == "unreadable"
    assert diagnostic.text_provenance == {
        "method": "near_page_low_complexity_rejection",
        "source_region_ids": ["alpha", "beta"],
    }
    assert diagnostic.structure == {
        "role": "table_candidate",
        "status": "rejected",
        "reason": "near_page_low_complexity_grid",
        "row_count": 2,
        "column_count": 2,
        "grid_cells": 4,
        "predicted_cells": 4,
        "occupied_cells": 2,
        "cell_coverage": 0.5,
        "table_area_ratio": 1.0,
        "model": {"id": "grid", "origin": "test"},
        "detection_confidence": 0.99,
    }


def test_near_page_one_by_nine_keeps_primary_text(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (100, 100))
    source = [
        _region(
            f"cell-{column}",
            str(column),
            column + 1,
            (column * 11 + 1, 2, column * 11 + 10, 20),
            0.95,
        )
        for column in range(9)
    ]

    result = process_document(
        image_path,
        FixedReader(source),
        stages=[TatrTableStage(GridExtractor(1, 9))],
    )

    page = result.pages[0]
    assert page.text.value == "0 1 2 3 4 5 6 7 8"
    assert page.text.evidence_ids == [f"cell-{column}" for column in range(9)]
    assert not any(region.kind == "table" for region in page.regions)
    diagnostic = next(
        region for region in page.regions if region.kind == "table_candidate"
    )
    assert diagnostic.structure["row_count"] == 1
    assert diagnostic.structure["column_count"] == 9
    assert diagnostic.structure["grid_cells"] == 9
    assert diagnostic.structure["predicted_cells"] == 9


def test_near_page_dense_grid_remains_a_table(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (100, 100))
    source = [
        _region(
            f"cell-{row}-{column}",
            f"{row},{column}",
            row * 4 + column + 1,
            (column * 25 + 2, row * 25 + 2, column * 25 + 20, row * 25 + 20),
            0.95,
        )
        for row in range(4)
        for column in range(4)
    ]

    output = TatrTableStage(GridExtractor(4, 4)).apply(image_path, 1, source)

    assert any(region.kind == "table" for region in output)
    assert not any(region.kind == "table_candidate" for region in output)


def test_near_page_table_keeps_text_outside_recognized_cells(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (100, 100))
    source = [
        _region("upper", "Upper", 1, (5, 5, 30, 10), 0.95),
        _region("cell", "Cell", 2, (5, 35, 20, 45), 0.95),
        _region("lower", "Lower", 3, (5, 90, 30, 95), 0.95),
    ]

    result = process_document(
        image_path,
        FixedReader(source),
        stages=[TatrTableStage(BandGridExtractor())],
    )

    page = result.pages[0]
    table = next(region for region in page.regions if region.kind == "table")
    sources = {
        region.id
        for region in page.regions
        if (region.structure or {}).get("role") == "table_source"
    }
    assert table.structure["row_count"] == 4
    assert table.structure["column_count"] == 4
    assert table.structure["cells"][0]["text"] == "Cell"
    assert table.text_provenance["source_region_ids"] == ["cell"]
    assert sources == {"cell"}
    assert source[0].structure is None
    assert source[2].structure is None
    assert page.text.value == f"Upper {table.text} Lower"
    assert page.text.evidence_ids == ["upper", table.id, "lower"]


def test_author_and_abstract_layout_is_rejected_before_challengers(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path, (867, 1122))
    extractor = SpanningLayoutExtractor()
    cells = extractor.extract(image_path, [])[0].cells
    occupied_columns = {0: {0, 1, 2}, 1: {0, 1, 3}, 2: {0, 2}, 4: {0, 1, 3}}
    source = []
    for cell_index, cell in enumerate(cells):
        row = cell.row_nums[0]
        column = cell.column_nums[0]
        if len(cell.column_nums) == 1 and column not in occupied_columns.get(
            row, set()
        ):
            continue
        box = cell.bounding_box
        source.append(
            _region(
                f"source-{cell_index}",
                "prose",
                len(source) + 1,
                (box.left + 4, box.top + 4, box.right - 4, box.bottom - 4),
                0.95,
            )
        )
    challenger = FixedReader(
        [_region("challenger", "should not run", 1, (1, 1, 10, 10), 0.9)]
    )

    output = TatrTableStage(
        extractor,
        challengers=[TableChallenger("tesseract_raw", challenger)],
    ).apply(image_path, 1, source)

    assert challenger.image_sizes == []
    assert not any(region.kind == "table" for region in output)
    assert all(region.structure is None for region in source)
    diagnostic = next(region for region in output if region.kind == "table_candidate")
    assert diagnostic.text_provenance["method"] == "table_semantics_rejection"
    assert diagnostic.structure == {
        "role": "table_candidate",
        "status": "rejected",
        "reason": "unsupported_spanning_layout",
        "row_count": 6,
        "column_count": 4,
        "grid_cells": 24,
        "predicted_cells": 18,
        "occupied_cells": 19,
        "cell_coverage": 0.791667,
        "table_area_ratio": 0.241469,
        "model": {"id": "spanning-layout", "origin": "test"},
        "detection_confidence": 0.93598,
        "occupied_predicted_cells": 13,
        "complete_independent_rows": 0,
        "full_width_rows": 2,
    }


def test_single_column_prose_list_is_rejected(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (1000, 1000))
    extractor = GridExtractor(4, 1, BoundingBox(600, 100, 800, 500))
    cells = extractor.extract(image_path, [])[0].cells
    source = [
        _region(
            f"line-{row}",
            "prose line",
            row + 1,
            (
                cell.bounding_box.left + 5,
                cell.bounding_box.top + 5,
                cell.bounding_box.right - 5,
                cell.bounding_box.bottom - 5,
            ),
            0.95,
        )
        for row, cell in enumerate(cells)
    ]

    output = TatrTableStage(extractor).apply(image_path, 1, source)

    assert not any(region.kind == "table" for region in output)
    diagnostic = next(region for region in output if region.kind == "table_candidate")
    assert diagnostic.structure["reason"] == "single_axis_list"
    assert diagnostic.structure["row_count"] == 4
    assert diagnostic.structure["column_count"] == 1


def test_single_row_card_layout_is_rejected(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (900, 600))
    extractor = GridExtractor(1, 3, BoundingBox(60, 180, 840, 420))
    cells = extractor.extract(image_path, [])[0].cells
    source = [
        _region(
            f"card-{column}",
            f"Card {column}",
            column + 1,
            (
                cell.bounding_box.left + 8,
                cell.bounding_box.top + 8,
                cell.bounding_box.right - 8,
                cell.bounding_box.bottom - 8,
            ),
            0.95,
        )
        for column, cell in enumerate(cells)
    ]

    output = TatrTableStage(extractor).apply(image_path, 1, source)

    assert not any(region.kind == "table" for region in output)
    diagnostic = next(region for region in output if region.kind == "table_candidate")
    assert diagnostic.structure["reason"] == "single_axis_list"
    assert diagnostic.structure["row_count"] == 1
    assert diagnostic.structure["column_count"] == 3


def test_overlapping_logical_cells_are_rejected(tmp_path: Path) -> None:
    class OverlappingGridExtractor:
        name = "fake-tables"

        def extract(
            self,
            image_path: Path,
            tokens: list[dict[str, object]],
        ) -> list[TablePrediction]:
            return [
                TablePrediction(
                    BoundingBox(100, 100, 700, 500),
                    (
                        TableCell(BoundingBox(100, 100, 300, 250), (0,), (0,)),
                        TableCell(BoundingBox(180, 100, 500, 250), (0,), (1,)),
                        TableCell(BoundingBox(100, 250, 300, 400), (1,), (0,)),
                        TableCell(BoundingBox(300, 250, 500, 400), (1,), (1,)),
                    ),
                    0.99,
                    {"id": "overlapping-grid", "origin": "test"},
                )
            ]

    image_path = _image(tmp_path, (1000, 1000))
    source = [
        _region("left", "Left", 1, (120, 120, 260, 180), 0.95),
        _region("right", "Right", 2, (320, 120, 460, 180), 0.95),
    ]

    output = TatrTableStage(OverlappingGridExtractor()).apply(image_path, 1, source)

    assert not any(region.kind == "table" for region in output)
    diagnostic = next(region for region in output if region.kind == "table_candidate")
    assert diagnostic.structure["reason"] == "invalid_cell_topology"
    assert diagnostic.structure["topology_conflicts"] == 1


def test_dense_financial_grid_remains_a_table_with_exact_currency(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path, (3024, 1964))
    extractor = HeaderGridExtractor(17, 5, BoundingBox(336, 647, 2268, 1835))
    cells = extractor.extract(image_path, [])[0].cells
    exact_values = {
        (4, 4): "$ 7,500,542",
        (5, 4): "$ 3,058,647",
        (8, 4): "$ 1,941,102",
        (11, 4): "$ 8,371,855",
    }
    source = []
    for index, cell in enumerate(cells):
        row = cell.row_nums[0]
        column = cell.column_nums[0]
        if (row, column) == (0, 0):
            continue
        box = cell.bounding_box
        source.append(
            _region(
                f"cell-{row}-{column}",
                exact_values.get((row, column), f"value {row} {column}"),
                index + 1,
                (box.left + 4, box.top + 4, box.right - 4, box.bottom - 4),
                0.98,
            )
        )

    output = TatrTableStage(extractor).apply(image_path, 1, source)

    table = next(region for region in output if region.kind == "table")
    assert table.structure["row_count"] == 17
    assert table.structure["column_count"] == 5
    assert table.structure["cells"][0] == {
        "id": "p1-tables-table-1-cell-1",
        "bbox": {
            "left": 336,
            "top": 647,
            "right": 722,
            "bottom": 717,
        },
        "row_nums": [0],
        "column_nums": [0],
        "text": "",
        "source": "fake-tables",
        "confidence": None,
        "resolution": "resolved",
        "alternatives": [],
        "evidence_ids": [],
        "supporters": [],
        "column_header": True,
        "projected_row_header": False,
        "span_bboxes": [],
        "decision": "structural_blank_corner",
    }
    assert [table.structure["cells"][row * 5 + 4]["text"] for row in (4, 5, 8, 11)] == [
        "$ 7,500,542",
        "$ 3,058,647",
        "$ 1,941,102",
        "$ 8,371,855",
    ]
    assert not any(region.kind == "table_candidate" for region in output)


def test_blank_corner_stays_unreadable_when_row_label_evidence_is_missing(
    tmp_path: Path,
) -> None:
    image_path = _image(tmp_path, (1000, 1000))
    extractor = HeaderGridExtractor(3, 3, BoundingBox(100, 100, 700, 700))
    cells = extractor.extract(image_path, [])[0].cells
    source = []
    for index, cell in enumerate(cells):
        row = cell.row_nums[0]
        column = cell.column_nums[0]
        if (row, column) in {(0, 0), (2, 0)}:
            continue
        box = cell.bounding_box
        source.append(
            _region(
                f"cell-{row}-{column}",
                f"value {row} {column}",
                index + 1,
                (box.left + 4, box.top + 4, box.right - 4, box.bottom - 4),
                0.98,
            )
        )

    output = TatrTableStage(extractor).apply(image_path, 1, source)

    table = next(region for region in output if region.kind == "table")
    corner = table.structure["cells"][0]
    missing_row_label = table.structure["cells"][6]
    assert corner["resolution"] == "unreadable"
    assert corner["decision"] == "no_cell_evidence"
    assert missing_row_label["resolution"] == "unreadable"
    assert missing_row_label["decision"] == "no_cell_evidence"


def test_small_sparse_grid_remains_a_table(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (100, 100))
    source = [_region("alpha", "Alpha", 1, (12, 12, 25, 25), 0.95)]
    extractor = GridExtractor(2, 2, BoundingBox(10, 10, 60, 60))

    output = TatrTableStage(extractor).apply(image_path, 1, source)

    assert any(region.kind == "table" for region in output)
    assert not any(region.kind == "table_candidate" for region in output)


def test_empty_table_structure_is_rejected(tmp_path: Path) -> None:
    class EmptyTableExtractor:
        name = "fake-tables"

        def extract(
            self,
            image_path: Path,
            tokens: list[dict[str, object]],
        ) -> list[TablePrediction]:
            return [
                TablePrediction(
                    BoundingBox(10, 10, 90, 40),
                    (),
                    0.8,
                    {"id": "fake"},
                )
            ]

    image_path = _image(tmp_path, (100, 100))
    source = [_region("footer", "About News", 1, (15, 15, 85, 35), 0.95)]

    output = TatrTableStage(EmptyTableExtractor()).apply(image_path, 1, source)

    assert not any(region.kind == "table" for region in output)
    candidate = next(region for region in output if region.kind == "table_candidate")
    assert candidate.structure["reason"] == "empty_table_structure"
    assert candidate.resolution == "unreadable"
    assert [region.text for region in output if region.kind == "word"] == ["About News"]


def test_stage_reader_error_keeps_base_page(tmp_path: Path) -> None:
    image_path = _image(tmp_path)

    result = process_document(
        image_path,
        FixedReader([_region("base", "literal", 1, (10, 10, 30, 25), 0.9)]),
        stages=[TatrTableStage(FailingExtractor())],
    )

    assert [region.text for region in result.pages[0].regions] == ["literal"]
    assert result.pages[0].failure_ids == ["failure-1"]
    assert result.failures[0].stage == "tables"
    assert result.failures[0].code == "table_model_unavailable"


def test_sauvola_view_returns_a_binary_local_threshold() -> None:
    source = Image.new("L", (40, 20), 230)
    for x in range(12, 28):
        for y in range(7, 13):
            source.putpixel((x, y), 40)

    result = sauvola_view(source)

    assert result.mode == "L"
    assert set(result.getdata()) <= {0, 255}
    assert result.getpixel((20, 10)) == 0
    assert result.getpixel((2, 2)) == 255


class FixedExtractor:
    name = "fake-tables"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        return [
            TablePrediction(
                BoundingBox(5, 5, 95, 40),
                (
                    TableCell(BoundingBox(5, 5, 50, 40), (0,), (0,), True),
                    TableCell(BoundingBox(50, 5, 95, 40), (0,), (1,), True),
                ),
                0.99,
                {"id": "fake", "origin": "test"},
            )
        ]


class OneCellExtractor:
    name = "fake-tables"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        return [
            TablePrediction(
                BoundingBox(50, 5, 95, 40),
                (TableCell(BoundingBox(50, 5, 95, 40), (0,), (0,)),),
                0.99,
                {"id": "fake", "origin": "test"},
            )
        ]


class TwoTableExtractor:
    name = "fake-tables"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        return [
            TablePrediction(
                BoundingBox(5, 5, 45, 40),
                (TableCell(BoundingBox(5, 5, 45, 40), (0,), (0,)),),
                0.99,
                {"id": "fake"},
            ),
            TablePrediction(
                BoundingBox(60, 5, 100, 40),
                (TableCell(BoundingBox(60, 5, 100, 40), (0,), (0,)),),
                0.99,
                {"id": "fake"},
            ),
        ]


class OverlapExtractor:
    name = "fake-tables"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        return [
            TablePrediction(
                BoundingBox(0, 0, 100, 60),
                (
                    TableCell(
                        BoundingBox(0, 0, 100, 20),
                        (0,),
                        (0,),
                        span_boxes=(BoundingBox(10, 2, 30, 18),),
                    ),
                    TableCell(
                        BoundingBox(0, 0, 100, 60),
                        (1,),
                        (0,),
                        span_boxes=(BoundingBox(10, 22, 30, 38),),
                    ),
                    TableCell(
                        BoundingBox(0, 0, 100, 60),
                        (2,),
                        (0,),
                        span_boxes=(BoundingBox(10, 42, 30, 58),),
                    ),
                ),
                0.99,
                {"id": "fake"},
            )
        ]


class GridExtractor:
    name = "fake-tables"

    def __init__(
        self,
        rows: int,
        columns: int,
        box: BoundingBox = BoundingBox(0, 0, 100, 100),
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.box = box

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        width = self.box.right - self.box.left
        height = self.box.bottom - self.box.top
        return [
            TablePrediction(
                self.box,
                tuple(
                    TableCell(
                        BoundingBox(
                            round(self.box.left + column * width / self.columns),
                            round(self.box.top + row * height / self.rows),
                            round(self.box.left + (column + 1) * width / self.columns),
                            round(self.box.top + (row + 1) * height / self.rows),
                        ),
                        (row,),
                        (column,),
                    )
                    for row in range(self.rows)
                    for column in range(self.columns)
                ),
                0.99,
                {"id": "grid", "origin": "test"},
            )
        ]


class HeaderGridExtractor(GridExtractor):
    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        prediction = super().extract(image_path, tokens)[0]
        return [
            TablePrediction(
                prediction.bounding_box,
                tuple(
                    TableCell(
                        cell.bounding_box,
                        cell.row_nums,
                        cell.column_nums,
                        column_header=cell.row_nums == (0,),
                    )
                    for cell in prediction.cells
                ),
                prediction.confidence,
                prediction.model,
            )
        ]


class BandGridExtractor:
    name = "band-grid"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        cells = tuple(
            TableCell(
                BoundingBox(
                    column * 25, 30 + row * 10, (column + 1) * 25, 40 + row * 10
                ),
                (row,),
                (column,),
            )
            for row in range(4)
            for column in range(4)
        )
        return [
            TablePrediction(
                BoundingBox(0, 0, 100, 100),
                cells,
                0.99,
                {"id": "band-grid", "origin": "test"},
            )
        ]


class SpanningLayoutExtractor:
    name = "fake-tables"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        box = BoundingBox(161, 243, 706, 674)
        row_edges = [
            round(box.top + row * (box.bottom - box.top) / 6) for row in range(7)
        ]
        column_edges = [
            round(box.left + column * (box.right - box.left) / 4) for column in range(5)
        ]
        cells = []
        for row in range(6):
            if row in {3, 5}:
                cells.append(
                    TableCell(
                        BoundingBox(
                            box.left,
                            row_edges[row],
                            box.right,
                            row_edges[row + 1],
                        ),
                        (row,),
                        (0, 1, 2, 3),
                        projected_row_header=True,
                    )
                )
                continue
            cells.extend(
                TableCell(
                    BoundingBox(
                        column_edges[column],
                        row_edges[row],
                        column_edges[column + 1],
                        row_edges[row + 1],
                    ),
                    (row,),
                    (column,),
                    column_header=row == 0,
                )
                for column in range(4)
            )
        return [
            TablePrediction(
                box,
                tuple(cells),
                0.93598,
                {"id": "spanning-layout", "origin": "test"},
            )
        ]


class FailingExtractor:
    name = "unavailable"

    def extract(
        self, image_path: Path, tokens: list[dict[str, object]]
    ) -> list[TablePrediction]:
        raise ReaderError("table_model_unavailable", "weights are absent")


class FixedReader:
    name = "base"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions
        self.image_sizes: list[tuple[int, int]] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            self.image_sizes.append(image.size)
        return self.regions


class BarrierReader(FixedReader):
    def __init__(
        self,
        regions: list[TextRegion],
        barrier: threading.Barrier,
    ) -> None:
        super().__init__(regions)
        self.barrier = barrier

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.barrier.wait(timeout=5)
        return super().read(image_path, page_number)


class FakeTatrPipeline:
    det_class_thresholds = {"table": 0.5, "table rotated": 0.5}

    def __init__(self) -> None:
        self.detect_tokens: list[dict[str, object]] = []

    def detect(self, image: Image.Image, **options: object) -> dict[str, object]:
        self.detect_tokens = options["tokens"]
        return {
            "objects": [{"label": "table", "score": 0.98, "bbox": [50, 20, 150, 100]}],
            "crops": [
                {
                    "image": image.crop((45, 15, 155, 105)),
                    "tokens": self.detect_tokens,
                }
            ],
        }

    def recognize(self, image: Image.Image, *args: object, **kwargs: object) -> dict:
        return {
            "cells": [
                [
                    {
                        "bbox": [10, 10, 55, 45],
                        "row_nums": [0],
                        "column_nums": [0],
                        "column header": True,
                    },
                    {
                        "bbox": [55, 10, 110, 45],
                        "row_nums": [0],
                        "column_nums": [1],
                        "column header": True,
                    },
                ]
            ]
        }


class ClinicalFormTatrPipeline(FakeTatrPipeline):
    def __init__(self, row_tops: tuple[int, ...]) -> None:
        super().__init__()
        self.row_tops = row_tops

    def detect(self, image: Image.Image, **options: object) -> dict[str, object]:
        return {
            "objects": [
                {
                    "label": "table",
                    "score": 0.98,
                    "bbox": [104, 368, 2426, 2801],
                }
            ],
            "crops": [
                {
                    "image": image.crop((104, 368, 2426, 2801)),
                    "tokens": options["tokens"],
                }
            ],
        }

    def recognize(self, image: Image.Image, *args: object, **kwargs: object) -> dict:
        return {
            "cells": [
                [
                    {
                        "bbox": [left, top, left + 420, top + 60],
                        "row_nums": [row],
                        "column_nums": [column],
                    }
                    for row, top in enumerate(self.row_tops)
                    for column, left in enumerate(range(10, 2210, 440))
                ]
            ]
        }


class ScoreTatrPipeline(FakeTatrPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.recognized_sizes: list[tuple[int, int]] = []

    def detect(self, image: Image.Image, **options: object) -> dict[str, object]:
        return {
            "objects": [
                {"label": "table", "score": 0.6, "bbox": [10, 20, 100, 100]},
                {"label": "table", "score": 0.9, "bbox": [110, 20, 200, 100]},
            ],
            "crops": [
                {"image": image.crop((5, 15, 105, 105)), "tokens": []},
                {"image": image.crop((105, 15, 205, 105)), "tokens": []},
            ],
        }

    def recognize(self, image: Image.Image, *args: object, **kwargs: object) -> dict:
        self.recognized_sizes.append(image.size)
        return super().recognize(image, *args, **kwargs)


class NestedDuplicateTatrPipeline(ScoreTatrPipeline):
    def detect(self, image: Image.Image, **options: object) -> dict[str, object]:
        return {
            "objects": [
                {"label": "table", "score": 0.97, "bbox": [10, 10, 210, 110]},
                {"label": "table", "score": 0.65, "bbox": [10, 35, 210, 110]},
            ],
            "crops": [
                {"image": image.crop((5, 5, 215, 120)), "tokens": []},
                {"image": image.crop((5, 30, 215, 120)), "tokens": []},
            ],
        }


class RuledTatrPipeline:
    det_class_thresholds = {"table": 0.5, "table rotated": 0.5}

    def __init__(
        self,
        *,
        rows: int,
        columns: int,
        detection_box: tuple[int, int, int, int] | None = None,
        spanning: bool = False,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.detection_box = detection_box
        self.spanning = spanning
        self.recognized_sizes: list[tuple[int, int]] = []
        self.recognized_tokens: list[list[dict[str, object]]] = []
        self.token_object_ids: list[int] = []
        self.first_pixels: list[tuple[int, int, int]] = []

    def detect(self, image: Image.Image, **options: object) -> dict[str, object]:
        if self.detection_box is None:
            return {"objects": [], "crops": []}
        return {
            "objects": [
                {
                    "label": "table",
                    "score": 0.98,
                    "bbox": list(self.detection_box),
                }
            ],
            "crops": [
                {
                    "image": image.crop(self.detection_box),
                    "tokens": options["tokens"],
                }
            ],
        }

    def recognize(
        self,
        image: Image.Image,
        tokens: list[dict[str, object]],
        **options: object,
    ) -> dict[str, object]:
        self.recognized_sizes.append(image.size)
        self.recognized_tokens.append(
            [dict(token, bbox=list(token["bbox"])) for token in tokens]
        )
        self.token_object_ids.append(id(tokens[0]) if tokens else 0)
        self.first_pixels.append(image.getpixel((20, 20)))
        if self.rows == 0 or self.columns == 0:
            return {"cells": [[]]}
        width, height = image.size
        cells = []
        for row in range(self.rows):
            for column in range(self.columns):
                if self.spanning and row == 0:
                    if column == 0:
                        cells.append(
                            {
                                "bbox": [0, 0, width, height / self.rows],
                                "row_nums": [0],
                                "column_nums": list(range(self.columns)),
                            }
                        )
                    continue
                cells.append(
                    {
                        "bbox": [
                            column * width / self.columns,
                            row * height / self.rows,
                            (column + 1) * width / self.columns,
                            (row + 1) * height / self.rows,
                        ],
                        "row_nums": [row],
                        "column_nums": [column],
                    }
                )
        return {"cells": [cells]}


class PanelTatrPipeline(RuledTatrPipeline):
    def __init__(self) -> None:
        super().__init__(
            rows=2,
            columns=2,
            detection_box=(20, 20, 220, 200),
        )

    def recognize(
        self,
        image: Image.Image,
        tokens: list[dict[str, object]],
        **options: object,
    ) -> dict[str, object]:
        if image.height <= 70:
            return super().recognize(image, tokens, **options)
        self.recognized_sizes.append(image.size)
        return {
            "cells": [
                [
                    {
                        "bbox": [0, 0, image.width, image.height],
                        "row_nums": [0],
                        "column_nums": [0],
                    },
                    {
                        "bbox": [0, 0, image.width, image.height],
                        "row_nums": [1],
                        "column_nums": [1],
                    },
                ]
            ]
        }


def _ruled_extractor(tmp_path: Path, pipeline: object) -> TatrTableExtractor:
    return TatrTableExtractor(
        tmp_path,
        tmp_path / "detection.pth",
        tmp_path / "structure.pth",
        device="cpu",
        crop_padding=0,
        enable_ruled_table_proposals=True,
        pipeline=pipeline,
    )


def _ruled_image(
    tmp_path: Path,
    grids: list[tuple[tuple[int, int, int, int], int, int]],
    *,
    marker: tuple[int, int, tuple[int, int, int]] | None = None,
) -> Path:
    image = Image.new("RGB", (240, 210), "white")
    draw = ImageDraw.Draw(image)
    for (left, top, right, bottom), rows, columns in grids:
        for row in range(rows + 1):
            y = round(top + row * (bottom - top) / rows)
            draw.line((left, y, right, y), fill="black", width=2)
        for column in range(columns + 1):
            x = round(left + column * (right - left) / columns)
            draw.line((x, top, x, bottom), fill="black", width=2)
    if marker is not None:
        image.putpixel((marker[0], marker[1]), marker[2])
    path = tmp_path / "ruled.png"
    image.save(path)
    return path


def _region(
    region_id: str,
    text: str,
    order: int,
    box: tuple[int, int, int, int],
    confidence: float,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="word",
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(*box),
        reading_order=order,
        provider="base",
    )


def _image(tmp_path: Path, size: tuple[int, int] = (120, 100)) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", size, "white").save(path)
    return path
