from __future__ import annotations

from pathlib import Path

from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.tables import (
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


def test_small_sparse_grid_remains_a_table(tmp_path: Path) -> None:
    image_path = _image(tmp_path, (100, 100))
    source = [_region("alpha", "Alpha", 1, (12, 12, 25, 25), 0.95)]
    extractor = GridExtractor(2, 2, BoundingBox(10, 10, 60, 60))

    output = TatrTableStage(extractor).apply(image_path, 1, source)

    assert any(region.kind == "table" for region in output)
    assert not any(region.kind == "table_candidate" for region in output)


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
