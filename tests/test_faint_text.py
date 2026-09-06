from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.faint_text import FaintTinyTextStage
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import ReaderError


class FixedReader:
    name = "first-pass"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [_heading()]


class CropReader:
    name = "crop-reader"

    def __init__(self, *, confidence: float = 0.96) -> None:
        self.confidence = confidence
        self.calls: list[tuple[int, int]] = []

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            width, height = image.size
        self.calls.append((width, height))
        return [
            TextRegion(
                id="reread-1",
                kind="text",
                text="fax ref 2048",
                confidence=self.confidence,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
                text_provenance={"method": "controlled-reread"},
            )
        ]


def test_selective_high_resolution_recovers_pixels_missed_by_first_pass(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "faint-tiny.png")
    reader = CropReader()

    baseline = process_document(source, FixedReader())
    recovered = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(reader)],
    )

    assert baseline.pages[0].text.value == "DISCHARGE SUMMARY"
    assert recovered.pages[0].text.value == "DISCHARGE SUMMARY fax ref 2048"
    assert len(reader.calls) == 1
    assert reader.calls[0][0] < 500 * 3
    region = recovered.pages[0].regions[-1]
    assert region.text == "fax ref 2048"
    assert region.resolution == "resolved"
    assert region.provider == "crop-reader"
    assert region.bounding_box.left <= 315
    assert region.bounding_box.right >= 368
    provenance = region.text_provenance
    assert provenance["method"] == ("unowned_pixel_proposal_and_high_resolution_reread")
    assert provenance["page_number"] == 1
    assert provenance["view"] == "selective_crop_high_resolution"
    assert provenance["scale"] == 3
    assert provenance["reader"] == "crop-reader"
    assert provenance["proposal"]["component_count"] >= 2
    assert provenance["proposal"]["ink_pixels"] > 0
    assert provenance["source_region_id"] == "reread-1"
    assert provenance["source_provider"] == "crop-reader"
    assert provenance["source_text_provenance"]["method"] == "controlled-reread"
    assert provenance["source_text_provenance"]["faint_tiny_view"]["scale"] == 3


def test_late_faint_recovery_fills_the_containing_unreadable_table_cell(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "faint-table-value.png")
    crop_reader = CropReader()

    class BlankTableStage:
        name = "tables"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions + [
                TextRegion(
                    id="table-1",
                    kind="table",
                    text="|  |\n| --- |",
                    confidence=None,
                    bounding_box=BoundingBox(280, 160, 400, 210),
                    reading_order=2,
                    provider="table-model",
                    text_provenance={"source_region_ids": []},
                    structure={
                        "role": "table",
                        "row_count": 1,
                        "column_count": 1,
                        "cells": [
                            {
                                "id": "table-1-cell-1",
                                "bbox": {
                                    "left": 280,
                                    "top": 160,
                                    "right": 400,
                                    "bottom": 210,
                                },
                                "row_nums": [0],
                                "column_nums": [0],
                                "text": "",
                                "source": "table-model",
                                "confidence": None,
                                "resolution": "unreadable",
                                "alternatives": [],
                                "evidence_ids": [],
                                "supporters": [],
                                "decision": "no_cell_evidence",
                            }
                        ],
                    },
                )
            ]

    result = process_document(
        source,
        FixedReader(),
        stages=[BlankTableStage(), FaintTinyTextStage(crop_reader)],
    )

    page = result.pages[0]
    table = next(region for region in page.regions if region.kind == "table")
    recovered = next(
        region for region in page.regions if region.id.startswith("p1-faint")
    )
    cell = table.structure["cells"][0]
    assert table.text == "| fax ref 2048 |\n| --- |"
    assert cell["text"] == "fax ref 2048"
    assert cell["resolution"] == "resolved"
    assert cell["decision"] == "recovered_missing_cell"
    assert cell["evidence_ids"] == [recovered.id]
    assert recovered.structure["role"] == "table_source"
    assert recovered.structure["source_role"] == "tiny_text_candidate"
    assert recovered.structure["parent_id"] == table.id
    assert len(crop_reader.calls) == 1


def test_blank_unreadable_table_cell_does_not_trigger_reread(tmp_path: Path) -> None:
    source = tmp_path / "blank-table-cell.png"
    Image.new("L", (500, 220), "white").save(source)
    reader = CropReader()
    table = TextRegion(
        id="table-1",
        kind="table",
        text="|  |\n| --- |",
        confidence=None,
        bounding_box=BoundingBox(280, 160, 400, 210),
        reading_order=1,
        provider="table-model",
        structure={
            "role": "table",
            "row_count": 1,
            "column_count": 1,
            "cells": [
                {
                    "bbox": {
                        "left": 280,
                        "top": 160,
                        "right": 400,
                        "bottom": 210,
                    },
                    "text": "",
                    "row_nums": [0],
                    "column_nums": [0],
                    "resolution": "unreadable",
                    "decision": "no_cell_evidence",
                }
            ],
        },
    )

    class TableReader:
        name = "table-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [table]

    result = process_document(
        source,
        TableReader(),
        stages=[FaintTinyTextStage(reader)],
    )

    assert result.pages[0].regions == [table]
    assert reader.calls == []
    assert table.structure["cells"][0]["text"] == ""


def test_native_global_and_selective_views_are_ablatable_on_same_page(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "same-page.png")
    observed = {}

    class AblationReader(CropReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            with Image.open(image_path) as image:
                width, height = image.size
            self.calls.append((width, height))
            if (width, height) == (500, 220):
                box = BoundingBox(315, 188, 369, 199)
            elif (width, height) == (1500, 660):
                box = BoundingBox(945, 564, 1107, 597)
            else:
                box = BoundingBox(0, 0, width, height)
            return [
                TextRegion(
                    id="reread-1",
                    kind="text",
                    text="fax ref 2048",
                    confidence=0.96,
                    bounding_box=box,
                    reading_order=1,
                    provider=self.name,
                )
            ]

    for view in (
        "native",
        "global_high_resolution",
        "selective_crop_high_resolution",
    ):
        reader = AblationReader()
        result = process_document(
            source,
            FixedReader(),
            stages=[FaintTinyTextStage(reader, view=view)],
        )
        observed[view] = {
            "text": result.pages[0].text.value,
            "size": reader.calls[0],
            "view": result.pages[0].regions[-1].structure["recovery_view"],
        }

    assert {item["text"] for item in observed.values()} == {
        "DISCHARGE SUMMARY fax ref 2048"
    }
    assert observed["native"]["size"] == (500, 220)
    assert observed["global_high_resolution"]["size"] == (1500, 660)
    assert observed["selective_crop_high_resolution"]["size"][0] < 1500
    assert [observed[view]["view"] for view in observed] == list(observed)


def test_blank_page_and_ruling_do_not_create_text_proposals(tmp_path: Path) -> None:
    for name, ruling in (("blank.png", False), ("ruling.png", True)):
        source = tmp_path / name
        image = Image.new("L", (500, 220), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
        if ruling:
            draw.line((260, 188, 460, 188), fill=175, width=1)
        image.save(source)
        reader = CropReader()

        result = process_document(
            source,
            FixedReader(),
            stages=[FaintTinyTextStage(reader)],
        )

        assert result.pages[0].regions == [_heading()]
        assert reader.calls == []


def test_low_confidence_reread_remains_unreadable_alternative(tmp_path: Path) -> None:
    source = _page(tmp_path / "uncertain.png")

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(CropReader(confidence=0.6))],
    )

    candidate = result.pages[0].regions[-1]
    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "DISCHARGE SUMMARY"
    assert candidate.text == ""
    assert candidate.confidence is None
    assert candidate.resolution == "unreadable"
    assert [
        (alternative.text, alternative.confidence)
        for alternative in candidate.alternatives
    ] == [("fax ref 2048", 0.6)]


def test_reread_failure_preserves_first_pass_and_is_reported(tmp_path: Path) -> None:
    source = _page(tmp_path / "failure.png")

    class FailingReader:
        name = "failed-reread"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            raise ReaderError("reread_unavailable", "reader failed")

    executions: list[dict[str, object]] = []
    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(FailingReader())],
        stage_execution=executions,
    )

    assert result.pages[0].text.value == "DISCHARGE SUMMARY"
    assert result.pages[0].regions == [_heading()]
    assert result.pages[0].route == "review"
    assert [(failure.stage, failure.code) for failure in result.failures] == [
        ("faint-tiny-text", "reread_unavailable")
    ]
    assert executions[0]["status"] == "failed"
    assert executions[0]["failure_code"] == "reread_unavailable"


def test_selective_crops_use_one_reader_batch(tmp_path: Path) -> None:
    source = tmp_path / "two-lines.png"
    image = Image.new("L", (500, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 170), "fax ref 2048", fill=185)
    draw.text((36, 210), "copy to care", fill=185)
    image.save(source)

    class BatchReader(CropReader):
        batch_size = 8

        def __init__(self) -> None:
            super().__init__()
            self.batches: list[int] = []

        def read_batch(
            self,
            paths: list[Path],
            page_numbers: list[int],
        ) -> list[list[TextRegion]]:
            self.batches.append(len(paths))
            assert page_numbers == [1] * len(paths)
            return [self.read(path, 1) for path in paths]

    reader = BatchReader()
    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(reader)],
    )

    assert reader.batches == [2]
    assert len(result.pages[0].regions) == 3


def test_partial_reader_batch_preserves_success_and_records_failure_risk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "partial-batch.png"
    image = Image.new("L", (500, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 170), "fax ref 2048", fill=185)
    draw.text((36, 210), "copy to care", fill=185)
    image.save(source)

    class PartialBatchReader(CropReader):
        batch_size = 8

        def read_batch(
            self,
            paths: list[Path],
            page_numbers: list[int],
        ) -> list[list[TextRegion] | ReaderError]:
            assert len(paths) == 2
            assert page_numbers == [1, 1]
            return [
                self.read(paths[0], 1),
                ReaderError("reread_unavailable", "reader failed"),
            ]

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(PartialBatchReader())],
    )

    assert result.failures == []
    recovered = [
        region
        for region in result.pages[0].regions
        if region.kind == "text" and region.provider == "crop-reader"
    ]
    assert len(recovered) == 1
    risk = next(
        region for region in result.pages[0].regions if region.kind == "coverage_risk"
    )
    assert risk.structure["reasons"] == ["reread_failed"]
    assert risk.structure["region_risks"][0]["failure_code"] == ("reread_unavailable")


def test_partial_sequential_reads_preserve_success_and_record_failure_risk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "partial-sequential.png"
    image = Image.new("L", (500, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 170), "fax ref 2048", fill=185)
    draw.text((36, 210), "copy to care", fill=185)
    image.save(source)

    class PartialReader(CropReader):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            self.attempts += 1
            if self.attempts == 2:
                raise ReaderError("reread_unavailable", "reader failed")
            return super().read(image_path, page_number)

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(PartialReader())],
    )

    assert result.failures == []
    recovered = [
        region
        for region in result.pages[0].regions
        if region.kind == "text" and region.provider == "crop-reader"
    ]
    assert len(recovered) == 1
    risk = next(
        region for region in result.pages[0].regions if region.kind == "coverage_risk"
    )
    assert risk.structure["reasons"] == ["reread_failed"]
    assert risk.structure["region_risks"][0]["failure_code"] == "reread_unavailable"


def test_empty_reader_recovery_preserves_failure_and_empty_evidence(
    tmp_path: Path,
) -> None:
    source = _faint_only_page(tmp_path / "empty-reader.png")

    class EmptyReader:
        name = "empty-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return []

    result = process_document(
        source,
        EmptyReader(),
        stages=[FaintTinyTextStage(CropReader())],
    )

    page = result.pages[0]
    assert [failure.code for failure in result.failures] == ["no_text_detected"]
    assert page.failure_ids == [result.failures[0].id]
    assert page.route == "review"
    assert page.regions[0] == TextRegion(
        id="p1-empty-page-1",
        kind="page_text",
        text="",
        confidence=None,
        bounding_box=BoundingBox(0, 0, 500, 220),
        reading_order=1,
        provider="empty-reader",
    )
    assert page.regions[1].text == "fax ref 2048"
    assert page.text.value == " fax ref 2048"


def test_unresolved_full_page_region_does_not_claim_residual_pixels(
    tmp_path: Path,
) -> None:
    source = _faint_only_page(tmp_path / "unresolved-page.png")

    class UnresolvedReader:
        name = "unresolved-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="unresolved-page",
                    kind="text",
                    text="unreadable evidence",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 500, 220),
                    reading_order=1,
                    provider=self.name,
                    resolution="unreadable",
                )
            ]

    result = process_document(
        source,
        UnresolvedReader(),
        stages=[FaintTinyTextStage(CropReader())],
    )

    assert [region.id for region in result.pages[0].regions] == [
        "unresolved-page",
        "p1-faint-tiny-1-1",
    ]
    assert result.pages[0].regions[0].resolution == "unreadable"
    assert result.pages[0].regions[1].text == "fax ref 2048"


def test_recovered_region_is_inserted_between_existing_reading_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "middle.png"
    image = Image.new("L", (500, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 105), "fax ref 2048", fill=185)
    draw.text((24, 188), "END OF PAGE", fill=0)
    image.save(source)

    class TopBottomReader:
        name = "top-bottom"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                _heading(),
                TextRegion(
                    id="footer",
                    kind="text",
                    text="END OF PAGE",
                    confidence=0.99,
                    bounding_box=BoundingBox(24, 188, 130, 204),
                    reading_order=2,
                    provider=self.name,
                ),
            ]

    result = process_document(
        source,
        TopBottomReader(),
        stages=[FaintTinyTextStage(CropReader())],
    )

    page = result.pages[0]
    assert [region.id for region in page.regions] == [
        "heading",
        "p1-faint-tiny-1-1",
        "footer",
    ]
    assert [region.reading_order for region in page.regions] == [1, 2, 2]
    assert page.text.value == "DISCHARGE SUMMARY fax ref 2048 END OF PAGE"


def test_unresolved_reread_preserves_and_deduplicates_all_alternatives(
    tmp_path: Path,
) -> None:
    source = _page(tmp_path / "alternatives.png")

    class AlternativeReader(CropReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            with Image.open(image_path) as image:
                width, height = image.size
            return [
                TextRegion(
                    id="reread-1",
                    kind="text",
                    text="",
                    confidence=0.6,
                    bounding_box=BoundingBox(0, 0, width, height),
                    reading_order=1,
                    provider=self.name,
                    text_provenance={"method": "primary"},
                    resolution="unreadable",
                    alternatives=[
                        TextAlternative(
                            "fax ref 2048",
                            0.6,
                            self.name,
                            {"method": "primary"},
                        ),
                        TextAlternative(
                            "FAX  REF  2048",
                            0.5,
                            self.name,
                            {"method": "duplicate"},
                        ),
                        TextAlternative(
                            "fax ref 204B",
                            0.4,
                            "second-reader",
                            {"method": "alternate"},
                            "rejected",
                        ),
                    ],
                )
            ]

    result = process_document(
        source,
        FixedReader(),
        stages=[FaintTinyTextStage(AlternativeReader())],
    )

    recovered = result.pages[0].regions[-1]
    assert recovered.resolution == "unreadable"
    assert [(item.text, item.provider) for item in recovered.alternatives] == [
        ("fax ref 2048", "crop-reader"),
        ("fax ref 204B", "second-reader"),
    ]
    assert recovered.alternatives[0].text_provenance["method"] == "primary"
    assert recovered.alternatives[1].text_provenance["method"] == "alternate"
    assert recovered.alternatives[1].decision_state == "rejected"
    assert all(
        "faint_tiny_view" in item.text_provenance
        and "faint_tiny_recovery" in item.text_provenance
        for item in recovered.alternatives
    )


def test_budget_recovers_two_of_three_table_cells_and_records_coverage_risk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "three-table-cells.png"
    image = Image.new("L", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    boxes = [
        BoundingBox(20, 80, 180, 140),
        BoundingBox(220, 80, 380, 140),
        BoundingBox(420, 80, 580, 140),
    ]
    for index, box in enumerate(boxes, start=1):
        draw.text((box.left + 20, box.top + 20), f"value {index}", fill=0)
    image.save(source)
    rereader = CropReader()
    table = TextRegion(
        id="table-1",
        kind="table",
        text="|  |  |  |\n| --- | --- | --- |",
        confidence=None,
        bounding_box=BoundingBox(10, 70, 590, 150),
        reading_order=1,
        provider="table-model",
        structure={
            "role": "table",
            "row_count": 1,
            "column_count": 3,
            "cells": [
                {
                    "bbox": {
                        "left": box.left,
                        "top": box.top,
                        "right": box.right,
                        "bottom": box.bottom,
                    },
                    "text": "",
                    "row_nums": [0],
                    "column_nums": [index],
                    "resolution": "unreadable",
                    "decision": "no_cell_evidence",
                    "evidence_ids": [],
                    "alternatives": [],
                }
                for index, box in enumerate(boxes)
            ],
        },
    )

    class TableReader:
        name = "table-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [table]

    result = process_document(
        source,
        TableReader(),
        stages=[FaintTinyTextStage(rereader, max_proposals=2)],
    )

    assert result.failures == []
    assert len(rereader.calls) == 2
    result_table = next(
        region for region in result.pages[0].regions if region.kind == "table"
    )
    assert [cell["resolution"] for cell in result_table.structure["cells"]] == [
        "resolved",
        "resolved",
        "unreadable",
    ]
    risk = next(
        region for region in result.pages[0].regions if region.kind == "coverage_risk"
    )
    assert risk.structure["reasons"] == ["proposal_budget_exceeded"]
    assert risk.structure["region_risks"] == [
        {
            "reason": "proposal_budget_exceeded",
            "bounding_box": {
                "left": boxes[2].left,
                "top": boxes[2].top,
                "right": boxes[2].right,
                "bottom": boxes[2].bottom,
            },
            "source": "table_cell",
        }
    ]


def test_table_cell_recovery_uses_budget_before_optional_residuals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "structured-recovery-priority.png"
    image = Image.new("L", (600, 420), "white")
    draw = ImageDraw.Draw(image)
    for top in range(20, 300, 30):
        draw.text((40, top), f"faint line {top}", fill=175)
    draw.text((430, 350), "$187", fill=0)
    image.save(source)
    rereader = CropReader()
    table = TextRegion(
        id="table-1",
        kind="table",
        text="|  |\n| --- |",
        confidence=None,
        bounding_box=BoundingBox(400, 320, 500, 390),
        reading_order=1,
        provider="table-model",
        structure={
            "role": "table",
            "row_count": 1,
            "column_count": 1,
            "cells": [
                {
                    "bbox": {
                        "left": 400,
                        "top": 320,
                        "right": 500,
                        "bottom": 390,
                    },
                    "text": "",
                    "row_nums": [0],
                    "column_nums": [0],
                    "resolution": "unreadable",
                    "decision": "no_cell_evidence",
                    "evidence_ids": [],
                    "alternatives": [],
                }
            ],
        },
    )

    class TableReader:
        name = "table-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [table]

    result = process_document(
        source,
        TableReader(),
        stages=[FaintTinyTextStage(rereader, max_proposals=2)],
    )

    assert result.failures == []
    assert len(rereader.calls) == 2
    result_table = next(
        region for region in result.pages[0].regions if region.kind == "table"
    )
    assert result_table.structure["cells"][0]["text"] == "fax ref 2048"
    recovered = next(
        region
        for region in result.pages[0].regions
        if region.id.startswith("p1-faint")
        and region.kind == "text"
        and region.text_provenance["proposal"]["source"] == "table_cell"
    )
    assert recovered.structure["role"] == "table_source"
    assert recovered.structure["parent_id"] == result_table.id
    risk = next(
        region for region in result.pages[0].regions if region.kind == "coverage_risk"
    )
    assert risk.structure["reasons"] == ["proposal_budget_exceeded"]
    assert all(
        item["source"] == "unowned_pixels" for item in risk.structure["region_risks"]
    )


def _page(path: Path) -> Path:
    image = Image.new("L", (500, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 26), "DISCHARGE SUMMARY", fill=0)
    draw.text((315, 188), "fax ref 2048", fill=185)
    image.save(path)
    return path


def _faint_only_page(path: Path) -> Path:
    image = Image.new("L", (500, 220), "white")
    ImageDraw.Draw(image).text((315, 188), "fax ref 2048", fill=185)
    image.save(path)
    return path


def _heading() -> TextRegion:
    return TextRegion(
        id="heading",
        kind="text",
        text="DISCHARGE SUMMARY",
        confidence=0.99,
        bounding_box=BoundingBox(24, 26, 210, 42),
        reading_order=1,
        provider="first-pass",
    )
