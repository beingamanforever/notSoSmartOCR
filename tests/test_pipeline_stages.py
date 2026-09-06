from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.contracts import (
    BoundingBox,
    TextAlternative,
    TextRegion,
)
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.rendering import render_evidence


def test_contract_defaults_serialize_as_schema_v2(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    result = process_document(source, FixedReader())
    payload = result.to_dict()

    assert payload["schema_version"] == 2
    assert payload["pages"][0]["regions"][0]["resolution"] == "resolved"
    assert payload["pages"][0]["regions"][0]["alternatives"] == []
    assert payload["pages"][0]["regions"][0]["structure"] is None


def test_render_evidence_excludes_unresolved_regions() -> None:
    resolved = _region("resolved", "literal", 1)
    unreadable = _region("unreadable", "guess", 2, resolution="unreadable")
    conflicting = _region("conflicting", "choice", 3, resolution="conflicting")

    evidence = render_evidence([resolved, unreadable, conflicting])

    assert evidence.value == "literal"
    assert evidence.evidence_ids == ["resolved"]


def test_render_evidence_keeps_table_once_without_dropping_source_regions() -> None:
    source = _region("source", "cell text", 1)
    source.structure = {"role": "table_source", "parent_id": "table"}
    table = _region("table", "| cell |\n| --- |", 2)
    table.kind = "table"
    table.structure = {"role": "table", "cells": []}

    evidence = render_evidence([source, table])

    assert evidence.value == "| cell |\n| --- |"
    assert evidence.evidence_ids == ["table"]


def test_render_evidence_keeps_controls_structured_without_duplicate_text() -> None:
    label = _region("label", "Fall prevention", 1)
    control = _region("control", "[x] Fall prevention", 1)
    control.kind = "checkbox"
    control.structure = {
        "role": "control",
        "state": "selected",
        "label_evidence_ids": ["label"],
    }

    evidence = render_evidence([label, control])

    assert evidence.value == "Fall prevention"
    assert evidence.evidence_ids == ["label"]


def test_resolved_region_with_conflicting_alternative_routes_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class ConflictStage:
        name = "tiny-text"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].alternatives.append(TextAlternative("different", 0.9, self.name))
            return regions

    result = process_document(source, FixedReader(), stages=[ConflictStage()])

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "base"


@pytest.mark.parametrize(
    ("structure", "expected_route"),
    [
        ({"block_type": "formula", "formula_recognition": "heuristic"}, "review"),
        (
            {
                "block_type": "formula",
                "formula_attempt": {"outcome": "invalid_output"},
            },
            "review",
        ),
        (
            {
                "block_type": "formula",
                "formula_attempt": {
                    "outcome": "same_reader_repeat",
                    "reason": "reader_not_independent",
                },
            },
            "review",
        ),
        (
            {
                "block_type": "formula",
                "formula_recognition": "specialist_supported",
                "formula_attempt": {"outcome": "supported"},
            },
            "accept_local",
        ),
        (
            {
                "block_type": "formula",
                "formula_recognition": "human_accepted",
                "formula_attempt": {"outcome": "same_reader_repeat"},
            },
            "accept_local",
        ),
    ],
)
def test_formula_route_requires_specialist_support_or_human_acceptance(
    tmp_path: Path,
    structure: dict[str, object],
    expected_route: str,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class FormulaStage:
        name = "formula"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].kind = "layout_block"
            regions[0].structure = structure
            return regions

    result = process_document(source, FixedReader(), stages=[FormulaStage()])

    assert result.pages[0].route == expected_route


def test_resolved_region_and_cell_history_do_not_route_review(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class HistoryStage:
        name = "history"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].alternatives.extend(
                [
                    TextAlternative(
                        "old", 0.7, "reader-a", decision_state="superseded"
                    ),
                    TextAlternative("bad", 0.6, "reader-b", decision_state="rejected"),
                ]
            )
            regions[0].structure = {
                "cells": [
                    {
                        "text": "base",
                        "resolution": "resolved",
                        "alternatives": [
                            {"text": "older", "decision_state": "superseded"},
                            {"text": "wrong", "decision_state": "rejected"},
                        ],
                    }
                ]
            }
            return regions

    result = process_document(source, FixedReader(), stages=[HistoryStage()])

    assert result.pages[0].route == "accept_local"


def test_resolved_table_cell_with_pending_alternative_routes_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class CellConflictStage:
        name = "cell-conflict"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].structure = {
                "cells": [
                    {
                        "text": "base",
                        "resolution": "resolved",
                        "alternatives": [
                            {"text": "different", "decision_state": "pending"}
                        ],
                    }
                ]
            }
            return regions

    result = process_document(source, FixedReader(), stages=[CellConflictStage()])

    assert result.pages[0].route == "review"


def test_nested_table_cell_conflict_routes_review_without_hiding_table(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class TableStage:
        name = "tables"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].kind = "table"
            regions[0].text = "| primary |"
            regions[0].structure = {
                "role": "table",
                "cells": [{"text": "primary", "resolution": "conflicting"}],
            }
            return regions

    result = process_document(source, FixedReader(), stages=[TableStage()])

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "| primary |"


def test_post_restore_recovery_failure_preserves_last_good_regions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class FailedRecoveryReader(FixedReader):
        def restore_regions(
            self,
            regions: list[TextRegion],
            page_number: int,
        ) -> list[TextRegion]:
            regions[0].text = "restored canonical"
            return regions

        def recover_regions(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].text = "mutated before failure"
            raise ReaderError("recovery_failed", "controlled recovery failure")

    result = process_document(source, FailedRecoveryReader())

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "restored canonical"
    assert result.pages[0].regions[0].text == "restored canonical"
    assert result.pages[0].failure_ids == ["failure-1"]
    assert result.failures[0].stage == "ocr"
    assert result.failures[0].code == "recovery_failed"


def test_final_restored_tail_repetition_routes_review_without_rewriting(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    repeated = "stable prefix " + "AB12|" * 8

    class RestoredTailReader(FixedReader):
        def restore_regions(
            self,
            regions: list[TextRegion],
            page_number: int,
        ) -> list[TextRegion]:
            regions[0].text = repeated
            return regions

    result = process_document(source, RestoredTailReader())

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == repeated
    assert result.pages[0].regions[0].text == repeated
    assert result.pages[0].text.evidence_ids == ["p1-word-1"]
    assert result.pages[0].regions[1].structure == {
        "role": "coverage_risk",
        "reasons": ["tail_repetition"],
        "region_risks": [{"region_id": "p1-word-1", "reasons": ["tail_repetition"]}],
    }


def test_final_literal_date_risk_routes_review_with_result_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class InvalidDateStage:
        name = "literal-validation"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].text = "DOB: 02/31/2024"
            return regions

    result = process_document(source, FixedReader(), stages=[InvalidDateStage()])

    assert result.pages[0].route == "review"
    assert result.pages[0].text.value == "DOB: 02/31/2024"
    assert result.pages[0].text.evidence_ids == ["p1-word-1"]
    assert result.pages[0].regions[1].structure == {
        "role": "coverage_risk",
        "reasons": ["invalid_calendar_date"],
        "region_risks": [
            {"region_id": "p1-word-1", "reasons": ["invalid_calendar_date"]}
        ],
    }


def test_stages_run_in_order_and_keep_native_batching(tmp_path: Path) -> None:
    source = tmp_path / "pages.tiff"
    page = Image.new("RGB", (40, 20), "white")
    page.save(source, format="TIFF", save_all=True, append_images=[page])
    reader = BatchReader()
    calls: list[tuple[str, int, str]] = []

    class FirstStage:
        name = "first"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            calls.append((self.name, page_number, regions[0].text))
            regions[0].text += " first"
            return regions

    class SecondStage:
        name = "second"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            calls.append((self.name, page_number, regions[0].text))
            regions[0].text += " second"
            return regions

    result = process_document(source, reader, stages=[FirstStage(), SecondStage()])

    assert reader.batch_calls == 1
    assert reader.read_calls == 0
    assert calls == [
        ("first", 1, "page 1"),
        ("second", 1, "page 1 first"),
        ("first", 2, "page 2"),
        ("second", 2, "page 2 first"),
    ]
    assert [page.text.value for page in result.pages] == [
        "page 1 first second",
        "page 2 first second",
    ]


def test_pipeline_reports_timings_outside_the_result_schema(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    timings: dict[str, float] = {"stale": 1.0}

    class MarkerStage:
        name = "marker"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

    result = process_document(
        source,
        FixedReader(),
        stages=[MarkerStage()],
        timings=timings,
    )

    assert "timing" not in result.to_dict()
    assert "stale" not in timings
    assert set(timings) == {"prepare", "reader", "stage_view", "stage.marker"}
    assert all(seconds >= 0 for seconds in timings.values())


def test_pipeline_reports_page_queue_and_execution_outside_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.tiff"
    page = Image.new("RGB", (40, 20), "white")
    page.save(source, format="TIFF", save_all=True, append_images=[page])
    page_execution: list[dict[str, object]] = [{"page_number": 99}]

    result = process_document(
        source,
        FixedReader(),
        page_execution=page_execution,
    )

    assert "page_execution" not in result.to_dict()
    assert [run["page_number"] for run in page_execution] == [1, 2]
    assert all(float(run["queue_seconds"]) >= 0 for run in page_execution)
    assert all(float(run["execution_seconds"]) >= 0 for run in page_execution)
    assert all(run["batched_reader"] is False for run in page_execution)
    assert all(set(run["steps"]) == {"reader", "stage_view"} for run in page_execution)
    assert float(page_execution[1]["queue_seconds"]) >= float(
        page_execution[0]["queue_seconds"]
    )


def test_pipeline_reports_truthful_stage_execution_outside_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    execution: list[dict[str, object]] = [{"stage": "stale"}]

    class NoopStage:
        name = "tables"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

    class ProductiveStage:
        name = "controls"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].text += " changed"
            regions.append(_region("control", "[x] selected", 2))
            return regions

    result = process_document(
        source,
        FixedReader(),
        stages=[NoopStage(), ProductiveStage()],
        stage_execution=execution,
    )

    assert "stage_execution" not in result.to_dict()
    assert [run["status"] for run in execution] == ["fired", "productive"]
    assert execution[0]["input_regions"] == execution[0]["output_regions"] == 1
    assert execution[1]["input_regions"] == 1
    assert execution[1]["output_regions"] == 2
    assert execution[1]["added_regions"] == 1
    assert execution[1]["modified_regions"] == 1
    assert all(float(run["elapsed_seconds"]) >= 0 for run in execution)


def test_pipeline_marks_never_reached_stages_as_skipped(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    execution: list[dict[str, object]] = []

    class BrokenStageViewReader(FixedReader):
        def stage_view(self, image_path: Path, page_number: int) -> None:
            raise ReaderError("orientation_failed", "oriented view is unavailable")

    class NoopStage:
        name = "tables"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            raise AssertionError("stage must not run")

    result = process_document(
        source,
        BrokenStageViewReader(),
        stages=[NoopStage()],
        stage_execution=execution,
    )

    assert result.pages[0].route == "review"
    assert execution == [
        {
            "page_number": 1,
            "stage": "tables",
            "status": "skipped",
            "input_regions": 1,
            "output_regions": 1,
            "added_regions": 0,
            "removed_regions": 0,
            "modified_regions": 0,
            "elapsed_seconds": 0.0,
            "skip_reason": "orientation_failed",
        }
    ]


def test_pipeline_records_prepare_time_when_rendering_fails(tmp_path: Path) -> None:
    source = tmp_path / "page.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    timings: dict[str, float] = {}

    result = process_document(
        source,
        FixedReader(),
        pdftoppm_executable="missing-pdftoppm-for-test",
        timings=timings,
    )

    assert result.status == "failed"
    assert result.failures[0].code == "renderer_unavailable"
    assert set(timings) == {"prepare"}
    assert timings["prepare"] >= 0


def test_stage_reader_error_keeps_last_good_regions_and_continues(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    seen: list[str] = []

    class FailingStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            regions[0].text = "mutated before failure"
            raise ReaderError("model_unavailable", "weights are unavailable")

    class LaterStage:
        name = "controls"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            seen.append(regions[0].text)
            regions[0].alternatives.append(TextAlternative("alternate", 0.7, self.name))
            return regions

    execution: list[dict[str, object]] = []
    result = process_document(
        source,
        FixedReader(),
        stages=[FailingStage(), LaterStage()],
        stage_execution=execution,
    )

    assert seen == ["base"]
    assert result.pages[0].regions[0].text == "base"
    assert result.pages[0].regions[0].alternatives == [
        TextAlternative("alternate", 0.7, "controls")
    ]
    assert [run["status"] for run in execution] == ["failed", "productive"]
    assert execution[0]["failure_code"] == "model_unavailable"
    assert result.pages[0].text.value == "base"
    assert result.pages[0].route == "review"
    assert result.pages[0].failure_ids == ["failure-1"]
    assert result.failures[0].stage == "handwriting"
    assert result.failures[0].code == "model_unavailable"
    assert result.status == "failed"


def test_empty_reader_failure_is_preserved_before_stages(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    seen: list[str] = []

    class EmptyReader:
        name = "empty"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return []

    class RecoveryStage:
        name = "recovery"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            seen.extend(region.id for region in regions)
            regions[0].text = "recovered"
            return regions

    result = process_document(source, EmptyReader(), stages=[RecoveryStage()])

    assert seen == ["p1-empty-page-1"]
    assert result.pages[0].text.value == "recovered"
    assert result.failures[0].code == "no_text_detected"
    assert result.pages[0].failure_ids == [result.failures[0].id]


def test_unexpected_stage_error_escapes(tmp_path: Path) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)

    class BrokenStage:
        name = "broken"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            raise ValueError("programming error")

    with pytest.raises(ValueError, match="programming error"):
        process_document(source, FixedReader(), stages=[BrokenStage()])


class FixedReader:
    name = "fixed"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [_region(f"p{page_number}-word-1", "base", 1)]


class BatchReader(FixedReader):
    batch_size = 2

    def __init__(self) -> None:
        self.batch_calls = 0
        self.read_calls = 0

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        self.read_calls += 1
        return super().read(image_path, page_number)

    def read_batch(
        self,
        image_paths: list[Path],
        page_numbers: list[int],
    ) -> list[list[TextRegion] | ReaderError]:
        self.batch_calls += 1
        return [
            [_region(f"p{page_number}-word-1", f"page {page_number}", 1)]
            for page_number in page_numbers
        ]


def _region(
    region_id: str,
    text: str,
    reading_order: int,
    *,
    resolution: str = "resolved",
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="word",
        text=text,
        confidence=0.9,
        bounding_box=BoundingBox(0, 0, 20, 10),
        reading_order=reading_order,
        provider="fixed",
        resolution=resolution,
    )
