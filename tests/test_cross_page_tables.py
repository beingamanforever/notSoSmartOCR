from __future__ import annotations

import copy
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.cross_page_tables import (
    CrossPageTableStage,
    TableContinuationPrediction,
)
from ocr_pipeline.demo import create_app
from ocr_pipeline.pipeline import process_document


class _TableReader:
    name = "table-fixture"

    def __init__(self, tables: list[TextRegion]) -> None:
        self.tables = tables

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [copy.deepcopy(self.tables[page_number - 1])]


class _OrderedTableReader(_TableReader):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        table = copy.deepcopy(self.tables[page_number - 1])
        if page_number != 1:
            return [table]
        table.reading_order = 99
        table.structure = {**(table.structure or {}), "presentation_rank": 1}
        return [
            TextRegion(
                id="p1-title",
                kind="text",
                text="Continuation title",
                confidence=0.99,
                bounding_box=BoundingBox(5, 1, 95, 4),
                reading_order=1,
                provider=self.name,
                structure={"role": "title", "presentation_rank": 0},
            ),
            table,
            TextRegion(
                id="p1-body",
                kind="text",
                text="Continuation body",
                confidence=0.99,
                bounding_box=BoundingBox(5, 96, 95, 99),
                reading_order=2,
                provider=self.name,
                structure={"role": "paragraph", "presentation_rank": 2},
            ),
        ]


class _Classifier:
    name = "pairwise-test-classifier"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[int, int]] = []

    def classify(
        self,
        first_page_path: Path,
        second_page_path: Path,
        first_table: TextRegion,
        second_table: TextRegion,
    ) -> TableContinuationPrediction:
        self.calls.append(
            (
                int(first_table.id.split("-")[0][1:]),
                int(second_table.id.split("-")[0][1:]),
            )
        )
        return TableContinuationPrediction(
            self.scores[len(self.calls) - 1],
            {"checkpoint": "fixture"},
        )


class _ConstantClassifier:
    name = "constant-test-classifier"

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def classify(
        self,
        first_page_path: Path,
        second_page_path: Path,
        first_table: TextRegion,
        second_table: TextRegion,
    ) -> TableContinuationPrediction:
        self.calls += 1
        return TableContinuationPrediction(self.score, {"checkpoint": "fixture"})


def test_process_document_without_stage_preserves_default_behavior(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)

    result = process_document(
        source,
        _TableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
    )

    assert result.status == "success"
    assert result.table_continuations == []
    assert [page.regions[0].id for page in result.pages] == [
        "p1-table",
        "p2-table",
    ]


def test_process_document_adds_cross_page_table_without_mutating_sources(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    tables = [_table(1, "Aspirin"), _table(2, "Metformin")]
    tables[1].structure["cells"][-1]["evidence_ids"] = ["p2-dose-word"]  # type: ignore[index]
    baseline = process_document(source, _TableReader(tables))
    classifier = _Classifier([0.93])

    result = process_document(
        source,
        _TableReader(tables),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert [page.regions for page in result.pages] == [
        page.regions for page in baseline.pages
    ]
    assert classifier.calls == [(1, 2)]
    assert len(result.table_continuations) == 1
    continuation = result.table_continuations[0]
    assert continuation.source_table_ids == ["p1-table", "p2-table"]
    assert continuation.source_page_numbers == [1, 2]
    assert continuation.score == 0.93
    assert continuation.header_row_count == 1
    assert continuation.row_count == 3
    assert continuation.column_count == 2
    assert [cell["text"] for cell in continuation.cells] == [
        "Medication",
        "Dose",
        "Aspirin",
        "10 mg",
        "Metformin",
        "10 mg",
    ]
    assert [cell["row_nums"] for cell in continuation.cells] == [
        [0],
        [0],
        [1],
        [1],
        [2],
        [2],
    ]
    assert continuation.cells[-1]["continuation_source"] == {
        "page_number": 2,
        "table_id": "p2-table",
        "cell_id": "p2-cell-4",
        "evidence_ids": ["p2-dose-word"],
        "row_nums": [1],
    }
    assert continuation.provenance["guards"] == {
        "adjacent_pages": True,
        "valid_source_topology": True,
        "matching_column_count": True,
        "compatible_explicit_headers": True,
        "repeated_headers_omitted": 1,
        "valid_merged_topology": True,
    }
    assert continuation.provenance["pair_predictions"][0]["classifier"] == {
        "checkpoint": "fixture"
    }


def test_visually_similar_independent_tables_are_not_joined(tmp_path: Path) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    classifier = _Classifier([0.12])

    result = process_document(
        source,
        _TableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert classifier.calls == [(1, 2)]
    assert result.table_continuations == []


def test_incompatible_headers_fail_closed_before_classifier(tmp_path: Path) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    classifier = _Classifier([0.99])

    result = process_document(
        source,
        _TableReader(
            [_table(1, "Aspirin"), _table(2, "Metformin", header="Procedure")]
        ),
        cross_page_table_stage=CrossPageTableStage(classifier),
    )

    assert classifier.calls == []
    assert result.table_continuations == []


def test_headerless_continuation_reaches_classifier_and_retains_first_data_row(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    second = _table(2, "Metformin")
    second.structure = {
        "role": "table",
        "row_count": 2,
        "column_count": 2,
        "cells": [
            _cell(2, 1, 0, 0, "Metformin"),
            _cell(2, 2, 0, 1, "10 mg"),
            _cell(2, 3, 1, 0, "Lisinopril"),
            _cell(2, 4, 1, 1, "20 mg"),
        ],
    }
    classifier = _Classifier([0.93])

    result = process_document(
        source,
        _TableReader([_table(1, "Aspirin"), second]),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert classifier.calls == [(1, 2)]
    assert len(result.table_continuations) == 1
    continuation = result.table_continuations[0]
    assert continuation.header_row_count == 1
    assert continuation.row_count == 4
    assert [cell["text"] for cell in continuation.cells] == [
        "Medication",
        "Dose",
        "Aspirin",
        "10 mg",
        "Metformin",
        "10 mg",
        "Lisinopril",
        "20 mg",
    ]
    assert continuation.provenance["guards"]["repeated_headers_omitted"] == 0
    assert continuation.cells[4]["continuation_source"] == {
        "page_number": 2,
        "table_id": "p2-table",
        "cell_id": "p2-cell-1",
        "evidence_ids": [],
        "row_nums": [0],
    }


def test_headerless_first_table_rejects_later_explicit_header(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    classifier = _Classifier([0.93])

    result = process_document(
        source,
        _TableReader([_headerless_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert classifier.calls == []
    assert result.table_continuations == []


def test_all_headerless_continuation_renders_data_rows_as_html_cells(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    app = create_app(
        _TableReader(
            [_headerless_table(1, "Aspirin"), _headerless_table(2, "Metformin")]
        ),
        cross_page_table_stage=CrossPageTableStage(
            _Classifier([0.93]),
            minimum_score=0.8,
        ),
    )

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("pages.tiff", source.read_bytes(), "image/tiff")},
        )
        assert processed.status_code == 200
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    continuation = payload["result"]["table_continuations"][0]
    assert continuation["header_row_count"] == 0
    assert "<td>Aspirin</td>" in markdown
    assert "<td>Metformin</td>" in markdown
    assert "<th>Aspirin</th>" not in markdown
    assert "| Aspirin | 10 mg |" not in markdown


def test_invalid_topology_fails_closed_before_classifier(tmp_path: Path) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    second = _table(2, "Metformin")
    second.structure["cells"].pop()
    classifier = _Classifier([0.99])

    result = process_document(
        source,
        _TableReader([_table(1, "Aspirin"), second]),
        cross_page_table_stage=CrossPageTableStage(classifier),
    )

    assert classifier.calls == []
    assert result.table_continuations == []


def test_invalid_classifier_score_is_an_explicit_document_failure(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)

    result = process_document(
        source,
        _TableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(_Classifier([float("nan")])),
    )

    assert result.status == "partial"
    assert result.table_continuations == []
    assert [page.route for page in result.pages] == ["review", "review"]
    assert [(failure.stage, failure.code) for failure in result.failures] == [
        ("cross-page-tables", "invalid_table_continuation_score")
    ]


def test_invalid_classifier_output_is_an_explicit_document_failure(
    tmp_path: Path,
) -> None:
    class InvalidClassifier:
        name = "invalid-test-classifier"

        def classify(self, *args: object) -> object:
            return {"score": 0.99}

    source = _multi_page_tiff(tmp_path, 2)

    result = process_document(
        source,
        _TableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(InvalidClassifier()),  # type: ignore[arg-type]
    )

    assert result.status == "partial"
    assert result.table_continuations == []
    assert [(failure.stage, failure.code) for failure in result.failures] == [
        ("cross-page-tables", "invalid_table_continuation_prediction")
    ]


def test_classifier_runtime_failure_preserves_completed_ocr_and_requests_review(
    tmp_path: Path,
) -> None:
    class FailingClassifier:
        name = "failing-test-classifier"

        def classify(self, *args: object) -> TableContinuationPrediction:
            raise RuntimeError("private runtime detail")

    source = _multi_page_tiff(tmp_path, 2)
    tables = [_table(1, "Aspirin"), _table(2, "Metformin")]
    baseline = process_document(source, _TableReader(tables))
    execution: list[dict[str, object]] = []

    result = process_document(
        source,
        _TableReader(tables),
        cross_page_table_stage=CrossPageTableStage(FailingClassifier()),
        stage_execution=execution,
    )

    assert result.status == "partial"
    assert result.table_continuations == []
    assert [page.regions for page in result.pages] == [
        page.regions for page in baseline.pages
    ]
    assert [page.text for page in result.pages] == [
        page.text for page in baseline.pages
    ]
    assert [page.text.value for page in result.pages] == [
        tables[0].text,
        tables[1].text,
    ]
    assert [page.route for page in result.pages] == ["review", "review"]
    assert [
        (failure.stage, failure.code, failure.message) for failure in result.failures
    ] == [
        (
            "cross-page-tables",
            "table_continuation_classifier_failed",
            "Table continuation classifier failed during inference",
        )
    ]
    assert "private runtime detail" not in result.to_dict().__repr__()
    assert execution[0]["status"] == "failed"
    assert execution[0]["failure_code"] == "table_continuation_classifier_failed"
    assert execution[0]["input_regions"] == execution[0]["output_regions"] == 2


def test_three_page_continuation_is_one_vertical_chain(tmp_path: Path) -> None:
    source = _multi_page_tiff(tmp_path, 3)
    classifier = _Classifier([0.91, 0.87])

    result = process_document(
        source,
        _TableReader(
            [_table(1, "Aspirin"), _table(2, "Metformin"), _table(3, "Lisinopril")]
        ),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert classifier.calls == [(1, 2), (2, 3)]
    assert len(result.table_continuations) == 1
    continuation = result.table_continuations[0]
    assert continuation.source_page_numbers == [1, 2, 3]
    assert continuation.score == 0.87
    assert continuation.row_count == 4
    assert [
        cell["text"] for cell in continuation.cells if cell["column_nums"] == [0]
    ] == ["Medication", "Aspirin", "Metformin", "Lisinopril"]


def test_three_page_chain_allows_headerless_continuation_segments(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 3)
    classifier = _Classifier([0.91, 0.87])

    result = process_document(
        source,
        _TableReader(
            [
                _table(1, "Aspirin"),
                _headerless_table(2, "Metformin"),
                _headerless_table(3, "Lisinopril"),
            ]
        ),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert classifier.calls == [(1, 2), (2, 3)]
    assert len(result.table_continuations) == 1
    continuation = result.table_continuations[0]
    assert continuation.header_row_count == 1
    assert continuation.row_count == 6
    assert [
        cell["text"] for cell in continuation.cells if cell["column_nums"] == [0]
    ] == [
        "Medication",
        "Aspirin",
        "Metformin",
        "Metformin second row",
        "Lisinopril",
        "Lisinopril second row",
    ]


def test_duplicate_table_ids_on_different_pages_do_not_merge_separate_chains(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 4)
    tables = [
        _table(1, "Aspirin", table_id="table"),
        _table(2, "Metformin", table_id="table"),
        _table(3, "X-ray", header="Procedure", table_id="table"),
        _table(4, "MRI", header="Procedure", table_id="table"),
    ]
    classifier = _ConstantClassifier(0.99)

    result = process_document(
        source,
        _TableReader(tables),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    assert classifier.calls == 2
    assert [
        continuation.source_page_numbers for continuation in result.table_continuations
    ] == [[1, 2], [3, 4]]


def test_demo_renders_stitched_table_once_and_retains_source_evidence(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    classifier = _Classifier([0.93])
    app = create_app(
        _TableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(classifier, minimum_score=0.8),
    )

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("pages.tiff", source.read_bytes(), "image/tiff")},
        )
        assert processed.status_code == 200
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        index = client.get("/").text

    assert payload["pipeline_stages"] == ["cross-page-tables"]
    assert payload["stage_execution"] == [
        {
            "page_number": None,
            "stage": "cross-page-tables",
            "status": "productive",
            "input_regions": 2,
            "output_regions": 2,
            "added_regions": 0,
            "removed_regions": 0,
            "modified_regions": 0,
            "added_artifacts": 1,
            "elapsed_seconds": payload["stage_execution"][0]["elapsed_seconds"],
        }
    ]
    assert [page["regions"][0]["id"] for page in payload["result"]["pages"]] == [
        "p1-table",
        "p2-table",
    ]
    continuation = payload["result"]["table_continuations"][0]
    assert continuation["source_table_ids"] == ["p1-table", "p2-table"]
    assert continuation["source_page_numbers"] == [1, 2]
    assert continuation["provenance"]["classifier"] == classifier.name
    assert all("continuation_source" in cell for cell in continuation["cells"])
    assert continuation["cells"][-1]["continuation_source"] == {
        "page_number": 2,
        "table_id": "p2-table",
        "cell_id": "p2-cell-4",
        "evidence_ids": [],
        "row_nums": [1],
    }
    assert markdown.count("| Medication | Dose |") == 1
    assert markdown.count("Aspirin") == 1
    assert markdown.count("Metformin") == 1
    assert "pagesWithTableContinuations" in index
    assert "cell.continuation_source?.page_number" in index
    assert "cell.continuation_source?.cell_id" in index
    assert "evidencePageNumber(target)" in index


def test_demo_keeps_stitched_table_in_corrected_presentation_order(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    app = create_app(
        _OrderedTableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(
            _Classifier([0.93]),
            minimum_score=0.8,
        ),
    )

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("pages.tiff", source.read_bytes(), "image/tiff")},
        )
        assert processed.status_code == 200
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        index = client.get("/").text

    title_index = markdown.index("Continuation title")
    table_index = markdown.index("| Medication | Dose |")
    body_index = markdown.index("Continuation body")
    assert title_index < table_index < body_index
    assert "...(first.region.structure || {})" in index


def test_demo_preserves_separate_tables_when_classifier_rejects_continuation(
    tmp_path: Path,
) -> None:
    source = _multi_page_tiff(tmp_path, 2)
    app = create_app(
        _TableReader([_table(1, "Aspirin"), _table(2, "Metformin")]),
        cross_page_table_stage=CrossPageTableStage(
            _ConstantClassifier(0.1),
            minimum_score=0.8,
        ),
    )

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("pages.tiff", source.read_bytes(), "image/tiff")},
        )
        assert processed.status_code == 200
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    assert payload["result"]["table_continuations"] == []
    assert markdown.count("| Medication | Dose |") == 2
    assert markdown.count("Aspirin") == 1
    assert markdown.count("Metformin") == 1


def _multi_page_tiff(tmp_path: Path, count: int) -> Path:
    source = tmp_path / "pages.tiff"
    pages = [Image.new("RGB", (100, 100), "white") for _ in range(count)]
    pages[0].save(source, save_all=True, append_images=pages[1:], format="TIFF")
    return source


def _table(
    page_number: int,
    value: str,
    *,
    header: str = "Medication",
    table_id: str | None = None,
) -> TextRegion:
    cells = [
        _cell(page_number, 1, 0, 0, header, column_header=True),
        _cell(page_number, 2, 0, 1, "Dose", column_header=True),
        _cell(page_number, 3, 1, 0, value),
        _cell(page_number, 4, 1, 1, "10 mg"),
    ]
    return TextRegion(
        id=table_id or f"p{page_number}-table",
        kind="table",
        text=(f"| {header} | Dose |\n| --- | --- |\n| {value} | 10 mg |"),
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


def _headerless_table(page_number: int, value: str) -> TextRegion:
    cells = [
        _cell(page_number, 1, 0, 0, value),
        _cell(page_number, 2, 0, 1, "10 mg"),
        _cell(page_number, 3, 1, 0, f"{value} second row"),
        _cell(page_number, 4, 1, 1, "20 mg"),
    ]
    return TextRegion(
        id=f"p{page_number}-table",
        kind="table",
        text=f"{value}\t10 mg\n{value} second row\t20 mg",
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


def _cell(
    page_number: int,
    index: int,
    row: int,
    column: int,
    text: str,
    *,
    column_header: bool = False,
) -> dict[str, object]:
    return {
        "id": f"p{page_number}-cell-{index}",
        "row_nums": [row],
        "column_nums": [column],
        "text": text,
        "resolution": "resolved",
        "column_header": column_header,
    }
