from __future__ import annotations

import io
import json
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import ocr_pipeline.demo as demo_module
from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.orientation import OrientationReader
from ocr_pipeline.providers import ReaderError
from ocr_pipeline.rendering import INLINE_MAX_GAP_HEIGHTS, render_page_markdown
from ocr_pipeline.tables import TableCell, TablePrediction, TatrTableStage


class ControlledReader:
    name = "controlled-reader"
    version = "test-1"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            assert image.size == (120, 80)
        return [
            TextRegion(
                id=f"page-{page_number}-region-1",
                kind="paragraph",
                text=f"Controlled page {page_number}",
                confidence=0.91,
                bounding_box=BoundingBox(10, 12, 90, 36),
                reading_order=1,
                provider=self.name,
                text_provenance={"source": "controlled-test"},
            )
        ]


class SecondPageFailureReader(ControlledReader):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        if page_number == 2:
            raise ReaderError("controlled_failure", "Second page OCR failed")
        return super().read(image_path, page_number)


class ReviewEvidenceReader(ControlledReader):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [
            TextRegion(
                id="paragraph",
                kind="paragraph",
                text="Primary text",
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 50, 10),
                reading_order=1,
                provider=self.name,
                alternatives=[
                    TextAlternative("Primary text", 0.8, "supporter"),
                    TextAlternative("Different text", 0.8, "challenger"),
                ],
            ),
            TextRegion(
                id="field",
                kind="form_field",
                text="Value",
                confidence=None,
                bounding_box=BoundingBox(0, 12, 50, 22),
                reading_order=2,
                provider=self.name,
            ),
            TextRegion(
                id="handwriting",
                kind="handwriting",
                text="uncertain note",
                confidence=0.4,
                bounding_box=BoundingBox(0, 24, 50, 34),
                reading_order=3,
                provider=self.name,
                resolution="conflicting",
                structure={
                    "handwriting_review": {
                        "required": True,
                        "reason": "crop_disagreement",
                    }
                },
            ),
            TextRegion(
                id="control",
                kind="checkbox",
                text="[?] Fall risk",
                confidence=0.5,
                bounding_box=BoundingBox(0, 36, 50, 46),
                reading_order=4,
                provider=self.name,
                resolution="unreadable",
            ),
            TextRegion(
                id="table",
                kind="table",
                text="| Value |\n| --- |\n| 42 |",
                confidence=0.95,
                bounding_box=BoundingBox(52, 0, 110, 46),
                reading_order=5,
                provider=self.name,
                structure={
                    "role": "table",
                    "cells": [
                        {
                            "text": "42",
                            "resolution": "conflicting",
                            "alternatives": [{"text": "43"}],
                        }
                    ],
                },
            ),
            TextRegion(
                id="risk",
                kind="coverage_risk",
                text="",
                confidence=None,
                bounding_box=BoundingBox(0, 0, 120, 80),
                reading_order=6,
                provider="deterministic-evidence-risk",
                resolution="unreadable",
                structure={
                    "role": "coverage_risk",
                    "reasons": ["small_text_evidence", "low_mean_confidence"],
                },
            ),
        ]


class RejectedTableReader(ControlledReader):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [
            *super().read(image_path, page_number),
            TextRegion(
                id="rejected-table",
                kind="table_candidate",
                text="",
                confidence=0.93,
                bounding_box=BoundingBox(0, 0, 120, 80),
                reading_order=2,
                provider="table-model",
                resolution="unreadable",
                structure={
                    "role": "table_candidate",
                    "status": "rejected",
                    "reason": "near_page_low_complexity_grid",
                },
            ),
        ]


class StructuredDocumentReader(ControlledReader):
    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [
            TextRegion(
                id="label",
                kind="text",
                text="Fall risk",
                confidence=0.97,
                bounding_box=BoundingBox(0, 50, 30, 60),
                reading_order=3,
                provider=self.name,
            ),
            TextRegion(
                id="table",
                kind="table",
                text="flattened fallback",
                confidence=0.94,
                bounding_box=BoundingBox(0, 20, 110, 48),
                reading_order=2,
                provider=self.name,
                structure={
                    "role": "table",
                    "row_count": 2,
                    "column_count": 2,
                    "cells": [
                        {
                            "id": "name-header",
                            "text": "Name",
                            "row_nums": [0],
                            "column_nums": [0],
                            "column_header": True,
                            "resolution": "resolved",
                        },
                        {
                            "id": "value-header",
                            "text": "Value",
                            "row_nums": [0],
                            "column_nums": [1],
                            "column_header": True,
                            "resolution": "resolved",
                        },
                        {
                            "id": "measure",
                            "text": "Pulse",
                            "row_nums": [1],
                            "column_nums": [0],
                            "resolution": "resolved",
                        },
                        {
                            "id": "measure-value",
                            "text": "72",
                            "row_nums": [1],
                            "column_nums": [1],
                            "resolution": "conflicting",
                            "alternatives": [{"text": "77"}],
                        },
                    ],
                },
            ),
            TextRegion(
                id="title",
                kind="title",
                text="Visit summary",
                confidence=0.99,
                bounding_box=BoundingBox(0, 0, 90, 18),
                reading_order=1,
                provider=self.name,
            ),
            TextRegion(
                id="control",
                kind="checkbox",
                text="[x] Fall risk",
                confidence=0.93,
                bounding_box=BoundingBox(0, 50, 90, 60),
                reading_order=3,
                provider="control-reader",
                structure={
                    "role": "control",
                    "state": "selected",
                    "label_evidence_ids": ["label"],
                },
            ),
            TextRegion(
                id="handwriting",
                kind="handwriting",
                text="Return in 2 weeks",
                confidence=0.51,
                bounding_box=BoundingBox(0, 62, 100, 76),
                reading_order=4,
                provider=self.name,
                resolution="conflicting",
                alternatives=[TextAlternative("Return in 3 weeks", 0.5, "challenger")],
            ),
        ]


class GeneratedTableReader:
    name = "generated-table-reader"
    version = "test-1"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            assert image.size == (1200, 720)
            assert image.getpixel((80, 190)) == (21, 36, 59)
        columns = (80, 460, 680, 900, 1120)
        return [
            TextRegion(
                id=f"cell-{row_index}-{column_index}",
                kind="word",
                text=f"R{row_index}C{column_index}",
                confidence=0.99,
                bounding_box=BoundingBox(
                    columns[column_index] + 12,
                    190 + row_index * 76 + 12,
                    columns[column_index + 1] - 12,
                    190 + (row_index + 1) * 76 - 12,
                ),
                reading_order=row_index * 4 + column_index + 1,
                provider=self.name,
            )
            for row_index in range(5)
            for column_index in range(4)
        ]


class GeneratedTableExtractor:
    name = "generated-table-layout"

    def extract(
        self, image_path: Path, tokens: list[dict[str, Any]]
    ) -> list[TablePrediction]:
        assert len(tokens) == 20
        columns = (80, 460, 680, 900, 1120)
        return [
            TablePrediction(
                bounding_box=BoundingBox(80, 190, 1120, 570),
                cells=tuple(
                    TableCell(
                        bounding_box=BoundingBox(
                            columns[column_index],
                            190 + row_index * 76,
                            columns[column_index + 1],
                            190 + (row_index + 1) * 76,
                        ),
                        row_nums=(row_index,),
                        column_nums=(column_index,),
                        column_header=row_index == 0,
                    )
                    for row_index in range(5)
                    for column_index in range(4)
                ),
                confidence=0.99,
                model={"id": "generated-table-layout", "origin": "test"},
            )
        ]


class ControlledStage:
    name = "table-structure"

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        regions[0].text += " with table"
        return regions


class DemoMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def test_demo_processes_multi_page_tiff_and_clears_session() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["cache-control"] == "no-store"
        assert "Not So Smart OCR" in index.text
        assert 'id="page-canvas"' in index.text
        assert 'id="download-markdown"' in index.text
        assert 'canvas.addEventListener("click"' in index.text
        assert 'id="panel-rendered"' in index.text
        assert 'id="panel-layout"' in index.text
        assert 'id="panel-processed"' in index.text
        assert 'id="panel-raw"' in index.text

        processed = client.post(
            "/api/process",
            files={"file": ("visit.tiff", _two_page_tiff(), "image/tiff")},
        )

        assert processed.status_code == 200
        assert processed.headers["cache-control"] == "no-store"
        payload = processed.json()
        session_id = payload["session_id"]
        assert "/" not in session_id
        assert payload["filename"] == "visit.tiff"
        assert payload["backend"] == "controlled-reader"
        assert payload["backend_version"] == "test-1"
        assert payload["pipeline_stages"] == []
        assert payload["elapsed_seconds"] >= 0
        assert set(payload["timing"]) == {
            "receive_seconds",
            "save_seconds",
            "preview_queue_seconds",
            "queue_seconds",
            "pipeline_seconds",
            "preview_seconds",
            "total_seconds",
            "pipeline_steps",
        }
        assert payload["timing"]["pipeline_seconds"] == payload["elapsed_seconds"]
        assert set(payload["timing"]["pipeline_steps"]) == {
            "reader",
            "stage_view",
        }
        assert all(
            seconds >= 0
            for name, seconds in payload["timing"].items()
            if name != "pipeline_steps"
        )
        assert payload["geometry"] == "pixel coordinates"
        assert payload["coverage_assessment"] == {
            "status": "not_assessed",
            "message": "Extraction completeness was not independently assessed.",
            "pages": [],
        }
        assert payload["uncertainty"] == {
            "interpretation": (
                "Evidence diagnostics for review routing, not calibrated probabilities."
            ),
            "pages": [
                {
                    "page_number": 1,
                    "review_required": False,
                    "mean_primary_confidence": 0.91,
                    "primary_regions": 1,
                    "confidence_regions": 1,
                    "unresolved_evidence": 0,
                    "conflicting_evidence": 0,
                    "disagreeing_alternatives": 0,
                    "failure_count": 0,
                    "risk_reasons": [],
                    "category_counts": {"paragraph": 1},
                },
                {
                    "page_number": 2,
                    "review_required": False,
                    "mean_primary_confidence": 0.91,
                    "primary_regions": 1,
                    "confidence_regions": 1,
                    "unresolved_evidence": 0,
                    "conflicting_evidence": 0,
                    "disagreeing_alternatives": 0,
                    "failure_count": 0,
                    "risk_reasons": [],
                    "category_counts": {"paragraph": 1},
                },
            ],
        }
        assert payload["preview_failure"] is None
        assert payload["page_images"] == [
            f"/api/sessions/{session_id}/pages/1",
            f"/api/sessions/{session_id}/pages/2",
        ]

        result = payload["result"]
        assert result["status"] == "success"
        assert result["document_id"] == "visit"
        assert result["source"] == {"name": "visit.tiff", "kind": "image"}
        assert str(app.state.session_root) not in processed.text
        assert [page["page_number"] for page in result["pages"]] == [1, 2]
        first_region = result["pages"][0]["regions"][0]
        assert first_region == {
            "id": "page-1-region-1",
            "kind": "paragraph",
            "text": "Controlled page 1",
            "confidence": 0.91,
            "bounding_box": {"left": 10, "top": 12, "right": 90, "bottom": 36},
            "reading_order": 1,
            "provider": "controlled-reader",
            "text_provenance": {"source": "controlled-test"},
            "resolution": "resolved",
            "alternatives": [],
            "structure": None,
        }

        preview = client.get(payload["page_images"][1])
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(preview.content)) as page_image:
            assert page_image.size == (120, 80)
            assert page_image.getpixel((0, 0)) == (220, 220, 220)

        json_download = client.get(f"/api/sessions/{session_id}/result.json")
        assert json_download.status_code == 200
        assert json_download.headers["content-type"] == "application/json"
        assert json.loads(json_download.text) == result
        assert "ocr-result.json" in json_download.headers["content-disposition"]

        markdown = client.get(f"/api/sessions/{session_id}/result.md")
        assert markdown.status_code == 200
        assert markdown.headers["content-type"].startswith("text/markdown")
        assert "ocr-result.md" in markdown.headers["content-disposition"]
        assert "# OCR result" in markdown.text
        assert "## Page 1" in markdown.text
        assert "Controlled page 2" in markdown.text
        assert "Backend: controlled-reader" in markdown.text
        assert "Coverage: not_assessed" in markdown.text

        session_dir = app.state.session_root / session_id
        assert session_dir.is_dir()
        cleared = client.delete(f"/api/sessions/{session_id}")
        assert cleared.status_code == 204
        assert not session_dir.exists()
        assert client.get(payload["page_images"][0]).status_code == 404
        assert client.get(f"/api/sessions/{session_id}/result.json").status_code == 404

        example = client.get("/api/examples/architecture")
        assert example.status_code == 200
        assert example.headers["content-type"] == "application/pdf"
        assert example.content.startswith(b"%PDF")
        assert client.get("/api/examples/../../AGENTS.md").status_code == 404
        assert client.get("/api/examples/private-case").status_code == 404

        assert "state.imageRequest += 1" in index.text
        assert (
            'byId("panel-json").querySelector("pre").textContent = "{}"' in index.text
        )
        assert 'kind: "table_cell"' in index.text
        assert "function currentOverlays()" in index.text
        assert "boxArea(left.region.bounding_box)" in index.text
        assert 'id="region-order"' in index.text
        assert 'id="region-alternatives"' in index.text
        assert 'id="region-structure"' in index.text
        assert 'id="risk-card"' in index.text
        assert 'id="copy-json"' in index.text
        assert 'id="category-list"' in index.text
        assert 'id="timing-card"' in index.text
        assert "function renderUncertainty()" in index.text
        assert "function renderCategories()" in index.text
        assert "function renderTiming(timing)" in index.text
        assert 'state.hiddenKinds.add("table_candidate")' in index.text
        assert 'textContent = "Loading page preview..."' in index.text
        assert (
            'setStatus(`${error.message}${cleanupFailed ? cleanupWarning : ""}`, true);'
            in index.text
        )


def test_demo_sanitizes_nested_paths_in_exported_json() -> None:
    class PathProvenanceReader(ControlledReader):
        session_root: Path

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            region = super().read(image_path, page_number)[0]
            region.text_provenance = {
                "model": {
                    "adapter": {
                        "source": str(
                            (self.session_root / "private" / "adapter.pt").resolve()
                        ),
                        "validated": True,
                    },
                    "debug": [
                        str(
                            (Path(tempfile.gettempdir()) / "private-crop.png").resolve()
                        )
                    ],
                }
            }
            return [region]

    reader = PathProvenanceReader()
    app = create_app(reader)
    reader.session_root = app.state.session_root
    session_path = str(app.state.session_root)
    resolved_session_path = str(app.state.session_root.resolve())
    temporary_path = str(Path(tempfile.gettempdir()))
    resolved_temporary_path = str(Path(tempfile.gettempdir()).resolve())

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("visit.png", _page_png(), "image/png")},
        )
        session_id = processed.json()["session_id"]
        downloaded = client.get(f"/api/sessions/{session_id}/result.json")

    assert processed.status_code == 200
    assert downloaded.status_code == 200
    exported_json = downloaded.text
    assert session_path not in exported_json
    assert resolved_session_path not in exported_json
    assert temporary_path not in exported_json
    assert resolved_temporary_path not in exported_json
    assert "[session]/private/adapter.pt" in exported_json
    assert "[temporary directory]/private-crop.png" in exported_json


def test_demo_exposes_browser_testable_timer_copy_and_output_states() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert html.count('class="tab" role="tab"') == 6
    assert html.count("Not So Smart OCR") == 2
    assert "!SoSmartOCR" not in html
    assert 'data-example="clinical-table">Clinical table</button>' in html
    assert "Table-cell comparison" not in html

    assert 'id="client-timer" data-state="idle"' in html
    assert 'id="client-elapsed">0.0 s' in html
    assert "function startClientTimer()" in html
    assert "function stopClientTimer(outcome)" in html
    assert 'let timerOutcome = "error"' in html
    assert 'timerOutcome = "complete"' in html
    assert html.index("startClientTimer();") < html.index("resetResult();")
    assert html.index("startClientTimer();") < html.index(
        'fetch("/api/process", { method: "POST", body })'
    )
    assert "stopClientTimer(timerOutcome);" in html
    assert "Backend stage timing" in html
    assert '"Backend pipeline"' in html
    assert '["Preview queue", timing.preview_queue_seconds]' in html
    assert '["Preview processing", timing.preview_seconds]' in html
    assert '["OCR queue", timing.queue_seconds]' in html
    assert '["OCR pipeline", timing.pipeline_seconds]' in html

    assert 'id="copy-json" type="button" data-copy-state="idle"' in html
    assert 'id="copy-status" role="status" aria-live="polite"' in html
    assert "navigator.clipboard.writeText(content)" in html
    assert "function copyTextWithTextarea(content)" in html
    assert 'document.execCommand("copy")' in html
    assert 'showCopyStatus("JSON copied", "copied")' in html
    assert 'showCopyStatus("Copy failed", "error")' in html
    assert 'button.textContent = "Copying..."' in html
    assert 'button.setAttribute("aria-busy", "true")' in html
    assert 'button.setAttribute("aria-busy", "false")' in html
    assert (
        'button.textContent = copyState === "copied" ? "Copied" : "Try copy again"'
        in html
    )
    assert "textarea.focus({ preventScroll: true })" in html
    assert (
        'id="reread-handwriting" type="button" hidden disabled>'
        "Reread handwriting</button>" in html
    )
    assert "function handleRereadHandwriting()" in html
    assert "function updateRereadAction(region)" in html
    assert "Rereading the selected region locally..." in html
    assert "Handwriting reread complete." in html
    assert "`/api/sessions/${sessionId}/handwriting`" in html
    assert (
        "JSON.stringify({ page_number: page.page_number, region_id: region.id })"
        in html
    )
    assert "state.response = payload;\n        renderResult();" in html

    assert 'form.setAttribute("aria-busy", String(busy))' in html
    assert 'parseButton.textContent = busy ? "Parsing..." : "Parse"' in html
    assert "function activateTab(tab)" in html
    assert "panel.hidden = !active" in html
    assert 'id="panel-layout" aria-labelledby="tab-layout" hidden' in html

    assert "Evidence diagnostics" in html
    assert "Provider-reported" in html
    assert 'card.dataset.state = page.review_required ? "review" : "clear"' in html
    assert "Review required · page ${page.page_number}" in html
    assert 'region?.structure?.role || ""' in html
    assert "block.dataset.sourceKind = group.sourceKind" in html
    assert 'region.text_provenance?.merge_level === "word"' in html


def test_generated_table_example_runs_through_table_pipeline() -> None:
    app = create_app(
        GeneratedTableReader(),
        stages=[TatrTableStage(GeneratedTableExtractor())],
    )

    with TestClient(app) as client:
        example = client.get("/api/examples/clinical-table")
        assert example.status_code == 200
        assert example.headers["content-type"] == "image/png"
        assert "synthetic-clinical-table.png" in example.headers["content-disposition"]
        with Image.open(io.BytesIO(example.content)) as image:
            assert image.size == (1200, 720)

        processed = client.post(
            "/api/process",
            files={
                "file": (
                    "synthetic-clinical-table.png",
                    example.content,
                    "image/png",
                )
            },
        )
        assert processed.status_code == 200
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    tables = [
        region
        for region in payload["result"]["pages"][0]["regions"]
        if region["kind"] == "table"
    ]
    assert payload["pipeline_stages"] == ["tables"]
    assert len(tables) == 1
    assert tables[0]["structure"]["row_count"] == 5
    assert tables[0]["structure"]["column_count"] == 4
    assert len(tables[0]["structure"]["cells"]) == 20
    assert payload["result"]["pages"][0]["text"]["evidence_ids"] == [tables[0]["id"]]
    table_lines = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert len(table_lines) == 6
    assert table_lines[1].count("---") == 4


def test_demo_uses_neutral_navy_and_amber_visual_roles() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        html = client.get("/").text

    for declaration in (
        "--background: #f4f2ed",
        "--panel: #ffffff",
        "--panel-raised: #eeece5",
        "--text: #15243b",
        "--muted: #5b6678",
        "--accent: #a85b00",
        "--accent-soft: #fff2d5",
    ):
        assert declaration in html
    assert "--danger:" not in html
    assert "--success:" not in html
    assert ".status-line.busy, .status-line.error { color: var(--accent);" in html
    assert (
        ".status-line.success { color: var(--text); border-color: var(--line-strong);"
        in html
    )
    assert 'const structureColor = rootStyle.getPropertyValue("--text").trim()' in html
    assert 'const actionColor = rootStyle.getPropertyValue("--accent").trim()' in html
    assert "return structureColor;" in html


def test_demo_markup_links_controls_tabs_and_output_panels() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    parser = DemoMarkupParser()
    parser.feed(response.text)
    elements = parser.elements
    ids = [attrs["id"] for _, attrs in elements if "id" in attrs]
    assert len(ids) == len(set(ids))

    by_element_id = {
        attrs["id"]: (tag, attrs) for tag, attrs in elements if "id" in attrs
    }
    assert by_element_id["source-file"][0] == "input"
    assert any(
        tag == "label" and attrs.get("for") == "source-file" for tag, attrs in elements
    )
    assert by_element_id["status-line"][1] == {
        "class": "status-line",
        "id": "status-line",
        "role": "status",
        "aria-live": "polite",
        "aria-atomic": "true",
    }
    assert by_element_id["client-timer"][1]["hidden"] is None
    reread_tag, reread_attrs = by_element_id["reread-handwriting"]
    assert reread_tag == "button"
    assert reread_attrs["hidden"] is None
    assert reread_attrs["disabled"] is None
    assert by_element_id["reread-status"][1]["role"] == "status"

    tabs = [attrs for _, attrs in elements if attrs.get("role") == "tab"]
    panels = {
        attrs["id"]: attrs for _, attrs in elements if attrs.get("role") == "tabpanel"
    }
    assert [tab["id"] for tab in tabs] == [
        "tab-rendered",
        "tab-layout",
        "tab-processed",
        "tab-raw",
        "tab-json",
        "tab-failures",
    ]
    assert len(panels) == len(tabs)
    assert [tab["id"] for tab in tabs if tab["aria-selected"] == "true"] == [
        "tab-rendered"
    ]
    for tab in tabs:
        panel = panels[tab["aria-controls"]]
        assert panel["aria-labelledby"] == tab["id"]
        assert ("hidden" in panel) is (tab["aria-selected"] == "false")

    for link_id in ("download-json", "download-markdown"):
        tag, attrs = by_element_id[link_id]
        assert tag == "a"
        assert "download" in attrs
        assert attrs["aria-disabled"] == "true"

    assert "@media (max-width: 900px)" in response.text
    assert ".workspace { grid-template-columns: 1fr; }" in response.text


def test_demo_prepares_each_page_once_for_ocr_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    prepare_pages = demo_module._prepare_pages

    def count_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return prepare_pages(*args, **kwargs)

    monkeypatch.setattr(demo_module, "_prepare_pages", count_prepare)
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        )

    assert processed.status_code == 200
    assert len(calls) == 1
    assert processed.json()["timing"]["preview_seconds"] >= 0


def test_demo_renders_structured_evidence_and_downloads_the_same_order() -> None:
    app = create_app(StructuredDocumentReader())

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("structured.png", _page_png(), "image/png")},
        )
        assert processed.status_code == 200
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        html = client.get("/").text

    assert markdown.index("### Visit summary") < markdown.index("| Name | Value |")
    assert markdown.index("| Name | Value |") < markdown.index("- [x] Fall risk")
    assert "flattened fallback" not in markdown
    assert markdown.count("Fall risk") == 1
    assert "| Pulse | 72 |" in markdown
    assert "**Conflicting table cell evidence (row 2, column 2)**" in markdown
    assert "> Alternative evidence: 77" in markdown
    assert "**Conflicting handwriting evidence**" in markdown
    assert "> Alternative evidence: Return in 3 weeks" in markdown

    assert "function displayRegions(page)" in html
    assert 'document.createElement("table")' in html
    assert 'table.className = "document-table"' in html
    assert "element.rowSpan = rowSpan" in html
    assert "element.colSpan = columnSpan" in html
    assert "appendEvidenceState(element, resolution, cell.alternatives)" in html
    assert 'block.dataset.resolution = group.region.resolution || "resolved"' in html
    assert "function sameTextLine(first, second)" in html


def test_markdown_preserves_physical_rows_and_literal_table_characters() -> None:
    regions = [
        {
            "id": "table",
            "kind": "table",
            "text": "fallback",
            "reading_order": 1,
            "resolution": "resolved",
            "structure": {
                "role": "table",
                "row_count": 3,
                "column_count": 2,
                "cells": [
                    {"text": r"A\|B", "row_nums": [0], "column_nums": [0]},
                    {"text": "first", "row_nums": [1], "column_nums": [0]},
                    {
                        "text": "Subsection",
                        "row_nums": [2],
                        "column_nums": [0],
                        "projected_row_header": True,
                    },
                ],
            },
        }
    ]

    markdown = render_page_markdown(regions, ["table"])

    assert markdown.splitlines() == [
        r"| A\\\|B |  |",
        "| --- | --- |",
        "| first |  |",
        "| Subsection |  |",
    ]


def test_markdown_groups_word_evidence_into_readable_lines() -> None:
    regions = [
        {
            "id": "one",
            "kind": "word",
            "text": "local",
            "reading_order": 1,
            "resolution": "resolved",
            "bounding_box": {"left": 0, "top": 0, "right": 20, "bottom": 10},
        },
        {
            "id": "two",
            "kind": "word",
            "text": "evidence",
            "reading_order": 2,
            "resolution": "resolved",
            "bounding_box": {"left": 22, "top": 1, "right": 50, "bottom": 11},
        },
        {
            "id": "three",
            "kind": "word",
            "text": "renderer",
            "reading_order": 3,
            "resolution": "resolved",
            "bounding_box": {"left": 52, "top": 0, "right": 82, "bottom": 10},
        },
        {
            "id": "four",
            "kind": "word",
            "text": "path",
            "reading_order": 4,
            "resolution": "resolved",
            "bounding_box": {"left": 0, "top": 18, "right": 20, "bottom": 28},
        },
    ]

    markdown = render_page_markdown(regions, ["one", "two", "three", "four"])

    assert markdown == "local evidence renderer\n\npath"


def test_markdown_keeps_independent_columns_and_reverse_order_separate() -> None:
    regions = [
        {
            "id": "left",
            "kind": "word",
            "text": "Left",
            "reading_order": 1,
            "resolution": "resolved",
            "bounding_box": {"left": 0, "top": 0, "right": 20, "bottom": 10},
        },
        {
            "id": "right",
            "kind": "word",
            "text": "Right",
            "reading_order": 2,
            "resolution": "resolved",
            "bounding_box": {
                "left": 120,
                "top": 0,
                "right": 150,
                "bottom": 10,
            },
        },
        {
            "id": "reverse",
            "kind": "word",
            "text": "Reverse",
            "reading_order": 3,
            "resolution": "resolved",
            "bounding_box": {"left": 0, "top": 0, "right": 35, "bottom": 10},
        },
    ]

    markdown = render_page_markdown(regions, ["left", "right", "reverse"])

    assert markdown == "Left\n\nRight\n\nReverse"


def test_browser_line_grouping_uses_the_same_horizontal_bounds() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        html = client.get("/").text

    assert f"const INLINE_MAX_GAP_HEIGHTS = {INLINE_MAX_GAP_HEIGHTS};" in html
    assert "const horizontalGap = secondBox.left - firstBox.right;" in html
    assert "if (secondBox.left < firstBox.left) return false;" in html
    assert (
        "horizontalGap >= -height && horizontalGap <= height * INLINE_MAX_GAP_HEIGHTS"
    ) in html


def test_demo_returns_page_failure_without_hiding_other_pages() -> None:
    app = create_app(SecondPageFailureReader())

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("failure.tif", _two_page_tiff(), "image/tiff")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["result"]["status"] == "partial"
        assert len(payload["page_images"]) == 2
        assert payload["result"]["pages"][0]["text"]["value"] == "Controlled page 1"
        assert payload["result"]["pages"][1]["route"] == "review"
        assert payload["result"]["pages"][1]["regions"] == []
        assert payload["result"]["failures"] == [
            {
                "id": "failure-1",
                "stage": "ocr",
                "code": "controlled_failure",
                "message": "Second page OCR failed",
                "page_number": 2,
            }
        ]
        assert payload["uncertainty"]["pages"][1] == {
            "page_number": 2,
            "review_required": True,
            "mean_primary_confidence": None,
            "primary_regions": 0,
            "confidence_regions": 0,
            "unresolved_evidence": 0,
            "conflicting_evidence": 0,
            "disagreeing_alternatives": 0,
            "failure_count": 1,
            "risk_reasons": [],
            "category_counts": {},
        }

        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        assert "`ocr/controlled_failure` on page 2" in markdown
        assert "Second page OCR failed" in markdown


def test_demo_reports_evidence_uncertainty_without_changing_result_schema() -> None:
    app = create_app(ReviewEvidenceReader())

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert "uncertainty" not in payload["result"]
        assert payload["uncertainty"] == {
            "interpretation": (
                "Evidence diagnostics for review routing, not calibrated probabilities."
            ),
            "pages": [
                {
                    "page_number": 1,
                    "review_required": True,
                    "mean_primary_confidence": 0.925,
                    "primary_regions": 3,
                    "confidence_regions": 2,
                    "unresolved_evidence": 1,
                    "conflicting_evidence": 2,
                    "disagreeing_alternatives": 2,
                    "failure_count": 0,
                    "risk_reasons": [
                        "crop_disagreement",
                        "small_text_evidence",
                        "low_mean_confidence",
                    ],
                    "category_counts": {
                        "checkbox": 1,
                        "coverage_risk": 1,
                        "form_field": 1,
                        "handwriting": 1,
                        "paragraph": 1,
                        "table": 1,
                    },
                }
            ],
        }


def test_demo_exposes_rejected_table_candidate_as_review_evidence() -> None:
    app = create_app(RejectedTableReader())

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("screen.png", _page_png(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    page = payload["result"]["pages"][0]
    uncertainty = payload["uncertainty"]["pages"][0]
    assert page["text"]["value"] == "Controlled page 1"
    assert page["route"] == "review"
    assert uncertainty["risk_reasons"] == ["rejected_table_candidate"]
    assert uncertainty["category_counts"] == {
        "paragraph": 1,
        "table_candidate": 1,
    }


def test_demo_shows_reason_for_orientation_routed_page() -> None:
    class RotatedPageReader:
        name = "rotated-page-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="rotated-text",
                    kind="paragraph",
                    text="Rotated page text",
                    confidence=0.95,
                    bounding_box=BoundingBox(5, 5, 40, 15),
                    reading_order=1,
                    provider=self.name,
                )
            ]

    reader = OrientationReader(
        RotatedPageReader(),
        orientation_detector=lambda _: {"angle": 90, "confidence": 0.99},
        osd_detector=lambda _: {"angle": 90, "confidence": 20.0},
    )
    app = create_app(reader)

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("rotated.png", _page_png(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    page = payload["uncertainty"]["pages"][0]
    assert payload["result"]["pages"][0]["route"] == "review"
    assert page["review_required"] is True
    assert page["risk_reasons"] == ["orientation_rotated_90_degrees"]


def test_demo_runs_injected_specialist_stages() -> None:
    app = create_app(ControlledReader(), stages=[ControlledStage()])

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["pipeline_stages"] == ["table-structure"]
        assert payload["result"]["pages"][0]["text"]["value"] == (
            "Controlled page 1 with table"
        )
        assert "stage.table-structure" in payload["timing"]["pipeline_steps"]


def test_demo_bounds_preparation_and_serializes_shared_reader_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SerializedReader(ControlledReader):
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.04)
                return super().read(image_path, page_number)
            finally:
                with self.lock:
                    self.active -= 1

    reader = SerializedReader()
    prepare_pages = demo_module._prepare_pages
    prepare_active = 0
    maximum_prepare_active = 0
    prepare_calls = 0
    prepare_state_lock = threading.Lock()

    def track_prepare(*args, **kwargs):
        nonlocal prepare_active, maximum_prepare_active, prepare_calls
        with prepare_state_lock:
            prepare_active += 1
            prepare_calls += 1
            maximum_prepare_active = max(maximum_prepare_active, prepare_active)
        try:
            time.sleep(0.04)
            return prepare_pages(*args, **kwargs)
        finally:
            with prepare_state_lock:
                prepare_active -= 1

    monkeypatch.setattr(demo_module, "_prepare_pages", track_prepare)
    app = create_app(reader)

    with TestClient(app) as client:

        def post_page(index: int) -> Any:
            return client.post(
                "/api/process",
                files={
                    "file": (
                        f"page-{index}.png",
                        _page_png(),
                        "image/png",
                    )
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(post_page, range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    assert prepare_calls == 2
    assert maximum_prepare_active == 1
    assert reader.maximum_active == 1
    assert all(
        response.json()["timing"]["queue_seconds"] >= 0 for response in responses
    )
    assert all(
        response.json()["timing"]["preview_seconds"] >= 0 for response in responses
    )
    assert (
        max(
            response.json()["timing"]["preview_queue_seconds"] for response in responses
        )
        >= 0.02
    )


@pytest.mark.skipif(
    not shutil.which("pdfinfo") or not shutil.which("pdftoppm"),
    reason="Poppler is required for PDF resource-bound testing",
)
def test_demo_rejects_pdf_above_page_limit_before_ocr() -> None:
    class CountingReader(ControlledReader):
        def __init__(self) -> None:
            self.calls = 0

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            self.calls += 1
            return super().read(image_path, page_number)

    reader = CountingReader()
    app = create_app(reader, max_pages=1)

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("two-pages.pdf", _two_page_pdf(), "application/pdf")},
        )
        session_entries = list(app.state.session_root.iterdir())

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Document has 2 pages; the limit is 1",
    }
    assert reader.calls == 0
    assert session_entries == []


@pytest.mark.skipif(
    not shutil.which("pdfinfo") or not shutil.which("pdftoppm"),
    reason="Poppler is required for PDF resource-bound testing",
)
def test_demo_rejects_pdf_above_decoded_pixel_limit_before_ocr() -> None:
    reader = ControlledReader()
    app = create_app(reader, max_decoded_pixels=160_000)

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("page.pdf", _one_page_pdf(), "application/pdf")},
        )
        session_entries = list(app.state.session_root.iterdir())

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail.startswith("Document decodes to ")
    assert detail.endswith(" pixels; the limit is 160000")
    assert session_entries == []


def test_demo_rejects_image_above_decoded_pixel_limit_before_ocr() -> None:
    class CountingReader(ControlledReader):
        def __init__(self) -> None:
            self.calls = 0

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            self.calls += 1
            return super().read(image_path, page_number)

    reader = CountingReader()
    app = create_app(reader, max_decoded_pixels=9_000)

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("large.png", _page_png(), "image/png")},
        )
        session_entries = list(app.state.session_root.iterdir())

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Document decodes to 9600 pixels; the limit is 9000",
    }
    assert reader.calls == 0
    assert session_entries == []


def test_demo_rejects_invalid_empty_and_oversized_uploads() -> None:
    app = create_app(ControlledReader(), max_upload_bytes=4)

    with TestClient(app) as client:
        unsupported = client.post(
            "/api/process",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert unsupported.status_code == 415
        assert unsupported.json()["detail"].startswith("Unsupported file type")

        empty = client.post(
            "/api/process",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert empty.status_code == 400
        assert empty.json() == {"detail": "The uploaded file is empty"}

        oversized = client.post(
            "/api/process",
            files={"file": ("large.png", b"12345", "image/png")},
        )
        assert oversized.status_code == 413
        assert "Upload exceeds" in oversized.json()["detail"]
        assert list(app.state.session_root.iterdir()) == []


def test_demo_bounds_request_before_multipart_parsing() -> None:
    class CountingReader(ControlledReader):
        def __init__(self) -> None:
            self.calls = 0

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            self.calls += 1
            return super().read(image_path, page_number)

    reader = CountingReader()
    app = create_app(reader, max_upload_bytes=8)
    boundary = "bounded-upload"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="large.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    def body_chunks() -> Any:
        yield prefix
        for _ in range(18):
            yield b"x" * 4096
        yield suffix

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            content=body_chunks(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        session_entries = list(app.state.session_root.iterdir())

    assert response.status_code == 413
    assert "transfer limit" in response.json()["detail"]
    assert reader.calls == 0
    assert session_entries == []


def _two_page_tiff() -> bytes:
    first = Image.new("RGB", (120, 80), (30, 30, 30))
    second = Image.new("RGB", (120, 80), (220, 220, 220))
    output = io.BytesIO()
    first.save(output, format="TIFF", save_all=True, append_images=[second])
    return output.getvalue()


def _two_page_pdf() -> bytes:
    first = Image.new("RGB", (120, 80), (30, 30, 30))
    second = Image.new("RGB", (120, 80), (220, 220, 220))
    output = io.BytesIO()
    first.save(output, format="PDF", save_all=True, append_images=[second])
    return output.getvalue()


def _one_page_pdf() -> bytes:
    image = Image.new("RGB", (120, 80), (30, 30, 30))
    output = io.BytesIO()
    image.save(output, format="PDF")
    return output.getvalue()


def _page_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(output, format="PNG")
    return output.getvalue()
