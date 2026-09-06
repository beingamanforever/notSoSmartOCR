from __future__ import annotations

import io
import json
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import ocr_pipeline.demo as demo_module
from ocr_pipeline.contracts import BoundingBox, PageResult, TextAlternative, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.evidence_layout import EvidenceLayoutStage
from ocr_pipeline.orientation import OrientationReader
from ocr_pipeline.preprocessing import PageFrameReader
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


class PresentationReader:
    name = "local-page-presenter"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            width, height = image.size
        return [
            TextRegion(
                id=f"page-{page_number}-presentation",
                kind="page_text",
                text="# Visit summary\n\n| Field | Value |\n| --- | --- |\n| Name | Ana |",
                confidence=None,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
                text_provenance={"model": {"id": "local-test-model"}},
            )
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


def test_default_demo_reports_reduced_composition_without_banner() -> None:
    app = create_app()

    with TestClient(app) as client:
        index = client.get("/")
        composition = client.get("/api/composition").json()

    assert 'id="composition-banner"' not in index.text
    assert composition["id"] == "reduced-tesseract-workbench"
    assert composition["scope"] == "reduced"
    assert composition["primary_ocr"] == "tesseract-routed"
    assert composition["orientation"] == "not configured"
    assert composition["tesseract_roles"] == ["primary OCR"]
    assert composition["build_label"] == "local-workbench"
    assert composition["warmup_completed"] is False
    assert composition["service_started_at"].endswith("+00:00")


def test_workbench_reports_orientation_from_wrapped_reader() -> None:
    reader = PageFrameReader(
        OrientationReader(
            ControlledReader(),
            osd_detector=lambda _: {"angle": 0, "confidence": 0.0},
        )
    )
    app = create_app(reader)

    with TestClient(app) as client:
        composition = client.get("/api/composition").json()

    assert composition["id"] == "custom-local-workbench"
    assert composition["orientation"] == "configured via OrientationReader"
    assert composition["tesseract_roles"] == []


def test_demo_processes_multi_page_tiff_and_clears_session() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["cache-control"] == "no-store"
        assert "Not So Smart OCR" in index.text
        assert 'id="page-canvas"' in index.text
        assert 'id="copy-markdown"' in index.text
        assert 'canvas.addEventListener("click"' in index.text
        assert 'id="panel-rendered"' in index.text
        assert 'id="panel-visual"' in index.text
        assert 'id="panel-markdown"' in index.text
        source_start = index.text.index('<article class="card source-card"')
        output_start = index.text.index('<article class="card output-card"')
        visual_start = index.text.index('id="panel-visual"')
        visual_end = index.text.index('id="panel-markdown"')
        assert source_start < index.text.index('id="page-canvas"') < output_start
        assert 'id="page-canvas"' not in index.text[visual_start:visual_end]
        assert 'id="region-categories"' not in index.text[visual_start:visual_end]
        assert 'id="composition-banner"' not in index.text
        assert 'aria-live="polite"' in index.text
        assert "__OCR_COMPOSITION_JSON__" not in index.text

        composition = client.get("/api/composition")
        assert composition.status_code == 200
        assert composition.json() == {
            "id": "custom-local-workbench",
            "label": "Custom local workbench",
            "scope": "custom",
            "primary_ocr": "controlled-reader",
            "orientation": "not configured",
            "tesseract_roles": [],
            "stages": [],
            "handwriting": "not configured",
            "note": (
                "Configuration only. This is not the full verified GPU pipeline; "
                "readiness and execution are reported only after processing."
            ),
            "build_label": "local-workbench",
            "service_started_at": composition.json()["service_started_at"],
            "warmup_completed": False,
        }

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
        assert payload["composition"] == composition.json()
        assert payload["pipeline_stages"] == []
        assert payload["stage_execution"] == []
        assert [run["page_number"] for run in payload["page_execution"]] == [1, 2]
        assert all(run["execution_seconds"] >= 0 for run in payload["page_execution"])
        assert all(run["queue_seconds"] >= 0 for run in payload["page_execution"])
        assert all(run["batched_reader"] is False for run in payload["page_execution"])
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
        assert example.headers["content-disposition"].startswith("inline;")
        for example_name in (
            "contract-amendment",
            "academic-paper",
            "code",
            "financial-table",
            "scanned-form",
        ):
            gallery_image = client.get(f"/api/examples/{example_name}")
            assert gallery_image.status_code == 200
            assert gallery_image.headers["content-type"] == "image/png"
            assert gallery_image.headers["content-disposition"].startswith("inline;")
            with Image.open(io.BytesIO(gallery_image.content)) as image:
                image.verify()
        handwriting_image = client.get("/api/examples/handwriting")
        assert handwriting_image.status_code == 200
        assert handwriting_image.headers["content-type"] == "image/png"
        assert handwriting_image.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert client.get("/api/examples/../../AGENTS.md").status_code == 404
        assert client.get("/api/examples/private-case").status_code == 404

        assert "state.imageRequest += 1" in index.text
        assert 'setEmptyContent("markdown-source"' in index.text
        assert 'kind: "table_cell"' in index.text
        assert "function currentOverlays()" in index.text
        assert "boxArea(left.region.bounding_box)" in index.text
        assert 'id="region-order"' in index.text
        assert 'id="region-alternatives"' in index.text
        assert 'id="region-structure"' in index.text
        assert 'id="risk-card"' not in index.text
        assert 'id="copy-json"' in index.text
        assert 'id="category-list"' in index.text
        assert 'id="timing-card"' in index.text
        assert "function renderUncertainty()" not in index.text
        assert "function renderCategories()" in index.text
        assert (
            "function renderTiming(timing, pageExecution = [], stageExecution = [])"
            in index.text
        )
        assert 'state.hiddenKinds.add("text")' in index.text
        assert 'state.hiddenKinds.add("table_candidate")' in index.text
        assert "function normalPreviewRegion(region)" in index.text
        assert '["table_candidate", "coverage_risk", "layout_block"]' in index.text
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
    assert response.headers["content-encoding"] == "gzip"
    html = response.text
    assert html.count('class="tab" role="tab"') == 3
    assert ">Readable draft</button>" not in html
    assert html.count("Not So Smart OCR") == 2
    assert "!SoSmartOCR" not in html
    assert html.count('class="example-button" type="button" data-example=') == 6
    assert 'data-example="contract-agreement"' in html
    assert 'data-example="contract-amendment"' in html
    assert 'data-example="academic-paper"' in html
    assert 'data-example="code"' in html
    assert 'data-example="financial-table"' in html
    assert 'data-example="scanned-form"' in html
    assert "Handwritten notes" not in html
    assert "Local evidence viewer" not in html
    assert "Local only" not in html
    assert "25 MB max" not in html
    assert "Total time taken" in html
    assert html.count('class="example-preview"') == 6
    assert "Table-cell comparison" not in html
    handle_example_start = html.index("async function handleExample(button)")
    example_fetch = html.index("await fetch", handle_example_start)
    assert (
        html.index("state.selectedFile = null;", handle_example_start) < example_fetch
    )
    assert (
        html.index('byId("parse-button").disabled = true;', handle_example_start)
        < example_fetch
    )
    assert 'button.setAttribute("aria-busy", "true")' in html
    assert 'const extension = blob.type === "image/png" ? ".png" : ".pdf";' in html
    assert "renderUncertainty();" not in html
    assert 'class="inspector empty"' in html
    assert 'class="tab-panel active empty-panel"' in html
    assert "Ready to extract" in html
    assert ">Text output</button>" in html
    assert ">Visual</button>" in html
    assert ">Confidence</button>" in html
    assert "confidence-summary" not in html
    assert "High 90%+" not in html
    assert "Medium 70-89.99%" not in html
    assert "Low below 70%" not in html
    assert "Confidence is not correctness." in html
    assert "element.dataset.confidenceLevel = validConfidence(confidence)" in html
    assert (
        'if (value === null || value === undefined || value === "") return false;'
        in html
    )
    assert 'if (confidence >= 0.9) return "high";' in html
    assert 'if (confidence >= 0.7) return "medium";' in html
    assert "${target.dataset.confidenceKind}: ${percent} · Uncalibrated" in html
    assert 'provenance.method === "tesseract_tsv"' in html
    assert "line or region" in html
    assert "function markdownUrl()" in html
    assert "function renderReview(payload)" in html
    assert 'run.status === "failed"' in html
    assert "page.review_required && !notedPages.has(page.page_number)" in html
    assert "const markdown = await response.text();" in html
    assert 'list.className = "markdown-lines";' in html
    assert 'const item = document.createElement("li");' in html
    assert 'aria-controls="panel-markdown" id="tab-markdown"' in html

    assert 'id="client-timer" data-state="idle"' in html
    assert 'id="client-elapsed">0.0 s' in html
    assert "function startClientTimer()" in html
    assert "function stopClientTimer(outcome)" in html
    assert 'let timerOutcome = "error"' in html
    assert 'timerOutcome = "complete"' in html
    assert html.count("if (resultRequest === state.resultRequest) {") == 2
    assert html.index("startClientTimer();") < html.index("resetResult();")
    assert html.index("startClientTimer();") < html.index(
        'fetch("/api/process", { method: "POST", body })'
    )
    assert "stopClientTimer(timerOutcome);" in html
    assert "Pipeline execution" in html
    assert '"Backend pipeline"' not in html
    assert '["Preview queue", timing.preview_queue_seconds]' in html
    assert '["Preview processing", timing.preview_seconds]' in html
    assert '["OCR queue", timing.queue_seconds]' in html
    assert '["OCR pipeline", timing.pipeline_seconds]' in html
    assert (
        "function renderTiming(timing, pageExecution = [], stageExecution = [])" in html
    )
    assert "`Page ${run.page_number} total`" in html
    assert "function stageExecutionDetail(run)" in html
    assert ">Regions</button>" not in html

    assert 'id="copy-text" type="button" data-copy-state="idle"' in html
    assert 'id="copy-markdown" type="button" data-copy-state="idle"' in html
    assert 'id="copy-json" type="button" data-copy-state="idle"' in html
    assert ">Copy</summary>" in html
    assert ">Download</summary>" in html
    assert 'id="download-markdown" aria-disabled="true"' in html
    assert 'id="download-json" aria-disabled="true"' in html
    assert ">Result JSON</a>" in html
    assert 'id="copy-status" role="status" aria-live="polite"' in html
    assert html.count("JSON.stringify(state.response, null, 2)") == 1
    assert "navigator.clipboard.writeText(content)" in html
    assert "function copyTextWithTextarea(content)" in html
    assert 'document.execCommand("copy")' in html
    assert 'showCopyStatus(button, successMessage, "copied")' in html
    assert 'showCopyStatus(button, "Copy failed", "error")' in html
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
    assert "`/api/sessions/${sessionId}/${path}`" in html
    assert "revision," in html
    assert "request_id: requestId" in html
    assert 'payload.recovery_outcome?.status === "stale"' in html
    assert "function applyRevisionResponse(payload)" in html
    assert "function recoveryStatusMessage(payload, fallback)" in html
    assert "Alternative accepted as the canonical revision." in html
    assert "A candidate was added; review is still required." in html
    assert "state.response = payload;" in html
    assert 'id="review-actions"' in html
    assert 'id="accept-alternative"' in html
    assert 'id="keep-unresolved"' in html
    assert 'id="layers-control"' in html
    assert 'window.addEventListener("pagehide", cancelRecovery, { once: true })' in html
    assert "state.recoveryController?.abort();" in html
    assert "if (!target) {\n        clearInteraction();" in html
    assert "state.renderedMarkdownRevision === revision" in html
    assert "state.response?.session_id !== sessionId" in html
    assert 'byId("region-edit-text").value = "";' in html
    assert "scrollIntoView" not in html

    assert 'form.setAttribute("aria-busy", String(busy))' in html
    assert 'parseButton.textContent = busy ? "Parsing..." : "Parse"' in html
    assert "function activateTab(tab)" in html
    assert "dismissConfidenceTooltip();" in html
    assert "panel.hidden = !active" in html
    assert 'id="panel-visual" aria-labelledby="tab-visual" hidden' in html
    assert 'id="panel-markdown" aria-labelledby="tab-markdown" hidden' in html
    assert 'if (tab.id === "tab-markdown") void loadRenderedMarkdown();' in html
    assert "function documentMarkdownLines(markdown)" in html
    assert "return lines;" in html
    assert ">Technical details</summary>" in html
    assert 'id="region-crop" aria-label="Selected source crop" hidden' in html
    assert 'state.response?.composition?.handwriting === "configured"' in html
    assert "function alignedAlternatives(region)" in html
    assert "function clearRegionSelection()" in html
    assert '["table", "figure", "control"].includes(semanticKind(region))' in html
    assert ">Region JSON</button>" not in html
    assert ">Processed</button>" not in html
    assert ">Failures</button>" not in html
    assert 'state.hiddenKinds.add("text")' in html
    assert 'state.hiddenKinds.add("line")' in html
    assert 'state.hiddenKinds.add("table_cell")' in html
    assert 'state.hiddenKinds.add("table")' not in html
    assert "state.hiddenKinds.delete(categoryKey(source))" not in html
    assert "state.selectedRegionKey = regionKey(source);" in html
    assert 'if (kind === "table_cell") return kind;' in html
    assert 'if (kind === "table") return "table bounds";' in html
    assert 'if (kind === "table_cell") return "table cell bounds";' in html
    assert "const label = selected ? `${index + 1} ${kind}` : kind;" in html
    assert 'if (role === "layout_block") return true;' in html
    assert "function visualTabActive()" in html

    assert "Evidence diagnostics" not in html
    assert 'id="metadata"' not in html
    assert 'region?.structure?.role || ""' in html
    assert "block.dataset.sourceKind = group.sourceKind" in html
    assert "function confidenceScope(region)" in html
    assert "function decoratePresentationSource(element, block, source)" in html
    assert "rendering.source_confidence_scope" in html
    assert "function evidencePageNumber(target)" in html
    assert "function renderedLiteral(value, resolution)" in html
    assert "appendEvidenceState" not in html


def test_confidence_review_uses_reported_values_and_downloaded_markdown() -> None:
    class ConfidenceReader:
        name = "confidence-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id=f"page-{page_number}-high",
                    kind="word",
                    text="High",
                    confidence=0.95,
                    bounding_box=BoundingBox(5, 5, 30, 20),
                    reading_order=1,
                    provider=self.name,
                ),
                TextRegion(
                    id=f"page-{page_number}-medium",
                    kind="word",
                    text="Medium",
                    confidence=0.8,
                    bounding_box=BoundingBox(35, 5, 70, 20),
                    reading_order=2,
                    provider=self.name,
                ),
                TextRegion(
                    id=f"page-{page_number}-low",
                    kind="word",
                    text="Low",
                    confidence=0.4,
                    bounding_box=BoundingBox(75, 5, 95, 20),
                    reading_order=3,
                    provider=self.name,
                ),
                TextRegion(
                    id=f"page-{page_number}-unreported",
                    kind="word",
                    text="Unreported",
                    confidence=None,
                    bounding_box=BoundingBox(5, 30, 60, 45),
                    reading_order=4,
                    provider=self.name,
                ),
            ]

    app = create_app(ConfidenceReader())

    with TestClient(app) as client:
        index = client.get("/")
        processed = client.post(
            "/api/process",
            files={"file": ("confidence.png", _page_png(), "image/png")},
        )
        payload = processed.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    assert index.status_code == 200
    assert processed.status_code == 200
    assert [
        region["confidence"] for region in payload["result"]["pages"][0]["regions"]
    ] == [0.95, 0.8, 0.4, None]
    assert "High Medium Low" in markdown
    assert "Unreported" in markdown
    assert (
        "function decorateConfidence(element, confidence, inline = false, evidence = {})"
        in index.text
    )
    assert (
        "element.dataset.confidencePercent = validConfidence(confidence)" in index.text
    )
    assert "function markdownUrl()" in index.text


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
    assert payload["stage_execution"][0]["stage"] == "tables"
    assert payload["stage_execution"][0]["status"] == "productive"
    assert len(tables) == 1
    assert tables[0]["structure"]["row_count"] == 5
    assert tables[0]["structure"]["column_count"] == 4
    assert len(tables[0]["structure"]["cells"]) == 20
    assert payload["result"]["pages"][0]["text"]["evidence_ids"] == [tables[0]["id"]]
    table_lines = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert len(table_lines) == 6
    assert table_lines[1].count("---") == 4


def test_public_raster_examples_process_end_to_end() -> None:
    class PublicExampleReader(ControlledReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id=f"page-{page_number}-region-1",
                    kind="paragraph",
                    text="Public example",
                    confidence=0.91,
                    bounding_box=BoundingBox(1, 1, 20, 20),
                    reading_order=1,
                    provider=self.name,
                )
            ]

    app = create_app(PublicExampleReader())

    with TestClient(app) as client:
        for example_name in ("contract-amendment", "academic-paper"):
            example = client.get(f"/api/examples/{example_name}")
            assert example.status_code == 200
            processed = client.post(
                "/api/process",
                files={"file": (f"{example_name}.png", example.content, "image/png")},
            )
            assert processed.status_code == 200
            assert processed.json()["result"]["status"] == "success"


def test_documented_cpu_entrypoint_processes_bundled_complex_examples_as_reduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo_module.shutil, "which", lambda _: "/usr/bin/tesseract")
    app = create_app()

    with TestClient(app) as client:
        for example_name in ("financial-table", "handwriting"):
            example = client.get(f"/api/examples/{example_name}")
            assert example.status_code == 200
            with Image.open(io.BytesIO(example.content)) as image:
                image.verify()
            processed = client.post(
                "/api/process",
                files={
                    "file": (
                        f"{example_name}.png",
                        example.content,
                        "image/png",
                    )
                },
            )
            assert processed.status_code == 200
            payload = processed.json()
            assert payload["composition"]["id"] == "reduced-tesseract-workbench"
            assert payload["composition"]["handwriting"] == "not configured"
            assert payload["pipeline_stages"] == []


def test_demo_uses_neutral_navy_and_blue_visual_roles() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        html = client.get("/").text

    for declaration in (
        "--background: #f6f7f9",
        "--panel: #ffffff",
        "--panel-raised: #f3f4f6",
        "--text: #172033",
        "--muted: #667085",
        "--accent: #2563eb",
        "--accent-soft: #eff6ff",
    ):
        assert declaration in html
    assert "--danger:" not in html
    assert "--success:" not in html
    assert ".status-line.busy, .status-line.error { color: var(--accent);" in html
    assert (
        ".status-line.success { color: var(--text); border-color: var(--line-strong);"
        in html
    )
    assert 'title: "#1d4ed8"' in html
    assert 'paragraph: "#a21caf"' in html
    assert 'figure: "#047857"' in html
    assert 'control: "#c2410c"' in html
    assert 'const actionColor = rootStyle.getPropertyValue("--accent").trim()' in html
    assert (
        "return regionColors[categoryKey(regionOrKind)] || regionColors.text;" in html
    )
    assert "context.fillRect(box.left, box.top" not in html
    assert "context.strokeRect(box.left, box.top" in html


def test_form_rows_disclose_only_unresolved_handwriting_values() -> None:
    html = TestClient(create_app(ControlledReader())).get("/").text
    start = html.index("function renderFormRow")
    end = html.index("function canonicalLayoutText", start)
    renderer = html[start:end]

    assert "segment.label_evidence_ids || []" in renderer
    assert 'semanticKind(source) === "handwriting"' in renderer
    assert 'document.createTextNode(" [unreadable handwriting]")' in renderer


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
    assert by_element_id["rendered-toolbar"][1]["hidden"] is None
    assert by_element_id["confidence-toggle"][1]["aria-pressed"] == "false"
    assert (
        by_element_id["confidence-toggle"][1]["aria-label"]
        == "Turn provider confidence highlights on"
    )
    assert "confidenceReview: false" in response.text
    assert "state.confidenceReview = false;" in response.text
    assert (
        'element.dataset.confidenceScope = evidence.scope || "region";' in response.text
    )

    tabs = [attrs for _, attrs in elements if attrs.get("role") == "tab"]
    panels = {
        attrs["id"]: attrs for _, attrs in elements if attrs.get("role") == "tabpanel"
    }
    assert [tab["id"] for tab in tabs] == [
        "tab-rendered",
        "tab-visual",
        "tab-markdown",
    ]
    assert len(panels) == len(tabs)
    assert [tab["id"] for tab in tabs if tab["aria-selected"] == "true"] == [
        "tab-rendered"
    ]
    for tab in tabs:
        panel = panels[tab["aria-controls"]]
        assert panel["aria-labelledby"] == tab["id"]
        assert ("hidden" in panel) is (tab["aria-selected"] == "false")

    for button_id in ("copy-text", "copy-markdown", "copy-json"):
        tag, attrs = by_element_id[button_id]
        assert tag == "button"
        assert "disabled" in attrs

    assert "@media (max-width: 900px)" in response.text
    assert (
        "grid-template-columns: minmax(0, .95fr) minmax(420px, 1.05fr);"
        in response.text
    )
    assert ".workspace { grid-template-columns: 1fr; }" in response.text


def test_demo_keeps_uncertainty_styling_out_of_rendered_controls() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        html = client.get("/").text

    assert ".semantic-block.control {" in html
    assert "border-left: 3px solid var(--line-strong)" in html
    assert '.semantic-block.control[data-resolution="unreadable"],' not in html
    assert '.semantic-block.control[data-resolution="conflicting"] {' not in html
    assert "box-shadow: inset 3px 0 var(--accent);" not in html
    assert ".evidence-state" not in html
    assert ".semantic-block.control, .semantic-block.handwriting" not in html


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
    assert "| Pulse |  |" in markdown
    assert "Conflicting" not in markdown
    assert "Alternative evidence" not in markdown
    assert "Return in 2 weeks" not in markdown
    handwriting = next(
        region
        for region in payload["result"]["pages"][0]["regions"]
        if region["id"] == "handwriting"
    )
    assert handwriting["text"] == "Return in 2 weeks"
    assert handwriting["alternatives"][0]["text"] == "Return in 3 weeks"

    assert "function displayRegions(page)" in html
    assert 'document.createElement("table")' in html
    assert 'table.className = "document-table"' in html
    assert "element.rowSpan = rowSpan" in html
    assert "element.colSpan = columnSpan" in html
    assert "appendEvidenceState" not in html
    assert 'unreadable: "no reading recovered"' not in html
    assert 'conflicting: "conflicting readings"' not in html
    assert (
        'block.dataset.resolution = group.region.resolution || "resolved"' not in html
    )
    assert "function sameTextLine(first, second)" in html


def test_rendered_markdown_keeps_selected_literals_and_blanks_empty_uncertainty() -> (
    None
):
    regions = [
        {
            "id": "field",
            "kind": "form_field",
            "text": "state uncertain",
            "reading_order": 1,
            "resolution": "unreadable",
            "structure": {"role": "field", "label": "Name"},
        },
        {
            "id": "table",
            "kind": "table",
            "text": "fallback",
            "reading_order": 2,
            "resolution": "resolved",
            "structure": {
                "role": "table",
                "row_count": 1,
                "column_count": 2,
                "cells": [
                    {
                        "id": "selected",
                        "text": "72",
                        "row_nums": [0],
                        "column_nums": [0],
                        "resolution": "conflicting",
                        "alternatives": [{"text": "77"}],
                    },
                    {
                        "id": "empty",
                        "text": "unreadable evidence",
                        "row_nums": [0],
                        "column_nums": [1],
                        "resolution": "unreadable",
                    },
                ],
            },
        },
    ]

    cleaned = demo_module._clean_render_regions(regions)
    markdown = render_page_markdown(cleaned, ["field", "table"])

    assert "72" not in markdown
    assert "77" not in markdown
    assert "uncertain" not in markdown.casefold()
    assert "unreadable" not in markdown.casefold()
    assert cleaned[0]["text"] == ""
    assert cleaned[1]["structure"]["cells"][1]["text"] == ""


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
                    {"text": "", "row_nums": [0], "column_nums": [1]},
                    {"text": "first", "row_nums": [1], "column_nums": [0]},
                    {"text": "", "row_nums": [1], "column_nums": [1]},
                    {
                        "text": "Subsection",
                        "row_nums": [2],
                        "column_nums": [0],
                        "projected_row_header": True,
                    },
                    {"text": "", "row_nums": [2], "column_nums": [1]},
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
                    "primary_regions": 5,
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


def test_demo_adds_review_only_local_page_presentation_without_replacing_evidence() -> (
    None
):
    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=PresentationReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        html = client.get("/").text

    page = payload["result"]["pages"][0]
    assert page["text"]["value"] == (
        "Primary text Value [unreadable handwriting] [?] | Value |\n| --- |\n| 42 |"
    )
    assert "presentation" not in page
    assert payload["presentation"] == {
        "schema_version": 2,
        "pages": [
            {
                "page_number": 1,
                "status": "review_draft",
                "provider": "local-page-presenter",
                "blocks": [
                    {
                        "id": "page-1-presentation",
                        "reading_order": 1,
                        "category": "plain",
                        "bbox": {
                            "left": 0,
                            "top": 0,
                            "right": 120,
                            "bottom": 80,
                        },
                        "raw_text": (
                            "# Visit summary\n\n| Field | Value |\n"
                            "| --- | --- |\n| Name | Ana |"
                        ),
                        "provider": "local-page-presenter",
                        "provenance": {"model": {"id": "local-test-model"}},
                        "rendering": {
                            "status": "canonical_fallback",
                            "reason": "missing_source_region",
                        },
                        "validation": {
                            "status": "passed",
                            "failures": [],
                            "termination_observed": False,
                        },
                    }
                ],
                "validation": {
                    "status": "passed",
                    "failures": [],
                    "termination_observed": False,
                },
                "canonical_unchanged": True,
            }
        ],
    }
    assert payload["timing"]["pipeline_steps"]["stage.presentation"] >= 0
    assert "Primary text" in markdown
    assert "# Visit summary" not in markdown
    assert 'id="tab-presentation"' not in html
    assert 'id="panel-presentation"' not in html
    assert "function renderRendered(pages, presentations)" in html
    assert "function renderPresentationMarkdown(markdown)" in html
    assert "renderStructuredPresentation(page, presentation)" in html


def test_demo_does_not_run_page_presenter_on_clear_page() -> None:
    class UnexpectedPresentationReader(PresentationReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            raise AssertionError(
                "clear pages must not be sent to the presentation model"
            )

    app = create_app(
        ControlledReader(),
        presentation_reader=UnexpectedPresentationReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("clear.png", _page_png(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    page = payload["result"]["pages"][0]
    assert "presentation" not in page
    assert payload["presentation"] == {"schema_version": 2, "pages": []}


def test_demo_preserves_and_validates_category_routed_presentation_blocks() -> None:
    class ResolvedReviewEvidenceReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            handwriting = next(
                region for region in regions if region.id == "handwriting"
            )
            handwriting.resolution = "resolved"
            handwriting.text = "a / b"
            table = next(region for region in regions if region.id == "table")
            table.structure["cells"] = []  # type: ignore[index]
            return regions

    class StructuredPresentationReader:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="formula",
                    kind="page_text",
                    text=r"\frac{a}{b}",
                    confidence=None,
                    bounding_box=BoundingBox(5, 45, 115, 75),
                    reading_order=3,
                    provider=self.name,
                    text_provenance={
                        "generation": {"category": "formula"},
                        "model": {"id": "tiiuae/Falcon-OCR"},
                        "source_region_id": "handwriting",
                    },
                ),
                TextRegion(
                    id="title",
                    kind="page_text",
                    text="Primary text",
                    confidence=None,
                    bounding_box=BoundingBox(5, 2, 115, 15),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "title"},
                    text_provenance={"source_region_id": "paragraph"},
                ),
                TextRegion(
                    id="table",
                    kind="page_text",
                    text=(
                        "<table><thead><tr><th>Value</th></tr></thead>"
                        "<tbody><tr><td>■ 42</td></tr></tbody></table>"
                    ),
                    confidence=None,
                    bounding_box=BoundingBox(5, 16, 115, 44),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                ),
            ]

    app = create_app(
        ResolvedReviewEvidenceReader(),
        presentation_reader=StructuredPresentationReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        html = client.get("/").text
        markdown = client.get(
            f"/api/sessions/{response.json()['session_id']}/result.md"
        ).text

    assert response.status_code == 200
    payload = response.json()
    page = payload["result"]["pages"][0]
    presentation = payload["presentation"]["pages"][0]
    assert payload["presentation"]["schema_version"] == 2
    assert "presentation" not in page
    assert page["text"]["value"] == (
        "Primary text Value a / b [?] | Value |\n| --- |\n| 42 |"
    )
    assert presentation["canonical_unchanged"] is True
    assert presentation["validation"] == {
        "status": "passed",
        "failures": [],
        "termination_observed": False,
    }
    assert [block["id"] for block in presentation["blocks"]] == [
        "title",
        "table",
        "formula",
    ]
    assert [block["category"] for block in presentation["blocks"]] == [
        "title",
        "table",
        "formula",
    ]
    assert presentation["blocks"][1]["raw_text"].startswith("<table>")
    assert presentation["blocks"][2]["raw_text"] == r"\frac{a}{b}"
    assert [block["rendering"]["status"] for block in presentation["blocks"]] == [
        "selected",
        "selected",
        "selected",
    ]
    assert all(
        block["validation"]["termination_observed"] is False
        for block in presentation["blocks"]
    )
    assert "function renderSafePresentationTable(rawText)" in html
    assert 'new DOMParser().parseFromString(rawText, "text/html")' in html
    assert "clone.append(document.createTextNode(child.textContent))" in html
    assert "window.katex.render(rawText, formula" in html
    assert 'if (kind === "formula" && group.region.resolution === "resolved")' in html
    assert (
        '} else if (kind === "formula" && group.region.resolution === "resolved") {'
        in html
    )
    assert 'if (semanticKind(selectedRegion()) === "formula")' in html
    assert (
        'if (region.structure?.block_type === "formula") return region.text || "";'
        in html
    )
    assert "Local KaTeX is unavailable" in html
    assert "function usablePresentation(presentation)" in html
    assert "usableFalconPresentation" not in html
    assert "function renderStructuredPresentation(page, presentation)" in html
    assert "function renderPresentationFallback(regions)" in html
    assert "Presentation validation failed:" not in html
    assert "Rendered with validated Falcon-OCR review blocks" not in markdown
    assert markdown.count("Primary text") == 1
    assert "Primary text" in markdown
    assert r"\frac{a}{b}" in markdown
    assert "<td>■ 42</td>" in markdown
    assert "innerHTML" not in html


def test_demo_rejects_unrelated_structured_presentation_content() -> None:
    class ResolvedStructuredReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            next(
                region for region in regions if region.id == "paragraph"
            ).confidence = 0.8
            table = next(region for region in regions if region.id == "table")
            table.confidence = None
            table.structure["cells"] = []  # type: ignore[index]
            return regions

    class UnrelatedStructuredPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="table-output",
                    kind="page_text",
                    text="<table><tr><td>Ana</td></tr></table>",
                    confidence=None,
                    bounding_box=BoundingBox(52, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                ),
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text=r"\frac{x}{y}",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 50, 10),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "paragraph"},
                ),
            ]

    app = create_app(
        ResolvedStructuredReader(),
        presentation_reader=UnrelatedStructuredPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    blocks = payload["presentation"]["pages"][0]["blocks"]
    assert all(block["rendering"]["reason"] == "validation_failed" for block in blocks)
    assert all(
        block["validation"]["failures"][0]["code"] == "source_evidence_omitted"
        for block in blocks
    )
    assert "Ana" not in markdown
    assert r"\frac{x}{y}" not in markdown


def test_demo_rejects_additive_structured_presentation_content() -> None:
    class ResolvedStructuredReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            formula = next(region for region in regions if region.id == "handwriting")
            formula.text = "a / b"
            formula.resolution = "resolved"
            table = next(region for region in regions if region.id == "table")
            table.structure["cells"] = []  # type: ignore[index]
            return regions

    class AdditiveStructuredPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="table-output",
                    kind="page_text",
                    text=(
                        "<table><tr><th>Value</th></tr><tr><td>42</td></tr>"
                        "<tr><td>Diagnosis</td><td>Flu</td></tr></table>"
                    ),
                    confidence=None,
                    bounding_box=BoundingBox(52, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                ),
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text=r"\frac{a}{b} + \frac{x}{y}",
                    confidence=None,
                    bounding_box=BoundingBox(0, 24, 50, 34),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "handwriting"},
                ),
            ]

    app = create_app(
        ResolvedStructuredReader(),
        presentation_reader=AdditiveStructuredPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    blocks = payload["presentation"]["pages"][0]["blocks"]
    assert all(block["rendering"]["reason"] == "validation_failed" for block in blocks)
    assert all(
        any(
            failure["code"] == "unsupported_generated_content"
            for failure in block["validation"]["failures"]
        )
        for block in blocks
    )
    assert "Diagnosis" not in markdown
    assert "Flu" not in markdown
    assert r"\frac{x}{y}" not in markdown


def test_demo_rejects_unsupported_ordinary_presentation_words() -> None:
    class AdditivePresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="paragraph-output",
                    kind="page_text",
                    text="Primary text invented",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 50, 10),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "paragraph"},
                )
            ]

    app = create_app(ReviewEvidenceReader(), presentation_reader=AdditivePresenter())
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["rendering"]["status"] == "canonical_fallback"
    assert "unsupported_generated_content" in {
        failure["code"] for failure in block["validation"]["failures"]
    }
    assert "Primary text" in markdown
    assert "invented" not in markdown


def test_demo_rejects_generated_content_for_empty_structured_evidence() -> None:
    class EmptyStructuredReader(ControlledReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="empty-table",
                    kind="table",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"role": "table", "cells": []},
                )
            ]

    class FabricatedTablePresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="table-output",
                    kind="page_text",
                    text=("<table><tr><td>Diagnosis</td><td>Flu</td></tr></table>"),
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "empty-table"},
                )
            ]

    app = create_app(
        EmptyStructuredReader(),
        presentation_reader=FabricatedTablePresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    block = payload["presentation"]["pages"][0]["blocks"][0]
    assert block["rendering"] == {
        "status": "canonical_fallback",
        "reason": "validation_failed",
        "source_region_id": "empty-table",
    }
    assert any(
        failure["code"] == "unsupported_generated_content"
        for failure in block["validation"]["failures"]
    )
    assert "Diagnosis" not in markdown
    assert "Flu" not in markdown


def test_demo_rejects_partial_structured_presentation_content() -> None:
    class StructuredReader(ControlledReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="table",
                    kind="table",
                    text="Name Ada Allergy Penicillin Active",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"role": "table", "cells": []},
                ),
                TextRegion(
                    id="formula",
                    kind="formula",
                    text="a b c d e",
                    confidence=None,
                    bounding_box=BoundingBox(0, 48, 110, 70),
                    reading_order=2,
                    provider=self.name,
                ),
                TextRegion(
                    id="risk",
                    kind="coverage_risk",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 80),
                    reading_order=3,
                    provider="deterministic-evidence-risk",
                    resolution="unreadable",
                    structure={
                        "role": "coverage_risk",
                        "reasons": ["low_mean_confidence"],
                    },
                ),
            ]

    class PartialStructuredPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="table-output",
                    kind="page_text",
                    text=(
                        "<table><tr><td>Name</td><td>Ada</td></tr>"
                        "<tr><td>Allergy</td><td>Active</td></tr></table>"
                    ),
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                ),
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text="a+b+c+d",
                    confidence=None,
                    bounding_box=BoundingBox(0, 48, 110, 70),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "formula"},
                ),
            ]

    app = create_app(
        StructuredReader(),
        presentation_reader=PartialStructuredPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    blocks = payload["presentation"]["pages"][0]["blocks"]
    assert all(block["rendering"]["reason"] == "validation_failed" for block in blocks)
    assert all(
        any(
            failure["code"] == "source_evidence_omitted"
            for failure in block["validation"]["failures"]
        )
        for block in blocks
    )
    assert "Name Ada Allergy Penicillin Active" in markdown
    assert "a b c d e" in markdown


@pytest.mark.parametrize(
    ("generated_table", "expected_status"),
    [
        (
            "<table><tr><td>Name</td><td>Ada</td></tr>"
            "<tr><td>Diagnosis</td><td>Flu</td></tr></table>",
            "selected",
        ),
        (
            "<table><tr><td>Name</td><td>Flu</td></tr>"
            "<tr><td>Diagnosis</td><td>Ada</td></tr></table>",
            "canonical_fallback",
        ),
    ],
)
def test_demo_preserves_table_field_associations(
    generated_table: str, expected_status: str
) -> None:
    class TableReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            table = next(region for region in regions if region.id == "table")
            table.text = "Name Ada Diagnosis Flu"
            table.structure["cells"] = []  # type: ignore[index]
            return regions

    class TablePresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="table-output",
                    kind="page_text",
                    text=generated_table,
                    confidence=None,
                    bounding_box=BoundingBox(52, 0, 110, 46),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                )
            ]

    app = create_app(
        TableReader(),
        presentation_reader=TablePresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    block = payload["presentation"]["pages"][0]["blocks"][0]
    assert block["rendering"]["status"] == expected_status
    if expected_status == "selected":
        assert "table_structure_changed" not in {
            failure["code"] for failure in block["validation"]["failures"]
        }
        assert "<td>Ada</td>" in markdown
        return
    assert any(
        failure["code"] == "table_structure_changed"
        for failure in block["validation"]["failures"]
    )
    assert "Name Ada Diagnosis Flu" in markdown


@pytest.mark.parametrize(
    "generated_formula", ["a * b", "a + b", "a - b", "b / a", "a / (b+c)"]
)
def test_demo_rejects_formula_semantic_changes(generated_formula: str) -> None:
    class ResolvedFormulaReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            formula = next(region for region in regions if region.id == "handwriting")
            formula.text = "a / b"
            formula.resolution = "resolved"
            return regions

    class ChangedFormulaPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text=generated_formula,
                    confidence=None,
                    bounding_box=BoundingBox(0, 24, 50, 34),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "handwriting"},
                )
            ]

    app = create_app(
        ResolvedFormulaReader(),
        presentation_reader=ChangedFormulaPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    block = payload["presentation"]["pages"][0]["blocks"][0]
    assert block["rendering"]["reason"] == "validation_failed"
    assert any(
        failure["code"] == "formula_semantics_changed"
        for failure in block["validation"]["failures"]
    )
    assert f"$$\n{generated_formula}\n$$" not in markdown


def test_demo_accepts_equivalent_fraction_presentation() -> None:
    class ResolvedFormulaReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            formula = next(region for region in regions if region.id == "handwriting")
            formula.text = "a / b"
            formula.resolution = "resolved"
            return regions

    class EquivalentFormulaPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text=r"\frac{a}{b}",
                    confidence=None,
                    bounding_box=BoundingBox(0, 24, 50, 34),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "handwriting"},
                )
            ]

    app = create_app(
        ResolvedFormulaReader(),
        presentation_reader=EquivalentFormulaPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    block = payload["presentation"]["pages"][0]["blocks"][0]
    assert block["rendering"]["status"] == "selected"
    assert "formula_semantics_changed" not in {
        failure["code"] for failure in block["validation"]["failures"]
    }
    assert r"\frac{a}{b}" in markdown


@pytest.mark.parametrize("generated_formula", ["a / (b", "a / [b"])
def test_demo_rejects_unclosed_formula_groups(generated_formula: str) -> None:
    class ResolvedFormulaReader(ReviewEvidenceReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            regions = super().read(image_path, page_number)
            formula = next(region for region in regions if region.id == "handwriting")
            formula.text = "a / (b)"
            formula.resolution = "resolved"
            return regions

    class MalformedFormulaPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text=generated_formula,
                    confidence=None,
                    bounding_box=BoundingBox(0, 24, 50, 34),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "handwriting"},
                )
            ]

    app = create_app(
        ResolvedFormulaReader(),
        presentation_reader=MalformedFormulaPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    block = payload["presentation"]["pages"][0]["blocks"][0]
    assert block["rendering"]["reason"] == "validation_failed"
    assert any(
        failure["code"] == "unbalanced_formula_groups"
        for failure in block["validation"]["failures"]
    )
    assert f"$$\n{generated_formula}\n$$" not in markdown


def test_demo_rejected_alternative_cannot_authorize_a_changed_dosage() -> None:
    class DosageReader(ControlledReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="dosage",
                    kind="paragraph",
                    text="Take 10 mg daily with food for seven days",
                    confidence=0.99,
                    bounding_box=BoundingBox(0, 0, 110, 20),
                    reading_order=1,
                    provider=self.name,
                    alternatives=[
                        TextAlternative("Take 100 mg daily", 0.8, "challenger")
                    ],
                )
            ]

    class DosagePresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="dosage-output",
                    kind="page_text",
                    text="Take 100 mg daily with food for seven days",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 110, 20),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "dosage"},
                )
            ]

    app = create_app(DosageReader(), presentation_reader=DosagePresenter())
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("dose.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["rendering"]["reason"] == "validation_failed"
    assert block["validation"]["failures"][0]["code"] == ("unsupported_critical_token")
    assert "Take 10 mg" in markdown
    assert "Take 100 mg" not in markdown


def test_demo_presentation_cannot_synthesize_checkbox_state() -> None:
    class ProseReader(ControlledReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="allergy",
                    kind="paragraph",
                    text="Allergy aspirin",
                    confidence=0.99,
                    bounding_box=BoundingBox(0, 0, 110, 20),
                    reading_order=1,
                    provider=self.name,
                ),
                TextRegion(
                    id="risk",
                    kind="coverage_risk",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 80),
                    reading_order=2,
                    provider="deterministic-evidence-risk",
                    resolution="unreadable",
                    structure={"role": "coverage_risk"},
                ),
            ]

    class ChecklistPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="allergy-output",
                    kind="page_text",
                    text="- [x] Allergy aspirin",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 110, 20),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "allergy"},
                )
            ]

    app = create_app(ProseReader(), presentation_reader=ChecklistPresenter())
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("allergy.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        html = client.get("/").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["rendering"]["reason"] == "validation_failed"
    assert block["validation"]["failures"][0]["code"] == ("unsupported_control_syntax")
    assert "[x]" not in markdown
    assert "☑" not in markdown
    assert "const control = line.match(/^-" not in html


def test_demo_keeps_generated_text_out_of_unresolved_evidence() -> None:
    class UnresolvedPresentationReader:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="generated-handwriting",
                    kind="page_text",
                    text="Metformin 500 mg",
                    confidence=None,
                    bounding_box=BoundingBox(0, 24, 50, 34),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "handwriting"},
                ),
                TextRegion(
                    id="generated-table",
                    kind="page_text",
                    text="<table><tr><td>43</td></tr></table>",
                    confidence=None,
                    bounding_box=BoundingBox(52, 0, 110, 46),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                ),
            ]

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=UnresolvedPresentationReader(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    assert response.status_code == 200
    assert [
        block["rendering"] for block in payload["presentation"]["pages"][0]["blocks"]
    ] == [
        {
            "status": "canonical_fallback",
            "reason": "source_evidence_unresolved",
            "source_region_id": "handwriting",
        },
        {
            "status": "canonical_fallback",
            "reason": "source_evidence_unresolved",
            "source_region_id": "table",
        },
    ]
    assert "Metformin 500 mg" not in markdown
    assert "<td>43</td>" not in markdown
    assert payload["presentation"]["pages"][0]["blocks"][0]["raw_text"] == (
        "Metformin 500 mg"
    )


def test_demo_selects_verified_image_grounded_table_recovery() -> None:
    class VerifiedFalconPresenter:
        name = "falcon-presentation"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="generated-table",
                    kind="table",
                    text=(
                        "<table><thead><tr><th>Value</th></tr></thead>"
                        "<tbody><tr><td>42</td></tr></tbody></table>"
                    ),
                    confidence=None,
                    bounding_box=BoundingBox(52, 0, 110, 46),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={
                        "method": "falcon_core_category_crop_generation",
                        "source_region_id": "table",
                        "model": {
                            "id": "tiiuae/Falcon-OCR",
                            "loaded_from": "/models/falcon-ocr",
                            "revision": ("42ec56b72a23984ac059e7c8a6d397a8529423fe"),
                            "origin": "TII, UAE",
                            "license": "Apache-2.0",
                            "identity_verified": True,
                            "local_files_only": True,
                        },
                    },
                )
            ]

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=VerifiedFalconPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["rendering"] == {
        "status": "selected",
        "source_region_id": "table",
        "source_page_number": 1,
        "source_bounding_box": {"left": 52, "top": 0, "right": 110, "bottom": 46},
        "source_evidence_ids": ["table"],
        "source_confidence": 0.95,
        "source_confidence_kind": "Recognition score",
        "source_confidence_scope": "source region",
    }
    assert "<td>42</td>" in markdown


def test_demo_links_selected_formula_presentation_to_source_evidence() -> None:
    class FormulaReader:
        name = "formula-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="formula",
                    kind="formula",
                    text="x + y",
                    confidence=0.88,
                    bounding_box=BoundingBox(4, 8, 80, 32),
                    reading_order=1,
                    provider=self.name,
                ),
                TextRegion(
                    id="risk",
                    kind="coverage_risk",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 80),
                    reading_order=2,
                    provider="risk",
                    resolution="unreadable",
                    structure={"role": "coverage_risk", "reasons": ["formula_review"]},
                ),
            ]

    class FormulaPresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="formula-output",
                    kind="page_text",
                    text="x+y",
                    confidence=None,
                    bounding_box=BoundingBox(4, 8, 80, 32),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "formula"},
                )
            ]

    app = create_app(FormulaReader(), presentation_reader=FormulaPresenter())
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("formula.png", _page_png(), "image/png")},
        )

    [block] = response.json()["presentation"]["pages"][0]["blocks"]
    assert block["rendering"] == {
        "status": "selected",
        "source_region_id": "formula",
        "source_page_number": 1,
        "source_bounding_box": {"left": 4, "top": 8, "right": 80, "bottom": 32},
        "source_evidence_ids": ["formula"],
        "source_confidence": 0.88,
        "source_confidence_kind": "Recognition score",
        "source_confidence_scope": "source region",
    }


@pytest.mark.parametrize(
    ("generated_table", "expected_status", "expected_failure"),
    [
        (
            "<table><tr><th>Organization Name</th><th>Year</th></tr>"
            "<tr><td>Alpha Medical Center</td><td>2024</td></tr></table>",
            "selected",
            None,
        ),
        (
            "<table><tr><th>Organization Name</th><th>Year</th></tr></table>",
            "canonical_fallback",
            "table_topology_mismatch",
        ),
        (
            "<table><tr><th>Organization Name</th></tr>"
            "<tr><td>Alpha Medical Center</td></tr></table>",
            "canonical_fallback",
            "table_topology_mismatch",
        ),
        (
            "<table><tr><th>Organization Name</th><th>Year</th></tr>"
            "<tr><td>Alpha Medical Center</td><td>2025</td></tr></table>",
            "canonical_fallback",
            "unsupported_critical_token",
        ),
        (
            "<table><tr><th>Organization Name</th><th>Year</th></tr>"
            "<tr><td>Alpha</td><td>2024</td></tr></table>",
            "canonical_fallback",
            "source_evidence_omitted",
        ),
        (
            "<table><tr><th>Organization Name</th><th>Year</th></tr>"
            "<tr><td>2024</td><td>Alpha Medical Center</td></tr></table>",
            "canonical_fallback",
            "table_cell_association_changed",
        ),
    ],
)
def test_demo_arbitrates_verified_falcon_tables_against_canonical_structure(
    generated_table: str,
    expected_status: str,
    expected_failure: str | None,
) -> None:
    class CanonicalTableReader:
        name = "canonical-table-reader"
        version = "test-1"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            cells = [
                {
                    "text": "Organization Name",
                    "row_nums": [0],
                    "column_nums": [0],
                    "column_header": True,
                    "resolution": "resolved",
                },
                {
                    "text": "Year",
                    "row_nums": [0],
                    "column_nums": [1],
                    "column_header": True,
                    "resolution": "resolved",
                },
                {
                    "text": "Alpha Medical Center",
                    "row_nums": [1],
                    "column_nums": [0],
                    "resolution": "conflicting",
                },
                {
                    "text": "2024",
                    "row_nums": [1],
                    "column_nums": [1],
                    "resolution": "resolved",
                },
            ]
            return [
                TextRegion(
                    id="table",
                    kind="table",
                    text=(
                        "| Organization Name | Year |\n| --- | --- |\n"
                        "| Alpha Medical Center | 2024 |"
                    ),
                    confidence=0.91,
                    bounding_box=BoundingBox(0, 0, 120, 60),
                    reading_order=1,
                    provider=self.name,
                    structure={
                        "role": "table",
                        "row_count": 2,
                        "column_count": 2,
                        "cells": cells,
                    },
                ),
                TextRegion(
                    id="risk",
                    kind="coverage_risk",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 80),
                    reading_order=2,
                    provider="deterministic-evidence-risk",
                    resolution="unreadable",
                    structure={
                        "role": "coverage_risk",
                        "reasons": ["table_cell_conflict"],
                    },
                ),
            ]

    class VerifiedFalconPresenter:
        name = "falcon-presentation"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="generated-table",
                    kind="table",
                    text=generated_table,
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 60),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={
                        "method": "falcon_core_category_crop_generation",
                        "source_region_id": "table",
                        "model": {
                            "id": "tiiuae/Falcon-OCR",
                            "loaded_from": "/models/falcon-ocr",
                            "revision": ("42ec56b72a23984ac059e7c8a6d397a8529423fe"),
                            "origin": "TII, UAE",
                            "license": "Apache-2.0",
                            "identity_verified": True,
                            "local_files_only": True,
                        },
                    },
                )
            ]

    app = create_app(
        CanonicalTableReader(),
        presentation_reader=VerifiedFalconPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("table.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["raw_text"] == generated_table
    assert block["rendering"]["status"] == expected_status
    failure_codes = {failure["code"] for failure in block["validation"]["failures"]}
    if expected_failure is None:
        assert failure_codes == set()
        assert "<td>Alpha Medical Center</td>" in markdown
    else:
        assert expected_failure in failure_codes
        assert generated_table not in markdown


def test_presentation_canonical_table_text_uses_each_cell_once() -> None:
    source = TextRegion(
        id="table",
        kind="table",
        text="| Name | Year |\n| --- | --- |\n| Alpha | 2024 |",
        confidence=0.9,
        bounding_box=BoundingBox(0, 0, 120, 60),
        reading_order=1,
        provider="table-reader",
        structure={
            "role": "table",
            "row_count": 2,
            "column_count": 2,
            "child_evidence_ids": ["name", "year", "alpha", "2024"],
            "cells": [
                {"text": "Name", "row_nums": [0], "column_nums": [0]},
                {"text": "Year", "row_nums": [0], "column_nums": [1]},
                {"text": "Alpha", "row_nums": [1], "column_nums": [0]},
                {"text": "2024", "row_nums": [1], "column_nums": [1]},
            ],
        },
    )
    evidence = [
        TextRegion(
            id=region_id,
            kind="table_cell",
            text=text,
            confidence=0.9,
            bounding_box=BoundingBox(index * 10, 0, index * 10 + 9, 10),
            reading_order=index,
            provider="table-reader",
        )
        for index, (region_id, text) in enumerate(
            (("name", "Name"), ("year", "Year"), ("alpha", "Alpha"), ("2024", "2024")),
            start=1,
        )
    ]

    assert demo_module._presentation_canonical_text(source, evidence) == (
        "Name Year Alpha 2024"
    )


def test_demo_rejects_unsupported_control_from_verified_table_recovery() -> None:
    class VerifiedFalconPresenter:
        name = "falcon-presentation"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="generated-table",
                    kind="table",
                    text="<table><tr><td>☑ Value</td></tr></table>",
                    confidence=None,
                    bounding_box=BoundingBox(52, 0, 110, 46),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={
                        "method": "falcon_core_category_crop_generation",
                        "source_region_id": "table",
                        "model": {
                            "id": "tiiuae/Falcon-OCR",
                            "loaded_from": "/models/falcon-ocr",
                            "revision": ("42ec56b72a23984ac059e7c8a6d397a8529423fe"),
                            "origin": "TII, UAE",
                            "license": "Apache-2.0",
                            "identity_verified": True,
                            "local_files_only": True,
                        },
                    },
                )
            ]

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=VerifiedFalconPresenter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["rendering"]["reason"] == "validation_failed"
    assert {failure["code"] for failure in block["validation"]["failures"]} == {
        "unsupported_control_glyph"
    }
    assert "☑ Value" not in markdown


def test_demo_composes_only_top_level_layout_evidence() -> None:
    class LayoutReader:
        name = "layout-reader"

        def __init__(self, *, unresolved_child: bool = False) -> None:
            self.unresolved_child = unresolved_child

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            words = [
                ("a", "Clinical", (10, 10, 45, 20)),
                ("b", "summary", (50, 10, 90, 20)),
                ("c", "continues", (10, 24, 55, 34)),
                ("d", "here.", (60, 24, 90, 34)),
            ]
            return [
                TextRegion(
                    id=identifier,
                    kind="word",
                    text=text,
                    confidence=0.9,
                    bounding_box=BoundingBox(*box),
                    reading_order=order,
                    provider=self.name,
                    resolution=(
                        "conflicting"
                        if self.unresolved_child and identifier == "b"
                        else "resolved"
                    ),
                    alternatives=(
                        [TextAlternative("alternate", 0.8, "challenger")]
                        if identifier == "a"
                        else []
                    ),
                )
                for order, (identifier, text, box) in enumerate(words, start=1)
            ]

    class LayoutPresentationReader:
        name = "falcon-review"

        def __init__(self, *, use_child: bool = False) -> None:
            self.use_child = use_child

        def read_page(
            self,
            image_path: Path,
            page: PageResult,
        ) -> list[TextRegion]:
            del image_path
            owner = next(
                region
                for region in page.regions
                if (region.structure or {}).get("role") == "layout_block"
            )
            source_id = "b" if self.use_child else owner.id
            return [
                TextRegion(
                    id="presentation",
                    kind="page_text",
                    text="# Clinical summary continues here.",
                    confidence=None,
                    bounding_box=owner.bounding_box,
                    reading_order=owner.reading_order,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": source_id},
                )
            ]

    selected_app = create_app(
        LayoutReader(),
        stages=(EvidenceLayoutStage(),),
        presentation_reader=LayoutPresentationReader(),
    )
    with TestClient(selected_app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("layout.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [selected] = payload["presentation"]["pages"][0]["blocks"]
    assert selected["rendering"]["status"] == "selected"
    assert markdown.count("Clinical summary continues here.") == 1
    assert all(
        markdown.count(word) == 1
        for word in ("Clinical", "summary", "continues", "here.")
    )

    for reader, reason in (
        (LayoutReader(unresolved_child=True), "source_evidence_unresolved"),
        (LayoutReader(), "source_is_owned_by_layout"),
    ):
        use_child = reason == "source_is_owned_by_layout"
        app = create_app(
            reader,
            stages=(EvidenceLayoutStage(),),
            presentation_reader=LayoutPresentationReader(use_child=use_child),
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/process",
                files={"file": ("layout.png", _page_png(), "image/png")},
            )
            payload = response.json()
            markdown = client.get(
                f"/api/sessions/{payload['session_id']}/result.md"
            ).text

        [block] = payload["presentation"]["pages"][0]["blocks"]
        assert block["rendering"]["status"] == "canonical_fallback"
        assert block["rendering"]["reason"] == reason
        assert "# Clinical summary continues here." not in markdown


def test_demo_passes_canonical_page_to_category_routed_presenter() -> None:
    class CategoryRoutedReader:
        name = "falcon-review"

        def read_page(
            self,
            image_path: Path,
            page: PageResult,
        ) -> list[TextRegion]:
            assert image_path.name == "preview-1.png"
            assert page.route == "review"
            return [
                TextRegion(
                    id="title",
                    kind="page_text",
                    text="Visit summary",
                    confidence=None,
                    bounding_box=BoundingBox(5, 2, 115, 15),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "title"},
                )
            ]

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=CategoryRoutedReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )

    assert response.status_code == 200
    presentation = response.json()["presentation"]["pages"][0]
    assert presentation["status"] == "review_draft"
    assert presentation["blocks"][0]["raw_text"] == "Visit summary"


def test_demo_rejects_orphan_and_linked_control_label_presentation_blocks() -> None:
    class OrphanPresentationReader:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="orphan",
                    kind="page_text",
                    text="Orphan model text",
                    confidence=None,
                    bounding_box=BoundingBox(115, 70, 120, 80),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "title"},
                    text_provenance={
                        "model": {"id": "tiiuae/Falcon-OCR"},
                        "source_region_id": "missing-region",
                    },
                )
            ]

    class LinkedLabelPresentationReader:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="linked-label",
                    kind="page_text",
                    text="Duplicated model label",
                    confidence=None,
                    bounding_box=BoundingBox(0, 50, 30, 60),
                    reading_order=3,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={
                        "model": {"id": "tiiuae/Falcon-OCR"},
                        "source_region_id": "label",
                    },
                )
            ]

    for reader, excluded in (
        (OrphanPresentationReader(), "Orphan model text"),
        (LinkedLabelPresentationReader(), "Duplicated model label"),
    ):
        app = create_app(
            StructuredDocumentReader(),
            presentation_reader=reader,
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/process",
                files={"file": ("review.png", _page_png(), "image/png")},
            )
            markdown = client.get(
                f"/api/sessions/{response.json()['session_id']}/result.md"
            ).text

        assert response.status_code == 200
        assert excluded not in markdown
        assert markdown.count("Fall risk") == 1

    html = TestClient(create_app(ControlledReader())).get("/").text
    assert 'if (block.rendering?.status !== "selected") return;' in html
    assert "if (!source || claimedIds.has(sourceId)) return;" in html


def test_demo_reports_no_eligible_presentation_regions_as_skipped() -> None:
    class EmptyCategoryRoutedReader:
        name = "falcon-review"

        def read_page(
            self,
            image_path: Path,
            page: PageResult,
        ) -> list[TextRegion]:
            return []

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=EmptyCategoryRoutedReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )

    assert response.status_code == 200
    presentation = response.json()["presentation"]["pages"][0]
    assert presentation == {
        "page_number": 1,
        "status": "skipped",
        "provider": "falcon-review",
        "message": "No eligible positioned regions for presentation",
        "canonical_unchanged": True,
    }
    run = response.json()["stage_execution"][0]
    assert run == {
        "page_number": 1,
        "stage": "presentation",
        "status": "skipped",
        "input_regions": 6,
        "output_regions": 6,
        "added_regions": 0,
        "removed_regions": 0,
        "modified_regions": 0,
        "elapsed_seconds": run["elapsed_seconds"],
        "skip_reason": "No eligible positioned regions for presentation",
    }


def test_demo_surfaces_presentation_service_failure_and_page_limit() -> None:
    class OfflinePresentationReader:
        name = "falcon-review"

        def read_page(
            self,
            image_path: Path,
            page: PageResult,
        ) -> list[TextRegion]:
            raise ReaderError("service_unavailable", "Falcon service is offline")

    offline_app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=OfflinePresentationReader(),
    )
    with TestClient(offline_app) as client:
        offline_response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        html = client.get("/").text

    offline_payload = offline_response.json()
    assert offline_payload["presentation"]["pages"][0]["failure"] == {
        "code": "service_unavailable",
        "message": "Falcon service is offline",
    }
    assert offline_payload["stage_execution"][0]["status"] == "failed"
    assert offline_payload["stage_execution"][0]["failure_code"] == (
        "service_unavailable"
    )

    limited_app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=PresentationReader(),
        max_presentation_pages=1,
    )
    with TestClient(limited_app) as client:
        limited_response = client.post(
            "/api/process",
            files={"file": ("review.tif", _two_page_tiff(), "image/tiff")},
        )

    limited_payload = limited_response.json()
    assert [page["status"] for page in limited_payload["presentation"]["pages"]] == [
        "review_draft",
        "skipped",
    ]
    assert [run["status"] for run in limited_payload["stage_execution"]] == [
        "productive",
        "skipped",
    ]
    assert limited_payload["stage_execution"][1]["skip_reason"] == (
        "Review draft page limit reached"
    )
    assert ">Failures</button>" not in html
    assert 'id="panel-failures"' not in html


def test_demo_fails_closed_on_disallowed_table_html_and_bad_formula() -> None:
    class UnsafePresentationReader:
        name = "falcon-unsafe-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="unsafe-table",
                    kind="page_text",
                    text='<table onclick="steal()"><tr><td>A</td><script>x</script></tr></table>',
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 40),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "table"},
                    text_provenance={"source_region_id": "table"},
                ),
                TextRegion(
                    id="bad-formula",
                    kind="page_text",
                    text=r"\frac{a}{b",
                    confidence=None,
                    bounding_box=BoundingBox(0, 40, 120, 80),
                    reading_order=2,
                    provider=self.name,
                    structure={"category": "formula"},
                    text_provenance={"source_region_id": "handwriting"},
                ),
            ]

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=UnsafePresentationReader(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        markdown = client.get(
            f"/api/sessions/{response.json()['session_id']}/result.md"
        ).text

    assert response.status_code == 200
    presentation = response.json()["presentation"]["pages"][0]
    assert presentation["status"] == "review_draft"
    assert presentation["validation"]["status"] == "failed"
    table, formula = presentation["blocks"]
    assert table["raw_text"] == (
        '<table onclick="steal()"><tr><td>A</td><script>x</script></tr></table>'
    )
    assert {failure["code"] for failure in table["validation"]["failures"]} >= {
        "disallowed_html_attribute",
        "disallowed_html_element",
    }
    assert formula["raw_text"] == r"\frac{a}{b"
    assert formula["validation"]["failures"] == [
        {
            "code": "unbalanced_latex_braces",
            "message": "Formula output has unbalanced braces",
        }
    ]
    assert "Primary text" in markdown
    assert "onclick" not in markdown
    assert r"\frac{a}{b" not in markdown


def test_presentation_table_accepts_safe_semantic_scripts() -> None:
    table = (
        "<table><thead><tr><th>Metric<sup>1</sup></th></tr></thead>"
        "<tbody><tr><td>CO<sub>2</sub></td></tr></tbody></table>"
    )

    assert demo_module._validate_presentation_table(table) == []


def test_browser_table_allowlist_keeps_safe_semantic_scripts() -> None:
    html = Path(demo_module.__file__).with_name("demo.html").read_text(encoding="utf-8")

    assert '"SUP", "SUB"' in html
    assert '["TH", "TD"].includes(parent)' in html


def test_demo_rejects_implausibly_expanded_crop_output() -> None:
    class RunawayPresentationReader:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="runaway",
                    kind="page_text",
                    text="generated " * 80,
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 50, 10),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={
                        "model": {"id": "tiiuae/Falcon-OCR"},
                        "source_region_id": "paragraph",
                    },
                )
            ]

    app = create_app(
        ReviewEvidenceReader(),
        presentation_reader=RunawayPresentationReader(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("review.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text

    [block] = payload["presentation"]["pages"][0]["blocks"]
    assert block["rendering"] == {
        "status": "canonical_fallback",
        "reason": "validation_failed",
        "source_region_id": "paragraph",
    }
    assert block["validation"]["failures"] == [
        {
            "code": "implausible_output_expansion",
            "message": (
                "Generated text is implausibly long for its evidence-owned crop"
            ),
        }
    ]
    assert "Primary text" in markdown
    assert "generated generated" not in markdown


def test_demo_keeps_structured_rows_canonical_and_rejects_unsupported_claims() -> None:
    class EvidenceReader:
        name = "evidence-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            children = [
                TextRegion(
                    id="patient-label",
                    kind="word",
                    text="Patient Name:",
                    confidence=0.96,
                    bounding_box=BoundingBox(0, 0, 25, 10),
                    reading_order=1,
                    provider=self.name,
                    structure={"layout_owner_id": "form-row"},
                ),
                TextRegion(
                    id="patient-value",
                    kind="word",
                    text="Ada",
                    confidence=0.96,
                    bounding_box=BoundingBox(26, 0, 45, 10),
                    reading_order=2,
                    provider=self.name,
                    structure={"layout_owner_id": "form-row"},
                ),
                TextRegion(
                    id="plan-label",
                    kind="word",
                    text="Plan Name:",
                    confidence=0.96,
                    bounding_box=BoundingBox(60, 0, 82, 10),
                    reading_order=3,
                    provider=self.name,
                    structure={"layout_owner_id": "form-row"},
                ),
                TextRegion(
                    id="plan-value",
                    kind="word",
                    text="Care Plus",
                    confidence=0.96,
                    bounding_box=BoundingBox(83, 0, 115, 10),
                    reading_order=4,
                    provider=self.name,
                    structure={"layout_owner_id": "form-row"},
                ),
            ]
            owners = [
                TextRegion(
                    id="form-row",
                    kind="layout_block",
                    text="Patient Name: Ada | Plan Name: Care Plus",
                    confidence=0.96,
                    bounding_box=BoundingBox(0, 0, 115, 10),
                    reading_order=1,
                    provider="evidence-spatial-layout",
                    structure={
                        "role": "layout_block",
                        "block_type": "form_row",
                        "child_evidence_ids": [region.id for region in children],
                        "segments": [
                            {"evidence_ids": ["patient-label", "patient-value"]},
                            {"evidence_ids": ["plan-label", "plan-value"]},
                        ],
                    },
                ),
                TextRegion(
                    id="date-line",
                    kind="paragraph",
                    text="Date of service: 3/18/2025",
                    confidence=0.96,
                    bounding_box=BoundingBox(0, 20, 115, 30),
                    reading_order=5,
                    provider=self.name,
                ),
                TextRegion(
                    id="empty-fields",
                    kind="paragraph",
                    text="Alcohol: Drug:",
                    confidence=0.96,
                    bounding_box=BoundingBox(0, 40, 115, 50),
                    reading_order=6,
                    provider=self.name,
                ),
                TextRegion(
                    id="long-line",
                    kind="paragraph",
                    text="Recommendations for medical services remain in the patient chart",
                    confidence=0.96,
                    bounding_box=BoundingBox(0, 55, 115, 65),
                    reading_order=7,
                    provider=self.name,
                ),
                TextRegion(
                    id="risk",
                    kind="coverage_risk",
                    text="",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 120, 80),
                    reading_order=8,
                    provider="deterministic-evidence-risk",
                    resolution="unreadable",
                    structure={
                        "role": "coverage_risk",
                        "reasons": ["small_text_evidence"],
                    },
                ),
            ]
            return [*children, *owners]

    class UnsafePresenter:
        name = "falcon-review"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            return [
                TextRegion(
                    id="generated-form",
                    kind="page_text",
                    text="Patient Name: Ada Patient Name: Care Plus",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 115, 10),
                    reading_order=1,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "form-row"},
                ),
                TextRegion(
                    id="generated-date",
                    kind="page_text",
                    text="Date of service: 3/18/2015",
                    confidence=None,
                    bounding_box=BoundingBox(0, 20, 115, 30),
                    reading_order=5,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "date-line"},
                ),
                TextRegion(
                    id="generated-marks",
                    kind="page_text",
                    text="Alcohol: ♡ Drug: ♡",
                    confidence=None,
                    bounding_box=BoundingBox(0, 40, 115, 50),
                    reading_order=6,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "empty-fields"},
                ),
                TextRegion(
                    id="generated-truncated",
                    kind="page_text",
                    text="Recommendations",
                    confidence=None,
                    bounding_box=BoundingBox(0, 55, 115, 65),
                    reading_order=7,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={"source_region_id": "long-line"},
                ),
            ]

    app = create_app(EvidenceReader(), presentation_reader=UnsafePresenter())
    with TestClient(app) as client:
        response = client.post(
            "/api/process",
            files={"file": ("form.png", _page_png(), "image/png")},
        )
        payload = response.json()
        markdown = client.get(f"/api/sessions/{payload['session_id']}/result.md").text
        html = client.get("/").text

    blocks = payload["presentation"]["pages"][0]["blocks"]
    assert blocks[0]["rendering"]["reason"] == "structured_source_is_authoritative"
    assert blocks[1]["rendering"]["reason"] == "validation_failed"
    assert blocks[1]["validation"]["failures"][0]["code"] == (
        "unsupported_critical_token"
    )
    assert blocks[2]["rendering"]["reason"] == "validation_failed"
    assert blocks[2]["validation"]["failures"][0]["code"] == (
        "unsupported_control_glyph"
    )
    assert blocks[3]["rendering"]["reason"] == "validation_failed"
    assert blocks[3]["validation"]["failures"][0]["code"] == ("source_evidence_omitted")
    assert "Plan Name: Care Plus" in markdown
    assert "3/18/2025" in markdown
    assert "3/18/2015" not in markdown
    assert "♡" not in markdown
    assert "const outcome = renderPresentationBlock(block, source);" in html
    assert "decorateConfidence(marker, source.confidence" in html
    assert 'scope: "table detection"' in html
    assert "card.open = false;" in html
    assert 'button.addEventListener("focus"' in html
    assert "function appendPresentationInline(element, text)" in html
    assert 'document.createElement("strong")' in html


def test_demo_serves_only_explicit_local_katex_assets(tmp_path: Path) -> None:
    assets = tmp_path / "katex"
    fonts = assets / "fonts"
    fonts.mkdir(parents=True)
    (assets / "katex.js").write_text("window.katex = {};", encoding="utf-8")
    (assets / "katex.css").write_text(".katex {}", encoding="utf-8")
    (fonts / "KaTeX_Main-Regular.woff2").write_bytes(b"font")
    (assets / "secret.txt").write_text("secret", encoding="utf-8")
    app = create_app(ControlledReader(), katex_asset_root=assets)

    with TestClient(app) as client:
        index = client.get("/")
        javascript = client.get("/assets/katex/katex.js")
        stylesheet = client.get("/assets/katex/katex.css")
        font = client.get("/assets/katex/fonts/KaTeX_Main-Regular.woff2")
        secret = client.get("/assets/katex/fonts/../secret.txt")

    assert index.status_code == 200
    assert '<link rel="stylesheet" href="/assets/katex/katex.css">' in index.text
    assert '<script defer src="/assets/katex/katex.js"></script>' in index.text
    assert javascript.text == "window.katex = {};"
    assert stylesheet.text == ".katex {}"
    assert font.content == b"font"
    assert secret.status_code == 404


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
        assert payload["stage_execution"][0]["stage"] == "table-structure"
        assert payload["stage_execution"][0]["status"] == "productive"
        assert payload["result"]["pages"][0]["text"]["value"] == (
            "Controlled page 1 with table"
        )
        assert "stage.table-structure" in payload["timing"]["pipeline_steps"]


def test_demo_reports_manual_handwriting_reread_execution() -> None:
    class ManualRereadStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

        def review_region(
            self,
            image_path: Path,
            page_number: int,
            region: TextRegion,
        ) -> TextRegion:
            return replace(region, text="Corrected handwritten text")

    app = create_app(
        ControlledReader(),
        handwriting_stage=ManualRereadStage(),
    )

    with TestClient(app) as client:
        processed = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        )
        initial = processed.json()
        assert initial["composition"]["handwriting"] == "configured"
        assert initial["stage_execution"] == []
        reread = client.post(
            f"/api/sessions/{initial['session_id']}/handwriting",
            json={
                "page_number": 1,
                "region_id": "page-1-region-1",
                "revision": initial["revision"],
                "request_id": "manual-reread-1",
            },
        )

    assert reread.status_code == 200
    payload = reread.json()
    assert payload["result"]["pages"][0]["text"]["value"] == (
        "Corrected handwritten text"
    )
    assert payload["pipeline_stages"] == []
    manual_run = payload["stage_execution"][0]
    assert manual_run == {
        "page_number": 1,
        "stage": "handwriting.manual-reread",
        "status": "productive",
        "input_regions": 1,
        "output_regions": 1,
        "added_regions": 0,
        "removed_regions": 0,
        "modified_regions": 1,
        "elapsed_seconds": manual_run["elapsed_seconds"],
    }
    assert manual_run["elapsed_seconds"] >= 0
    assert (
        payload["timing"]["pipeline_steps"]["stage.handwriting.manual-reread"]
        == manual_run["elapsed_seconds"]
    )
    assert payload["timing"]["manual_reread_seconds"] == manual_run["elapsed_seconds"]
    assert payload["recovery_outcome"]["status"] == "corrected"


def test_manual_reread_refreshes_the_page_presentation() -> None:
    class ManualRereadStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

        def review_region(
            self,
            image_path: Path,
            page_number: int,
            region: TextRegion,
        ) -> TextRegion:
            return replace(region, text="Corrected handwritten text")

    class RefreshingPresentationReader:
        name = "falcon-review"

        def read_page(
            self,
            image_path: Path,
            page: PageResult,
        ) -> list[TextRegion]:
            source = page.regions[0]
            return [
                TextRegion(
                    id="refreshed-presentation",
                    kind="page_text",
                    text=f"Rendered: {source.text}",
                    confidence=None,
                    bounding_box=source.bounding_box,
                    reading_order=source.reading_order,
                    provider=self.name,
                    structure={"category": "text"},
                    text_provenance={
                        "model": {"id": "tiiuae/Falcon-OCR"},
                        "source_region_id": source.id,
                    },
                )
            ]

    app = create_app(
        ControlledReader(),
        handwriting_stage=ManualRereadStage(),
        presentation_reader=RefreshingPresentationReader(),
    )
    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        assert initial["presentation"]["pages"] == []
        reread = client.post(
            f"/api/sessions/{initial['session_id']}/handwriting",
            json={
                "page_number": 1,
                "region_id": "page-1-region-1",
                "revision": initial["revision"],
                "request_id": "presentation-reread-1",
            },
        )

    payload = reread.json()
    block = payload["presentation"]["pages"][0]["blocks"][0]
    assert block["raw_text"] == "Rendered: Corrected handwritten text"
    assert "Controlled page 1" not in block["raw_text"]
    assert [run["stage"] for run in payload["stage_execution"]] == [
        "handwriting.manual-reread",
        "presentation",
    ]


def test_manual_reread_skips_presentation_when_canonical_text_is_unchanged() -> None:
    class MetadataOnlyRereadStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

        def review_region(
            self,
            image_path: Path,
            page_number: int,
            region: TextRegion,
        ) -> TextRegion:
            return replace(region, confidence=0.99)

    class CountingPresentationReader:
        name = "falcon-review"

        def __init__(self) -> None:
            self.calls = 0

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            self.calls += 1
            return []

    presentation_reader = CountingPresentationReader()
    app = create_app(
        ControlledReader(),
        handwriting_stage=MetadataOnlyRereadStage(),
        presentation_reader=presentation_reader,
    )
    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        reread = client.post(
            f"/api/sessions/{initial['session_id']}/handwriting",
            json={
                "page_number": 1,
                "region_id": "page-1-region-1",
                "revision": initial["revision"],
                "request_id": "metadata-reread-1",
            },
        )

    payload = reread.json()
    assert presentation_reader.calls == 0
    assert payload["presentation"]["pages"] == []
    assert payload["uncertainty"]["pages"][0]["review_required"] is True
    assert payload["uncertainty"]["pages"][0]["risk_reasons"] == []
    assert [run["stage"] for run in payload["stage_execution"]] == [
        "handwriting.manual-reread"
    ]


def test_failed_manual_reread_cannot_mutate_live_session_evidence() -> None:
    class MutatingFailureStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

        def review_region(
            self,
            image_path: Path,
            page_number: int,
            region: TextRegion,
        ) -> TextRegion:
            region.text = "corrupted before failure"
            raise ReaderError("controlled_failure", "reread failed")

    app = create_app(
        ControlledReader(),
        handwriting_stage=MutatingFailureStage(),
    )

    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_id = initial["session_id"]
        failed = client.post(
            f"/api/sessions/{session_id}/handwriting",
            json={
                "page_number": 1,
                "region_id": "page-1-region-1",
                "revision": initial["revision"],
                "request_id": "failed-reread-1",
            },
        )
        stored = client.get(f"/api/sessions/{session_id}/result.json").json()

    assert failed.status_code == 422
    assert stored["pages"][0]["regions"][0]["text"] == "Controlled page 1"


def test_manual_reread_candidate_can_be_accepted_into_every_export() -> None:
    class CandidateRereadStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

        def review_region(
            self,
            image_path: Path,
            page_number: int,
            region: TextRegion,
        ) -> TextRegion:
            region.alternatives.append(
                TextAlternative(
                    "Corrected handwritten text",
                    0.88,
                    "candidate-reader",
                    {"method": "source_crop_reread"},
                )
            )
            region.structure = {
                "handwriting_review": {
                    "required": True,
                    "reason": "specialist_candidate",
                }
            }
            return region

    app = create_app(
        ControlledReader(),
        handwriting_stage=CandidateRereadStage(),
    )
    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        reread = client.post(
            f"/api/sessions/{initial['session_id']}/handwriting",
            json={
                "page_number": 1,
                "region_id": "page-1-region-1",
                "revision": 1,
                "request_id": "reread-1",
            },
        ).json()
        accepted = client.post(
            f"/api/sessions/{initial['session_id']}/corrections",
            json={
                "page_number": 1,
                "region_id": "page-1-region-1",
                "revision": 2,
                "request_id": "accept-reread-1",
                "action": "accept",
                "alternative_index": 0,
            },
        )
        exported = client.get(
            f"/api/sessions/{initial['session_id']}/result.json?revision=3"
        ).json()
        markdown = client.get(
            f"/api/sessions/{initial['session_id']}/result.md?revision=3"
        ).text

    assert reread["revision"] == 2
    assert reread["recovery_outcome"]["status"] == "candidate_pending"
    assert reread["result"]["pages"][0]["route"] == "review"
    assert accepted.status_code == 200
    payload = accepted.json()
    region = payload["result"]["pages"][0]["regions"][0]
    assert payload["revision"] == 3
    assert payload["result"]["pages"][0]["route"] == "accept_local"
    assert payload["uncertainty"]["pages"][0]["review_required"] is False
    assert region["text"] == "Corrected handwritten text"
    assert region["provider"] == "candidate-reader"
    assert region["confidence"] == 0.88
    assert region["structure"]["handwriting_review"]["required"] is False
    assert [item["decision_state"] for item in region["alternatives"]] == [
        "superseded",
        "accepted",
    ]
    assert exported["pages"][0]["text"]["value"] == "Corrected handwritten text"
    assert "Corrected handwritten text" in markdown


def test_human_correction_updates_every_export_at_one_revision() -> None:
    app = create_app(StructuredDocumentReader())

    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_id = initial["session_id"]
        corrected = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "handwriting",
                "revision": 1,
                "request_id": "accept-1",
                "action": "accept",
                "alternative_index": 0,
            },
        )
        current_json = client.get(f"/api/sessions/{session_id}/result.json?revision=2")
        current_markdown = client.get(
            f"/api/sessions/{session_id}/result.md?revision=2"
        )
        stale_export = client.get(f"/api/sessions/{session_id}/result.json?revision=1")
        stale_edit = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "handwriting",
                "revision": 1,
                "request_id": "late-edit-1",
                "action": "edit",
                "text": "Late overwrite",
            },
        )

    assert corrected.status_code == 200
    payload = corrected.json()
    assert payload["revision"] == payload["result"]["revision"] == 2
    assert payload["recovery_outcome"] == {
        "kind": "human_review",
        "status": "accept",
        "request_id": "accept-1",
        "page_number": 1,
        "region_id": "handwriting",
        "base_revision": 1,
        "revision": 2,
    }
    region = next(
        item
        for item in payload["result"]["pages"][0]["regions"]
        if item["id"] == "handwriting"
    )
    assert region["text"] == "Return in 3 weeks"
    assert region["provider"] == "challenger"
    assert region["confidence"] == 0.5
    assert [item["decision_state"] for item in region["alternatives"]] == [
        "superseded",
        "accepted",
    ]
    assert region["alternatives"][0]["text"] == "Return in 2 weeks"
    assert region["text_provenance"]["human_review"]["correcting_provider"] == (
        "challenger"
    )
    assert region["bounding_box"] == {
        "left": 0,
        "top": 62,
        "right": 100,
        "bottom": 76,
    }
    assert current_json.status_code == 200
    assert current_json.json()["revision"] == 2
    assert "Return in 3 weeks" in current_json.text
    assert current_markdown.status_code == 200
    assert "- Revision: 2" in current_markdown.text
    assert "Return in 3 weeks" in current_markdown.text
    assert stale_export.status_code == 409
    assert stale_edit.status_code == 409
    assert stale_edit.json()["recovery_outcome"]["status"] == "stale"


def test_accepted_formula_owner_supersedes_fragments_in_every_export() -> None:
    class FormulaReviewReader(ControlledReader):
        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            del image_path, page_number
            child = TextRegion(
                id="formula-fragment",
                kind="word",
                text="x + y",
                confidence=0.8,
                bounding_box=BoundingBox(5, 5, 45, 20),
                reading_order=1,
                provider=self.name,
                structure={
                    "layout_owner_id": "formula-owner",
                    "layout_owner_type": "formula",
                },
            )
            owner = TextRegion(
                id="formula-owner",
                kind="layout_block",
                text="x + y",
                confidence=0.8,
                bounding_box=BoundingBox(5, 5, 100, 30),
                reading_order=1,
                provider="evidence-spatial-layout",
                resolution="conflicting",
                alternatives=[
                    TextAlternative(
                        text=r"\frac{x^2 + y^2",
                        confidence=None,
                        provider="falcon-formula",
                        text_provenance={"category": "formula"},
                    ),
                    TextAlternative(
                        text=r"x^2 + y^2",
                        confidence=None,
                        provider="falcon-formula",
                        text_provenance={"category": "formula"},
                    ),
                    TextAlternative(
                        text="   ",
                        confidence=None,
                        provider="falcon-formula",
                        text_provenance={"category": "formula"},
                    ),
                ],
                structure={
                    "role": "layout_block",
                    "block_type": "formula",
                    "child_evidence_ids": [child.id],
                    "lines": [{"evidence_ids": [child.id]}],
                    "formula_review": {"required": True},
                },
            )
            return [child, owner]

    app = create_app(FormulaReviewReader())
    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("formula.png", _page_png(), "image/png")},
        ).json()
        session_id = initial["session_id"]
        invalid = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "formula-owner",
                "revision": 1,
                "request_id": "reject-invalid-formula",
                "action": "accept",
                "alternative_index": 0,
            },
        )
        invalid_edit = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "formula-owner",
                "revision": 1,
                "request_id": "reject-invalid-formula-edit",
                "action": "edit",
                "text": r"\frac{x^2 + y^2",
            },
        )
        blank = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "formula-owner",
                "revision": 1,
                "request_id": "reject-blank-formula",
                "action": "accept",
                "alternative_index": 2,
            },
        )
        corrected = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "formula-owner",
                "revision": 1,
                "request_id": "accept-formula",
                "action": "accept",
                "alternative_index": 1,
            },
        )
        current_json = client.get(f"/api/sessions/{session_id}/result.json?revision=2")
        current_markdown = client.get(
            f"/api/sessions/{session_id}/result.md?revision=2"
        )

    assert invalid.status_code == 422
    assert invalid.json()["detail"].endswith("unbalanced_latex_braces")
    assert invalid_edit.status_code == 422
    assert invalid_edit.json()["detail"].endswith("unbalanced_latex_braces")
    assert blank.status_code == 422
    assert blank.json()["detail"].endswith("empty_formula")
    assert corrected.status_code == 200
    payload = corrected.json()
    page = payload["result"]["pages"][0]
    owner = next(item for item in page["regions"] if item["id"] == "formula-owner")
    assert page["text"] == {
        "value": r"x^2 + y^2",
        "evidence_ids": ["formula-owner"],
    }
    assert owner["text"] == r"x^2 + y^2"
    assert owner["provider"] == "falcon-formula"
    assert owner["structure"]["formula_recognition"] == "human_accepted"
    assert owner["structure"]["formula_review"] == {
        "required": False,
        "resolved_by": "human_review",
    }
    assert [item["decision_state"] for item in owner["alternatives"]] == [
        "superseded",
        "rejected",
        "accepted",
        "rejected",
    ]
    assert current_json.json()["pages"][0]["text"] == page["text"]
    assert "$$\nx^2 + y^2\n$$" in current_markdown.text
    assert "x + y" not in current_markdown.text


def test_keep_unresolved_records_review_without_inventing_text() -> None:
    app = create_app(StructuredDocumentReader())

    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        response = client.post(
            f"/api/sessions/{initial['session_id']}/corrections",
            json={
                "page_number": 1,
                "region_id": "handwriting",
                "revision": 1,
                "request_id": "keep-1",
                "action": "keep_unresolved",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    region = next(
        item
        for item in payload["result"]["pages"][0]["regions"]
        if item["id"] == "handwriting"
    )
    assert payload["revision"] == 2
    assert payload["recovery_outcome"]["status"] == "keep_unresolved"
    assert region["resolution"] == "conflicting"
    assert region["provider"] == "controlled-reader"
    assert region["text_provenance"]["human_review"]["action"] == ("keep_unresolved")
    assert "Return in 2 weeks" not in payload["result"]["pages"][0]["text"]["value"]


def test_stale_tab_can_refresh_and_retry_against_current_revision() -> None:
    app = create_app(StructuredDocumentReader())

    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_id = initial["session_id"]
        kept = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "handwriting",
                "revision": 1,
                "request_id": "keep-1",
                "action": "keep_unresolved",
            },
        )
        stale = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "handwriting",
                "revision": 1,
                "request_id": "stale-edit",
                "action": "edit",
                "text": "Stale edit",
            },
        )
        retry = client.post(
            f"/api/sessions/{session_id}/corrections",
            json={
                "page_number": 1,
                "region_id": "handwriting",
                "revision": 2,
                "request_id": "fresh-edit",
                "action": "edit",
                "text": "Fresh edit",
            },
        )

    assert kept.status_code == 200
    assert stale.status_code == 409
    stale_payload = stale.json()
    assert stale_payload["session_id"] == session_id
    assert stale_payload["revision"] == 2
    assert stale_payload["result"]["revision"] == 2
    assert stale_payload["recovery_outcome"]["status"] == "stale"
    assert retry.status_code == 200
    retry_payload = retry.json()
    assert retry_payload["revision"] == 3
    assert retry_payload["result"]["pages"][0]["text"]["value"].endswith("Fresh edit")


def test_delayed_reread_cannot_overwrite_newer_human_correction() -> None:
    started = threading.Event()
    release = threading.Event()

    class DelayedRereadStage:
        name = "handwriting"

        def apply(
            self,
            image_path: Path,
            page_number: int,
            regions: list[TextRegion],
        ) -> list[TextRegion]:
            return regions

        def review_region(
            self,
            image_path: Path,
            page_number: int,
            region: TextRegion,
        ) -> TextRegion:
            started.set()
            assert release.wait(timeout=2)
            return replace(region, text="Late machine reread")

    app = create_app(
        StructuredDocumentReader(),
        handwriting_stage=DelayedRereadStage(),
    )
    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_id = initial["session_id"]

        def reread() -> Any:
            return client.post(
                f"/api/sessions/{session_id}/handwriting",
                json={
                    "page_number": 1,
                    "region_id": "handwriting",
                    "revision": 1,
                    "request_id": "reread-1",
                },
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(reread)
            assert started.wait(timeout=2)
            correction = client.post(
                f"/api/sessions/{session_id}/corrections",
                json={
                    "page_number": 1,
                    "region_id": "handwriting",
                    "revision": 1,
                    "request_id": "edit-1",
                    "action": "edit",
                    "text": "Human correction",
                },
            )
            release.set()
            delayed = future.result(timeout=2)
        stored = client.get(f"/api/sessions/{session_id}/result.json?revision=2").json()

    assert correction.status_code == 200
    assert correction.json()["revision"] == 2
    assert delayed.status_code == 409
    assert delayed.json()["recovery_outcome"]["status"] == "stale"
    assert "Human correction" in stored["pages"][0]["text"]["value"]
    assert "Late machine reread" not in json.dumps(stored)
    region = next(
        item for item in stored["pages"][0]["regions"] if item["id"] == "handwriting"
    )
    assert region["provider"] == "human-review"
    assert region["confidence"] is None
    assert [item["decision_state"] for item in region["alternatives"]] == [
        "superseded",
        "rejected",
    ]


def test_concurrent_revision_responses_keep_their_own_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_started = threading.Event()
    release_response = threading.Event()
    original_response = demo_module.JSONResponse

    class DelayedJSONResponse(original_response):
        def __init__(self, content: Any, *args: Any, **kwargs: Any) -> None:
            request_id = (content.get("recovery_outcome") or {}).get("request_id")
            if request_id == "edit-a":
                response_started.set()
                assert release_response.wait(timeout=2)
            super().__init__(content, *args, **kwargs)

    class TwoReviewReader:
        name = "two-review-reader"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id=region_id,
                    kind="handwriting",
                    text="uncertain",
                    confidence=0.4,
                    bounding_box=BoundingBox(0, top, 50, top + 10),
                    reading_order=index,
                    provider=self.name,
                    resolution="conflicting",
                )
                for index, (region_id, top) in enumerate(
                    (("region-a", 10), ("region-b", 30)), start=1
                )
            ]

    monkeypatch.setattr(demo_module, "JSONResponse", DelayedJSONResponse)
    app = create_app(TwoReviewReader())
    with TestClient(app) as client:
        initial = client.post(
            "/api/process",
            files={"file": ("page.png", _page_png(), "image/png")},
        ).json()
        session_id = initial["session_id"]

        def edit_first() -> Any:
            return client.post(
                f"/api/sessions/{session_id}/corrections",
                json={
                    "page_number": 1,
                    "region_id": "region-a",
                    "revision": 1,
                    "request_id": "edit-a",
                    "action": "edit",
                    "text": "first correction",
                },
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(edit_first)
            assert response_started.wait(timeout=2)
            second = client.post(
                f"/api/sessions/{session_id}/corrections",
                json={
                    "page_number": 1,
                    "region_id": "region-b",
                    "revision": 2,
                    "request_id": "edit-b",
                    "action": "edit",
                    "text": "second correction",
                },
            )
            release_response.set()
            first = future.result(timeout=2)

    assert first.json()["revision"] == 2
    assert first.json()["recovery_outcome"]["request_id"] == "edit-a"
    assert second.json()["revision"] == 3
    assert second.json()["recovery_outcome"]["request_id"] == "edit-b"


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
