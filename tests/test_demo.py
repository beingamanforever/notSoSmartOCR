from __future__ import annotations

import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextAlternative, TextRegion
from ocr_pipeline.demo import create_app
from ocr_pipeline.providers import ReaderError


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


def test_demo_processes_multi_page_tiff_and_clears_session() -> None:
    app = create_app(ControlledReader())

    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["cache-control"] == "no-store"
        assert "Clinical OCR Workbench" in index.text
        assert 'id="page-canvas"' in index.text
        assert 'id="download-markdown"' in index.text
        assert 'canvas.addEventListener("click"' in index.text

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
            "queue_seconds",
            "pipeline_seconds",
            "preview_seconds",
            "total_seconds",
            "pipeline_steps",
        }
        assert payload["timing"]["pipeline_seconds"] == payload["elapsed_seconds"]
        assert set(payload["timing"]["pipeline_steps"]) == {
            "prepare",
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
        assert json.loads(json_download.text) == result
        assert "ocr-result.json" in json_download.headers["content-disposition"]

        markdown = client.get(f"/api/sessions/{session_id}/result.md")
        assert markdown.status_code == 200
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


def test_demo_serializes_shared_reader_requests() -> None:
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
    assert reader.maximum_active == 1
    assert all(
        response.json()["timing"]["queue_seconds"] >= 0 for response in responses
    )


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


def _page_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(output, format="PNG")
    return output.getvalue()
