"""Local web demo for inspecting evidence-linked OCR results."""

from __future__ import annotations

import copy
import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contracts import BoundingBox, PageResult, RegionStage, TextAlternative, TextRegion
from .cross_page_tables import CrossPageTableStage
from .falcon import is_verified_falcon_model_provenance
from .pipeline import (
    IMAGE_SUFFIXES,
    PipelineError,
    _PreparedPages,
    _prepare_pages,
    _region_needs_review,
    _source_kind,
)
from .pipeline import process_document
from .orientation import OrientationReader
from .preprocessing import PageFrameReader, RoutedTesseractReader
from .providers import LocalReader, ReaderError, TesseractReader
from .rendering import render_evidence, render_page_markdown
from .table_topology import TableTopologyError, validate_table_topology

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from starlette.middleware.gzip import GZipMiddleware
except ImportError:  # pragma: no cover - exercised only without demo dependencies
    FastAPI = None
    File = None
    HTTPException = None
    Request = None
    UploadFile = None
    FileResponse = None
    HTMLResponse = None
    JSONResponse = None
    Response = None
    GZipMiddleware = None


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_PAGES = 32
MAX_PAGE_PIXELS = 32_000_000
MAX_DECODED_PIXELS = 200_000_000
ACCEPTED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
EXAMPLE_FILES = {
    "architecture": "artifacts/ocr-pipeline-architecture.pdf",
    "hard-case-routing": "artifacts/hard-case-routing.pdf",
    "handwriting": "artifacts/demo/handwriting_notes.png",
    "contract-agreement": "artifacts/demo/contract_agreement.png",
    "contract-amendment": "artifacts/demo/contract_amendment.png",
    "academic-paper": "artifacts/demo/academic_paper.png",
    "code": "artifacts/demo/code_document.png",
    "financial-table": "artifacts/demo/financial_table.png",
    "scanned-form": "artifacts/demo/scanned_form.png",
}
TABLE_EXAMPLE_NAME = "clinical-table"
PRESENTATION_CATEGORIES = {
    "plain",
    "text",
    "table",
    "formula",
    "caption",
    "footnote",
    "list-item",
    "page-footer",
    "page-header",
    "section-header",
    "title",
}
PRESENTATION_HTML_TAGS = {
    "table",
    "caption",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "sup",
    "sub",
}
PRESENTATION_SOURCE_CONFIDENCE = 0.9
PRESENTATION_SOURCE_COVERAGE = 0.8
PRESENTATION_TABLE_SOURCE_COVERAGE = 0.9
PRESENTATION_CONTROL_GLYPHS = frozenset("☑☐✓✔✗✘♡♥□◫")
PRESENTATION_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[./#@'-][^\W_]+)*", re.UNICODE)
PRESENTATION_CHECKLIST_PATTERN = re.compile(r"(?m)^\s*[-*]\s+\[[xX ?]\]\s+")
LOGGER = logging.getLogger(__name__)
FORMULA_OPERATOR_ALIASES = {
    r"\cdot": "*",
    r"\times": "*",
    r"\div": "/",
    r"\le": "<=",
    r"\leq": "<=",
    r"\ge": ">=",
    r"\geq": ">=",
    r"\ne": "!=",
    r"\neq": "!=",
}
FORMULA_FRACTION_COMMANDS = frozenset({r"\frac", r"\dfrac", r"\tfrac"})
FORMULA_FORMAT_COMMANDS = frozenset(
    {
        r"\mathbf",
        r"\mathrm",
        r"\mathit",
        r"\mathsf",
        r"\mathtt",
        r"\operatorname",
        r"\text",
    }
)
FORMULA_IGNORED_COMMANDS = frozenset(
    {
        r"\left",
        r"\right",
        r"\!",
        r"\,",
        r"\:",
        r"\;",
        r"\quad",
        r"\qquad",
    }
)


@dataclass(frozen=True)
class _PresentationTableCell:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class _PresentationTable:
    row_count: int
    column_count: int
    cells: tuple[_PresentationTableCell, ...]


class _PresentationTableParser(HTMLParser):
    """Validate one minimal HTML table without interpreting its text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self.rows: list[list[tuple[int, int]]] = []
        self.current_row: list[tuple[int, int]] | None = None
        self.table_count = 0
        self.failures: list[str] = []
        self.text_parts: list[str] = []
        self.cell_rows: list[list[dict[str, Any]]] = []
        self.current_cells: list[dict[str, Any]] | None = None
        self.current_cell: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        parent = self.stack[-1] if self.stack else None
        if tag not in PRESENTATION_HTML_TAGS:
            self._fail("disallowed_html_element")
        if not self._valid_parent(tag, parent):
            self._fail("malformed_html_table")
        attributes_valid = self._valid_attributes(tag, attrs)
        if not attributes_valid:
            self._fail("disallowed_html_attribute")
        if tag == "table":
            self.table_count += 1
        elif tag == "tr":
            if self.current_row is not None:
                self._fail("malformed_html_table")
            self.current_row = []
            self.current_cells = []
        elif tag in {"th", "td"} and self.current_row is not None:
            values = dict(attrs)
            colspan = int(values.get("colspan") or "1") if attributes_valid else 1
            rowspan = int(values.get("rowspan") or "1") if attributes_valid else 1
            self.current_row.append((colspan, rowspan))
            self.current_cell = {
                "colspan": colspan,
                "rowspan": rowspan,
                "text_parts": [],
            }
            if self.current_cells is not None:
                self.current_cells.append(self.current_cell)
        self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._fail("malformed_html_table")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if not self.stack or self.stack[-1] != tag:
            self._fail("malformed_html_table")
            return
        self.stack.pop()
        if tag == "tr":
            if self.current_row is None:
                self._fail("malformed_html_table")
            else:
                self.rows.append(self.current_row)
                self.cell_rows.append(self.current_cells or [])
            self.current_row = None
            self.current_cells = None
        elif tag in {"th", "td"}:
            self.current_cell = None

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if not self.stack or not {"caption", "th", "td"}.intersection(self.stack):
            self._fail("malformed_html_table")
            return
        self.text_parts.append(data)
        if self.current_cell is not None:
            self.current_cell["text_parts"].append(data)

    def handle_comment(self, data: str) -> None:
        self._fail("disallowed_html_content")

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")

    def handle_decl(self, decl: str) -> None:
        self._fail("disallowed_html_content")

    def unknown_decl(self, data: str) -> None:
        self._fail("disallowed_html_content")

    def valid(self) -> bool:
        if self.stack or self.table_count != 1 or not self.rows:
            self._fail("malformed_html_table")
        if not _rectangular_table(self.rows):
            self._fail("non_rectangular_html_table")
        return not self.failures

    def _fail(self, code: str) -> None:
        if code not in self.failures:
            self.failures.append(code)

    @staticmethod
    def _valid_parent(tag: str, parent: str | None) -> bool:
        if tag == "table":
            return parent is None
        if tag in {"caption", "thead", "tbody", "tfoot", "tr"}:
            return parent == "table" or (
                tag == "tr" and parent in {"thead", "tbody", "tfoot"}
            )
        if tag in {"sup", "sub"}:
            return parent in {"th", "td"}
        return tag in {"th", "td"} and parent == "tr"

    @staticmethod
    def _valid_attributes(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        names = [name.casefold() for name, _ in attrs]
        if len(names) != len(set(names)):
            return False
        if tag not in {"th", "td"}:
            return not attrs
        for name, value in attrs:
            if name.casefold() not in {"rowspan", "colspan"}:
                return False
            if value is None or not value.isdecimal() or not 1 <= int(value) <= 100:
                return False
        return True


@dataclass(frozen=True)
class CompositionDescriptor:
    """Static pipeline configuration shown before any model is exercised."""

    id: str
    label: str
    scope: str
    primary_ocr: str
    orientation: str
    tesseract_roles: tuple[str, ...]
    stages: tuple[str, ...]
    handwriting: str
    note: str
    build_label: str = "local-workbench"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RequestTooLarge(Exception):
    pass


class RequestLimitMiddleware:
    """Bound process-request bodies before multipart parsing can spool them."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if not _is_process_request(scope):
            await self.app(scope, receive, send)
            return

        scope.setdefault("state", {})["request_started"] = time.perf_counter()
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await _send_request_too_large(send, self.max_body_bytes)
                    return
            except ValueError:
                pass

        received_bytes = 0
        limit_exceeded = False

        async def handle_receive() -> Any:
            nonlocal limit_exceeded, received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    limit_exceeded = True
                    raise RequestTooLarge
            return message

        async def handle_send(message: Any) -> None:
            if not limit_exceeded:
                await send(message)

        try:
            await self.app(scope, handle_receive, handle_send)
        except RequestTooLarge:
            limit_exceeded = True
        if limit_exceeded:
            await _send_request_too_large(send, self.max_body_bytes)


def _is_process_request(scope: Any) -> bool:
    return (
        scope.get("type") == "http"
        and scope.get("method") == "POST"
        and scope.get("path") == "/api/process"
    )


async def _send_request_too_large(send: Any, max_body_bytes: int) -> None:
    limit_mb = max_body_bytes // (1024 * 1024)
    body = json.dumps(
        {"detail": f"Request exceeds the {limit_mb} MB transfer limit"}
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, Path]:
        while True:
            session_id = token_urlsafe(18)
            session_dir = self.root / session_id
            try:
                session_dir.mkdir()
            except FileExistsError:
                continue
            return session_id, session_dir

    def save(self, session_id: str, session: dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id] = session

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        shutil.rmtree(session["directory"], ignore_errors=True)
        return True

    def cleanup(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            shutil.rmtree(session["directory"], ignore_errors=True)


def _workbench_composition(
    reader: LocalReader,
    stages: Sequence[RegionStage],
    handwriting_stage: object | None = None,
) -> CompositionDescriptor:
    stage_names = tuple(stage.name for stage in stages)
    reader_chain = _reader_chain(reader)
    uses_tesseract = any(
        isinstance(item, (RoutedTesseractReader, TesseractReader))
        for item in reader_chain
    )
    uses_orientation = any(isinstance(item, OrientationReader) for item in reader_chain)
    return CompositionDescriptor(
        id=(
            "reduced-tesseract-workbench"
            if uses_tesseract
            else "custom-local-workbench"
        ),
        label=(
            "Reduced Tesseract workbench"
            if uses_tesseract
            else "Custom local workbench"
        ),
        scope="reduced" if uses_tesseract else "custom",
        primary_ocr=reader.name,
        orientation=(
            "configured via OrientationReader" if uses_orientation else "not configured"
        ),
        tesseract_roles=("primary OCR",) if uses_tesseract else (),
        stages=stage_names,
        handwriting=(
            "configured"
            if handwriting_stage is not None
            or any(callable(getattr(stage, "review_region", None)) for stage in stages)
            else "not configured"
        ),
        note=(
            "Configuration only. This is not the full verified GPU pipeline; "
            "readiness and execution are reported only after processing."
        ),
    )


def _reader_chain(reader: LocalReader) -> tuple[object, ...]:
    chain: list[object] = []
    current: object | None = reader
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        if isinstance(current, (PageFrameReader, OrientationReader)):
            current = current.reader
        else:
            current = None
    return tuple(chain)


def _katex_assets(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    resolved_root = root.resolve()
    javascript = resolved_root / "katex.js"
    stylesheet = resolved_root / "katex.css"
    if not javascript.is_file() or not stylesheet.is_file():
        raise ValueError("katex_asset_root must contain katex.js and katex.css")
    assets = {"katex.js": javascript, "katex.css": stylesheet}
    fonts = resolved_root / "fonts"
    if fonts.is_dir():
        for font in fonts.iterdir():
            if (
                font.is_file()
                and font.suffix.casefold() in {".woff", ".woff2", ".ttf"}
                and font.resolve().parent == fonts.resolve()
            ):
                assets[f"fonts/{font.name}"] = font
    return assets


def _katex_response(assets: dict[str, Path], asset_name: str, media_type: str) -> Any:
    asset = assets.get(asset_name)
    if asset is None:
        raise HTTPException(404, "KaTeX asset not found")
    return FileResponse(asset, media_type=media_type)


def create_app(
    reader: LocalReader | None = None,
    *,
    stages: Sequence[RegionStage] = (),
    cross_page_table_stage: CrossPageTableStage | None = None,
    handwriting_stage: object | None = None,
    presentation_reader: LocalReader | None = None,
    composition: CompositionDescriptor | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_pages: int = MAX_DOCUMENT_PAGES,
    max_page_pixels: int = MAX_PAGE_PIXELS,
    max_decoded_pixels: int = MAX_DECODED_PIXELS,
    max_presentation_pages: int = 4,
    katex_asset_root: Path | None = None,
    warmup_completed: bool = False,
) -> Any:
    """Create the local demo app with an injectable OCR reader."""
    if FastAPI is None:
        raise RuntimeError("Install FastAPI and python-multipart to run the OCR demo")
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if max_page_pixels <= 0:
        raise ValueError("max_page_pixels must be positive")
    if max_decoded_pixels <= 0:
        raise ValueError("max_decoded_pixels must be positive")
    if max_presentation_pages <= 0:
        raise ValueError("max_presentation_pages must be positive")
    katex_assets = _katex_assets(katex_asset_root)

    active_reader = reader or RoutedTesseractReader()
    active_stages = tuple(stages)
    active_composition = composition or _workbench_composition(
        active_reader,
        active_stages,
        handwriting_stage,
    )
    composition_payload = active_composition.to_dict()
    composition_payload["service_started_at"] = datetime.now(UTC).isoformat()
    composition_payload["warmup_completed"] = warmup_completed
    active_handwriting_stage = handwriting_stage or next(
        (
            stage
            for stage in active_stages
            if callable(getattr(stage, "review_region", None))
        ),
        None,
    )
    temporary_root = tempfile.TemporaryDirectory(prefix="ocr-demo-")
    session_root = Path(temporary_root.name)
    sessions = SessionStore(session_root)
    backend_version = _backend_version(active_reader)
    prepare_lock = threading.Lock()
    process_lock = threading.Lock()
    recovery_lock = threading.Lock()

    @asynccontextmanager
    async def handle_lifespan(_: Any):
        try:
            yield
        finally:
            sessions.cleanup()
            temporary_root.cleanup()

    app = FastAPI(title="Not So Smart OCR", lifespan=handle_lifespan)
    app.state.session_root = session_root
    app.state.sessions = sessions
    app.state.prepare_lock = prepare_lock
    app.state.process_lock = process_lock
    app.state.recovery_lock = recovery_lock
    app.state.composition = active_composition
    app.state.service_started_at = composition_payload["service_started_at"]
    app.add_middleware(
        RequestLimitMiddleware,
        max_body_bytes=max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def handle_no_store(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", response_class=HTMLResponse)
    def handle_index() -> Any:
        html = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
        asset_tags = ""
        if katex_assets:
            asset_tags = (
                '<link rel="stylesheet" href="/assets/katex/katex.css">\n'
                '  <script defer src="/assets/katex/katex.js"></script>'
            )
        return HTMLResponse(html.replace("__OCR_KATEX_ASSETS__", asset_tags))

    @app.get("/assets/katex/katex.css")
    def handle_katex_css() -> Any:
        return _katex_response(katex_assets, "katex.css", "text/css")

    @app.get("/assets/katex/katex.js")
    def handle_katex_javascript() -> Any:
        return _katex_response(
            katex_assets,
            "katex.js",
            "text/javascript",
        )

    @app.get("/assets/katex/fonts/{font_name}")
    def handle_katex_font(font_name: str) -> Any:
        return _katex_response(
            katex_assets,
            f"fonts/{font_name}",
            "font/woff2" if font_name.endswith(".woff2") else "font/woff",
        )

    @app.get("/api/composition")
    def handle_composition() -> Any:
        return JSONResponse(composition_payload)

    @app.get("/api/examples/{example_name}")
    def handle_example(example_name: str) -> Any:
        if example_name == TABLE_EXAMPLE_NAME:
            return Response(
                _table_example_png(),
                media_type="image/png",
                headers={
                    "Content-Disposition": (
                        'inline; filename="synthetic-clinical-table.png"'
                    )
                },
            )
        relative_path = EXAMPLE_FILES.get(example_name)
        if relative_path is None:
            raise HTTPException(404, "Example not found")
        example = Path(__file__).parents[2] / relative_path
        if not example.is_file():
            raise HTTPException(404, "Example not found")
        return FileResponse(
            example,
            filename=example.name,
            content_disposition_type="inline",
        )

    @app.post("/api/process")
    def handle_process(request: Request, file: UploadFile = File(...)) -> Any:
        original_name = _safe_filename(file.filename)
        suffix = Path(original_name).suffix.lower()
        if suffix not in ACCEPTED_SUFFIXES:
            accepted = ", ".join(sorted(ACCEPTED_SUFFIXES))
            raise HTTPException(415, f"Unsupported file type. Accepted: {accepted}")

        session_id, session_dir = sessions.create()
        source = session_dir / f"source{suffix}"
        try:
            total_started = getattr(
                request.state,
                "request_started",
                time.perf_counter(),
            )
            receive_seconds = time.perf_counter() - total_started
            save_started = time.perf_counter()
            _save_upload(file, source, max_upload_bytes)
            save_seconds = time.perf_counter() - save_started
            preview_queued = time.perf_counter()
            with prepare_lock:
                preview_queue_seconds = time.perf_counter() - preview_queued
                preview_started = time.perf_counter()
                prepared_pages, preview_failure = _render_previews(
                    source,
                    session_dir,
                    max_pages=max_pages,
                    max_page_pixels=max_page_pixels,
                    max_pixels=max_decoded_pixels,
                )
                preview_seconds = time.perf_counter() - preview_started
            preview_paths = list(prepared_pages.pages) if prepared_pages else []
            queued = time.perf_counter()
            pipeline_timings: dict[str, float] = {}
            stage_execution: list[dict[str, object]] = []
            page_execution: list[dict[str, object]] = []
            with process_lock:
                queue_seconds = time.perf_counter() - queued
                started = time.perf_counter()
                document = process_document(
                    source,
                    active_reader,
                    stages=active_stages,
                    cross_page_table_stage=cross_page_table_stage,
                    timings=pipeline_timings,
                    stage_execution=stage_execution,
                    page_execution=page_execution,
                    prepared_pages=prepared_pages,
                    max_pages=max_pages,
                    max_page_pixels=max_page_pixels,
                    max_pixels=max_decoded_pixels,
                )
                presentation_started = time.perf_counter()
                presentations = _read_presentations(
                    document.pages,
                    preview_paths,
                    presentation_reader,
                    max_pages=max_presentation_pages,
                    stage_execution=stage_execution,
                )
                if presentation_reader is not None:
                    pipeline_timings["stage.presentation"] = (
                        time.perf_counter() - presentation_started
                    )
                elapsed_seconds = time.perf_counter() - started
                coverage = _coverage_assessment(active_reader, len(document.pages))
            result = _sanitize_result(document.to_dict(), session_root)
            result["document_id"] = Path(original_name).stem
            result["source"]["name"] = original_name
            result["revision"] = 1
            total_seconds = time.perf_counter() - total_started
            presentation_payload = _sanitize_result(
                {
                    "schema_version": 2,
                    "pages": [
                        {"page_number": page_number, **presentation}
                        for page_number, presentation in sorted(presentations.items())
                    ],
                },
                session_root,
            )
            response = {
                "session_id": session_id,
                "revision": 1,
                "filename": original_name,
                "backend": active_reader.name,
                "backend_version": backend_version,
                "composition": composition_payload,
                "pipeline_stages": [
                    *(stage.name for stage in active_stages),
                    *(
                        [cross_page_table_stage.name]
                        if cross_page_table_stage is not None
                        else []
                    ),
                ],
                "stage_execution": [
                    {
                        **run,
                        "elapsed_seconds": round(
                            float(run.get("elapsed_seconds", 0.0)),
                            3,
                        ),
                    }
                    for run in stage_execution
                ],
                "page_execution": [
                    {
                        **run,
                        "queue_seconds": round(float(run["queue_seconds"]), 3),
                        "execution_seconds": round(float(run["execution_seconds"]), 3),
                        "steps": {
                            name: round(float(seconds), 3)
                            for name, seconds in dict(run["steps"]).items()
                        },
                    }
                    for run in page_execution
                ],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "timing": {
                    "receive_seconds": round(receive_seconds, 3),
                    "save_seconds": round(save_seconds, 3),
                    "preview_queue_seconds": round(preview_queue_seconds, 3),
                    "queue_seconds": round(queue_seconds, 3),
                    "pipeline_seconds": round(elapsed_seconds, 3),
                    "preview_seconds": round(preview_seconds, 3),
                    "total_seconds": round(total_seconds, 3),
                    "pipeline_steps": {
                        name: round(seconds, 3)
                        for name, seconds in pipeline_timings.items()
                    },
                },
                "geometry": "pixel coordinates",
                "coverage_assessment": coverage,
                "uncertainty": _uncertainty_summary(result),
                "recovery_outcome": None,
                "presentation": presentation_payload,
                "result": result,
                "page_images": [
                    f"/api/sessions/{session_id}/pages/{number}"
                    for number in range(1, len(preview_paths) + 1)
                ],
                "preview_failure": preview_failure,
            }
            sessions.save(
                session_id,
                {
                    "directory": session_dir,
                    "document": document,
                    "pages": preview_paths,
                    "response": response,
                    "revision": 1,
                },
            )
            return JSONResponse(response)
        except HTTPException:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        except Exception as error:
            shutil.rmtree(session_dir, ignore_errors=True)
            LOGGER.exception("OCR processing failed")
            raise HTTPException(
                500,
                f"OCR processing failed: {type(error).__name__}",
            ) from error
        finally:
            file.file.close()

    @app.get("/api/sessions/{session_id}/pages/{page_number}")
    def handle_page(session_id: str, page_number: int) -> Any:
        session = _get_session(sessions, session_id)
        pages = session["pages"]
        if page_number < 1 or page_number > len(pages):
            raise HTTPException(404, "Page not found")
        return FileResponse(pages[page_number - 1], media_type="image/png")

    @app.get("/api/sessions/{session_id}/result.json")
    def handle_json_download(session_id: str, revision: int | None = None) -> Any:
        with process_lock:
            session = _get_session(sessions, session_id)
            if revision is not None and revision != session["revision"]:
                raise HTTPException(409, "The requested revision is no longer current")
            content = json.dumps(
                session["response"]["result"], indent=2, ensure_ascii=False
            )
        return Response(
            content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="ocr-result.json"'},
        )

    @app.get("/api/sessions/{session_id}/result.md")
    def handle_markdown_download(session_id: str, revision: int | None = None) -> Any:
        with process_lock:
            session = _get_session(sessions, session_id)
            if revision is not None and revision != session["revision"]:
                raise HTTPException(409, "The requested revision is no longer current")
            content = _result_markdown(session["response"])
        return Response(
            content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="ocr-result.md"'},
        )

    @app.post("/api/sessions/{session_id}/handwriting")
    def handle_handwriting(
        session_id: str,
        payload: dict[str, Any],
    ) -> Any:
        if active_handwriting_stage is None:
            raise HTTPException(409, "Handwriting rereading is not configured")
        page_number, region_id, base_revision, request_id = _revision_request(payload)
        session = _get_session(sessions, session_id)
        with process_lock:
            if session["revision"] != base_revision:
                return _stale_recovery_response(
                    request_id,
                    page_number,
                    region_id,
                    base_revision,
                    session["revision"],
                    session["response"],
                )
            page, region_index = _session_page_region(session, page_number, region_id)
            if page_number > len(session["pages"]):
                raise HTTPException(409, "Page preview is unavailable")
            original_region = copy.deepcopy(page.regions[region_index])
            image_path = session["pages"][page_number - 1]

        reread_started = time.perf_counter()
        try:
            with recovery_lock:
                reviewed = active_handwriting_stage.review_region(
                    image_path,
                    page_number,
                    copy.deepcopy(original_region),
                )
        except ReaderError as error:
            with process_lock:
                current_session = _get_session(sessions, session_id)
                current_revision = current_session["revision"]
                current_response = copy.deepcopy(current_session["response"])
            if current_revision != base_revision:
                return _stale_recovery_response(
                    request_id,
                    page_number,
                    region_id,
                    base_revision,
                    current_revision,
                    current_response,
                )
            return JSONResponse(
                {
                    "revision": current_revision,
                    "recovery_outcome": {
                        "kind": "handwriting_reread",
                        "status": "failed",
                        "request_id": request_id,
                        "page_number": page_number,
                        "region_id": region_id,
                        "base_revision": base_revision,
                        "code": error.code,
                        "message": str(error),
                    },
                },
                status_code=422,
            )
        reread_seconds = time.perf_counter() - reread_started

        with process_lock:
            session = _get_session(sessions, session_id)
            if session["revision"] != base_revision:
                return _stale_recovery_response(
                    request_id,
                    page_number,
                    region_id,
                    base_revision,
                    session["revision"],
                    session["response"],
                )
            page, region_index = _session_page_region(session, page_number, region_id)
            changed = page.regions[region_index] != reviewed
            page.regions[region_index] = reviewed
            page.text = render_evidence(page.regions)
            page.route = "review"
            if changed:
                session["revision"] += 1

            response = session["response"]
            reread_stage = f"{active_handwriting_stage.name}.manual-reread"
            elapsed_seconds = round(reread_seconds, 3)
            response["stage_execution"].append(
                {
                    "page_number": page_number,
                    "stage": reread_stage,
                    "status": (
                        "productive" if original_region != reviewed else "fired"
                    ),
                    "input_regions": len(page.regions),
                    "output_regions": len(page.regions),
                    "added_regions": 0,
                    "removed_regions": 0,
                    "modified_regions": int(original_region != reviewed),
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            timing_name = f"stage.{reread_stage}"
            pipeline_steps = response["timing"]["pipeline_steps"]
            pipeline_steps[timing_name] = round(
                float(pipeline_steps.get(timing_name, 0.0)) + elapsed_seconds,
                3,
            )
            response["timing"]["manual_reread_seconds"] = round(
                float(response["timing"].get("manual_reread_seconds", 0.0))
                + elapsed_seconds,
                3,
            )
            _refresh_session_response(
                session,
                session_root,
                page,
                original_region,
                reviewed,
                presentation_reader,
            )
            attempt = (
                reviewed.structure.get("handwriting_attempt")
                if isinstance(reviewed.structure, dict)
                else None
            )
            recorded_outcome = (
                attempt.get("outcome") if isinstance(attempt, dict) else None
            )
            pending_candidate = any(
                alternative.decision_state == "pending"
                for alternative in reviewed.alternatives
            )
            recovery_status = (
                recorded_outcome
                if isinstance(recorded_outcome, str)
                and recorded_outcome
                in {
                    "unchanged_after_reread",
                    "candidate_pending",
                    "corrected",
                    "unresolved",
                    "failed",
                }
                else (
                    "corrected"
                    if original_region.text != reviewed.text
                    else (
                        "candidate_pending"
                        if pending_candidate
                        else "unchanged_after_reread"
                    )
                )
            )
            response["recovery_outcome"] = {
                "kind": "handwriting_reread",
                "status": recovery_status,
                "request_id": request_id,
                "page_number": page_number,
                "region_id": region_id,
                "base_revision": base_revision,
                "revision": session["revision"],
            }
            response_payload = copy.deepcopy(response)
        return JSONResponse(response_payload)

    @app.post("/api/sessions/{session_id}/corrections")
    def handle_correction(session_id: str, payload: dict[str, Any]) -> Any:
        page_number, region_id, base_revision, request_id = _revision_request(payload)
        action = payload.get("action")
        if action not in {"accept", "edit", "keep_unresolved"}:
            raise HTTPException(400, "action must be accept, edit, or keep_unresolved")

        session = _get_session(sessions, session_id)
        with process_lock:
            if session["revision"] != base_revision:
                return _stale_recovery_response(
                    request_id,
                    page_number,
                    region_id,
                    base_revision,
                    session["revision"],
                    session["response"],
                )
            page, region_index = _session_page_region(session, page_number, region_id)
            original_region = copy.deepcopy(page.regions[region_index])
            pending_alternatives = [
                alternative
                for alternative in original_region.alternatives
                if alternative.decision_state == "pending"
            ]
            if original_region.resolution == "resolved" and not pending_alternatives:
                raise HTTPException(409, "Region has no pending alternatives")
            reviewed = copy.deepcopy(original_region)
            accepted_alternative: TextAlternative | None = None
            accepted_index: int | None = None
            if action == "accept":
                alternative_index = payload.get("alternative_index", 0)
                if (
                    isinstance(alternative_index, bool)
                    or not isinstance(alternative_index, int)
                    or alternative_index < 0
                    or alternative_index >= len(reviewed.alternatives)
                ):
                    raise HTTPException(400, "alternative_index is invalid")
                accepted_alternative = reviewed.alternatives[alternative_index]
                if accepted_alternative.decision_state != "pending":
                    raise HTTPException(409, "Alternative is not pending")
                if _region_semantic_kind(reviewed) == "formula":
                    formula_failures = _validate_presentation_formula(
                        accepted_alternative.text
                    )
                    if formula_failures:
                        raise HTTPException(
                            422,
                            "Formula candidate failed syntax validation: "
                            f"{formula_failures[0]['code']}",
                        )
                accepted_index = alternative_index
                reviewed.text = accepted_alternative.text
            elif action == "edit":
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise HTTPException(400, "text is required for edit")
                if _region_semantic_kind(reviewed) == "formula":
                    formula_failures = _validate_presentation_formula(text)
                    if formula_failures:
                        raise HTTPException(
                            422,
                            "Formula edit failed syntax validation: "
                            f"{formula_failures[0]['code']}",
                        )
                reviewed.text = text

            review_record = {
                "action": action,
                "request_id": request_id,
                "base_revision": base_revision,
                "original_provider": original_region.provider,
                "correcting_provider": (
                    accepted_alternative.provider
                    if accepted_alternative is not None
                    else "human-review"
                ),
            }
            if accepted_alternative is not None:
                review_record["accepted_alternative"] = asdict(accepted_alternative)
            reviewed_structure = {
                **(reviewed.structure if isinstance(reviewed.structure, dict) else {}),
                "human_review": review_record,
            }
            if action != "keep_unresolved":
                for review_name in ("handwriting_review", "formula_review"):
                    review = reviewed_structure.get(review_name)
                    if isinstance(review, dict):
                        reviewed_structure[review_name] = {
                            **review,
                            "required": False,
                            "resolved_by": "human_review",
                        }
                if _region_semantic_kind(reviewed) == "formula":
                    reviewed_structure["formula_recognition"] = "human_accepted"
            reviewed.structure = reviewed_structure
            if action != "keep_unresolved":
                history = [
                    TextAlternative(
                        text=original_region.text,
                        confidence=original_region.confidence,
                        provider=original_region.provider,
                        text_provenance=copy.deepcopy(original_region.text_provenance),
                        decision_state="superseded",
                    )
                ]
                for index, alternative in enumerate(original_region.alternatives):
                    historical = copy.deepcopy(alternative)
                    if historical.decision_state == "pending":
                        historical.decision_state = (
                            "accepted" if index == accepted_index else "rejected"
                        )
                    history.append(historical)
                reviewed.resolution = "resolved"
                reviewed.alternatives = history
                if accepted_alternative is not None:
                    reviewed.confidence = accepted_alternative.confidence
                    reviewed.provider = accepted_alternative.provider
                    reviewed.text_provenance = {
                        **(
                            copy.deepcopy(accepted_alternative.text_provenance)
                            if isinstance(
                                accepted_alternative.text_provenance,
                                dict,
                            )
                            else {}
                        ),
                        "human_review": review_record,
                    }
                else:
                    reviewed.confidence = None
                    reviewed.provider = "human-review"
                    reviewed.text_provenance = {"human_review": review_record}
            else:
                reviewed.text_provenance = {
                    **(
                        reviewed.text_provenance
                        if isinstance(reviewed.text_provenance, dict)
                        else {}
                    ),
                    "human_review": review_record,
                }

            page.regions[region_index] = reviewed
            page.text = render_evidence(page.regions)
            reader_review = getattr(active_reader, "page_needs_review", None)
            page.route = (
                "review"
                if page.failure_ids
                or bool(callable(reader_review) and reader_review(page.page_number))
                or any(_region_needs_review(region) for region in page.regions)
                else "accept_local"
            )
            session["revision"] += 1
            _refresh_session_response(
                session,
                session_root,
                page,
                original_region,
                reviewed,
                presentation_reader,
            )
            response = session["response"]
            response["recovery_outcome"] = {
                "kind": "human_review",
                "status": action,
                "request_id": request_id,
                "page_number": page_number,
                "region_id": region_id,
                "base_revision": base_revision,
                "revision": session["revision"],
            }
            response_payload = copy.deepcopy(response)
        return JSONResponse(response_payload)

    @app.delete("/api/sessions/{session_id}")
    def handle_clear(session_id: str) -> Any:
        if not sessions.delete(session_id):
            raise HTTPException(404, "Session not found")
        return Response(status_code=204)

    return app


def _get_session(sessions: SessionStore, session_id: str) -> dict[str, Any]:
    try:
        return sessions.get(session_id)
    except KeyError as error:
        raise HTTPException(404, "Session not found") from error


def _revision_request(payload: dict[str, Any]) -> tuple[int, str, int, str]:
    page_number = payload.get("page_number")
    region_id = payload.get("region_id")
    revision = payload.get("revision")
    request_id = payload.get("request_id")
    if (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
        or not isinstance(region_id, str)
        or not region_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(request_id, str)
        or not request_id
    ):
        raise HTTPException(
            400,
            "page_number, region_id, revision, and request_id are required",
        )
    return page_number, region_id, revision, request_id


def _stale_recovery_response(
    request_id: str,
    page_number: int,
    region_id: str,
    base_revision: int,
    revision: int,
    response: dict[str, Any],
) -> JSONResponse:
    payload = copy.deepcopy(response)
    payload["revision"] = revision
    payload["recovery_outcome"] = {
        "kind": "revision_conflict",
        "status": "stale",
        "request_id": request_id,
        "page_number": page_number,
        "region_id": region_id,
        "base_revision": base_revision,
        "revision": revision,
    }
    return JSONResponse(
        payload,
        status_code=409,
    )


def _session_page_region(
    session: dict[str, Any],
    page_number: int,
    region_id: str,
) -> tuple[PageResult, int]:
    page = next(
        (
            candidate
            for candidate in session["document"].pages
            if candidate.page_number == page_number
        ),
        None,
    )
    if page is None:
        raise HTTPException(404, "Page not found")
    region_index = next(
        (index for index, region in enumerate(page.regions) if region.id == region_id),
        None,
    )
    if region_index is None:
        raise HTTPException(404, "Region not found")
    return page, region_index


def _refresh_session_response(
    session: dict[str, Any],
    session_root: Path,
    page: PageResult,
    original_region: TextRegion,
    reviewed: TextRegion,
    presentation_reader: LocalReader | None,
) -> None:
    response = session["response"]
    result = _sanitize_result(session["document"].to_dict(), session_root)
    result["document_id"] = Path(response["filename"]).stem
    result["source"]["name"] = response["filename"]
    result["revision"] = session["revision"]
    response["revision"] = session["revision"]
    response["result"] = result
    response["uncertainty"] = _uncertainty_summary(result)

    if _visible_literal(
        original_region.text,
        original_region.resolution,
    ) == _visible_literal(reviewed.text, reviewed.resolution):
        return
    if presentation_reader is None or page.page_number > len(session["pages"]):
        return

    refreshed = _read_presentations(
        [page],
        [session["pages"][page.page_number - 1]],
        presentation_reader,
        max_pages=1,
        stage_execution=response["stage_execution"],
    )
    presentation_pages = [
        item
        for item in response.get("presentation", {}).get("pages", [])
        if item.get("page_number") != page.page_number
    ]
    if page.page_number in refreshed:
        presentation_pages.append(
            {"page_number": page.page_number, **refreshed[page.page_number]}
        )
    response["presentation"] = _sanitize_result(
        {
            "schema_version": 2,
            "pages": sorted(
                presentation_pages,
                key=lambda item: item["page_number"],
            ),
        },
        session_root,
    )


def _safe_filename(filename: str | None) -> str:
    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return name or "upload"


def _table_example_png() -> bytes:
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=36)
    body_font = ImageFont.load_default(size=24)
    note_font = ImageFont.load_default(size=18)
    navy = (21, 36, 59)
    neutral = (238, 236, 229)
    amber = (255, 242, 213)
    rows = (
        ("Test", "Result", "Unit", "Flag"),
        ("Hemoglobin", "13.8", "g/dL", "Normal"),
        ("Platelets", "245", "10^3/uL", "Normal"),
        ("Glucose", "108", "mg/dL", "High"),
        ("Creatinine", "0.9", "mg/dL", "Normal"),
    )
    columns = (80, 460, 680, 900, 1120)
    row_height = 76
    table_top = 190

    draw.text((80, 60), "Synthetic lab results", fill=navy, font=title_font)
    draw.text(
        (80, 120),
        "Generated locally for table-structure testing",
        fill=navy,
        font=note_font,
    )
    for row_index, row in enumerate(rows):
        top = table_top + row_index * row_height
        bottom = top + row_height
        draw.rectangle(
            (columns[0], top, columns[-1], bottom),
            fill=amber
            if row_index == 0
            else neutral
            if row_index % 2 == 0
            else "white",
            outline=navy,
            width=3,
        )
        for column_index, text in enumerate(row):
            left = columns[column_index]
            right = columns[column_index + 1]
            draw.line((right, top, right, bottom), fill=navy, width=3)
            draw.text((left + 16, top + 22), text, fill=navy, font=body_font)
    draw.text(
        (80, 620),
        "Synthetic example - no patient data",
        fill=navy,
        font=note_font,
    )

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _save_upload(file: Any, destination: Path, max_upload_bytes: int) -> None:
    total_bytes = 0
    with destination.open("wb") as output:
        while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise HTTPException(
                    413,
                    f"Upload exceeds the {max_upload_bytes // (1024 * 1024)} MB limit",
                )
            output.write(chunk)
    if total_bytes == 0:
        raise HTTPException(400, "The uploaded file is empty")


def _render_previews(
    source: Path,
    session_dir: Path,
    *,
    max_pages: int,
    max_page_pixels: int,
    max_pixels: int,
) -> tuple[_PreparedPages | None, dict[str, str] | None]:
    preview_dir = session_dir / "pages"
    preview_dir.mkdir()
    try:
        prepared = _prepare_pages(
            source,
            _source_kind(source),
            preview_dir,
            300,
            "pdftoppm",
            max_pages=max_pages,
            max_page_pixels=max_page_pixels,
            max_pixels=max_pixels,
        )
    except PipelineError as error:
        if error.code in {"page_limit_exceeded", "pixel_limit_exceeded"}:
            raise HTTPException(413, str(error)) from error
        return None, {
            "stage": error.stage,
            "code": error.code,
            "message": "Page preview could not be rendered",
        }

    preview_paths: list[Path] = []
    for page_number, page in enumerate(prepared.pages, start=1):
        destination = preview_dir / f"preview-{page_number}.png"
        if page != destination:
            try:
                with Image.open(page) as image:
                    image.save(destination, format="PNG")
            except OSError:
                return None, {
                    "stage": "image",
                    "code": "invalid_image",
                    "message": "Page preview could not be rendered",
                }
        preview_paths.append(destination)
    return _PreparedPages(prepared.source, tuple(preview_paths)), None


def _read_presentations(
    pages: Sequence[PageResult],
    image_paths: Sequence[Path],
    reader: LocalReader | None,
    *,
    max_pages: int,
    stage_execution: list[dict[str, object]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Generate review-only page drafts without changing canonical evidence."""
    if reader is None:
        return {}

    presentations: dict[int, dict[str, Any]] = {}
    attempted = 0
    for page, image_path in zip(pages, image_paths, strict=False):
        if page.route != "review":
            continue
        page_started = time.perf_counter()
        if attempted >= max_pages:
            presentation = {
                "status": "skipped",
                "provider": reader.name,
                "message": "Review draft page limit reached",
                "canonical_unchanged": True,
            }
        else:
            attempted += 1
            presentation = _read_page_presentation(page, image_path, reader)
        presentations[page.page_number] = presentation
        if stage_execution is not None:
            _record_presentation_execution(
                stage_execution,
                page,
                presentation,
                time.perf_counter() - page_started,
            )
    return presentations


def _read_page_presentation(
    page: PageResult,
    image_path: Path,
    reader: LocalReader,
) -> dict[str, Any]:
    try:
        read_page = getattr(reader, "read_page", None)
        if callable(read_page):
            regions = read_page(image_path, page)
        else:
            regions = reader.read(image_path, page.page_number)
    except ReaderError as error:
        return {
            "status": "failed",
            "provider": reader.name,
            "failure": {"code": error.code, "message": str(error)},
            "canonical_unchanged": True,
        }
    if regions == []:
        return {
            "status": "skipped",
            "provider": reader.name,
            "message": "No eligible positioned regions for presentation",
            "canonical_unchanged": True,
        }
    if not _valid_presentation_regions(regions, page):
        return {
            "status": "failed",
            "provider": reader.name,
            "failure": {
                "code": "invalid_presentation_output",
                "message": (
                    "Presentation reader must return one full-page Markdown region "
                    "or positioned category regions"
                ),
            },
            "canonical_unchanged": True,
        }
    blocks = [
        _presentation_block(region, len(regions) == 1)
        for region in sorted(
            regions,
            key=lambda item: (item.reading_order, item.id),
        )
    ]
    _assign_presentation_rendering(blocks, page)
    failures = [
        failure for block in blocks for failure in block["validation"]["failures"]
    ]
    return {
        "status": "review_draft",
        "provider": reader.name,
        "blocks": blocks,
        "validation": {
            "status": "failed" if failures else "passed",
            "failures": failures,
            "termination_observed": False,
        },
        "canonical_unchanged": True,
    }


def _assign_presentation_rendering(
    blocks: list[dict[str, Any]], page: PageResult
) -> None:
    sources = {region.id: region for region in page.regions}
    linked_labels = {
        str(label_id)
        for region in page.regions
        if _region_semantic_kind(region) == "control"
        for label_id in (
            (region.structure or {}).get("label_evidence_ids", [])
            if isinstance(region.structure, dict)
            else []
        )
    }
    source_counts = Counter(
        source_id
        for block in blocks
        if isinstance(block.get("provenance"), dict)
        if isinstance(
            source_id := block.get("provenance", {}).get("source_region_id"), str
        )
    )
    claimed_sources: set[str] = set()
    for block in blocks:
        provenance = block.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        source_id = provenance.get("source_region_id")
        source = sources.get(source_id) if isinstance(source_id, str) else None
        _validate_presentation_source_alignment(block, source, sources)
        reason = _presentation_fallback_reason(
            block,
            source,
            sources,
            linked_labels,
            source_counts,
            claimed_sources,
        )
        if reason is None:
            claimed_sources.add(source_id)
            block["rendering"] = {
                "status": "selected",
                "source_region_id": source_id,
            }
            if block.get("category") in {"table", "formula"}:
                block["rendering"].update(
                    _presentation_source_metadata(page.page_number, source, sources)
                )
            continue
        block["rendering"] = {
            "status": "canonical_fallback",
            "reason": reason,
        }
        if isinstance(source_id, str):
            block["rendering"]["source_region_id"] = source_id


def _validate_presentation_source_alignment(
    block: dict[str, Any], source: TextRegion | None, sources: dict[str, TextRegion]
) -> None:
    if source is None:
        return
    raw_text = str(block.get("raw_text", "")).strip()
    validation = block.get("validation")
    if not isinstance(validation, dict):
        return
    failures = validation.get("failures")
    if not isinstance(failures, list):
        return
    evidence = _presentation_source_evidence(source, sources)
    supported_text = _presentation_canonical_text(source, evidence)
    unsupported_control = False
    if PRESENTATION_CHECKLIST_PATTERN.search(raw_text):
        unsupported_control = True
        failures.append(
            {
                "code": "unsupported_control_syntax",
                "message": (
                    "Generated checklist syntax is not supported by canonical control evidence"
                ),
            }
        )
    generated_glyphs = PRESENTATION_CONTROL_GLYPHS.intersection(raw_text)
    supported_glyphs = PRESENTATION_CONTROL_GLYPHS.intersection(supported_text)
    if generated_glyphs - supported_glyphs:
        unsupported_control = True
        failures.append(
            {
                "code": "unsupported_control_glyph",
                "message": (
                    "Generated checkbox or mark is not supported by canonical control evidence"
                ),
            }
        )
    if unsupported_control:
        validation["status"] = "failed"
        return
    if block.get("category") == "table":
        if _presentation_source_is_fully_resolved(
            source, sources
        ) or _verified_image_grounded_presentation(block):
            _validate_presentation_table_alignment(
                raw_text,
                source,
                evidence,
                sources,
                failures,
            )
        if failures:
            validation["status"] = "failed"
        return
    if failures:
        validation["status"] = "failed"
    aligned_text = _presentation_alignment_text(block.get("category"), raw_text)
    if not _presentation_source_is_fully_resolved(source, sources):
        return

    source_text = source.text.strip()
    maximum_length = max(256, len(source_text) * 3)
    if len(raw_text) > maximum_length:
        failures.append(
            {
                "code": "implausible_output_expansion",
                "message": (
                    "Generated text is implausibly long for its evidence-owned crop"
                ),
            }
        )
        validation["status"] = "failed"
        return

    supported_critical = Counter(_critical_presentation_tokens(supported_text))
    generated_critical = Counter(_critical_presentation_tokens(aligned_text))
    if generated_critical - supported_critical:
        failures.append(
            {
                "code": "unsupported_critical_token",
                "message": (
                    "Generated number or identifier is not present in canonical evidence"
                ),
            }
        )

    category = block.get("category")
    generated_tokens = Counter(_presentation_tokens(aligned_text))
    source_tokens = (
        Counter(_presentation_tokens(supported_text))
        if category in {"table", "formula"}
        else Counter(
            token
            for region in evidence
            if region.confidence is not None
            and region.confidence >= PRESENTATION_SOURCE_CONFIDENCE
            for token in _presentation_tokens(region.text)
        )
    )
    if source_tokens:
        preserved = sum((source_tokens & generated_tokens).values())
        coverage = preserved / sum(source_tokens.values())
        omitted_source = source_tokens - generated_tokens
        if (
            category in {"table", "formula"}
            and omitted_source
            or category not in {"table", "formula"}
            and coverage < PRESENTATION_SOURCE_COVERAGE
        ):
            failures.append(
                {
                    "code": "source_evidence_omitted",
                    "message": (
                        "Generated text omitted too much high-confidence canonical evidence"
                    ),
                }
            )
    if (
        not failures
        and (source.structure or {}).get("block_type") != "form_row"
        and not (source.structure or {}).get("layout_owner_id")
        and generated_tokens - Counter(_presentation_tokens(supported_text))
    ):
        failures.append(
            {
                "code": "unsupported_generated_content",
                "message": "Generated content is not present in canonical evidence",
            }
        )
    if category == "table" and _presentation_tokens(
        aligned_text
    ) != _presentation_tokens(supported_text):
        failures.append(
            {
                "code": "table_structure_changed",
                "message": (
                    "Generated table order or field association differs from canonical evidence"
                ),
            }
        )
    if category == "formula" and _formula_semantic_signature(
        raw_text
    ) != _formula_semantic_signature(supported_text):
        failures.append(
            {
                "code": "formula_semantics_changed",
                "message": (
                    "Generated formula structure differs from canonical evidence"
                ),
            }
        )
    if failures:
        validation["status"] = "failed"


def _validate_presentation_table_alignment(
    raw_text: str,
    source: TextRegion,
    evidence: list[TextRegion],
    sources: dict[str, TextRegion],
    failures: list[dict[str, str]],
) -> None:
    generated = _html_presentation_table(raw_text)
    canonical = _canonical_presentation_table(source, evidence)
    supported_text = (
        " ".join(cell.text for cell in canonical.cells)
        if canonical is not None
        else _presentation_canonical_text(source, evidence)
    )
    generated_text = (
        " ".join(cell.text for cell in generated.cells)
        if generated is not None
        else _presentation_alignment_text("table", raw_text)
    )
    supported_tokens = Counter(_presentation_tokens(supported_text))
    generated_tokens = Counter(_presentation_tokens(generated_text))
    supported_critical = Counter(_critical_presentation_tokens(supported_text))
    generated_critical = Counter(_critical_presentation_tokens(generated_text))

    if not supported_tokens and generated_tokens:
        _append_presentation_failure(
            failures,
            "unsupported_generated_content",
            "Generated structured content is not present in canonical evidence",
        )
    elif supported_tokens and _token_coverage(supported_tokens, generated_tokens) < (
        PRESENTATION_TABLE_SOURCE_COVERAGE
    ):
        _append_presentation_failure(
            failures,
            "source_evidence_omitted",
            "Generated table omitted too much canonical source evidence",
        )
    if generated_critical - supported_critical:
        _append_presentation_failure(
            failures,
            "unsupported_critical_token",
            "Generated number or identifier is not present in canonical evidence",
        )
    if supported_critical - generated_critical:
        _append_presentation_failure(
            failures,
            "critical_content_changed",
            "Generated table did not preserve every canonical number and identifier",
        )
    if generated_tokens - supported_tokens:
        _append_presentation_failure(
            failures,
            "unsupported_generated_content",
            "Generated structured content is not present in canonical evidence",
        )

    if generated is None:
        return

    if canonical is None:
        if _presentation_tokens(generated_text) != _presentation_tokens(supported_text):
            _append_presentation_failure(
                failures,
                "table_structure_changed",
                "Generated table order or field association differs from canonical evidence",
            )
        return

    if _table_topology_signature(generated) != _table_topology_signature(canonical):
        _append_presentation_failure(
            failures,
            "table_topology_mismatch",
            "Generated table rows, columns, or spans differ from canonical geometry",
        )
        return

    canonical_cells = {(cell.rows, cell.columns): cell for cell in canonical.cells}
    association_changed = False
    for generated_cell in generated.cells:
        canonical_cell = canonical_cells[(generated_cell.rows, generated_cell.columns)]
        source_cell_tokens = Counter(_presentation_tokens(canonical_cell.text))
        generated_cell_tokens = Counter(_presentation_tokens(generated_cell.text))
        if (
            source_cell_tokens
            and _token_coverage(source_cell_tokens, generated_cell_tokens)
            < PRESENTATION_TABLE_SOURCE_COVERAGE
        ):
            association_changed = True
            break
        if Counter(_critical_presentation_tokens(canonical_cell.text)) != Counter(
            _critical_presentation_tokens(generated_cell.text)
        ):
            association_changed = True
            break
    if association_changed:
        _append_presentation_failure(
            failures,
            "table_cell_association_changed",
            "Generated table moved or changed evidence associated with a canonical cell",
        )


def _html_presentation_table(raw_text: str) -> _PresentationTable | None:
    parser = _PresentationTableParser()
    try:
        parser.feed(raw_text)
        parser.close()
        if not parser.valid():
            return None
    except (TypeError, ValueError):
        return None
    return _position_presentation_cells(parser.cell_rows)


def _canonical_presentation_table(
    source: TextRegion, evidence: list[TextRegion]
) -> _PresentationTable | None:
    structure = source.structure if isinstance(source.structure, dict) else {}
    try:
        topology = validate_table_topology(structure)
    except TableTopologyError:
        return _markdown_presentation_table(source.text)

    evidence_index = {region.id: region for region in evidence}
    cells = tuple(
        _PresentationTableCell(
            rows=cell.rows,
            columns=cell.columns,
            text=_canonical_table_cell_text(cell.value, evidence_index),
        )
        for cell in topology.cells
    )
    return _PresentationTable(topology.row_count, topology.column_count, cells)


def _canonical_table_cell_text(cell: Any, evidence: dict[str, TextRegion]) -> str:
    text = str(cell.get("text", "")).strip()
    if text:
        return text
    evidence_ids = cell.get("evidence_ids", [])
    return " ".join(
        region.text.strip()
        for evidence_id in evidence_ids
        if isinstance(evidence_id, str)
        and (region := evidence.get(evidence_id)) is not None
        and region.text.strip()
    )


def _position_presentation_cells(
    rows: list[list[dict[str, Any]]],
) -> _PresentationTable | None:
    if not rows or any(not row for row in rows):
        return None
    occupied: set[tuple[int, int]] = set()
    cells: list[_PresentationTableCell] = []
    row_count = len(rows)
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row:
            while (row_index, column) in occupied:
                column += 1
            rowspan = int(cell["rowspan"])
            colspan = int(cell["colspan"])
            cell_rows = tuple(range(row_index, row_index + rowspan))
            cell_columns = tuple(range(column, column + colspan))
            if cell_rows[-1] >= row_count:
                return None
            occupied.update(
                (row, cell_column) for row in cell_rows for cell_column in cell_columns
            )
            cells.append(
                _PresentationTableCell(
                    rows=cell_rows,
                    columns=cell_columns,
                    text="".join(cell["text_parts"]).strip(),
                )
            )
            column += colspan
    column_count = max((column for _, column in occupied), default=-1) + 1
    if column_count <= 0 or len(occupied) != row_count * column_count:
        return None
    return _PresentationTable(row_count, column_count, tuple(cells))


def _markdown_presentation_table(text: str) -> _PresentationTable | None:
    rows = [
        _split_markdown_table_row(line)
        for line in text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    if len(rows) < 2 or not rows[0]:
        return None
    if len(rows[1]) != len(rows[0]) or not all(
        re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in rows[1]
    ):
        return None
    data_rows = [rows[0], *rows[2:]]
    column_count = len(data_rows[0])
    if any(len(row) != column_count for row in data_rows):
        return None
    cells = tuple(
        _PresentationTableCell((row,), (column,), cell.strip())
        for row, values in enumerate(data_rows)
        for column, cell in enumerate(values)
    )
    return _PresentationTable(len(data_rows), column_count, cells)


def _split_markdown_table_row(line: str) -> list[str]:
    values: list[str] = []
    value: list[str] = []
    escaped = False
    for character in line.strip()[1:-1]:
        if escaped:
            value.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            values.append("".join(value).strip())
            value = []
        else:
            value.append(character)
    if escaped:
        value.append("\\")
    values.append("".join(value).strip())
    return values


def _table_topology_signature(table: _PresentationTable) -> tuple[Any, ...]:
    return (
        table.row_count,
        table.column_count,
        tuple((cell.rows, cell.columns) for cell in table.cells),
    )


def _token_coverage(source: Counter[str], generated: Counter[str]) -> float:
    return sum((source & generated).values()) / sum(source.values())


def _append_presentation_failure(
    failures: list[dict[str, str]], code: str, message: str
) -> None:
    if not any(failure.get("code") == code for failure in failures):
        failures.append({"code": code, "message": message})


def _presentation_source_evidence(
    source: TextRegion, sources: dict[str, TextRegion]
) -> list[TextRegion]:
    structure = source.structure if isinstance(source.structure, dict) else {}
    child_ids = structure.get("child_evidence_ids", [])
    source_ids = {
        child_id
        for child_id in child_ids
        if isinstance(child_id, str) and child_id in sources
    }
    source_ids.update(
        region.id
        for region in sources.values()
        if isinstance(region.structure, dict)
        and region.structure.get("parent_id") == source.id
    )
    children = sorted(
        (sources[source_id] for source_id in source_ids),
        key=lambda region: (region.reading_order, region.id),
    )
    return children or [source]


def _presentation_source_metadata(
    page_number: int,
    source: TextRegion,
    sources: dict[str, TextRegion],
) -> dict[str, Any]:
    evidence = _presentation_source_evidence(source, sources)
    metadata: dict[str, Any] = {
        "source_page_number": page_number,
        "source_bounding_box": asdict(source.bounding_box),
        "source_evidence_ids": [region.id for region in evidence],
    }
    structure = source.structure if isinstance(source.structure, dict) else {}
    detection_confidence = structure.get("detection_confidence")
    if (
        _region_semantic_kind(source) == "table"
        and isinstance(detection_confidence, int | float)
        and not isinstance(detection_confidence, bool)
        and 0 <= detection_confidence <= 1
    ):
        metadata.update(
            {
                "source_confidence": float(detection_confidence),
                "source_confidence_kind": "Detection score",
                "source_confidence_scope": "table detection",
            }
        )
    elif source.confidence is not None:
        metadata.update(
            {
                "source_confidence": source.confidence,
                "source_confidence_kind": "Recognition score",
                "source_confidence_scope": "source region",
            }
        )
    return metadata


def _presentation_canonical_text(
    source: TextRegion,
    evidence: list[TextRegion],
) -> str:
    structure = source.structure if isinstance(source.structure, dict) else {}
    cells = structure.get("cells", [])
    if _region_semantic_kind(source) == "table" and isinstance(cells, list) and cells:
        try:
            topology = validate_table_topology(structure)
        except TableTopologyError:
            topology = None
        if topology is not None:
            parts = [
                str(cell.value.get("text", "")).strip()
                for cell in topology.cells
                if str(cell.value.get("text", "")).strip()
            ]
            if parts:
                return " ".join(parts)
    if evidence != [source]:
        return " ".join(
            region.text.strip()
            for region in evidence
            if region.resolution == "resolved" and region.text.strip()
        )
    return source.text.strip()


def _presentation_alignment_text(category: Any, raw_text: str) -> str:
    if category == "table":
        parser = _PresentationTableParser()
        parser.feed(raw_text)
        parser.close()
        return " ".join(parser.text_parts)
    if category == "formula":
        return re.sub(r"\\[A-Za-z]+|[{}_^$]", " ", raw_text)
    return raw_text


def _formula_semantic_signature(text: str) -> tuple[str, ...]:
    tokens = _formula_tokens(text)
    normalized, _ = _normalize_formula_tokens(tokens)
    return tuple(normalized)


def _formula_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace() or character == "$":
            index += 1
            continue
        if character == "\\":
            end = index + 1
            if end < len(text) and text[end].isalpha():
                while end < len(text) and text[end].isalpha():
                    end += 1
            elif end < len(text):
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if character.isalnum() or character == ".":
            end = index + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "."):
                end += 1
            tokens.append(text[index:end].casefold())
            index = end
            continue
        if index + 1 < len(text) and text[index : index + 2] in {
            "<=",
            ">=",
            "!=",
        }:
            tokens.append(text[index : index + 2])
            index += 2
            continue
        tokens.append(character)
        index += 1
    return tokens


def _normalize_formula_tokens(
    tokens: list[str], start: int = 0, closing: str | None = None
) -> tuple[list[str], int]:
    normalized: list[str] = []
    index = start
    while index < len(tokens):
        token = tokens[index]
        if token == closing:
            return normalized, index + 1
        if token in FORMULA_IGNORED_COMMANDS:
            index += 1
            continue
        if token in FORMULA_FRACTION_COMMANDS:
            numerator, index = _normalize_formula_group(tokens, index + 1)
            denominator, index = _normalize_formula_group(tokens, index)
            normalized.extend(_formula_group(numerator))
            normalized.append("/")
            normalized.extend(_formula_group(denominator))
            continue
        if token in FORMULA_FORMAT_COMMANDS:
            formatted, index = _normalize_formula_group(tokens, index + 1)
            normalized.extend(formatted)
            continue
        if token == r"\sqrt":
            radicand, index = _normalize_formula_group(tokens, index + 1)
            normalized.extend(["sqrt", "(", *radicand, ")"])
            continue
        if token in FORMULA_OPERATOR_ALIASES:
            normalized.append(FORMULA_OPERATOR_ALIASES[token])
            index += 1
            continue
        if token in {"{", "("}:
            group, index = _normalize_formula_tokens(
                tokens,
                index + 1,
                "}" if token == "{" else ")",
            )
            normalized.extend(["(", *group, ")"])
            continue
        if token in {"[", r"\[", r"\("}:
            closing_token = {
                "[": "]",
                r"\[": r"\]",
                r"\(": r"\)",
            }[token]
            group, index = _normalize_formula_tokens(tokens, index + 1, closing_token)
            normalized.extend(["(", *group, ")"])
            continue
        if token.startswith("\\"):
            normalized.append(token[1:].casefold())
        else:
            normalized.append(token.casefold())
        index += 1
    return normalized, index


def _normalize_formula_group(tokens: list[str], start: int) -> tuple[list[str], int]:
    if start >= len(tokens):
        return [], start
    opening = tokens[start]
    if opening == "{":
        return _normalize_formula_tokens(tokens, start + 1, "}")
    if opening == "(":
        return _normalize_formula_tokens(tokens, start + 1, ")")
    token, index = _normalize_formula_tokens(tokens[start : start + 1])
    return token, start + index


def _formula_group(tokens: list[str]) -> list[str]:
    if len(tokens) <= 1:
        return tokens
    return ["(", *tokens, ")"]


def _presentation_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in PRESENTATION_TOKEN_PATTERN.findall(text)
        if token.strip()
    ]


def _critical_presentation_tokens(text: str) -> list[str]:
    return [
        token
        for token in _presentation_tokens(text)
        if any(character.isdigit() for character in token)
    ]


def _presentation_fallback_reason(
    block: dict[str, Any],
    source: TextRegion | None,
    sources: dict[str, TextRegion],
    linked_labels: set[str],
    source_counts: Counter[str],
    claimed_sources: set[str],
) -> str | None:
    validation = block.get("validation", {})
    if validation.get("status") != "passed":
        return "validation_failed"
    provenance = block.get("provenance")
    source_id = (
        provenance.get("source_region_id") if isinstance(provenance, dict) else None
    )
    if not isinstance(source_id, str) or source is None:
        return "missing_source_region"
    if source_id in linked_labels or _region_semantic_kind(source) == "control":
        return "control_evidence_is_authoritative"
    structure = source.structure if isinstance(source.structure, dict) else {}
    role = str(structure.get("role", ""))
    if role in {"coverage_risk", "table_candidate", "table_source"}:
        return "source_is_not_renderable"
    owner_id = structure.get("layout_owner_id")
    if isinstance(owner_id, str) and owner_id in {
        region_id
        for region_id, region in sources.items()
        if isinstance(region.structure, dict)
        and region.structure.get("role") == "layout_block"
    }:
        return "source_is_owned_by_layout"
    source_resolved = _presentation_source_is_fully_resolved(source, sources)
    if not source_resolved and not _verified_image_grounded_presentation(block):
        return "source_evidence_unresolved"
    category = block.get("category")
    source_kind = _region_semantic_kind(source)
    if source_resolved and structure.get("block_type") == "form_row":
        return "structured_source_is_authoritative"
    if (
        source_resolved
        and category == "table"
        and (isinstance(structure.get("cells"), list) and structure["cells"])
    ):
        return "structured_source_is_authoritative"
    if category == "table" and source_kind != "table":
        return "table_source_mismatch"
    if category != "table" and source_kind == "table":
        return "table_source_mismatch"
    if source_counts[source_id] > 1:
        return "segmented_source_requires_canonical_rendering"
    if source_id in claimed_sources:
        return "source_already_claimed"
    return None


def _verified_image_grounded_presentation(block: dict[str, Any]) -> bool:
    provenance = block.get("provenance")
    if not isinstance(provenance, dict):
        return False
    return bool(
        block.get("provider") == "falcon-presentation"
        and provenance.get("method") == "falcon_core_category_crop_generation"
        and is_verified_falcon_model_provenance(provenance.get("model"))
    )


def _presentation_source_is_fully_resolved(
    source: TextRegion,
    sources: dict[str, TextRegion],
) -> bool:
    if source.resolution != "resolved":
        return False
    structure = source.structure if isinstance(source.structure, dict) else {}
    child_ids = structure.get("child_evidence_ids", [])
    if isinstance(child_ids, list):
        for child_id in child_ids:
            child = sources.get(child_id) if isinstance(child_id, str) else None
            if child is None or child.resolution != "resolved":
                return False
    cells = structure.get("cells", [])
    if not isinstance(cells, list):
        return True
    for cell in cells:
        if not isinstance(cell, dict):
            return False
        if str(cell.get("resolution", "resolved")) != "resolved":
            return False
        evidence_ids = cell.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            continue
        for evidence_id in evidence_ids:
            evidence = (
                sources.get(evidence_id) if isinstance(evidence_id, str) else None
            )
            if evidence is not None and evidence.resolution != "resolved":
                return False
    return True


def _region_semantic_kind(region: TextRegion) -> str:
    structure = region.structure if isinstance(region.structure, dict) else {}
    value = (
        f"{region.kind} {structure.get('role', '')} {structure.get('block_type', '')}"
    ).casefold()
    if "table" in value:
        return "table"
    if "figure" in value or "image" in value:
        return "figure"
    if "formula" in value or "equation" in value:
        return "formula"
    if "checkbox" in value or "radio" in value or "control" in value:
        return "control"
    return "text"


def _record_presentation_execution(
    executions: list[dict[str, object]],
    page: PageResult,
    presentation: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    presentation_status = presentation["status"]
    validation = presentation.get("validation", {})
    failed_validation = validation.get("status") == "failed"
    status = (
        "failed"
        if presentation_status == "failed" or failed_validation
        else "productive"
    )
    if presentation_status == "skipped":
        status = "skipped"
    failure = presentation.get("failure", {})
    run: dict[str, object] = {
        "page_number": page.page_number,
        "stage": "presentation",
        "status": status,
        "input_regions": len(page.regions),
        "output_regions": len(page.regions),
        "added_regions": 0,
        "removed_regions": 0,
        "modified_regions": 0,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    if status == "failed":
        run["failure_code"] = failure.get("code", "presentation_validation_failed")
    elif status == "skipped":
        run["skip_reason"] = presentation.get("message", "presentation_skipped")
    executions.append(run)


def _valid_presentation_regions(regions: Any, page: PageResult) -> bool:
    if not isinstance(regions, list) or not regions:
        return False
    if any(not isinstance(region, TextRegion) for region in regions):
        return False
    if len(regions) == 1 and _presentation_category(regions[0], True) == "plain":
        box = regions[0].bounding_box
        if (box.left, box.top, box.right, box.bottom) != (
            0,
            0,
            page.width,
            page.height,
        ):
            return False
    return all(
        _valid_presentation_box(region.bounding_box, page)
        and _presentation_category(region, len(regions) == 1) is not None
        for region in regions
    )


def _valid_presentation_box(box: BoundingBox, page: PageResult) -> bool:
    return (
        isinstance(box, BoundingBox)
        and 0 <= box.left < box.right <= page.width
        and 0 <= box.top < box.bottom <= page.height
    )


def _presentation_category(region: TextRegion, single_region: bool) -> str | None:
    structure = region.structure if isinstance(region.structure, dict) else {}
    provenance = (
        region.text_provenance if isinstance(region.text_provenance, dict) else {}
    )
    generation = provenance.get("generation")
    category = structure.get("category")
    if category is None and isinstance(generation, dict):
        category = generation.get("category")
    if category is None and single_region:
        category = "plain"
    return category if category in PRESENTATION_CATEGORIES else None


def _presentation_block(region: TextRegion, single_region: bool) -> dict[str, Any]:
    category = _presentation_category(region, single_region)
    if category is None:
        raise ValueError("Presentation category was not validated")
    failures = _presentation_validation_failures(category, region.text)
    return {
        "id": region.id,
        "reading_order": region.reading_order,
        "category": category,
        "bbox": asdict(region.bounding_box),
        "raw_text": region.text,
        "provider": region.provider,
        "provenance": copy.deepcopy(region.text_provenance),
        "validation": {
            "status": "failed" if failures else "passed",
            "failures": failures,
            "termination_observed": False,
        },
    }


def _presentation_validation_failures(
    category: str, raw_text: str
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    if not raw_text.strip():
        failures.append(
            {"code": "empty_output", "message": "The model returned no content"}
        )
    if _has_presentation_repetition(raw_text):
        failures.append(
            {
                "code": "repetition_loop",
                "message": "Repeated output indicates an incomplete generation",
            }
        )
    if category == "table":
        failures.extend(_validate_presentation_table(raw_text))
    if category == "formula":
        failures.extend(_validate_presentation_formula(raw_text))
    return failures


def _has_presentation_repetition(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    period = (normalized + normalized).find(normalized, 1)
    if (
        period < len(normalized)
        and len(normalized) % period == 0
        and len(normalized) // period >= 4
    ):
        return True
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return len(lines) >= 8 and len(set(lines[-8:])) == 1


def _validate_presentation_table(raw_text: str) -> list[dict[str, str]]:
    parser = _PresentationTableParser()
    try:
        parser.feed(raw_text)
        parser.close()
        parser.valid()
    except (ValueError, TypeError):
        parser._fail("malformed_html_table")
    messages = {
        "disallowed_html_element": "Table output contains a disallowed HTML element",
        "disallowed_html_attribute": "Table output contains a disallowed HTML attribute",
        "disallowed_html_content": "Table output contains disallowed HTML content",
        "malformed_html_table": "Table output is malformed",
        "non_rectangular_html_table": "Table rows do not form a rectangular grid",
    }
    return [{"code": code, "message": messages[code]} for code in parser.failures]


def _rectangular_table(rows: list[list[tuple[int, int]]]) -> bool:
    if not rows or any(not row for row in rows):
        return False
    occupied: set[tuple[int, int]] = set()
    row_count = len(rows)
    for row_index, cells in enumerate(rows):
        column = 0
        for colspan, rowspan in cells:
            while (row_index, column) in occupied:
                column += 1
            if row_index + rowspan > row_count:
                return False
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    point = (row_index + row_offset, column + column_offset)
                    if point in occupied:
                        return False
                    occupied.add(point)
            column += colspan
    width = max((column for _, column in occupied), default=-1) + 1
    return width > 0 and all(
        all((row, column) in occupied for column in range(width))
        for row in range(row_count)
    )


def _validate_presentation_formula(raw_text: str) -> list[dict[str, str]]:
    if not raw_text.strip():
        return [
            {
                "code": "empty_formula",
                "message": "Formula output has no content",
            }
        ]
    if not _balanced_latex_braces(raw_text):
        return [
            {
                "code": "unbalanced_latex_braces",
                "message": "Formula output has unbalanced braces",
            }
        ]
    if not _balanced_latex_delimiters(raw_text):
        return [
            {
                "code": "unbalanced_latex_delimiters",
                "message": "Formula output has unbalanced math delimiters",
            }
        ]
    if not _balanced_formula_groups(raw_text):
        return [
            {
                "code": "unbalanced_formula_groups",
                "message": "Formula output has unbalanced parentheses or brackets",
            }
        ]
    return []


def _balanced_formula_groups(text: str) -> bool:
    stack: list[str] = []
    closing_groups = {")": "(", "]": "["}
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character in {"(", "["}:
            stack.append(character)
            continue
        expected = closing_groups.get(character)
        if expected is not None and (not stack or stack.pop() != expected):
            return False
    return not stack


def _balanced_latex_braces(text: str) -> bool:
    depth = 0
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _balanced_latex_delimiters(text: str) -> bool:
    stack: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            token = text[index : index + 2]
            if token in {"\\(", "\\["}:
                stack.append({"\\(": "\\)", "\\[": "\\]"}[token])
            elif token in {"\\)", "\\]"}:
                if not stack or stack.pop() != token:
                    return False
            index += 2
            continue
        if text[index] == "$" and (index == 0 or text[index - 1] != "\\"):
            token = "$$" if text[index : index + 2] == "$$" else "$"
            if stack and stack[-1] == token:
                stack.pop()
            else:
                stack.append(token)
            index += len(token)
            continue
        index += 1
    return not stack


def _sanitize_result(result: dict[str, Any], session_root: Path) -> dict[str, Any]:
    temporary_root = Path(tempfile.gettempdir())
    replacements = (
        (str(session_root.resolve()), "[session]"),
        (str(session_root), "[session]"),
        (str(temporary_root.resolve()), "[temporary directory]"),
        (str(temporary_root), "[temporary directory]"),
    )

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(sanitize(item) for item in value)
        if isinstance(value, str):
            for root, replacement in replacements:
                value = value.replace(root, replacement)
        return value

    return sanitize(result)


def _uncertainty_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "interpretation": (
            "Evidence diagnostics for review routing, not calibrated probabilities."
        ),
        "pages": [_page_uncertainty(page) for page in result["pages"]],
    }


def _page_uncertainty(page: dict[str, Any]) -> dict[str, Any]:
    regions = page["regions"]
    evidence_ids = set(page["text"]["evidence_ids"])
    primary = [region for region in regions if region["id"] in evidence_ids]
    confidences = [
        region["confidence"]
        for region in primary
        if region["resolution"] == "resolved" and region["confidence"] is not None
    ]
    unresolved = 0
    conflicting = 0
    disagreements = 0
    risk_reasons: list[str] = []

    for region in regions:
        provenance = region.get("text_provenance") or {}
        orientation = provenance.get("orientation")
        if isinstance(orientation, dict):
            angle = orientation.get("angle")
            if angle in {90, 180, 270}:
                risk_reasons.append(f"orientation_rotated_{angle}_degrees")

        structure = region.get("structure") or {}
        if region["kind"] == "coverage_risk":
            risk_reasons.extend(structure.get("reasons", []))
            continue
        if (
            structure.get("role") == "table_candidate"
            and structure.get("status") == "rejected"
        ):
            risk_reasons.append("rejected_table_candidate")
        handwriting_review = structure.get("handwriting_review")
        if isinstance(handwriting_review, dict) and handwriting_review.get("required"):
            reason = handwriting_review.get("reason")
            if isinstance(reason, str) and reason:
                risk_reasons.append(reason)
        if structure.get("role") == "table_source":
            continue

        unresolved += region["resolution"] == "unreadable"
        conflicting += region["resolution"] == "conflicting"
        cells = structure.get("cells", [])
        if isinstance(cells, list) and cells:
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                unresolved += cell.get("resolution") == "unreadable"
                conflicting += cell.get("resolution") == "conflicting"
                disagreements += _different_alternatives(
                    cell.get("text", ""), cell.get("alternatives", [])
                )
            continue
        disagreements += _different_alternatives(
            region["text"], region.get("alternatives", [])
        )

    mean_confidence = (
        round(sum(confidences) / len(confidences), 6) if confidences else None
    )
    failure_count = len(page.get("failure_ids", []))
    return {
        "page_number": page["page_number"],
        "review_required": (
            page["route"] == "review"
            or failure_count > 0
            or unresolved > 0
            or conflicting > 0
        ),
        "mean_primary_confidence": mean_confidence,
        "primary_regions": len(primary),
        "confidence_regions": len(confidences),
        "unresolved_evidence": unresolved,
        "conflicting_evidence": conflicting,
        "disagreeing_alternatives": disagreements,
        "failure_count": failure_count,
        "risk_reasons": list(dict.fromkeys(risk_reasons)),
        "category_counts": dict(
            sorted(Counter(region["kind"] for region in regions).items())
        ),
    }


def _different_alternatives(text: str, alternatives: Any) -> int:
    if not isinstance(alternatives, list):
        return 0
    normalized = " ".join(text.casefold().split())
    return sum(
        " ".join(str(alternative.get("text", "")).casefold().split()) != normalized
        for alternative in alternatives
        if isinstance(alternative, dict)
        and alternative.get("decision_state", "pending") == "pending"
    )


def _backend_version(reader: LocalReader) -> str:
    configured_version = getattr(reader, "pipeline_version", None) or getattr(
        reader, "version", None
    )
    if configured_version is not None:
        return str(configured_version)

    executable = getattr(reader, "executable", None)
    if not executable:
        return "unknown"
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    first_line = (completed.stdout or completed.stderr).splitlines()
    return first_line[0].strip() if first_line else "unknown"


def _coverage_assessment(reader: LocalReader, page_count: int) -> dict[str, Any]:
    assessment = getattr(reader, "coverage_assessment", None)
    if callable(assessment):
        value = assessment(page_count)
        if isinstance(value, dict):
            return value
    return {
        "status": "not_assessed",
        "message": "Extraction completeness was not independently assessed.",
        "pages": [],
    }


def _pages_with_table_continuations(
    pages: list[dict[str, Any]],
    continuations: Any,
) -> list[dict[str, Any]]:
    if not isinstance(continuations, list) or not continuations:
        return pages

    rendered_pages = copy.deepcopy(pages)
    pages_by_number = {
        page.get("page_number"): page
        for page in rendered_pages
        if isinstance(page, dict)
    }
    changed_pages: set[int] = set()
    for continuation in continuations:
        if not isinstance(continuation, dict):
            continue
        source_page_numbers = continuation.get("source_page_numbers")
        source_table_ids = continuation.get("source_table_ids")
        cells = continuation.get("cells")
        if (
            not isinstance(source_page_numbers, list)
            or not isinstance(source_table_ids, list)
            or len(source_page_numbers) < 2
            or len(source_page_numbers) != len(source_table_ids)
            or not isinstance(cells, list)
        ):
            continue
        sources = []
        for page_number, table_id in zip(
            source_page_numbers,
            source_table_ids,
            strict=True,
        ):
            page = pages_by_number.get(page_number)
            matches = (
                [
                    region
                    for region in page.get("regions", [])
                    if isinstance(region, dict)
                    and region.get("id") == table_id
                    and _presentation_semantic_kind(region) == "table"
                ]
                if isinstance(page, dict)
                else []
            )
            if len(matches) != 1:
                sources = []
                break
            sources.append((page, matches[0]))
        if not sources:
            continue

        first_page, first_source = sources[0]
        replacement = {
            **copy.deepcopy(first_source),
            "id": str(continuation.get("id", "cross-page-table")),
            "text": "",
            "confidence": continuation.get("score"),
            "provider": "cross-page-tables",
            "text_provenance": {
                "method": "cross_page_table_continuation",
                "source_page_numbers": copy.deepcopy(source_page_numbers),
                "source_table_ids": copy.deepcopy(source_table_ids),
                "continuation_provenance": copy.deepcopy(
                    continuation.get("provenance", {})
                ),
            },
            "structure": {
                **(
                    copy.deepcopy(first_source.get("structure"))
                    if isinstance(first_source.get("structure"), dict)
                    else {}
                ),
                "role": "table",
                "header_row_count": continuation.get("header_row_count"),
                "row_count": continuation.get("row_count"),
                "column_count": continuation.get("column_count"),
                "cells": copy.deepcopy(cells),
                "cross_page_continuation_id": continuation.get("id"),
            },
        }
        first_regions = first_page["regions"]
        first_regions[first_regions.index(first_source)] = replacement
        for page, source in sources[1:]:
            page["regions"].remove(source)
        for page, source in sources:
            page_number = page.get("page_number")
            if isinstance(page_number, int):
                changed_pages.add(page_number)
            evidence_ids = page.get("text", {}).get("evidence_ids", [])
            if isinstance(evidence_ids, list):
                page["text"]["evidence_ids"] = [
                    evidence_id
                    for evidence_id in evidence_ids
                    if evidence_id != source.get("id")
                ]
        first_page["text"]["evidence_ids"].append(replacement["id"])

    for page_number in changed_pages:
        page = pages_by_number[page_number]
        evidence_ids = set(page.get("text", {}).get("evidence_ids", []))
        page["text"]["value"] = " ".join(
            str(region.get("text", ""))
            for region in sorted(
                page.get("regions", []),
                key=lambda region: _region_order(region),
            )
            if region.get("id") in evidence_ids
            and region.get("resolution", "resolved") == "resolved"
            and region.get("text")
        )
    return rendered_pages


def _result_markdown(response: dict[str, Any]) -> str:
    result = response["result"]
    lines = [
        "# OCR result",
        "",
        f"- File: {response['filename']}",
        f"- Revision: {result['revision']}",
        f"- Status: {result['status']}",
        f"- Backend: {response['backend']}",
        f"- Backend version: {response['backend_version']}",
        f"- Elapsed: {response['elapsed_seconds']:.3f} seconds",
        f"- Geometry: {response['geometry']}",
        f"- Coverage: {response['coverage_assessment']['status']}",
        f"- Coverage note: {response['coverage_assessment']['message']}",
    ]
    pages = _pages_with_table_continuations(
        result["pages"],
        result.get("table_continuations", []),
    )
    for page in pages:
        canonical_source_regions = _presentation_canonical_regions(page)
        canonical_regions = _clean_render_regions(canonical_source_regions)
        canonical_markdown = render_page_markdown(
            canonical_regions,
            [
                str(region["id"])
                for region in canonical_source_regions
                if region.get("id")
            ],
            fallback=_visible_literal(page["text"]["value"], "resolved"),
        )
        presentation = next(
            (
                item
                for item in response.get("presentation", {}).get("pages", [])
                if isinstance(item, dict)
                and item.get("page_number") == page["page_number"]
            ),
            None,
        )
        page_markdown = _presentation_page_markdown(
            page,
            presentation,
            canonical_markdown,
        )
        lines.extend(["", f"## Page {page['page_number']}", "", page_markdown])
    if result["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in result["failures"]:
            page = (
                f" on page {failure['page_number']}"
                if failure["page_number"] is not None
                else ""
            )
            lines.append(
                f"- `{failure['stage']}/{failure['code']}`{page}: {failure['message']}"
            )
    return "\n".join(lines) + "\n"


def _presentation_page_markdown(
    page: dict[str, Any],
    presentation: Any,
    canonical_markdown: str,
) -> str:
    if not isinstance(presentation, dict):
        return canonical_markdown
    blocks = presentation.get("blocks")
    if not isinstance(blocks, list):
        return canonical_markdown

    source_regions = _clean_render_regions(_presentation_canonical_regions(page))
    canonical_sources = _presentation_top_level_regions(source_regions)
    source_index = {
        str(region.get("id")): region for region in source_regions if region.get("id")
    }
    selected: dict[str, str] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rendering = block.get("rendering")
        if not isinstance(rendering, dict) or rendering.get("status") != "selected":
            continue
        source_id = rendering.get("source_region_id")
        if not isinstance(source_id, str) or source_id not in source_index:
            continue
        rendered = _presentation_markdown_block(block)
        if rendered:
            selected[source_id] = rendered

    if not selected:
        return canonical_markdown

    entries: list[tuple[int, int, str]] = []
    for index, region in enumerate(canonical_sources):
        source_id = str(region.get("id", ""))
        chosen = selected.get(source_id)
        if chosen is not None:
            entries.append(
                (
                    _region_order(region),
                    index,
                    chosen,
                )
            )
            continue
        rendered = _canonical_region_markdown(region, source_index)
        if rendered:
            entries.append((_region_order(region), index, rendered))
    entries.sort(key=lambda item: (item[0], item[1]))
    return "\n\n".join(text for _, _, text in entries) or canonical_markdown


def _presentation_markdown_block(block: dict[str, Any]) -> str:
    raw_text = str(block.get("raw_text", "")).strip()
    validation = block.get("validation")
    if (
        not raw_text
        or not isinstance(validation, dict)
        or validation.get("status") != "passed"
    ):
        return ""
    category = block.get("category")
    if category == "title":
        return f"### {raw_text}"
    if category == "section-header":
        return f"#### {raw_text}"
    if category == "list-item":
        return f"- {raw_text}"
    if category == "formula":
        return f"$$\n{raw_text}\n$$"
    return raw_text


def _canonical_region_markdown(
    region: dict[str, Any], source_index: dict[str, dict[str, Any]]
) -> str:
    structure = region.get("structure")
    child_ids = (
        structure.get("child_evidence_ids", []) if isinstance(structure, dict) else []
    )
    bundle = [region]
    bundle.extend(
        source_index[evidence_id]
        for evidence_id in child_ids
        if isinstance(evidence_id, str) and evidence_id in source_index
    )
    return render_page_markdown(bundle, [str(region.get("id", ""))])


def _region_order(region: dict[str, Any]) -> int:
    structure = region.get("structure")
    rank = structure.get("presentation_rank") if isinstance(structure, dict) else None
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank
    order = region.get("reading_order")
    return order if isinstance(order, int) else 2**31 - 1


def _presentation_canonical_regions(page: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_ids = set(page.get("text", {}).get("evidence_ids", []))
    layout_owner_ids = {
        str(region.get("id"))
        for region in page.get("regions", [])
        if isinstance(region, dict)
        and isinstance(region.get("structure"), dict)
        and region["structure"].get("role") == "layout_block"
    }
    regions = []
    for region in page.get("regions", []):
        if not isinstance(region, dict):
            continue
        structure = region.get("structure")
        role = structure.get("role") if isinstance(structure, dict) else None
        if role in {"table_source", "coverage_risk", "table_candidate"}:
            continue
        if region.get("kind") in {"coverage_risk", "table_candidate"}:
            continue
        kind = str(region.get("kind", "")).casefold()
        layout_owner_id = (
            structure.get("layout_owner_id") if isinstance(structure, dict) else None
        )
        if (
            region.get("id") in evidence_ids
            or layout_owner_id in layout_owner_ids
            or "control" in kind
            or "checkbox" in kind
            or region.get("resolution", "resolved") != "resolved"
        ):
            regions.append(region)
    linked_label_ids = {
        str(label_id)
        for region in regions
        if _presentation_semantic_kind(region) == "control" and region.get("text")
        for label_id in (
            region.get("structure", {}).get("label_evidence_ids", [])
            if isinstance(region.get("structure"), dict)
            else []
        )
    }
    return [
        region
        for region in regions
        if str(region.get("id")) not in linked_label_ids
        or _presentation_semantic_kind(region) in {"table", "figure", "control"}
    ]


def _presentation_top_level_regions(
    regions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    owner_ids = {
        str(region.get("id"))
        for region in regions
        if isinstance(region.get("structure"), dict)
        and region["structure"].get("role") == "layout_block"
    }
    return [
        region
        for region in regions
        if not (
            isinstance(region.get("structure"), dict)
            and region["structure"].get("layout_owner_id") in owner_ids
            and _presentation_semantic_kind(region)
            not in {"table", "figure", "control"}
        )
    ]


def _presentation_semantic_kind(region: dict[str, Any]) -> str:
    structure = region.get("structure")
    role = structure.get("role", "") if isinstance(structure, dict) else ""
    value = f"{region.get('kind', 'text')} {role}".casefold()
    if "table" in value:
        return "table"
    if "figure" in value or "image" in value:
        return "figure"
    if "formula" in value or "equation" in value:
        return "formula"
    if "checkbox" in value or "radio" in value or "control" in value:
        return "control"
    return "text"


def _clean_render_regions(
    regions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned = copy.deepcopy(regions)
    for region in cleaned:
        resolution = str(region.get("resolution", "resolved"))
        region["text"] = _visible_literal(region.get("text", ""), resolution)
        structure = region.get("structure")
        if not isinstance(structure, dict):
            continue
        for cell in structure.get("cells", []):
            if not isinstance(cell, dict):
                continue
            cell_resolution = str(cell.get("resolution", "resolved"))
            cell["text"] = _visible_literal(cell.get("text", ""), cell_resolution)
    return cleaned


def _visible_literal(value: Any, resolution: str) -> str:
    text = str(value or "")
    if resolution == "resolved":
        return text
    return ""


def main() -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install uvicorn to run the OCR demo") from error
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
