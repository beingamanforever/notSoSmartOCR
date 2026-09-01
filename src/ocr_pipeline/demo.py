"""Local web demo for inspecting evidence-linked OCR results."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

from PIL import Image

from .contracts import RegionStage
from .pipeline import IMAGE_SUFFIXES, PipelineError, _prepare_pages, _source_kind
from .pipeline import process_document
from .preprocessing import RoutedTesseractReader
from .providers import LocalReader

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ACCEPTED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024


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


def create_app(
    reader: LocalReader | None = None,
    *,
    stages: Sequence[RegionStage] = (),
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> Any:
    """Create the local demo app with an injectable OCR reader."""
    if FastAPI is None:
        raise RuntimeError("Install FastAPI and python-multipart to run the OCR demo")
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")

    active_reader = reader or RoutedTesseractReader()
    active_stages = tuple(stages)
    temporary_root = tempfile.TemporaryDirectory(prefix="ocr-demo-")
    session_root = Path(temporary_root.name)
    sessions = SessionStore(session_root)
    backend_version = _backend_version(active_reader)
    process_lock = threading.Lock()

    @asynccontextmanager
    async def handle_lifespan(_: Any):
        try:
            yield
        finally:
            sessions.cleanup()
            temporary_root.cleanup()

    app = FastAPI(title="Clinical OCR Workbench", lifespan=handle_lifespan)
    app.state.session_root = session_root
    app.state.sessions = sessions
    app.state.process_lock = process_lock
    app.add_middleware(
        RequestLimitMiddleware,
        max_body_bytes=max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )

    @app.middleware("http")
    async def handle_no_store(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", response_class=HTMLResponse)
    def handle_index() -> Any:
        html = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

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
            queued = time.perf_counter()
            pipeline_timings: dict[str, float] = {}
            with process_lock:
                queue_seconds = time.perf_counter() - queued
                started = time.perf_counter()
                document = process_document(
                    source,
                    active_reader,
                    stages=active_stages,
                    timings=pipeline_timings,
                )
                elapsed_seconds = time.perf_counter() - started
                coverage = _coverage_assessment(active_reader, len(document.pages))
            result = _sanitize_result(document.to_dict(), session_root)
            result["document_id"] = Path(original_name).stem
            result["source"]["name"] = original_name
            preview_started = time.perf_counter()
            preview_paths, preview_failure = _render_previews(source, session_dir)
            preview_seconds = time.perf_counter() - preview_started
            total_seconds = time.perf_counter() - total_started
            response = {
                "session_id": session_id,
                "filename": original_name,
                "backend": active_reader.name,
                "backend_version": backend_version,
                "pipeline_stages": [stage.name for stage in active_stages],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "timing": {
                    "receive_seconds": round(receive_seconds, 3),
                    "save_seconds": round(save_seconds, 3),
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
                    "pages": preview_paths,
                    "response": response,
                },
            )
            return JSONResponse(response)
        except HTTPException:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        except Exception as error:
            shutil.rmtree(session_dir, ignore_errors=True)
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
    def handle_json_download(session_id: str) -> Any:
        session = _get_session(sessions, session_id)
        content = json.dumps(
            session["response"]["result"], indent=2, ensure_ascii=False
        )
        return Response(
            content,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="ocr-result.json"'},
        )

    @app.get("/api/sessions/{session_id}/result.md")
    def handle_markdown_download(session_id: str) -> Any:
        session = _get_session(sessions, session_id)
        content = _result_markdown(session["response"])
        return Response(
            content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="ocr-result.md"'},
        )

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


def _safe_filename(filename: str | None) -> str:
    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return name or "upload"


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
) -> tuple[list[Path], dict[str, str] | None]:
    preview_dir = session_dir / "pages"
    preview_dir.mkdir()
    try:
        pages = _prepare_pages(
            source,
            _source_kind(source),
            preview_dir,
            300,
            "pdftoppm",
        )
    except PipelineError as error:
        return [], {
            "stage": error.stage,
            "code": error.code,
            "message": "Page preview could not be rendered",
        }

    preview_paths: list[Path] = []
    for page_number, page in enumerate(pages, start=1):
        destination = preview_dir / f"preview-{page_number}.png"
        if page != destination:
            try:
                with Image.open(page) as image:
                    image.save(destination, format="PNG")
            except OSError:
                return [], {
                    "stage": "image",
                    "code": "invalid_image",
                    "message": "Page preview could not be rendered",
                }
        preview_paths.append(destination)
    return preview_paths, None


def _sanitize_result(result: dict[str, Any], session_root: Path) -> dict[str, Any]:
    temporary_root = Path(tempfile.gettempdir())
    for failure in result["failures"]:
        message = failure["message"]
        message = message.replace(str(session_root), "[session]")
        failure["message"] = message.replace(
            str(temporary_root), "[temporary directory]"
        )
    return result


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
        region["confidence"] for region in primary if region["confidence"] is not None
    ]
    unresolved = 0
    conflicting = 0
    disagreements = 0
    risk_reasons: list[str] = []

    for region in regions:
        structure = region.get("structure") or {}
        if region["kind"] == "coverage_risk":
            risk_reasons.extend(structure.get("reasons", []))
            continue
        if (
            structure.get("role") == "table_candidate"
            and structure.get("status") == "rejected"
        ):
            risk_reasons.append("rejected_table_candidate")
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


def _result_markdown(response: dict[str, Any]) -> str:
    result = response["result"]
    lines = [
        "# OCR result",
        "",
        f"- File: {response['filename']}",
        f"- Status: {result['status']}",
        f"- Backend: {response['backend']}",
        f"- Backend version: {response['backend_version']}",
        f"- Elapsed: {response['elapsed_seconds']:.3f} seconds",
        f"- Geometry: {response['geometry']}",
        f"- Coverage: {response['coverage_assessment']['status']}",
        f"- Coverage note: {response['coverage_assessment']['message']}",
    ]
    for page in result["pages"]:
        lines.extend(
            [
                "",
                f"## Page {page['page_number']}",
                "",
                page["text"]["value"] or "_No text detected._",
            ]
        )
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


def main() -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install uvicorn to run the OCR demo") from error
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
