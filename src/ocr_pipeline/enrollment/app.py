"""Enrollment review app: ticket intake, background processing, review actions, audit
downloads and metrics. It reaches the OCR service over HTTP, so the two stay separate.

    python -m ocr_pipeline.enrollment.app --root DIR --ocr-url http://127.0.0.1:8193
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from ..openrouter import budgeted_transport
from ..page_arrangement import arrange_page
from . import case as cases
from . import metrics as case_metrics
from .bundle import build_bundle, export_zip
from .extract import classify_document, extract_document

LOGGER = logging.getLogger(__name__)
ACCEPTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_FILE_BYTES = 25 << 20
MAX_FILES = 20
REVIEW_HTML = Path(__file__).with_name("review.html")
ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


# The client's enrollment rules: what a group (PVEN) and a practitioner (PCEN) ticket need,
# what loads only when printed, and what is defaulted when absent.
PRACTITIONER_REQUIRED = {
    "first_name": "practitioner.first_name", "middle_name": "practitioner.middle_name",
    "last_name": "practitioner.last_name", "npi": "practitioner.npi", "dob": "practitioner.dob",
    "gender": "practitioner.gender", "credential": "practitioner.credential",
    "primary_specialty": "practitioner.primary_specialty", "group_npi": "group.npi", "group_tin": "group.tin",
}
PRACTITIONER_IF_PROVIDED = {
    "license_number": "practitioner.license_number", "license_state": "practitioner.license_state",
    "medicare_id": "practitioner.medicare_id", "medicaid_id": "practitioner.medicaid_id",
}
PRACTITIONER_DEFAULTED = {
    "panel_status": "practitioner.panel_status", "directory_indicator": "practitioner.directory_indicator",
    "pcp_or_specialist": "practitioner.pcp", "languages": "practitioner.languages",
    "age_restriction": "practitioner.age_restriction", "gender_restriction": "practitioner.gender_restriction",
    "telehealth": "practitioner.telehealth",
}
GROUP_REQUIRED = {
    "dba_name": "group.dba_name", "legal_name": "group.legal_name", "tin": "group.tin", "npi": "group.npi",
    "specialty": "group.specialty", "taxonomy": "group.taxonomy",
}


def client_fields(extraction) -> dict[str, Any]:
    """One document's values in the client's enrollment structure (group first)."""
    from .schema import FIELDS_BY_KEY

    by_key = {value.key: value for value in extraction.values if value.key in FIELDS_BY_KEY}

    def field(key: str) -> dict[str, Any] | None:
        value = by_key.get(key)
        if value is None:
            spec = FIELDS_BY_KEY[key]
            return {"value": spec.default, "defaulted": True} if spec.category == "defaulted" else None
        return {
            "value": value.value, "printed": value.raw_value if value.raw_value != value.value else None,
            "confidence": value.confidence, "page": value.evidence[0].page if value.evidence else None,
        }

    def location(value) -> dict[str, Any]:
        parts = value.parts or {}
        return {
            "address_line_1": parts.get("line1"), "address_line_2": parts.get("line2"), "city": parts.get("city"),
            "state": parts.get("state"), "zip": parts.get("zip"), "phone": parts.get("phone"),
            "confidence": value.confidence, "page": value.evidence[0].page if value.evidence else None,
        }

    locations = [value for value in extraction.values if value.key == "location"]
    service = [location(value) for value in locations if value.role == "service_location"]
    clean = lambda mapping: {name: field(key) for name, key in mapping.items()}  # noqa: E731
    return {
        "document": {"type": extraction.document_type, "identified_by": extraction.document_type_reason},
        "group_enrollment_pven": {**clean(GROUP_REQUIRED), "service_locations": service},
        "practitioner_enrollment_pcen": {
            "required": {**clean(PRACTITIONER_REQUIRED), "service_location": service[0] if service else None},
            "loaded_if_provided": clean(PRACTITIONER_IF_PROVIDED),
            "defaulted_if_not_provided": clean(PRACTITIONER_DEFAULTED),
        },
        "other_details": {
            **{key.split(".", 1)[1]: field(key) for key in by_key if key.startswith("coverage.")},
            **{name: field(key) for name, key in (
                ("dea_number", "practitioner.dea_number"), ("dea_expiration", "practitioner.dea_expiration"),
                ("caqh_id", "practitioner.caqh_id"), ("board_certification", "practitioner.board_certification"),
                ("license_expiration", "practitioner.license_expiration"),
            ) if key in by_key},
            "other_addresses": [{"role": value.role, **location(value)} for value in locations if value.role != "service_location"],
        },
    }


def create_app(root: Path, *, ocr, classify, extract, ocr_url: str | None = None) -> FastAPI:
    root = Path(root)
    app = FastAPI(title="HiLabs OCR enrollment fields")
    jobs = ThreadPoolExecutor(max_workers=2, thread_name_prefix="enrollment-case")
    stages = {"ocr": ocr, "classify": classify, "extract": extract}

    def case_dir(case_id: str) -> Path:
        if not _valid_id(case_id) or not (root / "cases" / case_id / "case.json").is_file():
            raise HTTPException(404, "No such case")
        return root / "cases" / case_id

    def bundle(case_id: str) -> dict[str, Any]:
        directory = case_dir(case_id)
        return build_bundle(cases.load_case(root, case_id), directory)

    def background(function, *args, **kwargs) -> None:
        def run() -> None:
            try:
                function(*args, **kwargs)
            except Exception:
                LOGGER.exception("Background case job failed")

        jobs.submit(run)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return REVIEW_HTML.read_text()

    @app.get("/api/cases")
    def list_cases() -> list[dict]:
        return cases.list_cases(root)

    @app.post("/api/cases", status_code=202)
    async def intake(
        files: list[UploadFile] = File(...),
        ticket_id: str = Form(...),
        enrollment_type: str = Form("practitioner"),
        actor: str = Form(...),
        synthetic: bool = Form(False),
    ) -> dict:
        if enrollment_type not in {"group", "practitioner"}:
            raise HTTPException(400, "enrollment_type must be group or practitioner")
        if not 0 < len(files) <= MAX_FILES:
            raise HTTPException(400, f"Attach 1 to {MAX_FILES} files")
        received = []
        for upload in files:
            name = Path(upload.filename or "upload").name
            if Path(name).suffix.lower() not in ACCEPTED_SUFFIXES:
                raise HTTPException(415, f"{name}: unsupported file type")
            content = await upload.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, f"{name} is larger than 25 MB")
            received.append((name, content))
        case = cases.create_case(
            root, ticket_id.strip()[:64] or "untitled", enrollment_type, received,
            actor=actor.strip()[:64] or "unknown", synthetic=synthetic,
        )
        background(cases.run_case, root, case.case_id, **stages)
        return {"case_id": case.case_id}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict:
        return bundle(case_id)

    @app.get("/api/cases/{case_id}/pages/{document_id}/{page}")
    def page_image(case_id: str, document_id: str, page: int) -> Response:
        if not _valid_id(document_id):
            raise HTTPException(404, "No such page")
        path = case_dir(case_id) / "pages" / f"{document_id}-p{page}.png"
        if not path.is_file():
            raise HTTPException(404, "No such page")
        return Response(path.read_bytes(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/cases/{case_id}/actions")
    async def act(case_id: str, request: Request) -> dict:
        case_dir(case_id)
        action = await request.json()
        if not isinstance(action, dict):
            raise HTTPException(400, "Send one action object")
        kind = action.pop("action", None) or action.get("type")
        if kind == "reclassify":
            document_id = str(action.get("document_id") or "")
            if not _valid_id(document_id):
                raise HTTPException(400, "reclassify needs a document_id")
            background(
                cases.reprocess_document, root, case_id, document_id,
                document_type=action.get("document_type") or None, **stages,
            )
            return bundle(case_id)
        action["type"] = kind
        if kind == "resolve_exception":
            action.setdefault("exception_id", action.get("value_id"))
        try:
            cases.apply_action(root, case_id, action)
        except cases.ReviewBlocked as error:
            blocked, message = error.args
            raise HTTPException(409, {"blocked": blocked, "message": message})
        except (ValueError, KeyError) as error:
            raise HTTPException(400, str(error))
        return bundle(case_id)

    session_fields: dict[str, tuple[int, dict]] = {}  # session_id -> (reviewer edits, fields)
    session_locks: dict[str, threading.Lock] = {}
    session_locks_guard = threading.Lock()

    @app.post("/api/sessions/{session_id}/fields")
    def fields_for_session(session_id: str) -> dict:
        """The reader's JSON view: a document read by the OCR service, extracted into the
        client's enrollment fields. One LLM call per document revision, then cached; a lock
        per session means two requests racing for the same document share one LLM run."""
        if ocr_url is None or not _valid_id(session_id):
            raise HTTPException(404, "No such document")
        with session_locks_guard:
            lock = session_locks.setdefault(session_id, threading.Lock())
        with lock:
            base = ocr_url.rstrip("/")
            with httpx.Client(timeout=300) as client:
                response = client.get(f"{base}/api/sessions/{session_id}/result.json", params={"compact": "true"})
                if response.status_code == 404:
                    raise HTTPException(404, "This document is no longer available; read it again")
                response.raise_for_status()
                result = response.json()
                # A reviewer's correction changes the values; the LLM's re-read of doubtful
                # regions does not (measured), so only corrections start a new extraction.
                edits = sum(
                    bool((region.get("text_provenance") or {}).get("human_review"))
                    for page in result.get("pages") or [] for region in page.get("regions") or []
                )
                cached = session_fields.get(session_id)
                if cached is not None and cached[0] == edits:
                    return cached[1]
                paths = []
                with tempfile.TemporaryDirectory(prefix="fields-") as work:
                    for page in result.get("pages") or []:
                        image = client.get(f"{base}/api/sessions/{session_id}/pages/{page['page_number']}")
                        image.raise_for_status()
                        path = Path(work) / f"page{page['page_number']}.png"
                        path.write_bytes(image.content)
                        paths.append(path)
                    # No classify call: the extraction reply itself returns document_type, and
                    # extraction_messages sends every type's guidance when it's unknown.
                    extraction = extract(session_id, result, paths, document_type=None, allow_jev=False, hints=[], entities=False)
            while len(session_fields) >= 200:
                session_fields.pop(next(iter(session_fields)))
            session_fields[session_id] = (edits, client_fields(extraction))
        return session_fields[session_id][1]

    # For other teams: send a document (PDF or image), get back one output. Each endpoint does
    # only the work its output needs: the fields don't wait for the LLM's re-read of doubtful
    # regions (values come from the OCR reading itself; measured, same accuracy either way).
    def read_upload(file: UploadFile, client: httpx.Client) -> str:
        if ocr_url is None:
            raise HTTPException(404, "No OCR service is configured")
        response = client.post(
            f"{ocr_url.rstrip('/')}/api/process", params={"defer_refinement": "true", "compact": "true"},
            files={"file": (file.filename or "document", file.file)},
        )
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.json().get("detail", "The document could not be read"))
        return response.json()["session_id"]

    def final_reading(session_id: str, client: httpx.Client) -> dict:
        """The reading after the LLM's re-read of its doubtful regions."""
        response = client.get(f"{ocr_url.rstrip('/')}/api/sessions/{session_id}/refinement", params={"compact": "true"})
        response.raise_for_status()
        return response.json()

    @app.post("/api/documents/markdown", response_class=PlainTextResponse)
    def document_markdown(file: UploadFile = File(...)) -> str:
        """The document as Markdown, in reading order; tables as HTML."""
        with httpx.Client(timeout=900) as client:
            session_id = read_upload(file, client)
            final_reading(session_id, client)
            markdown = client.get(f"{ocr_url.rstrip('/')}/api/sessions/{session_id}/result.md")
            markdown.raise_for_status()
        return PlainTextResponse(markdown.text, media_type="text/markdown")

    @app.post("/api/documents/fields")
    def document_fields(file: UploadFile = File(...)) -> dict:
        """The key-value JSON: the client's enrollment fields (group PVEN; practitioner PCEN
        required, loaded if provided, defaulted), each with value, confidence and page."""
        with httpx.Client(timeout=900) as client:
            session_id = read_upload(file, client)
        return fields_for_session(session_id)

    @app.post("/api/documents/regions")
    def document_regions(file: UploadFile = File(...)) -> dict:
        """Every region in reading order: text, pixel bounding box (left, top, right, bottom
        on the page image of the given width and height), confidence and who read it."""
        with httpx.Client(timeout=900) as client:
            final = final_reading(read_upload(file, client), client)
        return {
            "filename": final["filename"],
            "status": final["result"]["status"],
            "ocr_seconds": (final.get("timing") or {}).get("ocr_seconds"),
            "pages": [
                {
                    "page_number": page["page_number"], "width": page.get("width"), "height": page.get("height"),
                    # Layout blocks only group the readings below them; they would repeat their text.
                    "regions": [
                        region_summary(region)
                        for region in sorted(page["regions"], key=lambda region: region.get("reading_order") or 0)
                        if region.get("kind") != "layout_block"
                    ],
                }
                for page in final["result"]["pages"]
            ],
        }

    @app.post("/api/cases/{case_id}/group-ticket", status_code=202)
    async def group_ticket(case_id: str, request: Request) -> dict:
        """Client rule: the group is enrolled before its practitioners. Start the group
        (PVEN) ticket from this practitioner ticket's own documents."""
        directory = case_dir(case_id)
        body = await request.json() if await request.body() else {}
        practitioner = cases.load_case(root, case_id)
        files = [
            (document.file_name, next((directory / "files").glob(f"{document.document_id}.*")).read_bytes())
            for document in practitioner.documents
            if document.status != "duplicate"
        ]
        group = cases.create_case(
            root, f"{practitioner.ticket_id}-PVEN", "group", files,
            actor=str(body.get("actor") or practitioner.created_by)[:64], synthetic=practitioner.synthetic,
        )
        background(cases.run_case, root, group.case_id, **stages)
        return {"case_id": group.case_id}

    @app.post("/api/cases/{case_id}/documents/{document_id}/ocr-view")
    def ocr_view(case_id: str, document_id: str) -> dict:
        """Open a document in the OCR workbench: the stored file is read again, which the
        OCR service answers from its page cache, and the browser goes to that session."""
        if ocr_url is None or not _valid_id(document_id):
            raise HTTPException(404, "No OCR view for this document")
        files = sorted((case_dir(case_id) / "files").glob(f"{document_id}.*"))
        if not files:
            raise HTTPException(404, "No such document")
        with httpx.Client(timeout=600) as client, files[0].open("rb") as handle:
            response = client.post(f"{ocr_url.rstrip('/')}/api/process", files={"file": (files[0].name, handle)})
        response.raise_for_status()
        return {"url": f"/?session={response.json()['session_id']}"}

    @app.get("/api/cases/{case_id}/payload.json")
    def payload(case_id: str) -> Response:
        path = case_dir(case_id) / "payload.json"
        if not path.is_file():
            raise HTTPException(404, "The case has not been submitted")
        return Response(path.read_bytes(), media_type="application/json", headers=_download(f"{case_id}-payload.json"))

    @app.get("/api/cases/{case_id}/audit.jsonl")
    def audit(case_id: str) -> Response:
        path = case_dir(case_id) / "audit.jsonl"
        content = path.read_bytes() if path.is_file() else b""
        return Response(content, media_type="application/x-ndjson", headers=_download(f"{case_id}-audit.jsonl"))

    @app.get("/api/cases/{case_id}/bundle.zip")
    def bundle_zip(case_id: str) -> Response:
        directory = case_dir(case_id)
        content = export_zip(cases.load_case(root, case_id), directory, REVIEW_HTML)
        return Response(content, media_type="application/zip", headers=_download(f"{case_id}-bundle.zip"))

    @app.get("/api/metrics")
    def get_metrics() -> dict:
        return case_metrics.metrics(root)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("Request failed: %s", request.url.path)
        return JSONResponse({"detail": "Something went wrong; the case is unchanged."}, status_code=500)

    return app


def region_summary(region: dict[str, Any]) -> dict[str, Any]:
    """One region for API callers: what it says, where, how sure, and who read it."""
    provenance = region.get("text_provenance") or {}
    read_by = (
        "reviewer" if provenance.get("human_review")
        else "llm" if provenance.get("method") == "qwen_crop_refinement"
        else "text_layer" if provenance.get("method") == "pdf_text_layer"
        else "ocr"
    )
    confidence = region.get("confidence")
    return {
        "id": region["id"],
        "kind": region.get("kind"),
        "text": region.get("text"),
        "bounding_box": region.get("bounding_box"),
        "confidence": confidence if confidence is not None else provenance.get("llm_confidence"),
        "read_by": read_by,
        "reading_order": region.get("reading_order"),
    }


def ocr_over_http(base_url: str, path: Path, timeout: float = 900) -> dict[str, Any]:
    """Read one document through the OCR service; pages carry their arrangement."""
    base_url = base_url.rstrip("/")
    with httpx.Client(timeout=timeout) as client:
        with path.open("rb") as handle:
            # Values are extracted from the OCR reading itself; the LLM's crop re-read only
            # polishes the display text, so it finishes in the background (measured on 11
            # documents: same accuracy with or without it, and the W-9 re-read alone is ~60 s).
            response = client.post(
                f"{base_url}/api/process", params={"defer_refinement": "true"},
                files={"file": (path.name, handle)},
            )
        response.raise_for_status()
        processed = response.json()
        session_id = processed["session_id"]
        result = client.get(f"{base_url}/api/sessions/{session_id}/result.json", params={"compact": "true"})
        result.raise_for_status()
        result = result.json()
        images = []
        for page in result.get("pages") or []:
            image = client.get(f"{base_url}/api/sessions/{session_id}/pages/{page['page_number']}")
            image.raise_for_status()
            images.append(image.content)
    for page in result.get("pages") or []:
        try:
            page["arrangement"] = arrange_page(page)
        except Exception as error:  # the viewer falls back to plain boxes; recorded, not hidden
            LOGGER.exception("Page arrangement failed")
            page["arrangement"] = {"error": str(error)}
    return {
        "result": result,
        "session_id": session_id,
        "backend_version": processed.get("backend_version"),
        "elapsed_seconds": processed.get("elapsed_seconds"),
        "timing": processed.get("timing") or {},
        "page_images": images,
    }


def _valid_id(value: str) -> bool:
    return bool(ID.fullmatch(value))


def _download(name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


def build_app(
    root: Path, ocr_url: str, *, llm_budget_usd: float = 2.0, allow_jev: bool = False,
    jev_budget_usd: float = 0.5,
) -> FastAPI:
    """The enrollment app wired to the OCR service at ocr_url and to budgeted LLM calls."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required")
    root.mkdir(parents=True, exist_ok=True)
    llm = partial(budgeted_transport, database=root / "llm_spend.sqlite3", budget_usd=llm_budget_usd, reserve_usd=0.02)
    jev = (
        partial(budgeted_transport, database=root / "verifier_spend.sqlite3", budget_usd=jev_budget_usd, reserve_usd=0.01)
        if allow_jev
        else None
    )
    return create_app(
        root,
        ocr=partial(ocr_over_http, ocr_url),
        classify=partial(classify_document, llm_transport=llm, jev_transport=jev),
        extract=partial(extract_document, llm_transport=llm, jev_transport=jev),
        ocr_url=ocr_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Case storage, outside any repository")
    parser.add_argument("--ocr-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8195)
    parser.add_argument("--llm-budget-usd", type=float, default=2.0)
    parser.add_argument("--allow-jev", action="store_true", help="Verifier calls, for synthetic cases only")
    parser.add_argument("--jev-budget-usd", type=float, default=0.5)
    args = parser.parse_args()
    app = build_app(
        args.root, args.ocr_url, llm_budget_usd=args.llm_budget_usd, allow_jev=args.allow_jev,
        jev_budget_usd=args.jev_budget_usd,
    )
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
