"""The case store: intake, concurrent per-document processing, consolidation, review
actions and the submission payload.

A case lives at root/cases/{case_id}/: case.json (the current state, written
atomically), audit.jsonl (append-only, one AuditEvent per line), payload.json (once
submitted), and the per-document files/, ocr/, extract/ and pages/ directories. root is
chosen by the caller and is never inside this repository.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import secrets
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from . import metrics
from .schema import (
    DOCUMENT_TYPES,
    FIELDS,
    FIELDS_BY_KEY,
    Alternative,
    AuditEvent,
    Box,
    Case,
    CaseException,
    DocumentExtraction,
    DocumentRecord,
    Evidence,
    Review,
    Value,
)
from .validate import CHECKS

# Which document types must be present per enrollment type, and how to complain when
# they are not. Any one type in a tuple satisfies that rule.
# TODO: confirm with the client's analyst SOP
REQUIRED_DOCUMENTS: dict[str, list[tuple[tuple[str, ...], str, str]]] = {
    "practitioner": [
        (("practitioner_application",), "error", "Enrollment or credentialing application missing"),
        (("state_license",), "warning", "State license missing; credentialing will collect it"),
        (("w9",), "warning", "Group W-9 missing"),
        (("malpractice_certificate",), "warning", "Malpractice certificate missing"),
        (("dea_certificate",), "info", "DEA certificate missing"),
    ],
    "group": [
        (("group_application", "w9"), "error", "Enrollment or credentialing application missing"),
        (("w9",), "warning", "Group W-9 missing"),
    ],
}
# Catalog keys required for an enrollment type beyond that record's own required specs.
REQUIRED_EXTRA_KEYS: dict[str, tuple[str, ...]] = {
    "practitioner": ("group.tin", "group.npi"),
    "group": ("group.dba_name",),
}

# RLock: apply_action and reprocess_document hold this while calling save_case/_append_audit,
# which re-acquire it on the same thread.
# ponytail: one global lock, per-case locks if throughput matters
_LOCK = threading.RLock()


class ReviewBlocked(Exception):
    """Raised by apply_action's "submit" when the case is not ready to leave review.

    args[0] lists the blocking value or exception ids, args[1] says why.
    """


# --------------------------------------------------------------------------- utilities

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _case_dir(root, case_id: str) -> Path:
    return Path(root) / "cases" / case_id


def _new_case_id(root) -> str:
    while True:
        candidate = re.sub(r"[^A-Za-z0-9_-]", "", secrets.token_urlsafe(9))
        if candidate and not _case_dir(root, candidate).exists():
            return candidate


def _document_path(case_dir: Path, document_id: str) -> Path:
    matches = sorted((case_dir / "files").glob(f"{document_id}.*"))
    if not matches:
        raise FileNotFoundError(f"no stored file for {document_id}")
    return matches[0]


def _find_document(case: Case, document_id: str) -> DocumentRecord:
    for document in case.documents:
        if document.document_id == document_id:
            return document
    raise ValueError(f"no document {document_id} in case {case.case_id}")


def _find_value(case: Case, value_id: str) -> Value:
    for value in case.values:
        if value.id == value_id:
            return value
    raise ValueError(f"no value {value_id} in case {case.case_id}")


def _add_exception(case: Case, **fields) -> CaseException:
    exception = CaseException(id=f"exception-{len(case.exceptions) + 1}", **fields)
    case.exceptions.append(exception)
    return exception


def _append_audit(case_dir: Path, event: AuditEvent) -> None:
    with _LOCK:
        with open(case_dir / "audit.jsonl", "a") as handle:
            handle.write(event.model_dump_json() + "\n")


# ------------------------------------------------------------------------------ store

def save_case(root, case: Case) -> None:
    case_dir = _case_dir(root, case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    data = case.model_dump_json(indent=2)
    with _LOCK:
        temp_path = case_dir / "case.json.tmp"
        temp_path.write_text(data)
        os.replace(temp_path, case_dir / "case.json")


def load_case(root, case_id: str) -> Case:
    return Case.model_validate_json((_case_dir(root, case_id) / "case.json").read_text())


def list_cases(root) -> list[dict]:
    cases_dir = Path(root) / "cases"
    if not cases_dir.exists():
        return []
    rows = []
    for case_path in cases_dir.glob("*/case.json"):
        case = Case.model_validate_json(case_path.read_text())
        rows.append({
            "case_id": case.case_id,
            "ticket_id": case.ticket_id,
            "enrollment_type": case.enrollment_type,
            "status": case.status,
            "created_at": case.created_at,
            "documents": len(case.documents),
            "open_exceptions": sum(1 for exception in case.exceptions if exception.state == "open"),
            "needs_review": sum(
                1 for value in case.values
                if value.review.state == "pending" and value.route in ("priority_review", "exception")
            ),
        })
    return sorted(rows, key=lambda row: row["created_at"])


# ------------------------------------------------------------------------------ intake

def _sniff_suffix(content: bytes) -> str:
    if content.startswith(b"%PDF"):
        return ".pdf"
    with Image.open(io.BytesIO(content)) as image:
        return f".{(image.format or 'png').lower()}"


def _detect_format(path: Path) -> tuple[str, int | None]:
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            page_count = document.page_count
            text = document[0].get_text().strip() if page_count else ""
        return ("digital_pdf" if len(text) >= 50 else "scanned_pdf"), page_count
    with Image.open(path) as image:
        page_count = getattr(image, "n_frames", 1)
        if path.suffix.lower() in (".tif", ".tiff") or image.mode == "1":
            return "fax", page_count
        return "image", page_count


def create_case(
    root, ticket_id: str, enrollment_type: str, files: list[tuple[str, bytes]], *, actor: str,
    synthetic: bool = False,
) -> Case:
    case_id = _new_case_id(root)
    case_dir = _case_dir(root, case_id)
    (case_dir / "files").mkdir(parents=True, exist_ok=True)
    now = _now()
    case = Case(
        case_id=case_id, ticket_id=ticket_id, enrollment_type=enrollment_type,
        synthetic=synthetic, created_at=now, created_by=actor,
    )
    seen_by_hash: dict[str, str] = {}
    for index, (file_name, content) in enumerate(files, 1):
        document_id = f"d{index}"
        sha256 = hashlib.sha256(content).hexdigest()
        duplicate_of = seen_by_hash.get(sha256)
        seen_by_hash.setdefault(sha256, document_id)
        suffix = Path(file_name).suffix or _sniff_suffix(content)
        stored_path = case_dir / "files" / f"{document_id}{suffix}"
        stored_path.write_bytes(content)
        format_, page_count = _detect_format(stored_path)
        document = DocumentRecord(
            document_id=document_id, file_name=file_name, sha256=sha256, bytes=len(content),
            format=format_, intake_at=now, duplicate_of=duplicate_of,
            status="duplicate" if duplicate_of else "queued", page_count=page_count,
        )
        case.documents.append(document)
        if duplicate_of:
            _add_exception(
                case, kind="duplicate", severity="info", document_id=document_id,
                message=f"{file_name} is identical to {_find_document(case, duplicate_of).file_name}; not processed again",
            )
        _append_audit(case_dir, AuditEvent(
            ts=now, actor=actor, action="intake", case_id=case_id, document_id=document_id,
            after=document.model_dump(),
        ))
    save_case(root, case)
    return case


# ------------------------------------------------------------------------ processing

class _StageError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _render_first_page(source: Path, format_: str, destination: Path) -> str:
    """Renders page 1 to a PNG (long side <= 1024 px); returns its text layer, if any."""
    if format_ in ("digital_pdf", "scanned_pdf"):
        with fitz.open(source) as document:
            page = document[0]
            text_layer = page.get_text() if format_ == "digital_pdf" else ""
            zoom = 100 / 72  # 100 dpi
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    else:
        text_layer = ""
        with Image.open(source) as opened:
            image = opened.convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    image.save(destination, format="PNG")
    return text_layer


def _save_ocr_output(case_dir: Path, document_id: str, ocr_response: dict) -> list[Path]:
    pages_dir = case_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_paths = []
    for index, png_bytes in enumerate(ocr_response.get("page_images") or [], 1):
        page_path = pages_dir / f"{document_id}-p{index}.png"
        page_path.write_bytes(png_bytes)
        page_paths.append(page_path)
    ocr_dir = case_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    stored = {key: value for key, value in ocr_response.items() if key != "page_images"}
    (ocr_dir / f"{document_id}.json").write_text(json.dumps(stored))
    return page_paths


def _save_extraction(case_dir: Path, document_id: str, extraction: DocumentExtraction) -> None:
    extract_dir = case_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    (extract_dir / f"{document_id}.json").write_text(extraction.model_dump_json())


def _process_document(
    case_dir: Path, document: DocumentRecord, *, ocr, classify, extract, allow_jev: bool, hints,
) -> dict:
    """Runs off the case lock; returns what happened so the caller can apply it under lock."""
    path = _document_path(case_dir, document.document_id)
    timings: dict[str, float] = {}

    # Classify first: a page confidently classified as carrying no enrollment data never
    # reaches the GPU, whose reads are serialized across a ticket's documents.
    try:
        with tempfile.TemporaryDirectory(prefix="enrollment-classify-") as work_dir:
            first_page_png = Path(work_dir) / "page1.png"
            started = time.perf_counter()
            text_layer = _render_first_page(path, document.format, first_page_png)
            timings["render_first_page"] = time.perf_counter() - started

            started = time.perf_counter()
            classification = classify(first_page_png, text_layer, allow_jev=allow_jev)
            timings["classify"] = time.perf_counter() - started
    except Exception as error:
        raise _StageError("classification", str(error)) from error

    document_type = classification["document_type"]
    extracts_data = DOCUMENT_TYPES[document_type][2]
    if not extracts_data and classification["confidence"] == "high":
        return {
            "status": "skipped", "classification": classification, "timings": timings,
            "reason": f"classified as {classification['label']}",
        }

    try:
        started = time.perf_counter()
        ocr_response = ocr(path)
        timings["ocr"] = time.perf_counter() - started
    except Exception as error:
        raise _StageError("ocr", str(error)) from error

    for name, seconds in ((ocr_response.get("timing") or {}).get("pipeline_steps") or {}).items():
        # The UI never names a model: the OCR service's LLM step is "llm-correction" here.
        timings[f"ocr.{name.replace('qwen-crop-refinement', 'llm-correction')}"] = seconds
    total_seconds = (ocr_response.get("timing") or {}).get("total_seconds")
    if total_seconds is not None:
        timings["ocr.total_seconds"] = total_seconds

    page_paths = _save_ocr_output(case_dir, document.document_id, ocr_response)

    try:
        started = time.perf_counter()
        extraction = extract(
            document.document_id, ocr_response["result"], page_paths,
            document_type=document_type, allow_jev=allow_jev, hints=hints,
        )
        timings["extract"] = time.perf_counter() - started
    except Exception as error:
        raise _StageError("extraction_failed", str(error)) from error
    _save_extraction(case_dir, document.document_id, extraction)

    return {
        "status": "extracted", "classification": classification, "timings": timings,
        "ocr": ocr_response, "extraction": extraction,
    }


def _apply_outcome(case: Case, case_dir: Path, document: DocumentRecord, outcome: dict, extractions: dict) -> None:
    classification = outcome["classification"]
    document.classification = {key: value for key, value in classification.items() if key != "llm"}
    document.document_type = classification["document_type"]
    document.document_type_label = classification["label"]
    document.timings.update(outcome["timings"])
    document.llm = classification.get("llm") or {}
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor="system", action="classify", case_id=case.case_id, document_id=document.document_id,
        after={key: classification.get(key) for key in ("document_type", "label", "method", "confidence", "probability", "evidence")},
        confidence=classification.get("probability"),
    ))

    if outcome["status"] == "skipped":
        document.status = "skipped"
        _append_audit(case_dir, AuditEvent(
            ts=_now(), actor="system", action="skip", case_id=case.case_id,
            document_id=document.document_id, reason=outcome["reason"],
        ))
        _add_exception(
            case, kind="classification", severity="info", document_id=document.document_id,
            message=f"Not read: classified as {document.document_type_label}",
        )
        return

    ocr_response = outcome["ocr"]
    document.ocr = {
        key: ocr_response[key] for key in ("session_id", "backend_version", "elapsed_seconds") if key in ocr_response
    }
    document.ocr.update({
        key: (ocr_response.get("result") or {})[key]
        for key in ("status", "failures") if key in (ocr_response.get("result") or {})
    })
    result = ocr_response.get("result") or {}
    pages = result.get("pages") or []
    regions = sum(len(page.get("regions") or []) for page in pages)
    failures = result.get("failures") or []
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor="system", action="ocr", case_id=case.case_id, document_id=document.document_id,
        after={"status": result.get("status"), "pages": len(pages), "regions": regions, "failures": len(failures),
               "session_id": ocr_response.get("session_id"), "backend_version": ocr_response.get("backend_version")},
    ))
    if failures:
        _add_exception(
            case, kind="ocr", severity="error" if not regions else "warning", document_id=document.document_id,
            message=(f"{document.file_name} could not be read" if not regions
                     else f"Parts of {document.file_name} could not be read; check its highlighted regions"),
        )
    extraction = outcome["extraction"]
    document.llm = {**document.llm, **extraction.llm}
    document.status = "extracted"
    extractions[document.document_id] = extraction
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor="system", action="extract", case_id=case.case_id, document_id=document.document_id,
        after={"document_type": extraction.document_type, "values": len(extraction.values),
               "model": extraction.llm.get("model"), "prompt_version": extraction.llm.get("prompt_version"),
               "cost_usd": extraction.llm.get("cost"), "problems": extraction.problems},
    ))


def run_case(root, case_id: str, *, ocr, classify, extract, workers: int = 4) -> Case:
    case_dir = _case_dir(root, case_id)
    case = load_case(root, case_id)
    hints = metrics.reviewer_hints(root)
    thresholds = metrics.confidence_thresholds(root)
    documents = [document for document in case.documents if document.status != "duplicate"]
    extractions: dict[str, DocumentExtraction] = {}
    started = time.perf_counter()

    def work(document_id: str):
        return _process_document(
            case_dir, _find_document(case, document_id), ocr=ocr, classify=classify, extract=extract,
            allow_jev=case.synthetic, hints=hints,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, document.document_id): document.document_id for document in documents}
        for future in as_completed(futures):
            document_id = futures[future]
            with _LOCK:
                document = _find_document(case, document_id)
                try:
                    outcome = future.result()
                except Exception as error:
                    document.status = "failed"
                    document.error = str(error)
                    kind = error.kind if isinstance(error, _StageError) else "extraction_failed"
                    _add_exception(
                        case, kind=kind, severity="error", document_id=document_id,
                        message=f"{document.file_name}: {error}",
                    )
                else:
                    _apply_outcome(case, case_dir, document, outcome, extractions)
            save_case(root, case)

    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor="system", action="processed", case_id=case.case_id,
        after={"seconds": round(time.perf_counter() - started, 3), "documents": len(documents)},
    ))

    with _LOCK:
        if documents and all(_find_document(case, document.document_id).status == "failed" for document in documents):
            case.status = "failed"
        else:
            consolidate(case, extractions, thresholds)
            check_group_enrolled(root, case)
            case.status = "review"
    save_case(root, case)
    return case


def reprocess_document(
    root, case_id: str, document_id: str, *, ocr, classify, extract, document_type: str | None = None,
) -> Case:
    """Forces a re-read of one document (reviewer's "read anyway" / reclassify) and
    re-consolidates the case from every stored extraction."""
    case_dir = _case_dir(root, case_id)
    case = load_case(root, case_id)
    document = _find_document(case, document_id)
    path = _document_path(case_dir, document_id)
    hints = metrics.reviewer_hints(root)
    thresholds = metrics.confidence_thresholds(root)

    started = time.perf_counter()
    ocr_response = ocr(path)
    ocr_seconds = time.perf_counter() - started
    page_paths = _save_ocr_output(case_dir, document_id, ocr_response)

    started = time.perf_counter()
    extraction = extract(
        document_id, ocr_response["result"], page_paths,
        document_type=document_type or document.document_type,
        allow_jev=case.synthetic, hints=hints,
    )
    extract_seconds = time.perf_counter() - started
    _save_extraction(case_dir, document_id, extraction)

    with _LOCK:
        case = load_case(root, case_id)  # another action may have changed the case meanwhile
        document = _find_document(case, document_id)
        document.status = "extracted"
        document.error = None
        if document_type:
            document.document_type = document_type
            document.document_type_label = DOCUMENT_TYPES[document_type][0]
        document.timings.update({"ocr": ocr_seconds, "extract": extract_seconds})
        document.llm = {**document.llm, **extraction.llm}
        extractions = _load_all_extractions(case_dir, case)
        consolidate(case, extractions, thresholds)
        check_group_enrolled(root, case)
        save_case(root, case)
        _append_audit(case_dir, AuditEvent(
            ts=_now(), actor="system", action="reclassify" if document_type else "extract",
            case_id=case_id, document_id=document_id, after={"document_type": document.document_type},
        ))
    return case


def _load_all_extractions(case_dir: Path, case: Case) -> dict[str, DocumentExtraction]:
    extract_dir = case_dir / "extract"
    extractions = {}
    for document in case.documents:
        path = extract_dir / f"{document.document_id}.json"
        if path.exists():
            extractions[document.document_id] = DocumentExtraction.model_validate_json(path.read_text())
    return extractions


# --------------------------------------------------------------------- consolidation

def _alnum(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text)


def _group_key(value: Value) -> str:
    """Readings agree when their letters and digits agree: "R." and "R", or an address in
    capitals and in mixed case, are the same value."""
    return _alnum(value.value if value.value is not None else value.raw_value or "").casefold()


def _group_by(items: list[Value], key) -> list[list[Value]]:
    groups: dict = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return list(groups.values())


def _merge_group(group: list[Value]) -> Value:
    # Within a group every normalized value is equal, except a folded initial: the full
    # name then wins over "R". Agreeing readings prefer the printed mixed case ("Doe" over
    # a certificate's "DOE").
    representative = max(
        group,
        key=lambda value: (len(_group_key(value)), not str(value.value or "").isupper(), value.confidence or 0.0),
    )
    evidence = [item for value in group for item in value.evidence]
    documents = {item.document_id for item in evidence}
    has_confidence = any(value.confidence is not None for value in group)
    product = 1.0
    for value in group:
        product *= 1 - (value.confidence or 0.0)  # ponytail: noisy-OR assumes independent reads
    return representative.model_copy(update={
        "evidence": evidence,
        "confidence": round(1 - product, 4) if has_confidence else None,
        "components": representative.components.model_copy(update={"agreement": len(documents) or 1}),
    })


def _group_rank(group: list[Value]) -> tuple[int, float]:
    documents = {item.document_id for value in group for item in value.evidence}
    return len(documents), max((value.confidence or 0.0) for value in group)


def _merge_catalog(extractions: dict[str, DocumentExtraction]) -> list[Value]:
    by_key: dict[str, list[Value]] = defaultdict(list)
    for extraction in extractions.values():
        for value in extraction.values:
            if value.key in FIELDS_BY_KEY:
                by_key[value.key].append(value)
    merged = []
    for key, candidates in by_key.items():
        groups = _group_by(candidates, _group_key)
        if key in NAME_KEYS_WITH_INITIALS:
            groups = _fold_initials(groups)
        best = max(groups, key=_group_rank)
        winner = _merge_group(best)
        losers = [group for group in groups if group is not best]
        if losers:
            winner.problems = [*winner.problems, "documents disagree"]
            winner.alternatives = [
                Alternative(value=group[0].value, evidence=[item for value in group for item in value.evidence])
                for group in losers
            ]
        merged.append(winner)
    return merged


NAME_KEYS_WITH_INITIALS = {"practitioner.first_name", "practitioner.middle_name"}


def _fold_initials(groups: list[list[Value]]) -> list[list[Value]]:
    """An initial on one document ("R") agrees with the full name on another ("Renee")."""
    full = [group for group in groups if len(_group_key(group[0])) > 1]
    kept = []
    for group in groups:
        initial = _group_key(group[0])
        names = [other for other in full if len(initial) == 1 and _group_key(other[0]).startswith(initial)]
        if len(names) == 1:
            names[0].extend(group)
        else:
            kept.append(group)
    return kept


def _merge_grouped(extractions: dict[str, DocumentExtraction], key_field: str, id_prefix: str, group_key) -> list[Value]:
    candidates = [value for extraction in extractions.values() for value in extraction.values if value.key == key_field]
    groups = _group_by(candidates, group_key)
    merged = [_merge_group(group) for group in groups]
    for index, value in enumerate(merged, 1):
        value.id = f"{id_prefix}-{index}"
    return merged


def _defaults(present_keys: set[str]) -> list[Value]:
    return [
        Value(
            id=spec.key, key=spec.key, label=spec.label, record=spec.record, category=spec.category,
            source="default", value=spec.default, note="default, pending setup guide", route="quick_verify",
        )
        for spec in FIELDS
        if spec.category == "defaulted" and spec.key not in present_keys
    ]


def _missing_field_message(spec) -> str:
    if spec.check == "npi":
        return f"{spec.label} not found in documents; verify via NPPES"
    return f"{spec.label} not found in documents"


def _group_requirements(case: Case, values: list[Value]) -> list[Value]:
    """Client rule for a group (PVEN) ticket: DBA name, TIN, NPI, specialty or taxonomy,
    and every service location with its phone."""
    if case.enrollment_type != "group":
        return values
    present = {value.key: value for value in values if value.value}
    legal = present.get("group.legal_name")
    if "group.dba_name" not in present and legal is not None:
        values.append(legal.model_copy(update={
            "id": "group.dba_name", "key": "group.dba_name", "label": FIELDS_BY_KEY["group.dba_name"].label,
            "category": "required", "note": "No separate DBA printed; the legal name is used as the DBA.",
            "route": "priority_review",
        }))
    for value in values:
        if value.key == "location" and value.role == "service_location" and not (value.parts or {}).get("phone"):
            _add_exception(
                case, kind="missing_field", severity="error", value_id=value.id,
                message=f"Service location {value.value or value.raw_value} has no phone number",
            )
    return values


def _require_fields(case: Case, values: list[Value]) -> list[Value]:
    present = {value.key: value for value in values}
    required_keys = [spec.key for spec in FIELDS if spec.record == case.enrollment_type and spec.category == "required"]
    required_keys += REQUIRED_EXTRA_KEYS[case.enrollment_type]
    for key in required_keys:
        value = present.get(key)
        if value is not None and value.value is not None:
            continue
        if key == "group.specialty" and (present.get("group.taxonomy") or Value(id="", key="", label="", record="other", category="other")).value:
            continue  # the client accepts a specialty or a taxonomy code
        spec = FIELDS_BY_KEY[key]
        if value is not None:
            values.remove(value)
        values.append(Value(id=key, key=key, label=spec.label, record=spec.record, category="required", route="exception"))
        _add_exception(case, kind="missing_field", severity="error", value_id=key, message=_missing_field_message(spec))
    return values


def _require_service_location_phone(case: Case, values: list[Value]) -> list[Value]:
    has_phone = any(
        value.key == "location" and value.role == "service_location" and (value.parts or {}).get("phone")
        for value in values
    )
    if has_phone:
        return values
    value_id = "location.service_location.phone"
    record = "practitioner" if case.enrollment_type == "practitioner" else "group"
    values.append(Value(
        id=value_id, key="location", label="Service location phone", record=record,
        category="required", role="service_location", route="exception",
    ))
    _add_exception(
        case, kind="missing_field", severity="error", value_id=value_id,
        message="Service location phone not found in documents",
    )
    return values


def _carry_over(previous: dict[str, Value], value: Value) -> Value:
    """A reviewer's edit or rejection of a value sticks through later reconsolidation;
    anything else (pending or merely approved) is rebuilt from the current documents."""
    old = previous.get(value.id)
    if old is not None and old.review.state in ("edited", "rejected"):
        return old
    return value


def _require_documents(case: Case) -> None:
    present_types = {
        document.document_type for document in case.documents
        if document.document_type and document.status != "duplicate"
    }
    for document_types, severity, message in REQUIRED_DOCUMENTS[case.enrollment_type]:
        if not present_types & set(document_types):
            _add_exception(case, kind="missing_document", severity=severity, message=message)


def _flag_expirations(case: Case) -> None:
    today = date.today()
    for value in case.values:
        if not value.value or "expir" not in (value.key + value.label).casefold():
            continue
        try:
            expires = date.fromisoformat(value.value)
        except ValueError:
            continue
        if expires < today:
            _add_exception(
                case, kind="expired", severity="warning", value_id=value.id,
                message=f"{value.label} expired on {value.value}",
            )
        elif (expires - today).days <= 90:
            _add_exception(
                case, kind="expired", severity="warning", value_id=value.id,
                message=f"{value.label} expires within 90 days",
            )


def _route(case: Case, thresholds: dict[str, float]) -> None:
    for value in case.values:
        if value.review.state != "pending" or value.route == "exception" or value.source == "default":
            continue  # already decided: a human decided, required-and-missing, or a fixed default
        reasons = _reasons(value, thresholds.get(value.key, 0.80))
        value.problems = reasons
        value.route = "priority_review" if reasons else "quick_verify"


def _reasons(value: Value, threshold: float) -> list[str]:
    """What a reviewer must check, most decisive first, as "<Field>: <what is wrong>
    (<evidence>). <What to do>." Never scores or internal codes."""
    read = f"read '{value.raw_value}'" if value.raw_value else "no text"
    components, reasons = value.components, []
    if components.validator == "fail":
        detail = next((p for p in value.problems if "fails" in p or "must" in p or "invalid" in p or "unrecognized" in p or "incomplete" in p or "ambiguous" in p), "fails its check")
        reasons.append(f"{value.label}: {detail} ({read}). Compare with the crop and correct it.")
    if value.alternatives:
        others = ", ".join(f"'{alternative.value}'" for alternative in value.alternatives)
        reasons.append(f"{value.label}: documents disagree ('{value.value}' vs {others}). Pick the right source.")
    if components.grounding is not None and components.grounding < 0.9:
        reasons.append(f"{value.label}: not found in the cited text ({read}). Check the highlighted region or change the source.")
    if components.repaired:
        reasons.append(f"{value.label}: a misread was corrected ({read}, saved as '{value.value}'). Confirm against the crop.")
    if not reasons and value.confidence is None:
        reasons.append(f"{value.label}: the reading could not be scored. Confirm against the crop.")
    elif not reasons and value.confidence < threshold:
        reasons.append(f"{value.label}: the reading is uncertain ({read}). Confirm against the crop.")
    return reasons


def _sample_for_audit(case: Case) -> None:
    # Defaults are fixed values, not readings: auditing them measures nothing.
    quick_verify = [
        value for value in case.values
        if value.route == "quick_verify" and value.review.state == "pending" and value.source != "default"
    ]
    rng = random.Random(case.case_id)
    for value in quick_verify:
        value.audit_sample = rng.random() < 0.03
    if len(quick_verify) >= 10 and not any(value.audit_sample for value in quick_verify):
        rng.choice(quick_verify).audit_sample = True


def enrolled_groups(root) -> list[dict]:
    """Groups loaded by submitted group (PVEN) tickets.

    ponytail: a local list stands in for the client's provider system; query it instead
    when integration access exists.
    """
    path = Path(root) / "groups.json"
    return json.loads(path.read_text()) if path.is_file() else []


def _record_group(root, case: Case) -> None:
    values = {value.key: value.value for value in case.values if value.value}
    groups = [group for group in enrolled_groups(root) if group["case_id"] != case.case_id]
    groups.append({
        "case_id": case.case_id, "ticket_id": case.ticket_id, "submitted_at": case.submitted_at,
        "npi": values.get("group.npi"), "tin": values.get("group.tin"),
        "legal_name": values.get("group.legal_name"), "dba_name": values.get("group.dba_name"),
    })
    path = Path(root) / "groups.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(groups, indent=2))
    os.replace(temporary, path)


def check_group_enrolled(root, case: Case) -> None:
    """Client rule: a practitioner is enrolled only after their group. The group NPI and TIN
    on the ticket show whether the group is loaded; if not, a group ticket comes first."""
    if case.enrollment_type != "practitioner":
        return
    values = {value.key: value.value for value in case.values if value.value}
    npi, tin = values.get("group.npi"), values.get("group.tin")
    case.exceptions = [exception for exception in case.exceptions if exception.kind != "group_not_enrolled"]
    if not npi and not tin:
        return  # the missing group NPI/TIN is already an exception of its own
    known = next((group for group in enrolled_groups(root) if (npi and group["npi"] == npi) or (tin and group["tin"] == tin)), None)
    if known is None:
        _add_exception(
            case, kind="group_not_enrolled", severity="error",
            message=f"Group (NPI {npi or 'not found'}, TIN {tin or 'not found'}) is not enrolled yet; "
                    "complete a group enrollment (PVEN) first",
        )


def consolidate(case: Case, extractions: dict[str, DocumentExtraction], thresholds: dict[str, float]) -> None:
    previous = {value.id: value for value in case.values}
    case.exceptions = [
        exception for exception in case.exceptions
        if exception.kind not in ("missing_document", "missing_field", "conflict", "expired")
    ]
    kept = [value for value in case.values if value.category == "other"]  # reviewer "add"-ed values

    values = [
        *_merge_catalog(extractions),
        *_merge_grouped(extractions, "location", "location", lambda value: (value.role, _group_key(value))),
        *_merge_grouped(extractions, "entity", "entity", lambda value: (value.role, value.label.casefold(), _group_key(value))),
    ]
    values = [_carry_over(previous, value) for value in values]
    values += _defaults({value.key for value in values})
    values = _group_requirements(case, values)
    values = _require_fields(case, values)
    values = _require_service_location_phone(case, values)
    case.values = kept + values

    for value in case.values:
        if "documents disagree" in value.problems:
            _add_exception(case, kind="conflict", severity="warning", value_id=value.id, message=f"{value.label}: documents disagree")

    _require_documents(case)
    _flag_expirations(case)
    _route(case, thresholds)
    _sample_for_audit(case)


# -------------------------------------------------------------------------- payload

def submission_payload(case: Case) -> dict:
    records: dict[str, dict] = {"group": {}, "practitioner": {}, "coverage": {}}
    service_locations, other_locations, entities = [], [], []
    field_sources: dict[str, dict] = {}
    for value in case.values:
        if value.key in FIELDS_BY_KEY:
            spec = FIELDS_BY_KEY[value.key]
            records[spec.record][spec.target] = value.value
        elif value.key == "location":
            parts = value.parts or {}
            location = {
                "role": value.role,
                "line1": parts.get("line1", ""), "line2": parts.get("line2", ""),
                "city": parts.get("city", ""), "state": parts.get("state", ""), "zip": parts.get("zip", ""),
                "phone": parts.get("phone", ""), "primary": parts.get("primary") == "Yes",
            }
            (service_locations if value.role == "service_location" else other_locations).append(location)
        elif value.key == "entity":
            entities.append({"type": value.role, "label": value.label, "value": value.value})
        if value.evidence:
            source = value.evidence[0]
            field_sources[value.id] = {
                "source": value.source, "document_id": source.document_id, "page": source.page,
                "region_ids": source.region_ids, "confidence": value.confidence, "review_state": value.review.state,
            }
    return {
        "schema_version": 1, "case_id": case.case_id, "ticket_id": case.ticket_id,
        "enrollment_type": case.enrollment_type, "submitted_at": case.submitted_at,
        "submitted_by": case.submitted_by, "records": records,
        "service_locations": service_locations, "other_locations": other_locations, "entities": entities,
        "field_sources": field_sources,
        "exceptions": [exception.message for exception in case.exceptions if exception.state == "open"],
    }


# ---------------------------------------------------------------------- review actions

def _record_feedback(root, case: Case, value: Value, *, extracted, final, correct: bool) -> None:
    document_id = value.evidence[0].document_id if value.evidence else None
    document = next((d for d in case.documents if d.document_id == document_id), None) if document_id else None
    metrics.record_feedback(root, {
        "case_id": case.case_id, "value_id": value.id, "key": value.key,
        "document_type": document.document_type if document else None,
        "raw_value": value.raw_value, "extracted": extracted, "final": final,
        "correct": correct, "confidence": value.confidence,
    })


def _union_boxes(regions: list[dict]) -> Box:
    boxes = [region["bounding_box"] for region in regions if region.get("bounding_box")]
    return Box(
        left=min(box["left"] for box in boxes), top=min(box["top"] for box in boxes),
        right=max(box["right"] for box in boxes), bottom=max(box["bottom"] for box in boxes),
    )


def _action_edit(root, case_dir, case, action) -> None:
    value = _find_value(case, action["value_id"])
    before = value.model_dump()
    new_value = action["value"]
    value.raw_value, value.value = new_value, new_value
    value.review = Review(
        state="edited", by=action["actor"], at=_now(),
        reason_code=action.get("reason_code"), reason=action.get("reason"),
    )
    # Re-saving the same value confirms it; only a changed value is a correction.
    unchanged = _alnum(str(before["value"] or "")).casefold() == _alnum(str(new_value or "")).casefold()
    _record_feedback(root, case, value, extracted=before["value"], final=new_value, correct=unchanged)
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="edit", case_id=case.case_id, value_id=value.id,
        before=before, after=value.model_dump(), reason=action.get("reason"), versions=case.versions,
    ))


def _action_add(root, case_dir, case, action) -> None:
    index = sum(1 for value in case.values if value.key == "other") + 1
    value = Value(
        id=f"other-{index}", key="other", label=action["label"], value=action.get("value"),
        raw_value=action.get("value"), role=action.get("role"),
        record=action.get("record", "other"), category=action.get("category", "other"),
        source="reviewer", route="quick_verify", review=Review(state="approved", by=action["actor"], at=_now()),
    )
    case.values.append(value)
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="add", case_id=case.case_id, value_id=value.id,
        after=value.model_dump(), versions=case.versions,
    ))


def _action_change_source(root, case_dir, case, action) -> None:
    value = _find_value(case, action["value_id"])
    before = value.model_dump()
    source = action["source"]
    document_id, page, region_ids = source["document_id"], source["page"], source["region_ids"]
    ocr_result = json.loads((case_dir / "ocr" / f"{document_id}.json").read_text())["result"]
    regions = {
        region["id"]: region for doc_page in ocr_result.get("pages") or [] for region in doc_page.get("regions") or []
    }
    cited = [regions[region_id] for region_id in region_ids if region_id in regions]
    raw_value = action.get("value") or " ".join(region.get("text") or "" for region in cited).strip()
    value.evidence = [Evidence(document_id=document_id, page=page, region_ids=region_ids, box=_union_boxes(cited))]
    value.raw_value = raw_value
    spec = FIELDS_BY_KEY.get(value.key)
    normalized, problem = CHECKS[spec.check](raw_value) if spec and spec.check and raw_value else (None, None)
    value.value = normalized if normalized is not None else raw_value
    # A new source is a new reading: it is checked again and waits for an approval.
    value.components = value.components.model_copy(update={
        "grounding": 1.0, "ocr": None, "llm": None, "verifier": None,
        "validator": "none" if not (spec and spec.check) else "fail" if normalized is None else "pass",
    })
    value.confidence = None
    value.problems = ["source changed by reviewer", *([problem] if problem and normalized is None else [])]
    value.route = "priority_review"
    value.review = Review(state="pending")
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="change_source", case_id=case.case_id, value_id=value.id,
        before=before, after=value.model_dump(), source=value.evidence[0], versions=case.versions,
    ))


def _action_approve(root, case_dir, case, action) -> None:
    value = _find_value(case, action["value_id"])
    before = value.model_dump()
    value.review = Review(state="approved", by=action["actor"], at=_now())
    _record_feedback(root, case, value, extracted=value.value, final=value.value, correct=True)
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="approve", case_id=case.case_id, value_id=value.id,
        before=before, after=value.model_dump(), versions=case.versions,
    ))


WRONG_READING_CODES = {"wrong_value", "wrong_source", "illegible"}


def _action_reject(root, case_dir, case, action) -> None:
    if not action.get("reason"):
        raise ValueError("reject requires a reason")
    value = _find_value(case, action["value_id"])
    before = value.model_dump()
    value.review = Review(
        state="rejected", by=action["actor"], at=_now(),
        reason_code=action.get("reason_code"), reason=action["reason"],
    )
    # Only a wrong or unreadable reading is a label about extraction accuracy; "not
    # applicable" says the value is irrelevant, not misread.
    if action.get("reason_code") in WRONG_READING_CODES:
        _record_feedback(root, case, value, extracted=value.value, final=None, correct=False)
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="reject", case_id=case.case_id, value_id=value.id,
        before=before, after=value.model_dump(), reason=action["reason"], versions=case.versions,
    ))


def _action_reject_case(root, case_dir, case, action) -> None:
    if not action.get("reason"):
        raise ValueError("reject_case requires a reason")
    before = {"status": case.status}
    case.status = "rejected"
    case.rejection = {"reason_code": action.get("reason_code"), "reason": action["reason"], "by": action["actor"], "at": _now()}
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="reject_case", case_id=case.case_id,
        before=before, after=case.rejection, reason=action["reason"], versions=case.versions,
    ))


def _action_resolve_exception(root, case_dir, case, action) -> None:
    if not action.get("reason"):
        raise ValueError("resolve_exception requires a reason")
    exception = next((item for item in case.exceptions if item.id == action["exception_id"]), None)
    if exception is None:
        raise ValueError(f"no exception {action['exception_id']} in case {case.case_id}")
    before = exception.model_dump()
    exception.state = "resolved"
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="resolve_exception", case_id=case.case_id,
        before=before, after=exception.model_dump(), reason=action["reason"], versions=case.versions,
    ))


def _blocking_value_ids(case: Case) -> list[str]:
    return [
        value.id for value in case.values
        if value.review.state == "pending" and (value.route in ("priority_review", "exception") or value.audit_sample)
    ]


def _invalid_value_ids(case: Case) -> list[str]:
    """Identifiers, dates and codes the downstream load would refuse."""
    invalid = []
    for value in case.values:
        spec = FIELDS_BY_KEY.get(value.key)
        if spec and spec.check and value.value and value.review.state != "rejected":
            if CHECKS[spec.check](value.value)[0] is None:
                invalid.append(value.id)
    return invalid


def _action_submit(root, case_dir, case, action) -> None:
    check_group_enrolled(root, case)
    waiting_for_group = [e.id for e in case.exceptions if e.kind == "group_not_enrolled" and e.state == "open"]
    if waiting_for_group:
        raise ReviewBlocked(waiting_for_group, "The group is not enrolled yet; submit its group enrollment (PVEN) first")
    blocking = _blocking_value_ids(case)
    if blocking:
        raise ReviewBlocked(blocking, "Review the listed values before submitting")
    invalid = _invalid_value_ids(case)
    if invalid:
        raise ReviewBlocked(invalid, "These values fail their format or check-digit validation; edit or reject them")
    # Only missing_document errors can stay open at submit; they are cleared by resolve_exception.
    open_document_errors = [
        exception for exception in case.exceptions
        if exception.kind == "missing_document" and exception.severity == "error" and exception.state == "open"
    ]
    if open_document_errors:
        raise ReviewBlocked(
            [exception.id for exception in open_document_errors], "Resolve the missing-document exceptions first"
        )

    pending_quick_verify = [value for value in case.values if value.route == "quick_verify" and value.review.state == "pending"]
    if pending_quick_verify:
        now = _now()
        for value in pending_quick_verify:
            value.review = Review(state="approved", by=action["actor"], at=now)
        _append_audit(case_dir, AuditEvent(
            ts=now, actor=action["actor"], action="bulk_approve", case_id=case.case_id,
            after={"value_ids": [value.id for value in pending_quick_verify]}, versions=case.versions,
        ))

    case.status = "submitted"
    case.submitted_at = _now()
    case.submitted_by = action["actor"]
    if case.enrollment_type == "group":
        _record_group(root, case)
    payload = submission_payload(case)
    (case_dir / "payload.json").write_text(json.dumps(payload, indent=2))
    _append_audit(case_dir, AuditEvent(
        ts=_now(), actor=action["actor"], action="submit", case_id=case.case_id,
        after={"payload": "payload.json"}, versions=case.versions,
    ))


_ACTIONS = {
    "edit": _action_edit,
    "add": _action_add,
    "change_source": _action_change_source,
    "approve": _action_approve,
    "reject": _action_reject,
    "reject_case": _action_reject_case,
    "resolve_exception": _action_resolve_exception,
    "submit": _action_submit,
}


def apply_action(root, case_id: str, action: dict) -> Case:
    if not action.get("actor"):
        raise ValueError("action requires an actor")
    handler = _ACTIONS.get(action.get("type"))
    if handler is None:
        raise ValueError(f"unknown action type: {action.get('type')}")
    case_dir = _case_dir(root, case_id)
    with _LOCK:
        case = load_case(root, case_id)
        if case.status in ("submitted", "rejected"):
            raise ValueError(f"case {case_id} is {case.status} and cannot be changed")
        handler(root, case_dir, case, action)
        save_case(root, case)
    return case
