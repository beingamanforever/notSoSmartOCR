"""Portable case bundle: one JSON artifact (plus an offline zip) that lets anyone
re-render every page with its regions, reading order, confidence, provenance,
extracted values and the full audit trail, without the live service.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from .schema import Case

SCHEMA_VERSION = 1
PDF_FORMATS = {"digital_pdf", "scanned_pdf"}
SKIPPED_KINDS = {"coverage_risk", "table_candidate"}

COORDINATES_LEGEND = (
    "box_px = [left, top, right, bottom] in page image pixels, origin top-left; "
    "box = the same as fractions of page width and height"
)
CONFIDENCE_LEGEND = (
    "calibrated probability that the OCR text is within 10% normalized edit "
    "distance of the truth; LLM-refined regions have none"
)
README_TEXT = """Enrollment case bundle
=======================

bundle.json   Everything: case record, documents, per-page regions with boxes
              and provenance, extracted values, exceptions and the audit trail.
              See bundle.json's "legend" for coordinates and confidence methods.
viewer.html   Open this in a browser to re-render the pages offline; the
              bundle is embedded in it, no server needed.
pages/*.png   The page images regions and boxes are drawn against.
audit.jsonl   One AuditEvent JSON object per line, in order.
payload.json  The submitted payload, if this case was submitted.

Coordinates: box_px is [left, top, right, bottom] in page image pixels,
origin top-left. box is the same box as fractions of page width/height.
"""


def _region_box(bounding_box: dict, width: int, height: int) -> tuple[list[int], list[float]]:
    box_px = [bounding_box["left"], bounding_box["top"], bounding_box["right"], bounding_box["bottom"]]
    box = [
        round(box_px[0] / width, 5),
        round(box_px[1] / height, 5),
        round(box_px[2] / width, 5),
        round(box_px[3] / height, 5),
    ]
    return box_px, box


def _routing_by_region(page: dict) -> dict[str, list[str]]:
    """Map source region id -> LLM routing reasons, from refinement.crops."""
    routing: dict[str, list[str]] = {}
    for crop in (page.get("refinement") or {}).get("crops") or []:  # null until a deferred LLM re-read lands
        for source_id in crop.get("source_ids", []):
            routing[source_id] = crop.get("routing_reasons", [])
    return routing


def _bundle_region(
    region: dict, width: int, height: int, routing: dict[str, list[str]], methods: dict[str, dict]
) -> dict | None:
    structure = region.get("structure") or {}
    if structure.get("role") == "layout_block" or region.get("kind") in SKIPPED_KINDS:
        return None

    provenance = region.get("text_provenance") or {}
    method = provenance.get("method")
    if method not in methods:
        methods[method] = {"meaning": provenance.get("confidence_meaning")}

    # qwen's raw 0.0 confidence means "unscored", not zero confidence.
    confidence = None if method == "qwen_crop_refinement" else region.get("confidence")
    box_px, box = _region_box(region["bounding_box"], width, height)
    region_id = region["id"]
    return {
        "id": region_id,
        "kind": region.get("kind"),
        "text": region.get("text"),
        "confidence": confidence,
        "reading_order": region.get("reading_order"),
        "resolution": region.get("resolution"),
        "method": method,
        "llm_routed": region_id in routing,
        "llm_reasons": routing.get(region_id, []),
        "box_px": box_px,
        "box": box,
    }


def _bundle_pages(
    ocr_result: dict, document_id: str, pdf_dpi: int | None, methods: dict[str, dict]
) -> list[dict]:
    pages = []
    for page in ocr_result.get("pages") or []:
        width, height = page["width"], page["height"]
        routing = _routing_by_region(page)
        regions = [
            bundled
            for region in page.get("regions") or []
            if (bundled := _bundle_region(region, width, height, routing, methods)) is not None
        ]
        pages.append(
            {
                "page": page["page_number"],
                "image": f"pages/{document_id}-p{page['page_number']}.png",
                "width": width,
                "height": height,
                "dpi": pdf_dpi,
                "regions": regions,
                "arrangement": page.get("arrangement"),
            }
        )
    return pages


def _bundle_document(document, case_dir: Path, methods: dict[str, dict]) -> dict:
    bundled = document.model_dump()
    ocr_path = case_dir / "ocr" / f"{document.document_id}.json"
    if not ocr_path.exists():
        bundled["pages"] = []
        return bundled
    ocr_response = json.loads(ocr_path.read_text())
    pdf_dpi = 300 if document.format in PDF_FORMATS else None
    bundled["pages"] = _bundle_pages(ocr_response["result"], document.document_id, pdf_dpi, methods)
    return bundled


def _read_audit(case_dir: Path) -> list[dict]:
    audit_path = case_dir / "audit.jsonl"
    if not audit_path.exists():
        return []
    return [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]


def build_bundle(case: Case, case_dir: Path) -> dict:
    methods: dict[str, dict] = {}
    documents = [_bundle_document(document, case_dir, methods) for document in case.documents]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legend": {
            "coordinates": COORDINATES_LEGEND,
            "methods": methods,
            "confidence": CONFIDENCE_LEGEND,
        },
        "case": case.model_dump(exclude={"documents", "values", "exceptions"}),
        "documents": documents,
        "values": [value.model_dump() for value in case.values],
        "exceptions": [exception.model_dump() for exception in case.exceptions],
        "audit": _read_audit(case_dir),
    }


def export_zip(case: Case, case_dir: Path, viewer_template: Path) -> bytes:
    bundle = build_bundle(case, case_dir)
    bundle_json = json.dumps(bundle, indent=2)

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle.json", bundle_json)
        archive.writestr("README.txt", README_TEXT)

        audit_path = case_dir / "audit.jsonl"
        if audit_path.exists():
            archive.write(audit_path, "audit.jsonl")

        payload_path = case_dir / "payload.json"
        if payload_path.exists():
            archive.write(payload_path, "payload.json")

        for document in bundle["documents"]:
            for page in document["pages"]:
                image_path = case_dir / page["image"]
                if image_path.exists():
                    archive.write(image_path, page["image"])

        # < can't close a </script> tag; it is a valid JSON/JS string escape for "<".
        embedded_json = bundle_json.replace("<", "\\u003c")
        viewer_html = viewer_template.read_text().replace(
            "<!--ENROLLMENT_BUNDLE-->",
            f"<script>window.ENROLLMENT_BUNDLE = {embedded_json};</script>",
        )
        archive.writestr("viewer.html", viewer_html)

    return buffer.getvalue()
