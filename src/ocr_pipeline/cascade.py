"""Deterministic validation and evidence-scoped text patching."""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from typing import Any

from .contracts import DocumentResult, TextRegion
from .rendering import render_evidence
from .verification import literal_text_risks

NON_TEXT_VISUAL_KINDS = {
    "image",
    "figure",
    "chart",
    "header_image",
    "footer_image",
}
MAX_REPEATED_PHRASE_TOKENS = 32


def identify_risky_regions(document: DocumentResult) -> dict[str, list[str]]:
    risks: dict[str, list[str]] = {}
    for page in document.pages:
        order_counts = Counter(region.reading_order for region in page.regions)
        for region in page.regions:
            reasons: list[str] = []
            kind = region.kind.casefold().replace("-", "_")
            if not region.text.strip() and kind not in NON_TEXT_VISUAL_KINDS:
                reasons.append("empty_content")
            box = region.bounding_box
            if (
                box.left < 0
                or box.top < 0
                or box.right <= box.left
                or box.bottom <= box.top
                or box.right > page.width
                or box.bottom > page.height
            ):
                reasons.append("invalid_bbox")
            if order_counts[region.reading_order] > 1:
                reasons.append("duplicate_reading_order")
            if not 1 <= region.reading_order <= len(page.regions):
                reasons.append("out_of_range_reading_order")
            table_risk = _html_table_risk(region)
            if table_risk:
                reasons.append(table_risk)
            if _has_repeated_text(region):
                reasons.append("repeated_text")
            reasons.extend(literal_text_risks(region.text))
            if reasons:
                risks[region.id] = reasons
    return risks


def regions_have_fewer_risks(
    before: DocumentResult,
    after: DocumentResult,
    region_ids: list[str],
) -> bool:
    if not region_ids or len(set(region_ids)) != len(region_ids):
        return False
    before_risks = identify_risky_regions(before)
    after_risks = identify_risky_regions(after)
    return all(
        set(after_risks.get(region_id, [])) < set(before_risks.get(region_id, []))
        for region_id in region_ids
    )


def build_patch_request(
    document: DocumentResult,
    risks: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    risks = identify_risky_regions(document) if risks is None else risks
    regions = {region.id: region for page in document.pages for region in page.regions}
    unknown_ids = set(risks) - set(regions)
    if unknown_ids:
        raise ValueError(f"Unknown risky region IDs: {sorted(unknown_ids)}")

    authorized_ids = [
        region.id
        for page in document.pages
        for region in page.regions
        if region.id in risks
    ]
    return {
        "document_id": document.document_id,
        "authorized_region_ids": authorized_ids,
        "regions": [
            {
                "id": region_id,
                "kind": regions[region_id].kind,
                "text": regions[region_id].text,
                "bounding_box": asdict(regions[region_id].bounding_box),
                "provider": regions[region_id].provider,
                "reasons": list(risks[region_id]),
            }
            for region_id in authorized_ids
        ],
    }


def apply_region_patches(
    document: DocumentResult,
    request: dict[str, Any],
    patches: list[dict[str, Any]],
) -> DocumentResult:
    if request.get("document_id") != document.document_id:
        raise ValueError("Patch request document does not match")

    regions = [region for page in document.pages for region in page.regions]
    existing = {region.id: region for region in regions}
    if len(existing) != len(regions):
        raise ValueError("Document contains duplicate region IDs")

    authorized = request.get("authorized_region_ids")
    if not isinstance(authorized, list) or any(
        not isinstance(item, str) for item in authorized
    ):
        raise ValueError("Patch request has invalid authorized region IDs")
    if len(set(authorized)) != len(authorized) or not set(authorized) <= set(existing):
        raise ValueError("Patch request authorizes invalid region IDs")

    replacements: dict[str, str] = {}
    for patch in patches:
        _validate_patch(patch, authorized, existing, replacements)
        replacements[patch["id"]] = patch["text"]

    result = copy.deepcopy(document)
    for page in result.pages:
        changed = False
        for region in page.regions:
            if region.id in replacements:
                region.text = replacements[region.id]
                changed = True
        if changed:
            page.text = render_evidence(page.regions)
    return result


def _validate_patch(
    patch: dict[str, Any],
    authorized: list[str],
    existing: dict[str, TextRegion],
    replacements: dict[str, str],
) -> None:
    allowed_keys = {"id", "text", "kind", "bounding_box", "provider"}
    if not isinstance(patch, dict) or set(patch) - allowed_keys:
        raise ValueError("Patch contains unsupported fields")
    region_id = patch.get("id")
    if region_id not in authorized or region_id not in existing:
        raise ValueError(f"Region is not authorized: {region_id}")
    if region_id in replacements:
        raise ValueError(f"Duplicate patch for region: {region_id}")
    text = patch.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Replacement text is empty: {region_id}")

    region = existing[region_id]
    protected = {
        "kind": region.kind,
        "bounding_box": asdict(region.bounding_box),
        "provider": region.provider,
    }
    for field, expected in protected.items():
        if field in patch and patch[field] != expected:
            raise ValueError(f"Patch changes protected field {field}: {region_id}")


def _html_table_risk(region: TextRegion) -> str | None:
    text = region.text.strip()
    if "<table" not in text.lower():
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return "malformed_html_table"
    if _tag(root) != "table":
        return "malformed_html_table"

    widths: list[int] = []
    has_rowspan = False
    for row in (element for element in root.iter() if _tag(element) == "tr"):
        width = 0
        for cell in row:
            if _tag(cell) not in {"td", "th"}:
                continue
            try:
                colspan = int(cell.attrib.get("colspan", "1"))
                rowspan = int(cell.attrib.get("rowspan", "1"))
            except ValueError:
                return "malformed_html_table"
            if colspan < 1 or rowspan < 1:
                return "malformed_html_table"
            if rowspan > 1:
                has_rowspan = True
            width += colspan
        widths.append(width)
    if not widths or 0 in widths:
        return "malformed_html_table"
    if has_rowspan:
        return None
    return "non_rectangular_html_table" if len(set(widths)) > 1 else None


def _has_repeated_text(region: TextRegion) -> bool:
    kind = region.kind.casefold().replace("-", "_")
    if "table" in kind or "<table" in region.text.casefold():
        return False
    tokens = re.findall(r"\w+", region.text.casefold())
    maximum_size = min(MAX_REPEATED_PHRASE_TOKENS, len(tokens) // 3)
    for size in range(2, maximum_size + 1):
        for start in range(len(tokens) - (size * 3) + 1):
            phrase = tokens[start : start + size]
            if len(set(phrase)) < 2:
                continue
            if (
                tokens[start + size : start + (size * 2)] == phrase
                and tokens[start + (size * 2) : start + (size * 3)] == phrase
            ):
                return True
    return False


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()
