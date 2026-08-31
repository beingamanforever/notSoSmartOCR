"""Selective visual repair for deterministic OCR risks."""

from __future__ import annotations

import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image, UnidentifiedImageError

from .cascade import (
    apply_region_patches,
    build_patch_request,
    identify_risky_regions,
    regions_have_fewer_risks,
)
from .contracts import DocumentResult, TextRegion
from .openrouter import OpenRouterError

PATCHABLE_RISKS = {
    "empty_content",
    "malformed_html_table",
    "non_rectangular_html_table",
    "repeated_text",
}


def repair_risky_regions(
    document: DocumentResult,
    page_images: Mapping[int, Path],
    repair_image: Callable[[Path, str, Mapping[str, Any]], Any],
    verifier_image: Callable[[Path, str, Mapping[str, Any]], Any] | None = None,
) -> tuple[DocumentResult, list[dict[str, Any]]]:
    current = document
    records: list[dict[str, Any]] = []
    risks_by_id = identify_risky_regions(document)
    candidates = [
        (page.page_number, region.id)
        for page in document.pages
        for region in page.regions
        if PATCHABLE_RISKS.intersection(risks_by_id.get(region.id, []))
    ]

    with tempfile.TemporaryDirectory(prefix="ocr-repair-") as temporary_dir:
        for sequence, (page_number, region_id) in enumerate(candidates, start=1):
            reasons = risks_by_id.get(region_id, [])
            if not PATCHABLE_RISKS.intersection(reasons):
                continue
            if "invalid_bbox" in reasons:
                records.append(_record(region_id, page_number, "invalid_bbox"))
                continue

            page = next(
                page for page in current.pages if page.page_number == page_number
            )
            region = next(region for region in page.regions if region.id == region_id)
            image_path = page_images.get(page_number)
            if image_path is None:
                records.append(_record(region_id, page_number, "missing_page_image"))
                continue

            crop_path = Path(temporary_dir) / f"region-{sequence}.png"
            crop_error = _write_crop(Path(image_path), crop_path, region)
            if crop_error:
                records.append(_record(region_id, page_number, crop_error))
                continue

            try:
                result = repair_image(
                    crop_path,
                    _prompt(region_id, reasons),
                    _response_schema(region_id),
                )
            except OpenRouterError as error:
                records.append(
                    _record(
                        region_id,
                        page_number,
                        f"primary_{error.code}",
                        primary_error=_provider_error(error),
                    )
                )
                continue

            primary = _metadata(result)
            content = getattr(result, "content", None)
            if not _is_strict_response(content, region_id):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "invalid_response",
                        primary=primary,
                    )
                )
                continue
            if (
                _requires_table_html(region, reasons)
                and "<table" not in str(content.get("text", "")).casefold()
            ):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "invalid_table_repair",
                        primary=primary,
                    )
                )
                continue

            request = build_patch_request(current, {region_id: reasons})
            try:
                candidate = apply_region_patches(current, request, [content])
            except ValueError:
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "invalid_patch",
                        primary=primary,
                    )
                )
                continue
            if verifier_image is None:
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "unverified_candidate",
                        primary=primary,
                    )
                )
                continue

            try:
                verifier_result = verifier_image(
                    crop_path,
                    _verifier_prompt(region_id, reasons),
                    _response_schema(region_id),
                )
            except OpenRouterError as error:
                records.append(
                    _record(
                        region_id,
                        page_number,
                        f"verifier_{error.code}",
                        primary=primary,
                        verifier_error=_provider_error(error),
                    )
                )
                continue

            verifier = _metadata(verifier_result)
            verifier_content = getattr(verifier_result, "content", None)
            if not _is_strict_response(verifier_content, region_id):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "invalid_verifier_response",
                        primary=primary,
                        verifier=verifier,
                    )
                )
                continue
            if (
                _requires_table_html(region, reasons)
                and "<table" not in str(verifier_content.get("text", "")).casefold()
            ):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "invalid_verifier_table_repair",
                        primary=primary,
                        verifier=verifier,
                    )
                )
                continue
            if not _models_are_independent(primary, verifier):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "non_independent_models",
                        primary=primary,
                        verifier=verifier,
                    )
                )
                continue
            try:
                apply_region_patches(current, request, [verifier_content])
            except ValueError:
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "invalid_verifier_patch",
                        primary=primary,
                        verifier=verifier,
                    )
                )
                continue
            if _normalized_text(content["text"]) != _normalized_text(
                verifier_content["text"]
            ):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "verifier_disagreement",
                        primary=primary,
                        verifier=verifier,
                    )
                )
                continue
            if not regions_have_fewer_risks(current, candidate, [region_id]):
                records.append(
                    _record(
                        region_id,
                        page_number,
                        "no_deterministic_improvement",
                        primary=primary,
                        verifier=verifier,
                    )
                )
                continue

            _record_hosted_patch(candidate, page_number, region_id, primary, verifier)
            current = candidate
            risks_by_id = identify_risky_regions(current)
            records.append(
                _record(
                    region_id,
                    page_number,
                    "independent_agreement",
                    accepted=True,
                    primary=primary,
                    verifier=verifier,
                )
            )
    return current, records


def _write_crop(source: Path, destination: Path, region: TextRegion) -> str | None:
    try:
        with Image.open(source) as image:
            box = region.bounding_box
            if box.right > image.width or box.bottom > image.height:
                return "image_bbox_out_of_bounds"
            image.crop((box.left, box.top, box.right, box.bottom)).save(
                destination, format="PNG"
            )
    except (OSError, UnidentifiedImageError):
        return "invalid_page_image"
    return None


def _response_schema(region_id: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": [region_id]},
            "text": {"type": "string"},
        },
        "required": ["id", "text"],
        "additionalProperties": False,
    }


def _prompt(region_id: str, reasons: list[str]) -> str:
    return (
        f"Read only this cropped OCR region. Return region id {region_id!r} and its "
        f"literal replacement text. Preserve table HTML when present. Risks: "
        f"{', '.join(reasons)}."
    )


def _verifier_prompt(region_id: str, reasons: list[str]) -> str:
    return (
        f"Independently read only this cropped OCR region. Return region id "
        f"{region_id!r} and its literal text. Preserve table HTML when present. "
        f"Risks: {', '.join(reasons)}."
    )


def _normalized_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n")).strip()


def _is_strict_response(content: Any, region_id: str) -> bool:
    return (
        isinstance(content, dict)
        and set(content) == {"id", "text"}
        and content["id"] == region_id
        and isinstance(content["text"], str)
        and bool(content["text"].strip())
    )


def _metadata(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    metadata = {
        "model": getattr(result, "model", None),
        "provider": getattr(result, "provider", None),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "cost": getattr(result, "cost", None),
    }
    for field in (
        "attempts",
        "latency_ms",
        "finish_reason",
        "native_finish_reason",
    ):
        value = getattr(result, field, None)
        if value is not None:
            metadata[field] = value
    return metadata


def _record(
    region_id: str,
    page_number: int,
    reason: str,
    *,
    accepted: bool = False,
    primary: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
    primary_error: dict[str, Any] | None = None,
    verifier_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reported_costs = [
        metadata["cost"]
        for metadata in (primary, verifier)
        if metadata is not None
        and isinstance(metadata.get("cost"), (int, float))
        and not isinstance(metadata.get("cost"), bool)
    ]
    record = {
        "region_id": region_id,
        "page_number": page_number,
        "status": "accepted" if accepted else "abstained",
        "reason": reason,
        "primary": primary,
        "verifier": verifier,
        "reported_cost": round(sum(reported_costs), 10),
    }
    if primary_error is not None:
        record["primary_error"] = primary_error
    if verifier_error is not None:
        record["verifier_error"] = verifier_error
    return record


def _requires_table_html(region: TextRegion, reasons: list[str]) -> bool:
    kind = region.kind.casefold().replace("-", "_")
    return bool(
        "table" in kind
        or {"malformed_html_table", "non_rectangular_html_table"}.intersection(reasons)
    )


def _models_are_independent(
    primary: Mapping[str, Any], verifier: Mapping[str, Any]
) -> bool:
    primary_model = primary.get("model")
    verifier_model = verifier.get("model")
    return (
        isinstance(primary_model, str)
        and bool(primary_model.strip())
        and isinstance(verifier_model, str)
        and bool(verifier_model.strip())
        and primary_model != verifier_model
    )


def _provider_error(error: OpenRouterError) -> dict[str, Any]:
    return {
        "code": error.code,
        "message": str(error),
        "status_code": error.status_code,
        "attempts": error.attempts,
        "latency_ms": error.latency_ms,
    }


def _record_hosted_patch(
    document: DocumentResult,
    page_number: int,
    region_id: str,
    primary: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> None:
    page = next(page for page in document.pages if page.page_number == page_number)
    region = next(region for region in page.regions if region.id == region_id)
    region.text_provenance = {
        "mode": "hosted_patch",
        "primary_model": primary.get("model"),
        "primary_provider": primary.get("provider"),
        "verifier_model": verifier.get("model"),
        "verifier_provider": verifier.get("provider"),
    }
    page.route = "accept_hosted_patch"

    resolved_ids = {
        failure.id
        for failure in document.failures
        if failure.page_number == page_number and failure.code == "no_text_detected"
    }
    if resolved_ids and region.kind.casefold().replace("-", "_") == "page_text":
        document.failures = [
            failure for failure in document.failures if failure.id not in resolved_ids
        ]
        page.failure_ids = [
            failure_id
            for failure_id in page.failure_ids
            if failure_id not in resolved_ids
        ]

    if not document.failures:
        document.status = "success"
        return
    successful_pages = sum(not result.failure_ids for result in document.pages)
    document.status = "partial" if successful_pages else "failed"
