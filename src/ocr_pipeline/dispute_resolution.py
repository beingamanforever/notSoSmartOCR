"""Bulk visual resolution for existing OCR disputes."""

from __future__ import annotations

import copy
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping

from PIL import Image, ImageDraw, UnidentifiedImageError

from .contracts import (
    ALTERNATIVE_DECISION_STATES,
    AlternativeDecision,
    BoundingBox,
    TextAlternative,
    TextRegion,
)
from .openrouter import OpenRouterError

BulkImageCall = Callable[[Path, str, Mapping[str, Any]], Any]
DISPUTE_RESOLUTIONS = frozenset({"conflicting", "unreadable"})
NON_TEXT_DISPUTE_ROLES = frozenset({"coverage_risk", "table_candidate"})
NON_TEXT_DISPUTE_KINDS = frozenset(
    {"checkbox", "control", "radio", "coverage_risk", "table_candidate"}
)
DEFAULT_MAX_DISPUTES = 32
DEFAULT_MAX_SHEET_WIDTH = 4096
DEFAULT_MAX_SHEET_HEIGHT = 4096


@dataclass(frozen=True)
class _Reading:
    id: str
    text: str
    confidence: float | None
    provider: str
    provenance: dict[str, Any] | None
    cell_evidence: _CellEvidence | None = None


@dataclass(frozen=True)
class _CellEvidence:
    evidence_ids: tuple[str, ...]
    supporters: tuple[dict[str, Any], ...]
    decision: str | None
    resolution: str | None


@dataclass(frozen=True)
class _Dispute:
    id: str
    resolution: str
    bounding_box: BoundingBox
    readings: tuple[_Reading, ...]
    region_index: int
    cell_index: int | None = None


class DisputeResolutionStage:
    """Resolve all page disputes using one contact-sheet call per model."""

    name = "dispute_resolution"

    def __init__(
        self,
        resolve_image: BulkImageCall,
        verifier_image: BulkImageCall | None = None,
        *,
        crop_padding: int = 8,
        eligible_local_models: Collection[str] = (),
        max_disputes: int = DEFAULT_MAX_DISPUTES,
        max_sheet_width: int = DEFAULT_MAX_SHEET_WIDTH,
        max_sheet_height: int = DEFAULT_MAX_SHEET_HEIGHT,
    ) -> None:
        if (
            not isinstance(crop_padding, int)
            or isinstance(crop_padding, bool)
            or not 0 <= crop_padding <= 64
        ):
            raise ValueError("crop_padding must be from 0 to 64")
        if not _positive_int(max_disputes):
            raise ValueError("max_disputes must be a positive integer")
        if not _positive_int(max_sheet_width) or not _positive_int(max_sheet_height):
            raise ValueError("contact-sheet dimensions must be positive integers")
        if isinstance(eligible_local_models, str) or any(
            not isinstance(model, str) or not model or model != model.strip()
            for model in eligible_local_models
        ):
            raise ValueError("eligible_local_models must contain trimmed model IDs")
        self.resolve_image = resolve_image
        self.verifier_image = verifier_image
        self.crop_padding = crop_padding
        self.eligible_local_models = frozenset(eligible_local_models)
        self.max_disputes = max_disputes
        self.max_sheet_width = max_sheet_width
        self.max_sheet_height = max_sheet_height

    def apply(
        self,
        image_path: Path,
        page_number: int,
        regions: list[TextRegion],
    ) -> list[TextRegion]:
        disputes = _collect_disputes(regions)
        if disputes is None or not disputes:
            return regions
        if len(disputes) > self.max_disputes:
            return regions

        try:
            with Image.open(image_path) as opened:
                page = opened.convert("RGB")
        except (OSError, UnidentifiedImageError):
            return regions

        try:
            crop_boxes = [
                _padded_box(item.bounding_box, page.size, self.crop_padding)
                for item in disputes
            ]
            if any(box is None for box in crop_boxes):
                return regions
            with tempfile.TemporaryDirectory(prefix="ocr-disputes-") as temporary:
                sheet_path = Path(temporary) / f"page-{page_number}-disputes.png"
                if not _write_contact_sheet(
                    page,
                    disputes,
                    [box for box in crop_boxes if box is not None],
                    sheet_path,
                    self.max_sheet_width,
                    self.max_sheet_height,
                ):
                    return regions
                prompt = _prompt(disputes)
                schema = _response_schema(disputes)
                try:
                    primary_result = self.resolve_image(sheet_path, prompt, schema)
                except OpenRouterError:
                    return regions
                primary = _validated_response(primary_result, disputes)
                if primary is None:
                    return regions

                verifier = None
                verifier_metadata = None
                if self.verifier_image is not None:
                    try:
                        verifier_result = self.verifier_image(
                            sheet_path,
                            _verifier_prompt(disputes),
                            schema,
                        )
                    except OpenRouterError:
                        return regions
                    verifier = _validated_response(verifier_result, disputes)
                    if verifier is None:
                        return regions
                    verifier_metadata = _metadata(verifier_result)
        finally:
            page.close()

        updated = copy.deepcopy(regions)
        primary_metadata = _metadata(primary_result)
        for dispute in disputes:
            primary_decision = primary[dispute.id]
            verifier_decision = verifier[dispute.id] if verifier is not None else None
            _apply_decision(
                updated,
                dispute,
                primary_decision,
                verifier_decision,
                primary_metadata,
                verifier_metadata,
                self.eligible_local_models,
            )
        return updated


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _collect_disputes(regions: list[TextRegion]) -> list[_Dispute] | None:
    disputes = []
    seen_ids = set()
    for region_index, region in enumerate(regions):
        if region.resolution in DISPUTE_RESOLUTIONS and _text_dispute(region):
            item = _region_dispute(region, region_index)
            if item is None or item.id in seen_ids:
                return None
            disputes.append(item)
            seen_ids.add(item.id)

        cells = (region.structure or {}).get("cells", [])
        if not isinstance(cells, list):
            return None
        for cell_index, cell in enumerate(cells):
            if (
                not isinstance(cell, dict)
                or cell.get("resolution") not in DISPUTE_RESOLUTIONS
            ):
                continue
            item = _cell_dispute(region, region_index, cell, cell_index)
            if item is None or item.id in seen_ids:
                return None
            disputes.append(item)
            seen_ids.add(item.id)
    return disputes


def _text_dispute(region: TextRegion) -> bool:
    role = str((region.structure or {}).get("role", "")).casefold()
    return region.kind.casefold() not in NON_TEXT_DISPUTE_KINDS and (
        role not in NON_TEXT_DISPUTE_ROLES
    )


def _region_dispute(region: TextRegion, region_index: int) -> _Dispute | None:
    item_id = f"region:{region.id}"
    readings = [
        _Reading(
            id=f"{item_id}:candidate:0",
            text=region.text,
            confidence=region.confidence,
            provider=region.provider,
            provenance=copy.deepcopy(region.text_provenance),
        )
    ]
    for index, alternative in enumerate(region.alternatives, start=1):
        if alternative.decision_state not in ALTERNATIVE_DECISION_STATES:
            return None
        if alternative.decision_state != "pending":
            continue
        readings.append(
            _Reading(
                id=f"{item_id}:candidate:{index}",
                text=alternative.text,
                confidence=alternative.confidence,
                provider=alternative.provider,
                provenance=copy.deepcopy(alternative.text_provenance),
            )
        )
    readings = [reading for reading in readings if reading.text]
    if region.resolution == "conflicting" and not readings:
        return None
    return _Dispute(
        item_id,
        region.resolution,
        copy.deepcopy(region.bounding_box),
        tuple(readings),
        region_index,
    )


def _cell_dispute(
    region: TextRegion,
    region_index: int,
    cell: dict[str, Any],
    cell_index: int,
) -> _Dispute | None:
    cell_id = cell.get("id")
    box = _dict_box(cell.get("bbox"))
    alternatives = cell.get("alternatives", [])
    if not isinstance(cell_id, str) or not cell_id or box is None:
        return None
    if not isinstance(alternatives, list) or any(
        not isinstance(alternative, dict) for alternative in alternatives
    ):
        return None
    item_id = f"cell:{region.id}:{cell_id}"
    readings = []
    text = cell.get("text")
    if isinstance(text, str) and text:
        provider = str(cell.get("source") or region.provider)
        provenance = _cell_provenance(cell, cell_id)
        evidence = _cell_evidence(cell, provider, provenance)
        if evidence is None:
            return None
        readings.append(
            _Reading(
                id=f"{item_id}:candidate:0",
                text=text,
                confidence=_confidence(cell.get("confidence")),
                provider=provider,
                provenance=provenance,
                cell_evidence=evidence,
            )
        )
    for index, alternative in enumerate(alternatives, start=1):
        decision_state = alternative.get("decision_state", "pending")
        if decision_state not in ALTERNATIVE_DECISION_STATES:
            return None
        if decision_state != "pending":
            continue
        alternative_text = alternative.get("text")
        provider = alternative.get("provider")
        if not isinstance(alternative_text, str) or not alternative_text:
            return None
        provider = str(provider or "unknown")
        provenance = copy.deepcopy(alternative.get("text_provenance"))
        evidence = _cell_evidence(alternative, provider, provenance)
        if evidence is None:
            return None
        readings.append(
            _Reading(
                id=f"{item_id}:candidate:{index}",
                text=alternative_text,
                confidence=_confidence(alternative.get("confidence")),
                provider=provider,
                provenance=provenance,
                cell_evidence=evidence,
            )
        )
    if cell.get("resolution") == "conflicting" and not readings:
        return None
    return _Dispute(
        item_id,
        str(cell["resolution"]),
        box,
        tuple(readings),
        region_index,
        cell_index,
    )


def _cell_provenance(
    cell: dict[str, Any],
    cell_id: str,
) -> dict[str, Any] | None:
    provenance = cell.get("text_provenance")
    if provenance is not None:
        return copy.deepcopy(provenance) if isinstance(provenance, dict) else None
    evidence_ids = cell.get("evidence_ids", [])
    if not isinstance(evidence_ids, list) or not all(
        isinstance(value, str) for value in evidence_ids
    ):
        return None
    return {"cell_id": cell_id, "evidence_ids": copy.deepcopy(evidence_ids)}


def _cell_evidence(
    candidate: dict[str, Any],
    provider: str,
    provenance: dict[str, Any] | None,
) -> _CellEvidence | None:
    if candidate.get("text_provenance") is not None and not isinstance(
        candidate["text_provenance"], dict
    ):
        return None
    evidence_ids = candidate.get("evidence_ids")
    if evidence_ids is None and isinstance(provenance, dict):
        evidence_ids = provenance.get("evidence_ids", [])
    if not isinstance(evidence_ids, list) or not all(
        isinstance(value, str) for value in evidence_ids
    ):
        return None

    supporters = candidate.get("supporters")
    if supporters is None:
        supporters = (
            [{"provider": provider, "evidence_ids": copy.deepcopy(evidence_ids)}]
            if evidence_ids
            else []
        )
    if not isinstance(supporters, list) or not all(
        isinstance(value, dict) for value in supporters
    ):
        return None
    decision = candidate.get("decision")
    resolution = candidate.get("resolution")
    if decision is not None and not isinstance(decision, str):
        return None
    if resolution is not None and not isinstance(resolution, str):
        return None
    return _CellEvidence(
        tuple(evidence_ids),
        tuple(copy.deepcopy(supporters)),
        decision,
        resolution,
    )


def _dict_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, dict) or set(value) != {"left", "top", "right", "bottom"}:
        return None
    coordinates = [value[key] for key in ("left", "top", "right", "bottom")]
    if any(
        not isinstance(coordinate, int) or isinstance(coordinate, bool)
        for coordinate in coordinates
    ):
        return None
    return BoundingBox(*coordinates)


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0 <= value <= 1
    ):
        return float(value)
    return None


def _padded_box(
    box: BoundingBox,
    page_size: tuple[int, int],
    padding: int,
) -> BoundingBox | None:
    width, height = page_size
    coordinates = (box.left, box.top, box.right, box.bottom)
    if (
        any(
            not isinstance(coordinate, int) or isinstance(coordinate, bool)
            for coordinate in coordinates
        )
        or box.left < 0
        or box.top < 0
        or box.right > width
        or box.bottom > height
        or box.left >= box.right
        or box.top >= box.bottom
    ):
        return None
    return BoundingBox(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(width, box.right + padding),
        min(height, box.bottom + padding),
    )


def _write_contact_sheet(
    page: Image.Image,
    disputes: list[_Dispute],
    crop_boxes: list[BoundingBox],
    destination: Path,
    max_width: int,
    max_height: int,
) -> bool:
    header_height = 20
    margin = 4
    label_draw = ImageDraw.Draw(page)
    label_width = max(label_draw.textbbox((0, 0), item.id)[2] for item in disputes)
    width = (
        max(
            label_width,
            *(box.right - box.left for box in crop_boxes),
        )
        + margin * 2
    )
    height = (
        sum(header_height + box.bottom - box.top + margin for box in crop_boxes)
        + margin
    )
    if width > max_width or height > max_height:
        return False

    crops = [page.crop(_box_tuple(box)) for box in crop_boxes]
    try:
        sheet = Image.new("RGB", (width, height), "white")
        try:
            draw = ImageDraw.Draw(sheet)
            y = margin
            for dispute, crop in zip(disputes, crops, strict=True):
                draw.text((margin, y), dispute.id, fill="black")
                y += header_height
                sheet.paste(crop, (margin, y))
                y += crop.height + margin
            sheet.save(destination, format="PNG")
        finally:
            sheet.close()
    finally:
        for crop in crops:
            crop.close()
    return True


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.left, box.top, box.right, box.bottom


def _prompt(disputes: list[_Dispute]) -> str:
    lines = [
        "Resolve each labeled OCR dispute from the contact sheet.",
        "Select only a listed candidate for conflicting items.",
        "For unreadable items, select a listed candidate, transcribe the visible literal, or abstain.",
        "Return exactly one decision for every item and no extra items.",
    ]
    for dispute in disputes:
        candidates = "; ".join(
            f"{reading.id}={reading.text!r}" for reading in dispute.readings
        )
        lines.append(
            f"{dispute.id} [{dispute.resolution}]: {candidates or 'no candidates'}"
        )
    return "\n".join(lines)


def _verifier_prompt(disputes: list[_Dispute]) -> str:
    return (
        _prompt(disputes)
        + "\nRead independently without assuming another model's decisions."
    )


def _response_schema(disputes: list[_Dispute]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(disputes),
                "maxItems": len(disputes),
                "items": {"oneOf": [_decision_schema(item) for item in disputes]},
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _decision_schema(dispute: _Dispute) -> dict[str, Any]:
    actions = ["select", "abstain"]
    if dispute.resolution == "unreadable":
        actions.append("transcribe")
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": [dispute.id]},
            "action": {"type": "string", "enum": actions},
            "candidate_id": {
                "type": ["string", "null"],
                "enum": [reading.id for reading in dispute.readings] + [None],
            },
            "text": {"type": ["string", "null"], "maxLength": 512},
        },
        "required": ["id", "action", "candidate_id", "text"],
        "additionalProperties": False,
    }


def _validated_response(
    result: Any,
    disputes: list[_Dispute],
) -> dict[str, dict[str, Any]] | None:
    content = getattr(result, "content", None)
    if not isinstance(content, dict) or set(content) != {"decisions"}:
        return None
    decisions = content["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(disputes):
        return None
    by_id = {item.id: item for item in disputes}
    validated = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
            "id",
            "action",
            "candidate_id",
            "text",
        }:
            return None
        item_id = decision["id"]
        if not isinstance(item_id, str) or item_id not in by_id or item_id in validated:
            return None
        if not _valid_decision(decision, by_id[item_id]):
            return None
        validated[item_id] = decision
    return validated if set(validated) == set(by_id) else None


def _valid_decision(decision: dict[str, Any], dispute: _Dispute) -> bool:
    action = decision["action"]
    candidate_id = decision["candidate_id"]
    text = decision["text"]
    if action == "abstain":
        return candidate_id is None and text is None
    if action == "select":
        return (
            isinstance(candidate_id, str)
            and candidate_id in {reading.id for reading in dispute.readings}
            and text is None
        )
    if action != "transcribe" or dispute.resolution != "unreadable":
        return False
    if candidate_id is not None or not isinstance(text, str) or not text.strip():
        return False
    if len(text) > 512 or any(
        unicodedata.category(character) == "Cc" and character not in "\n\t"
        for character in text
    ):
        return False
    normalized = _normalized_text(text)
    return all(
        normalized != _normalized_text(reading.text) for reading in dispute.readings
    )


def _apply_decision(
    regions: list[TextRegion],
    dispute: _Dispute,
    primary: dict[str, Any],
    verifier: dict[str, Any] | None,
    primary_metadata: dict[str, str | None],
    verifier_metadata: dict[str, str | None] | None,
    eligible_local_models: frozenset[str],
) -> None:
    if (
        dispute.resolution == "conflicting"
        and verifier is not None
        and verifier_metadata is not None
        and primary["action"] == "select"
        and _decisions_agree(primary, verifier)
        and _eligible_independent_models(
            primary_metadata,
            verifier_metadata,
            eligible_local_models,
        )
    ):
        reading = next(
            reading
            for reading in dispute.readings
            if reading.id == primary["candidate_id"]
        )
        _select_reading(
            regions,
            dispute,
            reading,
            _decision_provenance(
                "selected_existing_candidate",
                primary,
                verifier,
                primary_metadata,
                verifier_metadata,
            ),
        )
        return

    review = _decision_provenance(
        "review_required",
        primary,
        verifier,
        primary_metadata,
        verifier_metadata,
    )
    _record_dispute_decision(regions, dispute, review)
    _append_transcription_evidence(
        regions,
        dispute,
        primary,
        verifier,
        primary_metadata,
        verifier_metadata,
    )


def _decisions_agree(primary: dict[str, Any], verifier: dict[str, Any]) -> bool:
    if primary["action"] != verifier["action"]:
        return False
    if primary["action"] == "select":
        return primary["candidate_id"] == verifier["candidate_id"]
    if primary["action"] == "transcribe":
        return _normalized_text(primary["text"]) == _normalized_text(verifier["text"])
    return True


def _select_reading(
    regions: list[TextRegion],
    dispute: _Dispute,
    selected: _Reading,
    provenance: dict[str, Any],
) -> None:
    incumbent_id = f"{dispute.id}:candidate:0"
    correction_selected = selected.id != incumbent_id
    if dispute.cell_index is None:
        region = regions[dispute.region_index]
        historical = [
            copy.deepcopy(alternative)
            for alternative in region.alternatives
            if alternative.decision_state != "pending"
        ]
        alternatives = [
            _text_alternative(
                reading,
                "superseded" if reading.id == incumbent_id else "rejected",
            )
            for reading in dispute.readings
            if reading.id != selected.id
        ]
        region.text = selected.text
        region.confidence = selected.confidence
        region.provider = selected.provider
        region.text_provenance = _accepted_provenance(
            selected,
            provenance,
            correction_selected,
        )
        region.alternatives = [*historical, *alternatives]
        region.resolution = "resolved"
        _record_region_decision(region, provenance)
        return
    cell = _target_cell(regions, dispute)
    historical = [
        copy.deepcopy(alternative)
        for alternative in cell.get("alternatives", [])
        if alternative.get("decision_state", "pending") != "pending"
    ]
    alternatives = [
        _cell_alternative(
            reading,
            "superseded" if reading.id == incumbent_id else "rejected",
        )
        for reading in dispute.readings
        if reading.id != selected.id
    ]
    cell["text"] = selected.text
    cell["confidence"] = selected.confidence
    cell["source"] = selected.provider
    cell["alternatives"] = [*historical, *alternatives]
    evidence = selected.cell_evidence
    assert evidence is not None
    cell["evidence_ids"] = list(evidence.evidence_ids)
    cell["supporters"] = copy.deepcopy(list(evidence.supporters))
    cell["decision"] = "independent_dispute_agreement"
    cell["text_provenance"] = _accepted_provenance(
        selected,
        provenance,
        correction_selected,
    )
    cell["resolution"] = "resolved"
    cell["dispute_resolution"] = provenance


def _append_transcription_evidence(
    regions: list[TextRegion],
    dispute: _Dispute,
    primary: dict[str, Any],
    verifier: dict[str, Any] | None,
    primary_metadata: dict[str, str | None],
    verifier_metadata: dict[str, str | None] | None,
) -> None:
    if primary["action"] != "transcribe":
        if verifier is not None and verifier["action"] == "transcribe":
            _append_transcription(
                regions,
                dispute,
                verifier["text"],
                _decision_provenance(
                    "visual_transcription_review",
                    primary,
                    verifier,
                    primary_metadata,
                    verifier_metadata,
                ),
                verifier_metadata or {},
            )
        return

    agreeing_transcription = (
        verifier is not None
        and verifier["action"] == "transcribe"
        and _normalized_text(primary["text"]) == _normalized_text(verifier["text"])
    )
    primary_provenance = _decision_provenance(
        "visual_transcription_review",
        primary,
        verifier if agreeing_transcription else None,
        primary_metadata,
        verifier_metadata if agreeing_transcription else None,
    )
    _append_transcription(
        regions,
        dispute,
        primary["text"],
        primary_provenance,
        primary_metadata,
    )
    if (
        verifier is not None
        and verifier["action"] == "transcribe"
        and not agreeing_transcription
    ):
        _append_transcription(
            regions,
            dispute,
            verifier["text"],
            _decision_provenance(
                "visual_transcription_review",
                primary,
                verifier,
                primary_metadata,
                verifier_metadata,
            ),
            verifier_metadata or {},
        )


def _append_transcription(
    regions: list[TextRegion],
    dispute: _Dispute,
    text: str,
    provenance: dict[str, Any],
    metadata: dict[str, str | None],
) -> None:
    provider = _provider_label(metadata)
    if dispute.cell_index is None:
        region = regions[dispute.region_index]
        region.alternatives.append(
            TextAlternative(text, None, provider, provenance, "pending")
        )
        return
    cell = _target_cell(regions, dispute)
    alternatives = cell.setdefault("alternatives", [])
    alternatives.append(
        {
            "text": text,
            "confidence": None,
            "provider": provider,
            "text_provenance": provenance,
            "decision_state": "pending",
        }
    )


def _record_dispute_decision(
    regions: list[TextRegion],
    dispute: _Dispute,
    provenance: dict[str, Any],
) -> None:
    if dispute.cell_index is None:
        _record_region_decision(regions[dispute.region_index], provenance)
        return
    _target_cell(regions, dispute)["dispute_resolution"] = provenance


def _target_cell(regions: list[TextRegion], dispute: _Dispute) -> dict[str, Any]:
    assert dispute.cell_index is not None
    cells = (regions[dispute.region_index].structure or {})["cells"]
    return cells[dispute.cell_index]


def _text_alternative(
    reading: _Reading,
    decision_state: AlternativeDecision,
) -> TextAlternative:
    return TextAlternative(
        reading.text,
        reading.confidence,
        reading.provider,
        copy.deepcopy(reading.provenance),
        decision_state,
    )


def _cell_alternative(
    reading: _Reading,
    decision_state: AlternativeDecision,
) -> dict[str, Any]:
    alternative = {
        "text": reading.text,
        "confidence": reading.confidence,
        "provider": reading.provider,
        "text_provenance": copy.deepcopy(reading.provenance),
        "decision_state": decision_state,
    }
    if reading.cell_evidence is None:
        return alternative
    alternative.update(
        {
            "evidence_ids": list(reading.cell_evidence.evidence_ids),
            "supporters": copy.deepcopy(list(reading.cell_evidence.supporters)),
            "decision": reading.cell_evidence.decision,
            "resolution": reading.cell_evidence.resolution,
        }
    )
    return alternative


def _accepted_provenance(
    reading: _Reading,
    resolution: dict[str, Any],
    correction_selected: bool,
) -> dict[str, Any] | None:
    provenance = copy.deepcopy(reading.provenance)
    if not correction_selected:
        return provenance
    provenance = provenance or {}
    provenance["accepted_correction"] = {
        "decision_state": "accepted",
        "provider": reading.provider,
        "resolution": copy.deepcopy(resolution),
    }
    return provenance


def _record_region_decision(
    region: TextRegion,
    provenance: dict[str, Any],
) -> None:
    structure = dict(region.structure or {})
    structure["dispute_resolution"] = provenance
    region.structure = structure


def _decision_provenance(
    decision: str,
    primary_decision: dict[str, Any],
    verifier_decision: dict[str, Any] | None,
    primary: dict[str, str | None],
    verifier: dict[str, str | None] | None,
) -> dict[str, Any]:
    return {
        "method": "bulk_visual_dispute_resolution",
        "decision": decision,
        "primary": copy.deepcopy(primary),
        "verifier": copy.deepcopy(verifier),
        "primary_decision": _decision_metadata(primary_decision),
        "verifier_decision": _decision_metadata(verifier_decision),
    }


def _decision_metadata(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "action": decision["action"],
        "candidate_id": decision["candidate_id"],
    }


def _metadata(result: Any) -> dict[str, str | None]:
    model = getattr(result, "model", None)
    provider = getattr(result, "provider", None)
    return {
        "model": model if isinstance(model, str) and model else None,
        "provider": provider if isinstance(provider, str) and provider else None,
    }


def _provider_label(metadata: dict[str, str | None]) -> str:
    return metadata["provider"] or metadata["model"] or DisputeResolutionStage.name


def _eligible_independent_models(
    primary: dict[str, str | None],
    verifier: dict[str, str | None],
    eligible_local_models: frozenset[str],
) -> bool:
    primary_model = primary["model"]
    verifier_model = verifier["model"]
    return bool(
        primary_model
        and verifier_model
        and primary_model != verifier_model
        and primary_model in eligible_local_models
        and verifier_model in eligible_local_models
    )


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())
