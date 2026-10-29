"""Classify a document and extract its enrollment values with evidence.

Two parallel LLM calls per document read the arranged OCR lines (every region prefixed by a
short id) and the downscaled page images; one replies with the fields (`FieldsReply`), the
other with the addresses and other facts (`LocationsEntitiesReply`). The LLM cites region
ids, never coordinates: boxes come from the cited regions, and every value
is checked against the cited text before it gets a confidence (LMDX-style grounding).
"""

from __future__ import annotations

import base64
import difflib
import io
import json
import math
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.request import Request

from PIL import Image
from pydantic import ValidationError

from ..openrouter import QWEN_37_FLASH_MODEL, OpenRouterError, Transport, _call_openrouter
from ..page_arrangement import arrange_page, block_text
from .schema import (
    DOCUMENT_TYPES,
    FIELDS,
    FIELDS_BY_KEY,
    Box,
    ClassificationReply,
    Components,
    DocumentExtraction,
    Evidence,
    FieldsReply,
    LocationsEntitiesReply,
    LocationsReply,
    Value,
)
from .validate import CHECKS, split_address

PROMPT_VERSION = 5
MODEL = QWEN_37_FLASH_MODEL
IMAGE_LONG_SIDE = 1024  # measured: as accurate as 1600 px with 23% fewer tokens; text-only misread ticks
MAX_IMAGED_PAGES = 4
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"
JEV_SIDE = (280, 1568)  # accepted image side range
GROUNDED = 0.9
REPAIRED_CEILING = 0.75  # a value we had to repair is always shown to a reviewer
LLM_DOUBT = 0.5  # below this token probability the LLM was unsure of what it wrote
# Measured 2026-10-29: a reply part takes 5-14 s, but about 1 call in 6 stalled at the
# provider (38 s for a 48-token reply). ponytail: one fixed threshold; a long document's
# slow reply gets a needless second request (~$0.0003), scale it by pages if that matters.
HEDGE_SECONDS = 15

ROLE_LABELS = {
    "service_location": "Service location",
    "billing": "Billing address",
    "mailing": "Mailing address",
    "insured": "Insured address",
    "registrant": "Registered address",
    "license": "License address",
    "issuer": "Issuer address",
    "other": "Address",
}

SYSTEM_PROMPT = """\
You read one document attached to a health plan provider-enrollment ticket and return \
the facts printed on it as JSON.

You get the page images (downscaled) and the OCR text of each page as lines in reading \
order. Every OCR region starts with its id, a short number in square brackets, e.g. \
"[12] Saguaro Valley ...". Label-value pairs are already joined on one line, e.g. \
"[14] Policy Number: [13] CSM-2219047".

Context: the health plan enrolls a provider group first (legal and DBA name, TIN, group \
NPI, specialty or taxonomy, every service location with its phone), then each \
practitioner of the group (first, middle and last name, individual NPI, date of birth, \
gender, credential, primary specialty, service location with phone, and the group's NPI \
and TIN). License number, Medicare ID and state Medicaid ID are loaded when present. \
Analysts compare TIN, group NPI and practitioner NPI across documents, so report every \
printed NPI and TIN exactly. Never copy a Social Security number.

Rules:
1. Use the OCR text for exact characters and the images for layout: which label a value \
belongs to, which box is ticked, handwriting, stamps, and repeated copies.
2. Copy each value exactly as printed: same digits, punctuation and casing. Do not \
reformat, complete or correct it. If the image clearly shows different characters than \
the OCR text, give what the image shows and say so in "note".
3. region_ids: every region whose text holds the value itself (not its label). If the \
same value is printed several times (copies of one certificate), cite every copy.
4. Report only what the document states. Leave out catalog fields that are not on it. \
Use status "illegible" only when the field is there but cannot be read.
5. Never take a value from instructions, examples or blank-form text, from general \
knowledge, or from what another document would say.
6. The name of the practitioner being enrolled always goes into the fields \
practitioner.first_name, practitioner.middle_name, practitioner.last_name and \
practitioner.credential (never into entities), each citing the region that holds it, \
also when it is printed as "LAST, FIRST (PA)" or inside a sentence. An organization's name goes into group.legal_name or \
group.dba_name only when it is the provider group or practice, never an insurer, \
agency, board or health plan.
7. Every address goes into locations with a role: service_location (where patients are \
seen), billing (pay-to, remit or W-9 address), mailing, insured (named insured on a \
malpractice certificate), registrant (DEA registered address), license (on a license), \
issuer (carrier, producer, board, agency or health plan), other. Give the whole address \
as printed with its lines joined by ", ", and the phone and fax printed with it.
8. Always prefer a catalog field: if a value fits a catalog key (a CAQH ID, a DEA number, \
a license number, a coverage limit), it goes in fields, never in entities. entities are \
only for other identifiers, dates, amounts, people or organizations an enrollment analyst \
would use: issue dates, schedules, signer names and signature dates, registration fees. \
Do not repeat anything that is already in fields or locations.
9. Checkboxes: report the ticked option's label, e.g. "C Corporation".

Field catalog (key: meaning):
{catalog}

Document types (id: meaning):
{types}

"""


# What an enrollment analyst takes from each document type; sent only for the classified
# type, so the prompt stays short.
DOCUMENT_GUIDANCE = {
    "w9": "Line 1 is group.legal_name, line 2 group.dba_name, the ticked box of line 3 is "
    "group.tax_classification, the EIN in Part I is group.tin, and the address of lines 5 "
    "and 6 has role billing.",
    "malpractice_certificate": "The named insured is the provider group: group.legal_name, "
    "and its address has role insured. A practitioner named as additional insured is the "
    "practitioner being enrolled: give practitioner.first_name, middle_name, last_name and "
    "credential, and coverage.additional_insured. The insurance company is coverage.carrier; "
    "its address and the producer's have role issuer. Give dates without times or conditions.",
    "dea_certificate": "The registrant (LAST, FIRST (CREDENTIAL)) is the practitioner: give "
    "practitioner.first_name, last_name and credential; the registered address has role registrant. Give practitioner.dea_number, dea_expiration "
    "and dea_state. The certificate is often printed several times; cite every copy.",
    "state_license": "The licensee is the practitioner: name parts, credential, "
    "practitioner.license_number, license_state and license_expiration.",
    "practitioner_application": "Give every catalog field present. Each practice location "
    "with its phone is a service_location; the group section gives group.legal_name, "
    "group.tin and group.npi.",
    "group_application": "Give the group fields. Every practice location with its phone is a "
    "service_location; a billing or pay-to address has role billing.",
    "npi_confirmation": "A Type 1 NPI is practitioner.npi; a Type 2 NPI is group.npi.",
}


def classify_document(
    first_page: Path,
    text_layer: str = "",
    *,
    llm_transport: Transport,
    jev_transport: Transport | None = None,
    allow_jev: bool = False,
) -> dict[str, Any]:
    """Document type of a file from its first page, before any OCR is spent on it."""
    if allow_jev and jev_transport is not None:
        started = time.perf_counter()
        criteria = {key: f"{label}: {meaning}" for key, (label, meaning, _) in DOCUMENT_TYPES.items()}
        answers, cost = _ask_jev(
            [_jev_image(Image.open(first_page)), "The first page of a provider enrollment attachment."],
            {"document_type": {"type": "choice", "instructions": "Which kind of document is this page?", "criteria": criteria}},
            jev_transport,
        )
        answer = answers["document_type"]
        probability = float(answer.get("probabilities", {}).get(answer["choice"], 0))
        return {
            "document_type": answer["choice"],
            "label": DOCUMENT_TYPES[answer["choice"]][0],
            "method": "verifier",
            "probability": probability,
            "confidence": _band(probability),
            "evidence": "",
            "llm": {"cost": cost, "latency_ms": (time.perf_counter() - started) * 1000},
        }
    prompt = (
        "Classify this provider-enrollment attachment from its first page.\n\nTypes:\n"
        + _types_text()
        + "\n\nReply with one JSON object matching this JSON schema:\n"
        + json.dumps(ClassificationReply.model_json_schema())
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if text_layer.strip():
        content.append({"type": "text", "text": "Text layer of the page:\n" + text_layer[:2000]})
    content.append(_image_part(Image.open(first_page), 1024))
    reply, llm, tokens = _ask_llm([{"role": "user", "content": content}], ClassificationReply, llm_transport, 1024)
    probability = _token_probabilities(tokens, [reply.document_type])[0]
    return {
        "document_type": reply.document_type,
        "label": DOCUMENT_TYPES[reply.document_type][0],
        "method": "llm",
        "probability": probability,
        "confidence": _band(probability) if probability is not None else reply.confidence,
        "evidence": reply.evidence,
        "llm": llm,
    }


def extract_document(
    document_id: str,
    result: dict[str, Any],
    page_paths: Sequence[Path],
    *,
    document_type: str | None = None,
    llm_transport: Transport,
    jev_transport: Transport | None = None,
    allow_jev: bool = False,
    hints: Sequence[str] = (),
    entities: bool = True,
) -> DocumentExtraction:
    """`entities=False` skips the other facts (dates, fees, signers): the reader's client
    format has no place for them, and they are the longest part of the reply."""
    pages = result.get("pages") or []
    regions = {
        region["id"]: (page["page_number"], region)
        for page in pages
        for region in page.get("regions") or []
        if region.get("id")
    }
    if not regions:
        # Without OCR regions the LLM could only invent ids; the case flags the read.
        raise ValueError("OCR returned no text regions for this document")
    messages, aliases = extraction_messages(pages, page_paths, document_type, hints)
    # Two calls over the same input, in parallel, each asked for its part only: about half the
    # completion tokens each. A failure in either raises, as a single call did.
    other = LocationsEntitiesReply if entities else LocationsReply
    with ThreadPoolExecutor(max_workers=2) as pool:
        fields_future = pool.submit(_ask_llm, _part_messages(messages, FieldsReply), FieldsReply, llm_transport, 8192)
        other_future = pool.submit(_ask_llm, _part_messages(messages, other), other, llm_transport, 8192)
        fields_reply, fields_llm, fields_tokens = fields_future.result()
        other_reply, other_llm, other_tokens = other_future.result()
    _resolve_aliases(fields_reply, aliases)
    _resolve_aliases(other_reply, aliases)

    field_values = [_field_value(document_id, reading, regions) for reading in fields_reply.fields if reading.status != "absent"]
    field_written = [reading.value for reading in fields_reply.fields if reading.status != "absent"]
    for value, probability in zip(field_values, _token_probabilities(fields_tokens, field_written), strict=True):
        value.components.llm = probability

    location_values = [_location_value(document_id, index, reading, regions) for index, reading in enumerate(other_reply.locations, 1)]
    entity_readings = getattr(other_reply, "entities", [])
    entity_values = [_entity_value(document_id, index, reading, regions) for index, reading in enumerate(entity_readings, 1)]
    other_values = location_values + entity_values
    other_written = [reading.address for reading in other_reply.locations] + [reading.value for reading in entity_readings]
    for value, probability in zip(other_values, _token_probabilities(other_tokens, other_written), strict=True):
        value.components.llm = probability

    # An entity that just repeats a field or location (the two calls can't see each other's
    # reply) is noise for the reviewer.
    taken = {_alnum(value.value) for value in field_values + location_values if value.value}
    values = field_values + location_values + [value for value in entity_values if _alnum(value.value or "") not in taken]

    llm = _merge_llm(fields_llm, other_llm)
    problems: list[str] = []
    if allow_jev and jev_transport is not None:
        try:
            _verify(values, page_paths, jev_transport, llm)
        except (OpenRouterError, OSError, ValueError, KeyError) as error:
            problems.append(f"verifier unavailable: {error}")
    for value in values:
        value.confidence = confidence(value.components)
    return DocumentExtraction(
        document_type=fields_reply.document_type,
        document_type_reason=fields_reply.document_type_reason,
        values=values,
        llm=llm,
        problems=problems,
    )


def extraction_messages(
    pages: Sequence[dict[str, Any]],
    page_paths: Sequence[Path],
    document_type: str | None,
    hints: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The LLM messages, plus the short alias assigned to every OCR region id in them (the
    reply cites aliases; extract_document maps them back before grounding)."""
    system = SYSTEM_PROMPT.format(
        catalog="\n".join(f"- {spec.key}: {spec.description}" for spec in FIELDS),
        types=_types_text(),
    )
    if hints:
        system += "\nCorrections reviewers made on earlier documents:\n" + "\n".join(f"- {hint}" for hint in hints)
    aliases: dict[str, str] = {}
    text = "\n\n".join(f"=== Page {page['page_number']} ===\n{page_text(page, aliases)}" for page in pages)
    if document_type in DOCUMENT_TYPES:
        known = (
            f"The classifier says this is: {DOCUMENT_TYPES[document_type][0]}. "
            f"{DOCUMENT_GUIDANCE.get(document_type, '')}\n\n"
        )
    elif document_type is None:
        # Unclassified (the reader skips the classify call): give every type's guidance so
        # accuracy doesn't depend on knowing the type up front.
        known = "Guidance per document type, use whichever matches:\n" + "\n".join(
            f"- {key}: {guidance}" for key, guidance in DOCUMENT_GUIDANCE.items()
        ) + "\n\n"
    else:
        known = ""
    content: list[dict[str, Any]] = [{"type": "text", "text": f"{known}OCR text:\n{text}"}]
    content += [_image_part(Image.open(path), IMAGE_LONG_SIDE) for path in page_paths[:MAX_IMAGED_PAGES]]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}], aliases


_REGION_ID = re.compile(r"\[([^\[\]]+)\]")


def page_text(page: dict[str, Any], aliases: dict[str, str]) -> str:
    """OCR text of one page with every region's real id swapped for a short alias.

    `aliases` maps alias -> real region id; it is extended in place and shared across every
    page of the document, so a region keeps one alias and the prompt stays short ("[12]"
    instead of "[p1-falcon-12]").
    """
    arrangement = arrange_page(page)
    known_ids = {region["id"] for region in page.get("regions") or []}
    assigned = {real: alias for alias, real in aliases.items()}

    def alias_for(match: re.Match) -> str:
        real_id = match.group(1)
        if real_id not in known_ids:
            return match.group(0)  # not a region id, e.g. a checkbox mark "[x]"
        alias = assigned.get(real_id)
        if alias is None:
            alias = str(len(aliases) + 1)
            aliases[alias] = real_id
            assigned[real_id] = alias
        return f"[{alias}]"

    return "\n".join(
        _REGION_ID.sub(alias_for, line)
        for block in arrangement["blocks"]
        if (line := block_text(page, block, with_ids=True, arrangement=arrangement))
    )


def confidence(components: Components) -> float | None:
    """Gate score: the weaker of the OCR and verifier readings times how well the value is
    grounded in the cited text; a doubtful LLM reading caps it below the review threshold."""
    signals = [score for score in (components.ocr, components.verifier) if score is not None]
    if not signals or components.grounding is None:
        return None
    score = min(signals) * components.grounding
    # The LLM's token probability is a weak signal on its own (a name's rarest token is often
    # below 0.8), so it only flags a reading the LLM itself was unsure of, e.g. an invented one.
    if components.llm is not None and components.llm < LLM_DOUBT:
        score = min(score, REPAIRED_CEILING)
    if components.validator == "fail":
        score = min(score, 0.3)
    if components.repaired:
        score = min(score, REPAIRED_CEILING)
    return round(score, 4)


def _field_value(document_id: str, reading, regions) -> Value:
    spec = FIELDS_BY_KEY[reading.key]
    value = Value(
        id=spec.key, key=spec.key, label=spec.label, record=spec.record, category=spec.category,
        raw_value=reading.value, value=reading.value, note=reading.note,
    )
    _ground(value, document_id, reading.value, reading.region_ids, regions)
    if reading.status == "illegible":
        value.problems.append("printed but illegible")
        value.components.grounding = 0.0
    elif spec.check and reading.value:
        _validate(value, spec.check, reading.value)
    return value


def _location_value(document_id: str, index: int, reading, regions) -> Value:
    value = Value(
        id=f"location-{index}", key="location", label=ROLE_LABELS[reading.role],
        record="other", category="required" if reading.role == "service_location" else "other",
        role=reading.role, raw_value=reading.address,
    )
    _ground(value, document_id, reading.address, reading.address_region_ids, regions)
    parts, problem = split_address(reading.address)
    problems = [problem] if problem else []
    for name, raw, ids in (("phone", reading.phone, reading.phone_region_ids), ("fax", reading.fax, reading.fax_region_ids)):
        if not raw:
            continue
        evidence, grounding, ocr, cited_problems = _cite(document_id, raw, ids, regions)
        value.evidence += evidence
        value.components.grounding = min(value.components.grounding or 0, grounding)
        value.components.ocr = _min(value.components.ocr, ocr)
        problems += [f"{name}: {item}" for item in cited_problems]
        number, phone_problem = CHECKS["phone"](raw)
        if parts is not None:
            parts[name] = number or raw
        if phone_problem:
            problems.append(f"{name}: {phone_problem}")
    if parts is not None:
        parts["primary"] = "Yes" if reading.primary else "No" if reading.primary is False else ""
        line = ", ".join(part for part in (parts["line1"], parts["line2"], parts["city"]) if part)
        value.value = f"{line}, {parts['state']} {parts['zip']}".strip(", ")
    value.parts = parts
    value.problems += problems
    value.components.validator = "fail" if problems else "pass"
    return value


def _entity_value(document_id: str, index: int, reading, regions) -> Value:
    value = Value(
        id=f"entity-{index}", key="entity", label=reading.label, record="other",
        category="other", role=reading.type, raw_value=reading.value, value=reading.value,
    )
    _ground(value, document_id, reading.value, reading.region_ids, regions)
    label = reading.label.casefold()
    check = (
        "dea" if "dea" in label and "number" in label
        else "npi" if "npi" in label
        else "date" if reading.type == "date"
        else "money" if reading.type == "amount"
        else "phone" if reading.type == "phone"
        else None
    )
    if check:
        _validate(value, check, reading.value)
    return value


def _ground(value: Value, document_id: str, raw: str | None, region_ids, regions) -> None:
    evidence, grounding, ocr, problems = _cite(document_id, raw, region_ids, regions)
    value.evidence += evidence
    value.components.grounding = grounding
    value.components.ocr = ocr
    value.problems += problems


def _cite(document_id: str, raw: str | None, region_ids, regions):
    """Evidence boxes, grounding score, OCR confidence and problems for one cited value."""
    cited = [regions[region_id] for region_id in dict.fromkeys(region_ids) if region_id in regions]
    missing = [region_id for region_id in region_ids if region_id not in regions]
    problems = [f"cited regions not on the document: {', '.join(missing)}"] if missing else []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for page_number, region in cited:
        by_page.setdefault(page_number, []).append(region)
    evidence = [
        Evidence(document_id=document_id, page=page_number, region_ids=[region["id"] for region in group], box=_union(group))
        for page_number, group in sorted(by_page.items())
        if any(region.get("bounding_box") for region in group)
    ]
    grounding = _grounding(raw or "", [region.get("text") or "" for _, region in cited])
    if raw and grounding < GROUNDED:
        problems.append("value not found in the cited text")
    return evidence, grounding, _ocr_confidence(raw or "", [region for _, region in cited]), problems


def _validate(value: Value, check: str, raw: str) -> None:
    normalized, problem = CHECKS[check](raw)
    if normalized is not None:
        value.value = normalized
    if problem and normalized is None:
        value.components.validator = "fail"
        value.problems.append(problem)
    else:
        value.components.validator = "pass"
        if problem:  # normalized past a misread, e.g. "$10.000.000" or "26IQP2300X"
            value.components.repaired = True
            value.problems.append(problem)


def _grounding(raw: str, texts: Sequence[str]) -> float:
    """Share of the value's characters found, in order, in the cited regions' text."""
    value, source = _alnum(raw), _alnum(" ".join(texts))
    if not value:
        return 0.0
    if value in source:
        return 1.0
    blocks = difflib.SequenceMatcher(None, value, source, autojunk=False).get_matching_blocks()
    return round(sum(block.size for block in blocks) / len(value), 4)


def _ocr_confidence(raw: str, cited: Sequence[dict[str, Any]]) -> float | None:
    """Calibrated reading score of a value.

    A value printed whole in several regions (the copies of a certificate) is as sure as
    its best copy; a value spread over regions is as sure as the weakest one holding it.
    """
    value = _alnum(raw)
    whole = [
        score
        for region in cited
        if value and value in _alnum(region.get("text") or "")
        and (score := _region_confidence(region)) is not None
    ]
    if whole:
        return round(max(whole), 4)
    tokens = set(re.findall(r"\w+", raw.casefold()))
    holding = [region for region in cited if tokens & set(re.findall(r"\w+", (region.get("text") or "").casefold()))]
    scores = [_region_confidence(region) for region in holding or cited]
    if not scores or any(score is None for score in scores):
        return None
    return round(min(scores), 4)


def _region_confidence(region: dict[str, Any]) -> float | None:
    provenance = region.get("text_provenance") or {}
    if provenance.get("method") == "pdf_text_layer":
        return 1.0  # characters read from the PDF itself
    if isinstance(region.get("confidence"), int | float):
        return float(region["confidence"])
    # An LLM re-read has no calibrated score; it inherits the OCR reading's score only
    # when both readers produced the same text.
    for alternative in region.get("alternatives") or []:
        if _alnum(alternative.get("text") or "") == _alnum(region.get("text") or "") and isinstance(
            alternative.get("confidence"), int | float
        ):
            return float(alternative["confidence"])
    return None


def _verify(values: list[Value], page_paths: Sequence[Path], transport: Transport, llm: dict) -> None:
    """Ask the verifier whether each value's evidence crop shows that value (synthetic only)."""
    pending = [value for value in values if value.evidence and value.raw_value and value.key != "location"]
    for start in range(0, len(pending), 8):
        batch = pending[start : start + 8]
        state, questions = {}, {}
        for index, value in enumerate(batch):
            evidence = value.evidence[0]
            with Image.open(page_paths[evidence.page - 1]) as page:
                state[f"crop{index}"] = _jev_image(_crop(page, evidence.box))
            questions[f"v{index}"] = {
                "type": "noul",
                "instructions": f"Image crop{index} is cut from a scanned page. Is {value.label} printed there exactly as '{value.raw_value}'?",
                "criteria": {"true": f"crop{index} shows exactly '{value.raw_value}'", "false": "It shows different characters or is not there"},
            }
        answers, cost = _ask_jev({"crops": state}, questions, transport)
        llm["verifier_cost"] = (llm.get("verifier_cost") or 0) + (cost or 0)
        for index, value in enumerate(batch):
            value.components.verifier = float(answers[f"v{index}"]["noul"])


# Said last, after the page text and images: a part named only in the system prompt was
# ignored (the addresses call wrote every field too).
PART_REQUESTS = {
    FieldsReply: "Now reply with the document type and the catalog fields only.",
    LocationsReply: 'Now reply with the addresses only: {"locations": [...]}, nothing else.',
    LocationsEntitiesReply: 'Now reply with the addresses and the other facts only: '
    '{"locations": [...], "entities": [...]}, nothing else.',
}


def _part_messages(messages: list[dict[str, Any]], part: type) -> list[dict[str, Any]]:
    """The same messages, asking for one part of the extraction in that part's schema."""
    system = messages[0]["content"] + (
        "\nReply with exactly one JSON object matching this JSON schema, with no prose and no "
        f"code fences:\n{json.dumps(part.model_json_schema())}"
    )
    user = [*messages[1]["content"], {"type": "text", "text": PART_REQUESTS[part]}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _resolve_aliases(reply, aliases: dict[str, str]) -> None:
    """Map the short aliases the LLM cited back to real OCR region ids, in place. An alias
    the LLM invented is left as-is, so it is an unknown id downstream exactly as an invented
    region id was before aliasing."""
    # The prompt prints an alias as "[12]"; the reply sometimes cites it with its brackets.
    resolve = lambda ids: [aliases.get(alias.strip("[] "), alias) for alias in ids]  # noqa: E731
    for reading in [*getattr(reply, "fields", []), *getattr(reply, "entities", [])]:
        reading.region_ids = resolve(reading.region_ids)
    for reading in getattr(reply, "locations", []):
        reading.address_region_ids = resolve(reading.address_region_ids)
        reading.phone_region_ids = resolve(reading.phone_region_ids)
        reading.fax_region_ids = resolve(reading.fax_region_ids)


def _merge_llm(fields: dict, other: dict) -> dict:
    """Metadata for the two concurrent calls: total cost, the slower call's wall time, and
    both usages kept separate so spend can still be attributed to each half."""
    return {
        "model": fields["model"], "cost": (fields["cost"] or 0) + (other["cost"] or 0),
        "latency_ms": max(fields["latency_ms"], other["latency_ms"]),
        "attempts": fields["attempts"] + other["attempts"], "prompt_version": PROMPT_VERSION,
        "usage": {"fields": fields["usage"], "other": other["usage"]},
    }


def _hedged(call):
    """call()'s result. When it has not answered within HEDGE_SECONDS, an identical request
    races it and the first to finish wins: a provider stall otherwise costs half a minute."""
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(call)]
        if not wait(futures, timeout=HEDGE_SECONDS).done:
            futures.append(pool.submit(call))
        return next(iter(wait(futures, return_when=FIRST_COMPLETED).done)).result()
    finally:
        pool.shutdown(wait=False)  # the slower request finishes on its own and is ignored


def _ask_llm(messages, model, transport: Transport, max_tokens: int):
    result = None
    for attempt in range(2):
        try:
            # Plain JSON with Pydantic as the gate: the Qwen provider has no structured
            # outputs, and in JSON mode it returns no logprobs.
            result = _hedged(lambda: _call_openrouter(
                MODEL, messages, {"type": "object"}, model.__name__, max_tokens, None, 180, 3,
                False, transport, time.sleep, zero_data_retention=False, reasoning_enabled=False,
                logprobs=True,
            ))
        except OpenRouterError as error:
            # Compact JSON occasionally comes back cut off a character short (no closing
            # brace); one retry asking for the complete object fixes it as cheaply as a
            # schema-validation retry does.
            if attempt or error.code != "invalid_response":
                raise
            messages = [*messages, {"role": "user", "content": "That reply was not valid JSON (it looked cut off). Reply again with the complete JSON object only."}]
            continue
        try:
            reply = model.model_validate(result.content)
            break
        except ValidationError as error:
            if attempt:
                raise OpenRouterError(f"LLM reply failed validation twice: {error.error_count()} errors") from error
            messages = [
                *messages,
                {"role": "assistant", "content": json.dumps(result.content)},
                {"role": "user", "content": "That JSON does not match the schema:\n"
                 + _errors(error) + "\nReply with the corrected JSON object only."},
            ]
    return reply, {
        "model": result.model, "cost": result.cost, "latency_ms": result.latency_ms,
        "attempts": result.attempts + attempt, "prompt_version": PROMPT_VERSION,
        "usage": result.usage,
    }, result.token_logprobs


def _token_probabilities(tokens, written: Sequence[str | None]) -> list[float | None]:
    """The LLM's probability for each value it wrote: that of its least likely token.

    Values are looked up in reply order, so a value written twice maps to its own place.
    """
    text, starts = "", []
    for token, _ in tokens:
        starts.append(len(text))
        text += token
    cursor, probabilities = 0, []
    for raw in written:
        needle = json.dumps(raw)[1:-1] if raw else ""
        at = text.find(needle, cursor) if needle else -1
        if at < 0:
            probabilities.append(None)
            continue
        end = cursor = at + len(needle)
        logprob = min(
            logprob for (token, logprob), start in zip(tokens, starts, strict=True)
            if start < end and start + len(token) > at
        )
        probabilities.append(round(math.exp(logprob), 4))
    return probabilities


def _band(probability: float) -> str:
    return "high" if probability >= 0.9 else "medium" if probability >= 0.6 else "low"


def _errors(error: ValidationError) -> str:
    return "\n".join(f"- {'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()[:20])


def _ask_jev(state: Any, questions: dict, transport: Transport) -> tuple[dict, float | None]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set")
    request = Request(
        JEV_URL,
        data=json.dumps({"model": JEV_MODEL, "state": state, "questions": questions}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    status, _, body = transport(request, 60)
    if status != 200:
        raise OpenRouterError(f"Verifier request failed with HTTP {status}")
    payload = json.loads(body)
    return payload["answers"], (payload.get("usage") or {}).get("cost")


def _image_part(image: Image.Image, long_side: int) -> dict[str, Any]:
    image = image.convert("RGB")
    image.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}


def _jev_image(image: Image.Image) -> dict[str, Any]:
    low, high = JEV_SIDE
    image = image.convert("RGB")
    image.thumbnail((high, high), Image.Resampling.LANCZOS)
    if min(image.size) < low:
        canvas = Image.new("RGB", (max(low, image.width), max(low, image.height)), "white")
        canvas.paste(image)
        image = canvas
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}}


def _crop(page: Image.Image, box: Box, pad: int = 24) -> Image.Image:
    return page.crop((max(0, box.left - pad), max(0, box.top - pad), min(page.width, box.right + pad), min(page.height, box.bottom + pad)))


def _union(regions: Sequence[dict[str, Any]]) -> Box:
    boxes = [region["bounding_box"] for region in regions if region.get("bounding_box")]
    return Box(
        left=min(box["left"] for box in boxes), top=min(box["top"] for box in boxes),
        right=max(box["right"] for box in boxes), bottom=max(box["bottom"] for box in boxes),
    )


def _types_text() -> str:
    return "\n".join(f"- {key}: {meaning}" for key, (_, meaning, _) in DOCUMENT_TYPES.items())


def _alnum(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def _min(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else min(first, second)

