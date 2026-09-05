"""Compare evidence-preserving structured rendering prompts on public pages."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ocr_pipeline.openrouter import (
    DEFAULT_MAX_TOKENS,
    MAX_TOKENS,
    QWEN_37_FLASH_MODEL,
    OpenRouterError,
    OpenRouterResult,
    _call_openrouter,
    _default_transport,
    _shape_error,
    repair_image,
)

MODES = ("image_evidence", "evidence_json", "flat_text")
PROMPT_VERSION = "structured-render-challenger-v1"
PAGE_LITERAL_ID = "page-literal"
KINDS = (
    "title",
    "heading",
    "header",
    "footer",
    "text",
    "list",
    "field",
    "handwriting",
    "table",
    "control",
    "figure",
)
STATES = ("none", "checked", "unchecked", "ambiguous")

BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "left": {"type": "integer"},
        "top": {"type": "integer"},
        "right": {"type": "integer"},
        "bottom": {"type": "integer"},
    },
    "required": ["left", "top", "right", "bottom"],
    "additionalProperties": False,
}
CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "row": {"type": "integer"},
        "column": {"type": "integer"},
        "text": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "bounding_box": BOX_SCHEMA,
    },
    "required": ["row", "column", "text", "evidence_ids", "bounding_box"],
    "additionalProperties": False,
}
STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["none", "table", "control"]},
        "row_count": {"type": "integer"},
        "column_count": {"type": "integer"},
        "cells": {"type": "array", "items": CELL_SCHEMA},
        "state": {"type": "string", "enum": list(STATES)},
        "label_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "type",
        "row_count",
        "column_count",
        "cells",
        "state",
        "label_evidence_ids",
    ],
    "additionalProperties": False,
}
REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "text": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "bounding_box": BOX_SCHEMA,
        "reading_order": {"type": "integer"},
        "structure": STRUCTURE_SCHEMA,
    },
    "required": [
        "id",
        "kind",
        "text",
        "evidence_ids",
        "bounding_box",
        "reading_order",
        "structure",
    ],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "page": {
            "type": "object",
            "properties": {
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "regions": {"type": "array", "items": REGION_SCHEMA},
                "rendered_markdown": {"type": "string"},
            },
            "required": ["width", "height", "regions", "rendered_markdown"],
            "additionalProperties": False,
        }
    },
    "required": ["page"],
    "additionalProperties": False,
}

Challenger = Callable[
    [str, Path | None, str, Mapping[str, Any], str, int], OpenRouterResult
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a non-eligible structured-rendering challenger"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--confirm-public-data",
        action="store_true",
        help="Confirm every page is public or generated and contains no private data",
    )
    args = parser.parse_args(argv)
    if not args.confirm_public_data:
        parser.error("--confirm-public-data is required")
    try:
        report = run_benchmark(
            args.manifest,
            provider_slug=args.provider,
            max_tokens=args.max_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    manifest_path: Path,
    *,
    provider_slug: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    challenger: Challenger | None = None,
) -> dict[str, Any]:
    if not provider_slug or provider_slug != provider_slug.strip():
        raise ValueError("Provider slug must be a non-empty trimmed string")
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f"Max tokens must be between 1 and {MAX_TOKENS}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a non-empty cases array")

    records: list[dict[str, Any]] = []
    for case in cases:
        loaded = _load_case(case, manifest_path.parent)
        for mode in MODES:
            records.append(
                _evaluate(
                    loaded,
                    mode,
                    provider_slug,
                    max_tokens,
                    challenger or _openrouter_challenger,
                )
            )
    return {
        "challenger": QWEN_37_FLASH_MODEL,
        "eligible_runtime": False,
        "prompt_version": PROMPT_VERSION,
        "provider_slug": provider_slug,
        "case_count": len(cases),
        "summary": {mode: _summarize(records, mode) for mode in MODES},
        "cases": records,
    }


def validate_output(content: Mapping[str, Any], source: Mapping[str, Any]) -> list[str]:
    shape_error = _shape_error(content, OUTPUT_SCHEMA)
    if shape_error:
        return [f"schema mismatch: {shape_error}"]
    errors: list[str] = []
    page = content.get("page")
    if not isinstance(page, dict):
        return ["page must be an object"]
    width, height = source["width"], source["height"]
    if page.get("width") != width or page.get("height") != height:
        errors.append("page dimensions differ from canonical evidence")
    regions = page.get("regions")
    if not isinstance(regions, list):
        return errors + ["page.regions must be an array"]

    evidence = {item["id"]: item for item in source["evidence"]}
    output_ids: set[str] = set()
    orders: set[int] = set()
    rendered_text: list[str] = []
    for index, region in enumerate(regions):
        path = f"page.regions[{index}]"
        if not isinstance(region, dict):
            errors.append(f"{path} must be an object")
            continue
        region_id = region.get("id")
        if not isinstance(region_id, str) or not region_id.strip():
            errors.append(f"{path}.id must be non-empty")
        elif region_id in output_ids:
            errors.append(f"{path}.id is duplicated")
        else:
            output_ids.add(region_id)
        order = region.get("reading_order")
        if not _integer(order) or order < 0 or order in orders:
            errors.append(f"{path}.reading_order must be unique and non-negative")
        else:
            orders.add(order)
        errors.extend(_validate_region(region, evidence, width, height, path))
        if isinstance(region.get("text"), str):
            rendered_text.append(region["text"])
        structure = region.get("structure")
        if isinstance(structure, dict):
            rendered_text.extend(
                cell.get("text", "")
                for cell in structure.get("cells", [])
                if isinstance(cell, dict) and isinstance(cell.get("text"), str)
            )

    markdown = page.get("rendered_markdown")
    if not isinstance(markdown, str):
        errors.append("page.rendered_markdown must be a string")
    elif not _tokens_supported(
        _strip_markdown_syntax(markdown), " ".join(rendered_text)
    ):
        errors.append("rendered Markdown contains unsupported literals")
    return errors


def _validate_region(
    region: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    width: int,
    height: int,
    path: str,
) -> list[str]:
    errors: list[str] = []
    box = region.get("bounding_box")
    if not _valid_box(box, width, height):
        errors.append(f"{path}.bounding_box is outside the page")
    evidence_ids = region.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        errors.append(f"{path}.evidence_ids must be non-empty")
        return errors
    unknown = [value for value in evidence_ids if value not in evidence]
    if unknown:
        errors.append(f"{path}.evidence_ids contains unknown IDs")
        return errors
    source_text = " ".join(str(evidence[value]["text"]) for value in evidence_ids)
    text = region.get("text")
    if not isinstance(text, str) or not _tokens_supported(text, source_text):
        errors.append(f"{path}.text contains unsupported literals")
    if isinstance(box, dict) and not any(
        _overlaps(box, evidence[value]["bounding_box"]) for value in evidence_ids
    ):
        errors.append(f"{path}.bounding_box does not overlap cited evidence")
    errors.extend(_validate_structure(region, evidence, width, height, path))
    return errors


def _validate_structure(
    region: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    width: int,
    height: int,
    path: str,
) -> list[str]:
    structure = region.get("structure")
    if not isinstance(structure, dict):
        return [f"{path}.structure must be an object"]
    structure_type = structure.get("type")
    kind = region.get("kind")
    if structure_type == "table" and kind != "table":
        return [f"{path}.table structure requires table kind"]
    if structure_type == "control" and kind != "control":
        return [f"{path}.control structure requires control kind"]
    if kind in {"table", "control"} and structure_type != kind:
        return [f"{path}.{kind} kind requires matching structure"]
    if structure_type == "table":
        errors = _validate_table(structure, evidence, width, height, path)
        if (
            structure.get("state") != "none"
            or structure.get("label_evidence_ids") != []
        ):
            errors.append(f"{path}.table structure has control fields")
        return errors
    if structure_type == "control":
        state = structure.get("state")
        labels = structure.get("label_evidence_ids")
        errors = []
        if state not in STATES[1:]:
            errors.append(f"{path}.structure.state must describe the control")
        if (
            not isinstance(labels, list)
            or not labels
            or any(label not in evidence for label in labels)
        ):
            errors.append(f"{path}.structure.label_evidence_ids are invalid")
        if (
            structure.get("row_count") != 0
            or structure.get("column_count") != 0
            or structure.get("cells") != []
        ):
            errors.append(f"{path}.control structure has table fields")
        return errors
    if structure_type != "none":
        return [f"{path}.structure.type is invalid"]
    if (
        structure.get("row_count") != 0
        or structure.get("column_count") != 0
        or structure.get("cells") != []
        or structure.get("state") != "none"
        or structure.get("label_evidence_ids") != []
    ):
        return [f"{path}.none structure must be empty"]
    return []


def _validate_table(
    structure: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    width: int,
    height: int,
    path: str,
) -> list[str]:
    rows = structure.get("row_count")
    columns = structure.get("column_count")
    cells = structure.get("cells")
    if not _integer(rows) or rows < 1 or not _integer(columns) or columns < 1:
        return [f"{path}.structure table dimensions must be positive"]
    if not isinstance(cells, list) or not cells:
        return [f"{path}.structure.cells must be non-empty"]
    errors: list[str] = []
    for index, cell in enumerate(cells):
        cell_path = f"{path}.structure.cells[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"{cell_path} must be an object")
            continue
        row, column = cell.get("row"), cell.get("column")
        if not _integer(row) or not 0 <= row < rows:
            errors.append(f"{cell_path}.row is invalid")
        if not _integer(column) or not 0 <= column < columns:
            errors.append(f"{cell_path}.column is invalid")
        box = cell.get("bounding_box")
        if not _valid_box(box, width, height):
            errors.append(f"{cell_path}.bounding_box is outside the page")
        ids = cell.get("evidence_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(value not in evidence for value in ids)
        ):
            errors.append(f"{cell_path}.evidence_ids are invalid")
            continue
        source_text = " ".join(str(evidence[value]["text"]) for value in ids)
        if not isinstance(cell.get("text"), str) or not _tokens_supported(
            cell["text"], source_text
        ):
            errors.append(f"{cell_path}.text contains unsupported literals")
        if isinstance(box, dict) and not any(
            _overlaps(box, evidence[value]["bounding_box"]) for value in ids
        ):
            errors.append(f"{cell_path}.bounding_box does not overlap cited evidence")
    return errors


def _evaluate(
    case: Mapping[str, Any],
    mode: str,
    provider_slug: str,
    max_tokens: int,
    challenger: Challenger,
) -> dict[str, Any]:
    source = _source_for_mode(case["page"], mode)
    prompt = _prompt(source, mode)
    image_path = case["image_path"] if mode == "image_evidence" else None
    started = time.perf_counter()
    try:
        result = challenger(
            mode, image_path, prompt, OUTPUT_SCHEMA, provider_slug, max_tokens
        )
        errors = validate_output(result.content, source)
        if errors:
            status = "rejected"
        elif not result.content["page"]["regions"]:
            status = "abstained"
        else:
            status = "accepted"
        content = result.content
        provider = result.provider
        cost = result.cost
        api_latency_ms = result.latency_ms
        failure = None
    except OpenRouterError as error:
        status = "failed"
        errors = [str(error)]
        content = None
        provider = None
        cost = None
        api_latency_ms = error.latency_ms
        failure = {"code": error.code, "message": str(error)}
    return {
        "id": case["id"],
        "classification": case["classification"],
        "provenance": case["provenance"],
        "mode": mode,
        "status": status,
        "validation_errors": errors,
        "prediction": content,
        "requested_model": QWEN_37_FLASH_MODEL,
        "provider": provider,
        "cost": cost,
        "api_latency_ms": api_latency_ms,
        "wall_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "failure": failure,
    }


def _load_case(case: Any, root: Path) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError("Every case must be an object")
    case_id = case.get("id")
    classification = case.get("classification")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Every case requires a non-empty id")
    if classification == "public":
        source_url = case.get("source_url")
        if not isinstance(source_url, str) or not source_url.startswith("https://"):
            raise ValueError(f"Public case {case_id} requires an HTTPS source_url")
        provenance = {"source_url": source_url}
    elif classification == "synthetic":
        if not isinstance(case.get("generator"), str) or not case["generator"].strip():
            raise ValueError(f"Synthetic case {case_id} requires a generator")
        provenance = {"generator": case["generator"]}
    else:
        raise ValueError(f"Case {case_id} must be classified public or synthetic")

    image_path = _within(root, case.get("image"), "image")
    canonical_path = _within(root, case.get("canonical"), "canonical")
    page = _canonical_page(canonical_path, int(case.get("page_number", 1)))
    return {
        "id": case_id,
        "classification": classification,
        "provenance": provenance,
        "image_path": image_path,
        "page": page,
    }


def _within(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Case {name} path is required")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Case {name} path must stay within the manifest directory"
        ) from error
    if not path.is_file():
        raise ValueError(f"Case {name} file does not exist: {path}")
    return path


def _canonical_page(path: Path, page_number: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    pages = result.get("pages") if isinstance(result, dict) else None
    if not isinstance(pages, list):
        raise ValueError(f"Canonical result has no pages: {path}")
    page = next(
        (
            value
            for value in pages
            if isinstance(value, dict) and value.get("page_number") == page_number
        ),
        None,
    )
    if page is None:
        raise ValueError(f"Canonical result has no page {page_number}: {path}")
    width, height, regions = page.get("width"), page.get("height"), page.get("regions")
    if not _integer(width) or width < 1 or not _integer(height) or height < 1:
        raise ValueError(f"Canonical page dimensions are invalid: {path}")
    if not isinstance(regions, list) or not regions:
        raise ValueError(f"Canonical page has no regions: {path}")
    evidence = []
    evidence_ids: set[str] = set()
    for region in regions:
        if not isinstance(region, dict) or not isinstance(region.get("id"), str):
            raise ValueError(f"Canonical region is invalid: {path}")
        if not region["id"] or region["id"] in evidence_ids:
            raise ValueError(
                f"Canonical region IDs must be unique and non-empty: {path}"
            )
        evidence_ids.add(region["id"])
        box = region.get("bounding_box")
        if not _valid_box(box, width, height):
            raise ValueError(f"Canonical region box is invalid: {path}")
        evidence.append(
            {
                "id": region["id"],
                "kind": str(region.get("kind", "text")),
                "text": str(region.get("text", "")),
                "reading_order": int(region.get("reading_order", len(evidence))),
                "bounding_box": box,
            }
        )
    return {"width": width, "height": height, "evidence": evidence}


def _source_for_mode(page: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode != "flat_text":
        return dict(page)
    ordered = sorted(page["evidence"], key=lambda value: value["reading_order"])
    return {
        "width": page["width"],
        "height": page["height"],
        "evidence": [
            {
                "id": PAGE_LITERAL_ID,
                "kind": "text",
                "text": " ".join(item["text"] for item in ordered),
                "reading_order": 0,
                "bounding_box": {
                    "left": 0,
                    "top": 0,
                    "right": page["width"],
                    "bottom": page["height"],
                },
            }
        ],
    }


def _prompt(source: Mapping[str, Any], mode: str) -> str:
    if mode == "flat_text":
        literal = source["evidence"][0]["text"]
        supplied_input = (
            f"Page width: {source['width']}. Page height: {source['height']}. "
            f"Evidence ID for the entire page: {PAGE_LITERAL_ID}. Flat literal OCR: "
            f"{json.dumps(literal, ensure_ascii=False)}"
        )
    else:
        evidence = json.dumps(source, ensure_ascii=False, separators=(",", ":"))
        supplied_input = f"Canonical evidence JSON: {evidence}"
    image_note = (
        "Use the attached image only to recover layout."
        if mode == "image_evidence"
        else "No image is provided."
    )
    return (
        "Convert the supplied OCR evidence into the requested strict JSON schema. "
        "Preserve literal text. Never correct, infer, or add a literal. Every output "
        "region and table cell must cite input evidence IDs. Emit table and control "
        "structure only when supported. Use zero-based reading order, rows, and "
        "columns. In Markdown, represent controls only with [x], [ ], or [?]. "
        f"{image_note} {supplied_input}"
    )


def _openrouter_challenger(
    mode: str,
    image_path: Path | None,
    prompt: str,
    schema: Mapping[str, Any],
    provider_slug: str,
    max_tokens: int,
) -> OpenRouterResult:
    if mode == "image_evidence":
        if image_path is None:
            raise OpenRouterError("Image evidence mode requires an image")
        return repair_image(
            image_path,
            prompt,
            schema,
            model=QWEN_37_FLASH_MODEL,
            schema_name="structured_render_challenger",
            max_tokens=max_tokens,
            provider_slug=provider_slug,
            public_benchmark=True,
        )
    return _call_openrouter(
        QWEN_37_FLASH_MODEL,
        [{"role": "user", "content": prompt}],
        schema,
        "structured_render_challenger",
        max_tokens,
        provider_slug,
        120,
        3,
        True,
        _default_transport,
        time.sleep,
    )


def _summarize(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [record for record in records if record["mode"] == mode]
    latencies = [record["wall_latency_ms"] for record in selected]
    accepted = sum(record["status"] == "accepted" for record in selected)
    costs = [record["cost"] for record in selected if record["cost"] is not None]
    accepted_predictions = [
        record["prediction"] for record in selected if record["status"] == "accepted"
    ]
    return {
        "cases": len(selected),
        "accepted": accepted,
        "acceptance_rate": accepted / len(selected),
        "rejected": sum(record["status"] == "rejected" for record in selected),
        "abstained": sum(record["status"] == "abstained" for record in selected),
        "failed": sum(record["status"] == "failed" for record in selected),
        "table_proposals": _proposal_count(accepted_predictions, "table"),
        "control_proposals": _proposal_count(accepted_predictions, "control"),
        "p50_wall_latency_ms": statistics.median(latencies),
        "p95_wall_latency_ms": _percentile(latencies, 0.95),
        "reported_cost": round(sum(costs), 8) if costs else None,
    }


def _proposal_count(predictions: list[Mapping[str, Any]], kind: str) -> int:
    return sum(
        region.get("kind") == kind
        for prediction in predictions
        for region in prediction["page"]["regions"]
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999))
    return ordered[index]


def _valid_box(value: Any, width: int, height: int) -> bool:
    if not isinstance(value, dict):
        return False
    parts = [value.get(name) for name in ("left", "top", "right", "bottom")]
    if any(not _integer(part) for part in parts):
        return False
    left, top, right, bottom = parts
    return 0 <= left < right <= width and 0 <= top < bottom <= height


def _overlaps(first: Mapping[str, int], second: Mapping[str, int]) -> bool:
    return min(first["right"], second["right"]) > max(
        first["left"], second["left"]
    ) and min(first["bottom"], second["bottom"]) > max(first["top"], second["top"])


def _tokens_supported(candidate: str, source: str) -> bool:
    source_tokens = _literal_tokens(source)
    source_index = 0
    for candidate_token in _literal_tokens(candidate):
        while (
            source_index < len(source_tokens)
            and source_tokens[source_index] != candidate_token
        ):
            source_index += 1
        if source_index == len(source_tokens):
            return False
        source_index += 1
    return True


def _strip_markdown_syntax(value: str) -> str:
    lines = value.splitlines()
    separator_rows = {
        index for index, line in enumerate(lines) if _markdown_table_separator(line)
    }
    table_rows = set(separator_rows)
    for index in separator_rows:
        for direction in (-1, 1):
            row = index + direction
            while (
                0 <= row < len(lines)
                and lines[row].strip()
                and re.search(r"(?<!\\)\|", lines[row])
            ):
                table_rows.add(row)
                row += direction

    rendered_lines = []
    for index, line in enumerate(lines):
        if index in separator_rows:
            continue
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+", "", line)
        line = re.sub(r"\[[ xX?]\]\s*", "", line)
        line = re.sub(r"(?<!!)\[([^]\n]+)\]\([^\n)]*\)", r"\1", line)
        if index in table_rows:
            line = re.sub(r"(?<!\\)\|", " ", line)
        previous = None
        while line != previous:
            previous = line
            line = re.sub(
                r"(?<!\\)(\*\*|__|~~|`+|\*|_)(\S(?:.*?\S)?)(?<!\\)\1",
                r"\2",
                line,
            )
        rendered_lines.append(re.sub(r"\\([\\`*{}\[\]()#+\-.!_|>~])", r"\1", line))
    return "\n".join(rendered_lines)


def _markdown_table_separator(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*",
            value,
        )
    )


def _literal_tokens(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", value.casefold(), flags=re.UNICODE)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
