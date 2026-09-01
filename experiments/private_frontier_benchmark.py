"""Create fail-closed OpenRouter predictions for the private OCR benchmark."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata
from typing import Any, Callable, Mapping, Sequence

from ocr_pipeline.openrouter import (
    DEFAULT_MAX_TOKENS,
    MAX_TOKENS,
    PUBLIC_BENCHMARK_IMAGE_MODELS,
    OpenRouterError,
    OpenRouterResult,
    repair_image,
)

PROMPTS = {
    "forms": (
        "List every visible checkbox or radio control. Return its visible label "
        "exactly and classify the mark as checked, unchecked, or uncertain. Do not "
        "infer omitted controls or correct text."
    ),
    "handwriting": (
        "Transcribe all visible handwritten text literally in reading order. Preserve "
        "line breaks. Report page legibility and copy any uncertain spans exactly. Do "
        "not explain, correct, complete, or infer text."
    ),
    "tables": (
        "Extract every visible table literally. Return visible table, row, and column "
        "labels in reading order, cell text using 1-based row and column indices, and "
        "merged ranges using 1-based indices. Do not invent labels, fill blank cells, "
        "correct values, or infer structure."
    ),
}

APPROVAL_FILENAME = "synthetic_external_processing_approval.json"
_DARWIN_RENAME_EXCL = 4
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1

FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "controls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": ["checked", "unchecked", "uncertain"],
                    },
                },
                "required": ["label", "state"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["controls"],
    "additionalProperties": False,
}

HANDWRITING_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "legibility": {
            "type": "string",
            "enum": ["legible", "partly_legible", "illegible"],
        },
        "uncertain_spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "legibility", "uncertain_spans"],
    "additionalProperties": False,
}

_LABEL = {"type": "string"}
_INDEX = {"type": "integer", "minimum": 1}
_MERGED_RANGE = {
    "type": "object",
    "properties": {
        "start_row_index": _INDEX,
        "end_row_index": _INDEX,
        "start_column_index": _INDEX,
        "end_column_index": _INDEX,
    },
    "required": [
        "start_row_index",
        "end_row_index",
        "start_column_index",
        "end_column_index",
    ],
    "additionalProperties": False,
}
TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": _LABEL,
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"label": _LABEL},
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"label": _LABEL},
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                    "cells": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_index": _INDEX,
                                "column_index": _INDEX,
                                "text": {"type": "string"},
                            },
                            "required": ["row_index", "column_index", "text"],
                            "additionalProperties": False,
                        },
                    },
                    "merged_ranges": {"type": "array", "items": _MERGED_RANGE},
                },
                "required": [
                    "label",
                    "rows",
                    "columns",
                    "cells",
                    "merged_ranges",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tables"],
    "additionalProperties": False,
}

SCHEMAS = {
    "forms": FORM_SCHEMA,
    "handwriting": HANDWRITING_SCHEMA,
    "tables": TABLE_SCHEMA,
}
RepairCall = Callable[..., OpenRouterResult]
SPLIT_FRACTIONS = {"development": 0.4, "calibration": 0.2, "eval": 0.4}
SPLIT_ORDER = ("eval", "development", "calibration")


class CaseFailure(ValueError):
    """A case that must abstain without writing a prediction."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Case:
    case_id: str
    source_group: str
    task: str
    images: tuple[dict[str, Any], ...]
    target: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create reviewed OpenRouter predictions for private OCR data"
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--model", required=True, choices=sorted(PUBLIC_BENCHMARK_IMAGE_MODELS)
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument(
        "--split",
        choices=("development", "calibration", "eval", "all"),
        default="development",
    )
    parser.add_argument("--review-passes", type=_positive_int, default=3)
    parser.add_argument("--max-tokens", type=_max_tokens, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)

    try:
        run_benchmark(
            args.data_root,
            args.output_dir,
            model=args.model,
            provider=args.provider,
            split=args.split,
            review_passes=args.review_passes,
            max_tokens=args.max_tokens,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def run_benchmark(
    data_root: Path,
    output_dir: Path,
    *,
    model: str,
    provider: str,
    split: str = "development",
    review_passes: int = 3,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    repair: RepairCall = repair_image,
) -> dict[str, int]:
    _validate_run(
        data_root,
        output_dir,
        model,
        provider,
        split,
        review_passes,
        max_tokens,
    )
    cases = _discover_cases(data_root, split)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        predictions_dir = staging / "predictions"
        audit_dir = staging / "audit"
        predictions_dir.mkdir()
        audit_dir.mkdir()
        submitted: list[str] = []
        for case in cases:
            prediction, audit = _run_case(
                case,
                model=model,
                provider=provider,
                review_passes=review_passes,
                max_tokens=max_tokens,
                repair=repair,
            )
            if prediction is not None:
                _write_json(predictions_dir / f"{case.case_id}.json", prediction)
                submitted.append(case.case_id)
            _write_json(audit_dir / f"{case.case_id}.json", audit)
        summary = {
            "eligible_cases": len(cases),
            "submitted_cases": len(submitted),
        }
        _publish_no_replace(staging, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_no_replace(source: Path, destination: Path) -> None:
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "Atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            os.fsencode(source), os.fsencode(destination), _DARWIN_RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "Atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            _LINUX_AT_FDCWD,
            os.fsencode(source),
            _LINUX_AT_FDCWD,
            os.fsencode(destination),
            _LINUX_RENAME_NOREPLACE,
        )
    elif os.name == "nt":
        source.rename(destination)
        return
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-replace rename is unavailable")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _validate_run(
    data_root: Path,
    output_dir: Path,
    model: str,
    provider: str,
    split: str,
    review_passes: int,
    max_tokens: int,
) -> None:
    if not data_root.is_dir():
        raise ValueError(f"Private data root is not a directory: {data_root}")
    if output_dir.exists():
        raise ValueError("Output directory must be new")
    if model not in PUBLIC_BENCHMARK_IMAGE_MODELS:
        raise ValueError(f"Unsupported frontier model: {model}")
    if not provider or provider != provider.strip():
        raise ValueError("Provider must be a non-empty trimmed slug")
    if split not in {*SPLIT_FRACTIONS, "all"}:
        raise ValueError(f"Unsupported split: {split}")
    _validate_synthetic_approval(data_root, provider)
    if isinstance(review_passes, bool) or review_passes < 1:
        raise ValueError("Review passes must be positive")
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f"Max tokens must be between 1 and {MAX_TOKENS}")


def _validate_synthetic_approval(data_root: Path, provider: str) -> None:
    path = data_root / APPROVAL_FILENAME
    if not path.is_file():
        raise ValueError(f"Missing dataset approval record: {path}")
    approval = _read_object(path)
    _exact_keys(
        approval,
        {"schema_version", "data_root", "data_classification", "provider_slug"},
    )
    if (
        type(approval["schema_version"]) is not int
        or approval["schema_version"] != 1
        or approval["data_root"] != str(data_root.resolve())
        or approval["data_classification"] != "synthetic"
        or approval["provider_slug"] != provider
    ):
        raise ValueError("Dataset approval does not match this root and provider")


def _discover_cases(data_root: Path, split: str) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for path in sorted(data_root.rglob("ground_truth.json")):
        raw = _read_object(path)
        source_reference = _required_string(raw, "source_reference")
        if raw.get("availability") != "ready":
            continue
        case_id = _required_string(raw, "case_id")
        if Path(case_id).name != case_id or case_id in {".", ".."}:
            raise ValueError(f"Unsafe case identifier: {case_id}")
        task = _required_string(raw, "task")
        if task not in SCHEMAS:
            raise ValueError(f"Unsupported task in {path}")
        if case_id in seen:
            raise ValueError(f"Duplicate case identifier: {case_id}")
        seen.add(case_id)
        images = raw.get("images")
        target = raw.get("target")
        if not isinstance(images, list) or not images or not isinstance(target, dict):
            raise ValueError(f"Invalid case structure: {case_id}")
        ordered = tuple(
            sorted(
                (_image(item, path.parent) for item in images), key=lambda x: x["page"]
            )
        )
        pages = [image["page"] for image in ordered]
        image_ids = [image["image_id"] for image in ordered]
        if len(pages) != len(set(pages)) or len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Duplicate image page or identifier: {case_id}")
        cases.append(Case(case_id, source_reference, task, ordered, target))
    if split == "all":
        return cases
    assignments = _assign_splits(cases)
    return [case for case in cases if assignments[case.source_group] == split]


def _assign_splits(cases: list[Case]) -> dict[str, str]:
    groups: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        groups[case.source_group].append(case)

    totals = Counter(case.task for case in cases)
    counts = {split: Counter() for split in SPLIT_ORDER}
    assignments: dict[str, str] = {}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -len({case.task for case in item[1]}),
            item[0],
        ),
    )
    for source_group, group_cases in ordered_groups:
        group_counts = Counter(case.task for case in group_cases)
        options = []
        for position, split in enumerate(SPLIT_ORDER):
            candidate = {name: value.copy() for name, value in counts.items()}
            candidate[split].update(group_counts)
            target_total = len(cases) * SPLIT_FRACTIONS[split]
            load = sum(counts[split].values()) / max(target_total, 1)
            options.append((_split_error(candidate, totals), load, position, split))
        selected = min(options)[-1]
        counts[selected].update(group_counts)
        assignments[source_group] = selected
    return assignments


def _split_error(counts: dict[str, Counter[str]], totals: Counter[str]) -> float:
    error = 0.0
    total_cases = sum(totals.values())
    for split in SPLIT_ORDER:
        fraction = SPLIT_FRACTIONS[split]
        for task, total in totals.items():
            target = total * fraction
            error += ((counts[split][task] - target) / max(target, 1)) ** 2
        target_total = total_cases * fraction
        error += ((sum(counts[split].values()) - target_total) / target_total) ** 2
    return error


def _image(raw: Any, root: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Image metadata must be an object")
    image_id = _required_string(raw, "image_id")
    filename = _required_string(raw, "file")
    page = raw.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Image page must be a positive integer")
    path = root / filename
    if not path.is_file() or path.parent != root:
        raise ValueError("A ready case image is missing or outside its case")
    return {"image_id": image_id, "page": page, "path": path}


def _run_case(
    case: Case,
    *,
    model: str,
    provider: str,
    review_passes: int,
    max_tokens: int,
    repair: RepairCall,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []
    failure: str | None = None

    for image in case.images:
        outputs: list[dict[str, Any]] = []
        for review_pass in range(1, review_passes + 1):
            try:
                result = repair(
                    image["path"],
                    PROMPTS[case.task],
                    SCHEMAS[case.task],
                    model=model,
                    schema_name=f"private_{case.task}",
                    max_tokens=max_tokens,
                    provider_slug=provider,
                    public_benchmark=True,
                )
                raw = _validate_raw(case.task, result.content)
                outputs.append(raw)
                calls.append(_call_audit(image["page"], review_pass, result, raw))
            except OpenRouterError as error:
                calls.append(
                    {
                        "page": image["page"],
                        "review_pass": review_pass,
                        "status": "failed",
                        "failure_code": error.code,
                    }
                )
                failure = "page_failure"
                break
            except (OSError, TypeError, ValueError, KeyError):
                calls.append(
                    {
                        "page": image["page"],
                        "review_pass": review_pass,
                        "status": "failed",
                        "failure_code": "invalid_output",
                    }
                )
                failure = "invalid_structure"
                break
        if failure:
            agreement.append({"page": image["page"], "agreed": False})
            break
        agreed = len({_canonical(output) for output in outputs}) == 1
        agreement.append({"page": image["page"], "agreed": agreed})
        if not agreed:
            failure = "review_disagreement"
            break
        accepted.append(outputs[0])

    prediction: dict[str, Any] | None = None
    if failure is None:
        try:
            content = _map_prediction(case, accepted)
            prediction = {
                "case_id": case.case_id,
                "task": case.task,
                "prediction": content,
            }
        except CaseFailure as error:
            failure = error.code
        except (TypeError, ValueError, KeyError):
            failure = "invalid_structure"

    audit = {
        "case_id": case.case_id,
        "task": case.task,
        "requested_model": model,
        "provider_slug": provider,
        "review_passes": review_passes,
        "review_agreement": agreement,
        "consistency_note": (
            "Repeated same-model outputs measure repeat consistency, not independent "
            "review or ground truth."
        ),
        "status": "submitted" if prediction is not None else "abstained",
        "failure_reasons": [] if failure is None else [failure],
        "usage": _sum_usage(calls),
        "cost": _sum_number(calls, "cost"),
        "latency_ms": _sum_number(calls, "latency_ms"),
        "calls": calls,
    }
    return prediction, audit


def _call_audit(
    page: int, review_pass: int, result: OpenRouterResult, raw: dict[str, Any]
) -> dict[str, Any]:
    return {
        "page": page,
        "review_pass": review_pass,
        "status": "success",
        "model": result.model,
        "provider": result.provider,
        "usage": result.usage,
        "cost": result.cost,
        "latency_ms": result.latency_ms,
        "raw_output": raw,
    }


def _map_prediction(case: Case, pages: list[dict[str, Any]]) -> dict[str, Any]:
    if case.task == "forms":
        return _map_forms(case, pages)
    if case.task == "handwriting":
        return _map_handwriting(pages)
    return _map_tables(case, pages)


def _map_forms(case: Case, pages: list[dict[str, Any]]) -> dict[str, Any]:
    controls = case.target.get("controls")
    if not isinstance(controls, list):
        raise CaseFailure("invalid_target_structure")
    resolved: dict[str, str] = {}
    for image, page in zip(case.images, pages, strict=True):
        eligible = _controls_for_image(controls, image["image_id"])
        for raw, control in _map_form_controls(page["controls"], eligible):
            if raw["state"] == "uncertain":
                raise CaseFailure("uncertain_control")
            control_id = _required_string(control, "control_id")
            previous = resolved.get(control_id)
            if previous is not None and previous != raw["state"]:
                raise CaseFailure("control_conflict")
            resolved[control_id] = raw["state"]
    expected_ids = {_required_string(control, "control_id") for control in controls}
    if len(expected_ids) != len(controls) or set(resolved) != expected_ids:
        raise CaseFailure("control_sequence_mismatch")
    return {
        "controls": [
            {"control_id": control_id, "state": state}
            for control_id, state in sorted(resolved.items())
        ]
    }


def _map_form_controls(
    raw_controls: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    expected = {_required_string(target, "control_id") for target in targets}
    if len(expected) != len(targets):
        raise CaseFailure("invalid_target_structure")
    identities: dict[str, str] = {}
    for target in targets:
        control_id = _required_string(target, "control_id")
        for label in _normalized_labels(target):
            if label in identities and identities[label] != control_id:
                raise CaseFailure("ambiguous_control_identity")
            identities[label] = control_id
    if len(raw_controls) != len(targets):
        raise CaseFailure("control_sequence_mismatch")
    used: set[str] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in raw_controls:
        matches = [target for target in targets if _label_matches(raw["label"], target)]
        if len(matches) != 1:
            raise CaseFailure("control_identity_mismatch")
        target = matches[0]
        control_id = _required_string(target, "control_id")
        if control_id in used:
            raise CaseFailure("control_identity_mismatch")
        used.add(control_id)
        pairs.append((raw, target))
    if used != expected:
        raise CaseFailure("control_identity_mismatch")
    return pairs


def _map_handwriting(pages: list[dict[str, Any]]) -> dict[str, Any]:
    text: list[str] = []
    for page in pages:
        if (
            page["legibility"] != "legible"
            or page["uncertain_spans"]
            or not page["text"].strip()
        ):
            raise CaseFailure("uncertain_handwriting")
        text.append(page["text"].strip())
    return {"text": " ".join(text)}


def _map_tables(case: Case, pages: list[dict[str, Any]]) -> dict[str, Any]:
    targets = case.target.get("tables")
    if not isinstance(targets, list):
        raise CaseFailure("invalid_target_structure")
    predictions: list[dict[str, Any]] = []
    used_tables: set[str] = set()
    for image, page in zip(case.images, pages, strict=True):
        expected = _tables_for_image(targets, image["image_id"])
        pairs = _ordered_pairs(page["tables"], expected, "table_sequence_mismatch")
        for raw_table, target in pairs:
            table_id = _required_string(target, "table_id")
            if table_id in used_tables:
                raise CaseFailure("invalid_target_structure")
            used_tables.add(table_id)
            predictions.append(
                _map_table(raw_table, target, table_id, image["image_id"])
            )
    expected_ids = {_required_string(table, "table_id") for table in targets}
    if len(expected_ids) != len(targets) or used_tables != expected_ids:
        raise CaseFailure("table_sequence_mismatch")
    return {"tables": predictions}


def _map_table(
    raw: dict[str, Any], target: dict[str, Any], table_id: str, image_id: str
) -> dict[str, Any]:
    target_rows = target.get("rows")
    target_columns = target.get("columns")
    if not isinstance(target_rows, list) or not isinstance(target_columns, list):
        raise CaseFailure("invalid_target_structure")
    rows = _map_axes(raw["rows"], target_rows, "row_id")
    columns = _map_axes(raw["columns"], target_columns, "column_id")
    row_ids = [row["row_id"] for row in rows]
    column_ids = [column["column_id"] for column in columns]
    allowed_cells = _target_cell_positions(target, row_ids, column_ids)
    cells: list[dict[str, str]] = []
    positions: set[tuple[str, str]] = set()
    for cell in raw["cells"]:
        row_id = row_ids[cell["row_index"] - 1]
        column_id = column_ids[cell["column_index"] - 1]
        position = (row_id, column_id)
        if position in positions or position not in allowed_cells:
            raise CaseFailure("unmapped_cell")
        positions.add(position)
        cells.append({"row_id": row_id, "column_id": column_id, "text": cell["text"]})
    allowed_ranges = _target_merged_ranges(target, row_ids, column_ids)
    ranges: list[dict[str, str]] = []
    used_ranges: set[tuple[str, str, str, str]] = set()
    for merged in raw["merged_ranges"]:
        mapped = {
            "start_row_id": row_ids[merged["start_row_index"] - 1],
            "end_row_id": row_ids[merged["end_row_index"] - 1],
            "start_column_id": column_ids[merged["start_column_index"] - 1],
            "end_column_id": column_ids[merged["end_column_index"] - 1],
        }
        position = tuple(mapped.values())
        if position not in allowed_ranges or position in used_ranges:
            raise CaseFailure("unmapped_merged_range")
        used_ranges.add(position)
        ranges.append(mapped)
    return {
        "table_id": table_id,
        "label": raw["label"],
        "image_id": image_id,
        "rows": rows,
        "columns": columns,
        "cells": cells,
        "merged_ranges": ranges,
    }


def _map_axes(
    raw_items: list[dict[str, str]],
    target_items: list[Any],
    id_key: str,
) -> list[dict[str, str]]:
    pairs = _ordered_pairs(raw_items, target_items, "axis_sequence_mismatch")
    mapped = [
        {id_key: _required_string(target, id_key), "label": raw["label"]}
        for raw, target in pairs
    ]
    if len({item[id_key] for item in mapped}) != len(mapped):
        raise CaseFailure("invalid_target_structure")
    return mapped


def _controls_for_image(controls: list[Any], image_id: str) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for control in controls:
        if not isinstance(control, dict) or not isinstance(
            control.get("image_ids"), list
        ):
            raise CaseFailure("invalid_target_structure")
        if image_id in control["image_ids"]:
            eligible.append(control)
    return eligible


def _tables_for_image(tables: list[Any], image_id: str) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for table in tables:
        if not isinstance(table, dict):
            raise CaseFailure("invalid_target_structure")
        if table.get("image_id") == image_id:
            expected.append(table)
    return expected


def _ordered_pairs(
    raw_items: list[dict[str, Any]], target_items: list[Any], failure: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(raw_items) != len(target_items):
        raise CaseFailure(failure)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw, target in zip(raw_items, target_items, strict=True):
        if not isinstance(target, dict) or not _label_matches(raw["label"], target):
            raise CaseFailure(failure)
        pairs.append((raw, target))
    return pairs


def _label_matches(label: str, target: dict[str, Any]) -> bool:
    return _normalize(label) in _normalized_labels(target)


def _normalized_labels(target: dict[str, Any]) -> set[str]:
    aliases = target.get("aliases")
    labels = [target.get("label"), *(aliases if isinstance(aliases, list) else [])]
    if not isinstance(aliases, list) or any(
        not isinstance(item, str) or not _normalize(item) for item in labels
    ):
        raise CaseFailure("invalid_target_structure")
    return {_normalize(item) for item in labels}


def _target_cell_positions(
    target: dict[str, Any], row_ids: list[str], column_ids: list[str]
) -> set[tuple[str, str]]:
    cells = target.get("cells")
    if not isinstance(cells, list):
        raise CaseFailure("invalid_target_structure")
    valid_rows = set(row_ids)
    valid_columns = set(column_ids)
    positions: set[tuple[str, str]] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise CaseFailure("invalid_target_structure")
        position = (
            _required_string(cell, "row_id"),
            _required_string(cell, "column_id"),
        )
        if (
            position[0] not in valid_rows
            or position[1] not in valid_columns
            or position in positions
        ):
            raise CaseFailure("invalid_target_structure")
        positions.add(position)
    return positions


def _target_merged_ranges(
    target: dict[str, Any], row_ids: list[str], column_ids: list[str]
) -> set[tuple[str, str, str, str]]:
    ranges = target.get("merged_ranges")
    if not isinstance(ranges, list):
        raise CaseFailure("invalid_target_structure")
    valid_rows = set(row_ids)
    valid_columns = set(column_ids)
    mapped: set[tuple[str, str, str, str]] = set()
    for item in ranges:
        if not isinstance(item, dict):
            raise CaseFailure("invalid_target_structure")
        position = (
            _required_string(item, "start_row_id"),
            _required_string(item, "end_row_id"),
            _required_string(item, "start_column_id"),
            _required_string(item, "end_column_id"),
        )
        if (
            position[0] not in valid_rows
            or position[1] not in valid_rows
            or position[2] not in valid_columns
            or position[3] not in valid_columns
            or position in mapped
        ):
            raise CaseFailure("invalid_target_structure")
        mapped.add(position)
    return mapped


def _validate_raw(task: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Structured output must be an object")
    if task == "forms":
        _exact_keys(raw, {"controls"})
        controls = _object_list(raw["controls"])
        for control in controls:
            _exact_keys(control, {"label", "state"})
            _nonblank(control["label"])
            if control["state"] not in {"checked", "unchecked", "uncertain"}:
                raise ValueError("Invalid control output")
    elif task == "handwriting":
        _exact_keys(raw, {"text", "legibility", "uncertain_spans"})
        if not isinstance(raw["text"], str) or raw["legibility"] not in {
            "legible",
            "partly_legible",
            "illegible",
        }:
            raise ValueError("Invalid handwriting output")
        spans = raw["uncertain_spans"]
        if not isinstance(spans, list) or any(
            not isinstance(item, str) for item in spans
        ):
            raise ValueError("Invalid uncertain spans")
    else:
        _exact_keys(raw, {"tables"})
        _validate_raw_tables(_object_list(raw["tables"]))
    return raw


def _validate_raw_tables(tables: list[dict[str, Any]]) -> None:
    for table in tables:
        _exact_keys(table, {"label", "rows", "columns", "cells", "merged_ranges"})
        _nonblank(table["label"])
        rows = _object_list(table["rows"])
        columns = _object_list(table["columns"])
        _validate_raw_axes(rows)
        _validate_raw_axes(columns)
        positions: set[tuple[int, int]] = set()
        for cell in _object_list(table["cells"]):
            _exact_keys(cell, {"row_index", "column_index", "text"})
            row = cell["row_index"]
            column = cell["column_index"]
            if (
                not isinstance(cell["text"], str)
                or not _valid_index(row, len(rows))
                or not _valid_index(column, len(columns))
            ):
                raise ValueError("Cell references an unknown axis")
            if (row, column) in positions:
                raise ValueError("Duplicate cell position")
            positions.add((row, column))
        for merged in _object_list(table["merged_ranges"]):
            keys = {
                "start_row_index",
                "end_row_index",
                "start_column_index",
                "end_column_index",
            }
            _exact_keys(merged, keys)
            values = {key: merged[key] for key in keys}
            if (
                not _valid_index(values["start_row_index"], len(rows))
                or not _valid_index(values["end_row_index"], len(rows))
                or not _valid_index(values["start_column_index"], len(columns))
                or not _valid_index(values["end_column_index"], len(columns))
            ):
                raise ValueError("Merged range references an unknown axis")


def _validate_raw_axes(items: list[dict[str, Any]]) -> None:
    for item in items:
        _exact_keys(item, {"label"})
        _nonblank(item["label"])


def _valid_index(value: Any, size: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= size


def _sum_usage(calls: list[dict[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return totals


def _sum_number(calls: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for call in calls
        if isinstance((value := call.get(key)), (int, float))
        and not isinstance(value, bool)
    ]
    return round(sum(values), 6) if values else None


def _read_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read case metadata: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Case metadata must be an object: {path}")
    return raw


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("Expected a list of objects")
    return value


def _exact_keys(raw: Mapping[str, Any], keys: set[str]) -> None:
    if set(raw) != keys:
        raise ValueError("Structured output has invalid fields")


def _nonblank(value: Any) -> str:
    if not isinstance(value, str) or not _normalize(value):
        raise ValueError("Expected non-blank text")
    return value


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _max_tokens(value: str) -> int:
    number = int(value)
    if not 1 <= number <= MAX_TOKENS:
        raise argparse.ArgumentTypeError(
            f"max tokens must be between 1 and {MAX_TOKENS}"
        )
    return number


if __name__ == "__main__":
    raise SystemExit(main())
