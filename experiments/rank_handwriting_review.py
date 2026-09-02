"""Rank unresolved handwriting crops for blind, private manual review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import unicodedata


REVIEW_REASON = "no_conservative_match"
RESULT_FILE = "ranked-review.json"
QUEUE_FILE = "ranked-review.jsonl"
PAGE_QUEUE_FILE = "selected-pages.jsonl"


@dataclass(frozen=True)
class ReviewCase:
    field_id: str
    case_id: str
    family_id: str
    category_id: str
    crop_path: str
    source_path: str | None
    top_match: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = rank_review_queue(
            args.review_queue,
            args.phi4_results,
            args.output,
            limit=args.limit,
            max_per_page=args.max_per_page,
            max_per_family=args.max_per_family,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("phi4_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=_positive_int, default=100)
    parser.add_argument("--max-per-page", type=_positive_int, default=2)
    parser.add_argument("--max-per-family", type=_positive_int, default=8)
    return parser


def rank_review_queue(
    review_queue: Path,
    phi4_results: Path,
    output: Path,
    *,
    limit: int = 100,
    max_per_page: int = 2,
    max_per_family: int = 8,
) -> dict[str, Any]:
    """Join local model evidence and publish a failure-inclusive review queue."""
    _validate_inputs(
        review_queue,
        phi4_results,
        output,
        limit,
        max_per_page,
        max_per_family,
    )
    cases = _load_cases(review_queue)
    phi4_rows = _load_phi4_rows(phi4_results)
    model_cache: dict[Path, tuple[dict[str, Any] | None, str | None]] = {}
    rows = [
        _rank_row(case, phi4_rows.get(case.field_id), review_queue, model_cache)
        for case in cases
    ]
    rows.sort(key=_rank_key)
    _select_rows(
        rows,
        limit=limit,
        max_per_page=max_per_page,
        max_per_family=max_per_family,
    )
    page_queue = _selected_pages(rows)
    summary = _summarize(
        rows,
        selected_source_pages=len(page_queue),
        input_phi4_rows=len(phi4_rows),
        unused_phi4_rows=len(set(phi4_rows) - {case.field_id for case in cases}),
        limit=limit,
        max_per_page=max_per_page,
        max_per_family=max_per_family,
    )
    payload = {
        "status": "complete",
        "privacy": {
            "execution": "private_local_only",
            "model_requests_made": False,
            "ground_truth_in_model_request": False,
            "ground_truth_in_output": False,
        },
        "ranking": {
            "formula": "model_disagreement * low_nemotron_confidence",
            "model_disagreement": "normalized character edit distance",
            "low_nemotron_confidence": "1 - provider confidence",
            "clinical_criticality_used": False,
            "tie_breakers": [
                "model_disagreement descending",
                "low_nemotron_confidence descending",
                "field_id ascending",
            ],
            "unscored_order": "after scored rows, field_id ascending",
        },
        "selection": {
            "limit": limit,
            "max_per_page": max_per_page,
            "max_per_family": max_per_family,
        },
        "inputs": {
            "review_queue": str(review_queue.resolve()),
            "phi4_results": str(phi4_results.resolve()),
        },
        "summary": summary,
        "rows": rows,
    }
    _publish(output, payload, rows, page_queue)
    return summary


def _validate_inputs(
    review_queue: Path,
    phi4_results: Path,
    output: Path,
    limit: int,
    max_per_page: int,
    max_per_family: int,
) -> None:
    if not review_queue.is_file():
        raise FileNotFoundError(f"review queue was not found: {review_queue}")
    if not phi4_results.is_file():
        raise FileNotFoundError(f"Phi-4 results were not found: {phi4_results}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if min(limit, max_per_page, max_per_family) <= 0:
        raise ValueError("selection limits must be positive")


def _load_cases(path: Path) -> list[ReviewCase]:
    cases = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid review JSON on line {line_number}") from error
        if not isinstance(record, dict) or record.get("review_reason") != REVIEW_REASON:
            continue
        case = _review_case(record, path.parent, line_number)
        if case.field_id in seen:
            raise ValueError(f"duplicate field_id: {case.field_id}")
        seen.add(case.field_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"review queue has no {REVIEW_REASON} rows")
    return sorted(cases, key=lambda case: case.field_id)


def _review_case(record: dict[str, Any], root: Path, line_number: int) -> ReviewCase:
    required = ("field_id", "case_id", "family_id", "category_id", "review_crop_path")
    values = {key: record.get(key) for key in required}
    if any(
        not isinstance(value, str) or not value.strip() for value in values.values()
    ):
        raise ValueError(f"invalid review identifiers on line {line_number}")
    crop = Path(values["review_crop_path"])
    crop_path = crop if crop.is_absolute() else root / crop
    if not crop_path.is_file():
        raise FileNotFoundError(f"review crop was not found on line {line_number}")
    top_matches = record.get("top_matches")
    if not isinstance(top_matches, list) or not top_matches:
        raise ValueError(f"review row has no top match on line {line_number}")
    top_match = top_matches[0]
    if not isinstance(top_match, dict):
        raise ValueError(f"invalid top match on line {line_number}")
    source_value = record.get("source_path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(f"review row has no source path on line {line_number}")
    source = Path(source_value)
    source_path = source if source.is_absolute() else root / source
    if not source_path.is_file():
        raise FileNotFoundError(f"source image was not found on line {line_number}")
    return ReviewCase(
        field_id=values["field_id"],
        case_id=values["case_id"],
        family_id=values["family_id"],
        category_id=values["category_id"],
        crop_path=str(crop_path.resolve()),
        source_path=str(source_path.resolve()),
        top_match=top_match,
    )


def _load_phi4_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Phi-4 results have no rows: {path}")
    indexed = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"invalid Phi-4 row {index}")
        field_id = row.get("field_id")
        if not isinstance(field_id, str) or not field_id:
            raise ValueError(f"Phi-4 row {index} has no field_id")
        if field_id in indexed:
            raise ValueError(f"duplicate Phi-4 field_id: {field_id}")
        indexed[field_id] = row
    return indexed


def _rank_row(
    case: ReviewCase,
    phi4_row: dict[str, Any] | None,
    review_queue: Path,
    model_cache: dict[Path, tuple[dict[str, Any] | None, str | None]],
) -> dict[str, Any]:
    phi4 = _phi4_evidence(phi4_row)
    nemotron = _nemotron_evidence(case.top_match, review_queue.parent, model_cache)
    disagreement = None
    if phi4["status"] == "success" and nemotron["status"] == "success":
        disagreement = normalized_edit_distance(
            nemotron["candidate_text"], phi4["prediction"]
        )
    low_confidence = None
    if nemotron["confidence"] is not None:
        low_confidence = round(1.0 - nemotron["confidence"], 6)
    priority = None
    if disagreement is not None and low_confidence is not None:
        priority = round(disagreement * low_confidence, 6)
    return {
        "field_id": case.field_id,
        "case_id": case.case_id,
        "family_id": case.family_id,
        "category_id": case.category_id,
        "review_crop_path": case.crop_path,
        "source_path": case.source_path,
        "priority_score": priority,
        "ranking_components": {
            "model_disagreement": disagreement,
            "low_nemotron_confidence": low_confidence,
        },
        "phi4": phi4,
        "nemotron": nemotron,
        "annotation": {
            "target_state": None,
            "corrected_text": None,
            "reviewer_id": None,
        },
    }


def _phi4_evidence(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "status": "missing",
            "prediction": "",
            "error_type": "missing_result",
            "error": "No Phi-4 result was recorded for this field.",
        }
    status = row.get("status")
    prediction = row.get("prediction")
    if status == "success" and isinstance(prediction, str):
        return {
            "status": "success",
            "prediction": prediction,
            "error_type": None,
            "error": None,
        }
    return {
        "status": "failed",
        "prediction": prediction if isinstance(prediction, str) else "",
        "error_type": row.get("error_type") or "invalid_result",
        "error": row.get("error") or "Phi-4 did not produce a valid prediction.",
    }


def _nemotron_evidence(
    top_match: dict[str, Any],
    root: Path,
    cache: dict[Path, tuple[dict[str, Any] | None, str | None]],
) -> dict[str, Any]:
    path_value = top_match.get("model_output_path")
    region_ids = top_match.get("region_ids")
    if not isinstance(path_value, str) or not path_value:
        return _nemotron_failure("missing_model_output_path")
    if (
        not isinstance(region_ids, list)
        or not region_ids
        or not all(isinstance(region_id, str) and region_id for region_id in region_ids)
    ):
        return _nemotron_failure("invalid_region_ids", model_output_path=path_value)
    path = Path(path_value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if path not in cache:
        cache[path] = _try_read_object(path)
    payload, load_error = cache[path]
    if payload is None:
        return _nemotron_failure(
            load_error or "invalid_model_output", model_output_path=str(path)
        )
    regions = _regions_by_id(payload)
    selected = [regions.get(region_id) for region_id in region_ids]
    if any(region is None for region in selected):
        return _nemotron_failure("region_not_found", model_output_path=str(path))
    typed_regions = [region for region in selected if region is not None]
    texts = [region.get("text") for region in typed_regions]
    if not all(isinstance(text, str) for text in texts):
        return _nemotron_failure("invalid_region_text", model_output_path=str(path))
    confidences = [region.get("confidence") for region in typed_regions]
    if any(not _valid_confidence(confidence) for confidence in confidences):
        return _nemotron_failure(
            "invalid_region_confidence", model_output_path=str(path)
        )
    numeric_confidences = [
        float(confidence) for confidence in confidences if confidence is not None
    ]
    confidence = None
    if numeric_confidences:
        confidence = round(sum(numeric_confidences) / len(numeric_confidences), 6)
    return {
        "status": "success",
        "candidate_text": " ".join(texts),
        "confidence": confidence,
        "confidence_aggregation": "mean",
        "provider": typed_regions[0].get("provider"),
        "region_ids": region_ids,
        "model_output_path": str(path),
        "error": None,
    }


def _try_read_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _read_object(path), None
    except (OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _regions_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = payload.get("result")
    pages = result.get("pages") if isinstance(result, dict) else None
    if not isinstance(pages, list):
        return {}
    regions = {}
    for page in pages:
        page_regions = page.get("regions") if isinstance(page, dict) else None
        if not isinstance(page_regions, list):
            continue
        for region in page_regions:
            if not isinstance(region, dict):
                continue
            region_id = region.get("id")
            if isinstance(region_id, str) and region_id:
                regions[region_id] = region
    return regions


def _nemotron_failure(
    error: str, *, model_output_path: str | None = None
) -> dict[str, Any]:
    return {
        "status": "failed",
        "candidate_text": "",
        "confidence": None,
        "confidence_aggregation": "mean",
        "provider": None,
        "region_ids": [],
        "model_output_path": model_output_path,
        "error": error,
    }


def normalized_edit_distance(left: str, right: str) -> float:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    denominator = max(len(left_normalized), len(right_normalized))
    if denominator == 0:
        return 0.0
    return round(_edit_distance(left_normalized, right_normalized) / denominator, 6)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _valid_confidence(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    priority = row["priority_score"]
    components = row["ranking_components"]
    return (
        priority is None,
        -(priority or 0.0),
        -(components["model_disagreement"] or 0.0),
        -(components["low_nemotron_confidence"] or 0.0),
        row["field_id"],
    )


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    max_per_page: int,
    max_per_family: int,
) -> None:
    page_counts: defaultdict[str, int] = defaultdict(int)
    family_counts: defaultdict[str, int] = defaultdict(int)
    selected = 0
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        reasons = []
        if selected >= limit:
            reasons.append("selection_limit")
        if page_counts[row["case_id"]] >= max_per_page:
            reasons.append("page_cap")
        if family_counts[row["family_id"]] >= max_per_family:
            reasons.append("family_cap")
        row["selected"] = not reasons
        row["selection_reasons"] = reasons
        row["selected_rank"] = None
        if reasons:
            continue
        selected += 1
        page_counts[row["case_id"]] += 1
        family_counts[row["family_id"]] += 1
        row["selected_rank"] = selected


def _summarize(
    rows: list[dict[str, Any]],
    *,
    selected_source_pages: int,
    input_phi4_rows: int,
    unused_phi4_rows: int,
    limit: int,
    max_per_page: int,
    max_per_family: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["selected"]]
    reasons = Counter(reason for row in rows for reason in row["selection_reasons"])
    return {
        "unresolved_fields": len(rows),
        "selected_fields": len(selected),
        "selected_pages": len({row["case_id"] for row in selected}),
        "selected_source_pages": selected_source_pages,
        "selected_families": len({row["family_id"] for row in selected}),
        "scored_fields": sum(row["priority_score"] is not None for row in rows),
        "unscored_fields": sum(row["priority_score"] is None for row in rows),
        "phi4_input_rows": input_phi4_rows,
        "phi4_unused_rows": unused_phi4_rows,
        "phi4_failures_or_missing": sum(
            row["phi4"]["status"] != "success" for row in rows
        ),
        "nemotron_failures": sum(
            row["nemotron"]["status"] != "success" for row in rows
        ),
        "selection_caps": {
            "limit": limit,
            "max_per_page": max_per_page,
            "max_per_family": max_per_family,
        },
        "not_selected_reasons": dict(sorted(reasons.items())),
        "failures_remain_in_denominator": True,
    }


def _selected_pages(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    paths: dict[str, str] = {}
    for row in rows:
        if not row["selected"]:
            continue
        page_id = row["case_id"]
        image_path = row["source_path"]
        previous = paths.get(page_id)
        if previous is not None and previous != image_path:
            raise ValueError(f"selected page has conflicting source paths: {page_id}")
        if previous is not None:
            continue
        paths[page_id] = image_path
        pages.append({"page_id": page_id, "image_path": image_path})
    return pages


def _publish(
    output: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    page_queue: list[dict[str, str]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / RESULT_FILE).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / QUEUE_FILE).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        (temporary / PAGE_QUEUE_FILE).write_text(
            "".join(json.dumps(page, sort_keys=True) + "\n" for page in page_queue),
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
