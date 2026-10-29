"""Feedback log, reviewer hints, confidence thresholds and case metrics.

root/feedback.jsonl holds one line per reviewer correction or confirmation (written by
case.py's review actions). metrics() reads every case.json and audit.jsonl directly and
never imports case.py, so the two modules can't form an import cycle.
"""

from __future__ import annotations

import json
import statistics
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .schema import FIELDS_BY_KEY, Case

_FEEDBACK_LOCK = threading.Lock()  # ponytail: one global lock, per-root locks if throughput matters


def record_feedback(root, entry: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with _FEEDBACK_LOCK:
        with open(Path(root) / "feedback.jsonl", "a") as handle:
            handle.write(json.dumps(entry) + "\n")


def _read_feedback(root) -> list[dict]:
    path = Path(root) / "feedback.jsonl"
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reviewer_hints(root, per_key: int = 2, limit: int = 12) -> list[str]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for entry in _read_feedback(root):
        # Only a real correction teaches anything: a rejection has no right value, and an
        # edit back to the same text corrects nothing.
        final, extracted = entry.get("final"), entry.get("extracted")
        if entry.get("correct") is False and final not in (None, "") and str(final) != str(extracted):
            by_key[entry["key"]].append(entry)
    latest: list[dict] = []
    for entries in by_key.values():
        entries.sort(key=lambda entry: entry["ts"], reverse=True)
        latest += entries[:per_key]
    latest.sort(key=lambda entry: entry["ts"], reverse=True)
    return [_hint_sentence(entry) for entry in latest[:limit]]


def _hint_sentence(entry: dict) -> str:
    spec = FIELDS_BY_KEY.get(entry["key"])
    label = spec.label if spec else entry["key"].replace("_", " ").replace(".", " ").title()
    return f"On a {entry.get('document_type')}, '{label}' printed as '{entry.get('raw_value')}' is '{entry.get('final')}'."


def confidence_thresholds(
    root, min_labels: int = 20, default: float = 0.80, target: float = 0.95
) -> dict[str, float]:
    labeled = [
        entry
        for entry in _read_feedback(root)
        if entry.get("confidence") is not None and isinstance(entry.get("correct"), bool)
    ]
    by_key: dict[str, list[dict]] = defaultdict(list)
    for entry in labeled:
        by_key[entry["key"]].append(entry)
    # ponytail: one global threshold until a field has its own 20 labels; per document
    # type thresholds once volume allows.
    global_threshold = _lowest_confident_threshold(labeled, target, default) if len(labeled) >= min_labels else default
    return {
        key: _lowest_confident_threshold(entries, target, default) if len(entries) >= min_labels else global_threshold
        for key, entries in by_key.items()
    }


def _lowest_confident_threshold(entries: list[dict], target: float, default: float) -> float:
    """Lowest observed confidence at or above which accuracy-so-far is still >= target."""
    ordered = sorted(entries, key=lambda entry: entry["confidence"], reverse=True)
    best, correct = default, 0
    for count, entry in enumerate(ordered, 1):
        correct += entry["correct"]
        if correct / count >= target:
            best = entry["confidence"]
    return best


_REVIEW_ACTIONS = {
    "edit", "add", "change_source", "approve", "reject", "bulk_approve",
    "reject_case", "resolve_exception",
}
_EDIT_ACTIONS = {"edit", "add", "change_source"}


def metrics(root) -> dict:
    cases, audit_by_case = _load_cases(root)
    documents = [document for case in cases for document in case.documents]
    values = [value for case in cases for value in case.values]
    reviewed = [value for value in values if value.review.state != "pending"]
    audit_samples = [value for value in reviewed if value.audit_sample]
    touches = sum(
        1 for events in audit_by_case.values() for event in events if event.get("action") in _EDIT_ACTIONS
    )

    histogram = [0] * 10
    for value in values:
        if value.confidence is not None:
            histogram[min(9, int(value.confidence * 10))] += 1

    routes = {"quick_verify": 0, "priority_review": 0, "exception": 0}
    for value in values:
        routes[value.route] = routes.get(value.route, 0) + 1

    volume_by_day: dict[str, int] = defaultdict(int)
    for document in documents:
        volume_by_day[document.intake_at[:10]] += 1

    stage_values: dict[str, list[float]] = defaultdict(list)
    for document in documents:
        for stage, seconds in document.timings.items():
            stage_values[stage].append(seconds)
    stage_seconds = {stage: statistics.median(seconds) for stage, seconds in stage_values.items()}

    return {
        "cases": len(cases),
        "submitted": sum(1 for case in cases if case.status == "submitted"),
        "documents": len(documents),
        "pages": sum(document.page_count or 0 for document in documents),
        "exception_rate": _exception_rate(cases),
        "manual_touch_rate": touches / len(values) if values else 0.0,
        "field_accuracy": _accuracy(reviewed),
        "audit_accuracy": _accuracy(audit_samples),
        "confidence_histogram": histogram,
        "routes": routes,
        "volume_by_day": dict(sorted(volume_by_day.items())),
        "stage_seconds": stage_seconds,
        "by_case": [_case_summary(case, audit_by_case.get(case.case_id, [])) for case in cases],
    }


def _accuracy(reviewed_values: list) -> float | None:
    if not reviewed_values:
        return None  # nothing reviewed yet: no accuracy to report
    return sum(1 for value in reviewed_values if value.review.state == "approved") / len(reviewed_values)


def _exception_rate(cases: list[Case]) -> float:
    """Cases with an open error exception now (submitted cases never have one: submit blocks on it)."""
    if not cases:
        return 0.0
    with_open_error = sum(
        1 for case in cases if any(e.severity == "error" and e.state == "open" for e in case.exceptions)
    )
    return with_open_error / len(cases)


def _case_summary(case: Case, audit: list[dict]) -> dict:
    review_events = [event for event in audit if event.get("action") in _REVIEW_ACTIONS]
    first_review_ts = min((event["ts"] for event in review_events), default=None)
    touches = sum(1 for event in audit if event.get("action") in _EDIT_ACTIONS)
    return {
        "case_id": case.case_id,
        "ticket_id": case.ticket_id,
        "created_at": case.created_at,
        "submitted_at": case.submitted_at,
        "minutes": _minutes(case.created_at, case.submitted_at),
        "review_minutes": _minutes(first_review_ts, case.submitted_at),
        "touches": touches,
        "values": len(case.values),
        "exceptions": len(case.exceptions),
    }


def _minutes(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60, 2)


def _load_cases(root) -> tuple[list[Case], dict[str, list[dict]]]:
    cases_dir = Path(root) / "cases"
    if not cases_dir.exists():
        return [], {}
    cases = []
    audit_by_case: dict[str, list[dict]] = {}
    for case_path in cases_dir.glob("*/case.json"):
        case = Case.model_validate_json(case_path.read_text())
        cases.append(case)
        audit_path = case_path.parent / "audit.jsonl"
        if audit_path.exists():
            with open(audit_path) as handle:
                audit_by_case[case.case_id] = [json.loads(line) for line in handle if line.strip()]
        else:
            audit_by_case[case.case_id] = []
    cases.sort(key=lambda case: case.created_at)
    return cases, audit_by_case
