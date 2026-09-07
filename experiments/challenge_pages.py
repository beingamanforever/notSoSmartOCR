"""Frozen challenge-set page loading shared by page-level benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORY_SOURCES = {
    "C08": "C08-handwritten",
    "C14": "C14-user-reported",
}
EXPECTED_PAGE_COUNTS = {"C08": 20, "C14": 2}


@dataclass(frozen=True)
class PageCase:
    id: str
    source: Path
    source_file: str
    reference: str
    reference_scope: str
    challenges: tuple[str, ...]
    unresolved_spans: tuple[str, ...]
    handwriting: tuple[tuple[str, str], ...]


def load_page_cases(
    challenge_root: Path,
    track: str,
    triage_ids: tuple[str, ...],
) -> list[PageCase]:
    if track == "triage":
        return [_load_page_case(challenge_root, case_id) for case_id in triage_ids]

    pages: list[PageCase] = []
    for category in ("C14", "C08"):
        annotation_root = challenge_root / "annotations" / "primary" / category
        paths = sorted(annotation_root.glob("*.json"))
        expected = EXPECTED_PAGE_COUNTS[category]
        if len(paths) != expected:
            raise ValueError(
                f"expected {expected} frozen {category} pages, found {len(paths)}"
            )
        pages.extend(
            _load_page_case(challenge_root, path.stem, annotation_path=path)
            for path in paths
        )
    return pages


def _load_page_case(
    challenge_root: Path,
    case_id: str,
    *,
    annotation_path: Path | None = None,
) -> PageCase:
    category = case_id.split("-", 1)[0]
    source_dir = CATEGORY_SOURCES.get(category)
    if source_dir is None:
        raise ValueError(f"unsupported frozen category: {category}")
    annotation_path = annotation_path or (
        challenge_root / "annotations" / "primary" / category / f"{case_id}.json"
    )
    annotation = _read_json(annotation_path)
    if not isinstance(annotation, dict) or annotation.get("case_id") != case_id:
        raise ValueError(f"invalid annotation for {case_id}")
    if annotation.get("source_only") is not True:
        raise ValueError(f"annotation is not source-only: {case_id}")
    transcription = annotation.get("transcription")
    if not isinstance(transcription, dict):
        raise ValueError(f"missing transcription for {case_id}")
    reference = transcription.get("reading_order_text")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"missing reading-order text for {case_id}")
    unresolved = transcription.get("unresolved_spans") or []
    challenges = annotation.get("challenges") or []
    handwriting = annotation.get("handwriting") or []
    if not all(
        isinstance(value, list) for value in (unresolved, challenges, handwriting)
    ):
        raise ValueError(f"invalid annotation lists for {case_id}")
    source = challenge_root / "sources" / source_dir / f"{case_id}.png"
    if not source.is_file():
        raise FileNotFoundError(f"frozen source was not found: {source}")
    scope = (
        "complete"
        if annotation.get("page_legibility") == "complete" and not unresolved
        else "partial"
    )
    return PageCase(
        id=case_id,
        source=source,
        source_file=source.relative_to(challenge_root).as_posix(),
        reference=reference,
        reference_scope=scope,
        challenges=tuple(str(value) for value in challenges),
        unresolved_spans=tuple(str(value) for value in unresolved),
        handwriting=tuple(
            (str(item.get("text", "")), str(item.get("legibility", "unknown")))
            for item in handwriting
            if isinstance(item, dict)
        ),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error
