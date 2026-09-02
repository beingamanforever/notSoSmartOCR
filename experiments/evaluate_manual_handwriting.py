"""Evaluate reviewed handwriting boxes against local OCR run payloads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import unicodedata

from ocr_pipeline.verification import edit_counts, normalized_edit_distance


ELIGIBLE_REGION_TYPES = frozenset({"field", "line"})


@dataclass(frozen=True)
class Label:
    page_id: str
    box: tuple[float, float, float, float]
    text: str


@dataclass(frozen=True)
class Region:
    kind: str
    text: str
    provider: str
    box: tuple[float, float, float, float]
    order: tuple[float | str, ...]
    rendered_order: tuple[int, int] | None
    resolution: str


@dataclass(frozen=True)
class Run:
    path: Path
    state: str
    payload: dict[str, Any] | None
    regions: tuple[Region, ...] = ()


@dataclass(frozen=True)
class Candidate:
    text: str
    provider: str
    order: tuple[float | str, ...]
    rendered_order: tuple[int, int] | None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_manual_handwriting(
        args.labels,
        args.model_output,
        label_scale=args.label_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def evaluate_manual_handwriting(
    labels_path: Path,
    model_output_root: Path,
    *,
    label_scale: float = 1.0,
) -> dict[str, Any]:
    if not _number(label_scale) or label_scale <= 0:
        raise ValueError("label_scale must be positive and finite")
    label_scale = float(label_scale)
    labels, reviewed, excluded = _load_labels(labels_path, label_scale)
    runs = _load_runs(model_output_root)
    providers = sorted(
        {
            region.provider
            for run in runs.values()
            if run is not None
            for region in run.regions
        }
    )

    scored: list[tuple[str, str]] = []
    scored_by_provider: dict[str, list[tuple[str, str]]] = {
        provider: [] for provider in providers
    }
    proposed = 0
    proposed_by_provider: Counter[str] = Counter()
    failure_labels: Counter[str] = Counter()
    page_states: dict[str, str] = {}
    used_runs: dict[Path, Run] = {}

    for label in labels:
        run = runs.get(label.page_id)
        state = run.state if run is not None else "missing_run"
        page_states[label.page_id] = state
        if run is not None:
            used_runs[run.path] = run
        if state != "success":
            failure_labels[state] += 1
            proposals: list[Region] = []
        else:
            proposals = [
                region for region in run.regions if _proposal(label.box, region.box)
            ]

        if proposals:
            proposed += 1
        prediction = _primary_prediction(label, proposals)
        scored.append((label.text, prediction))
        for provider in providers:
            provider_proposals = [
                region for region in proposals if region.provider == provider
            ]
            if provider_proposals:
                proposed_by_provider[provider] += 1
            scored_by_provider[provider].append(
                (label.text, _provider_prediction(label, provider_proposals))
            )

    failures = {
        state: {
            "pages": sum(value == state for value in page_states.values()),
            "labels": failure_labels[state],
        }
        for state in ("missing_run", "invalid_run", "failed_request", "failed_pipeline")
    }
    return {
        "metadata": {"label_scale": label_scale},
        "labels": {
            "reviewed": reviewed,
            "eligible": len(labels),
            "excluded": dict(sorted(excluded.items())),
        },
        "proposal_recall": _ratio(proposed, len(labels)),
        "recognition": _recognition_metrics(scored),
        "providers": {
            provider: {
                "proposal_recall": _ratio(proposed_by_provider[provider], len(labels)),
                "recognition": _recognition_metrics(scored_by_provider[provider]),
            }
            for provider in providers
        },
        "run_failures": failures,
        "latency_seconds": _latency_metrics(list(used_runs.values())),
        "definitions": {
            "eligibility": "legibility is legible and region_type is field or line",
            "label_geometry": "label box coordinates multiplied by label_scale",
            "proposal": (
                "a text or word region has positive box overlap with the label, "
                "or its center is inside the label box"
            ),
            "prediction": (
                "resolved rendered evidence in page evidence order; otherwise the "
                "earliest deterministic provider prediction"
            ),
            "provider_prediction": (
                "reading-order join of resolved words centered in the label; otherwise "
                "the resolved region with greatest label coverage then tightness"
            ),
            "exact_match": "NFKC, casefolded, whitespace-normalized equality",
            "cer": "total character insertions, deletions, and substitutions / reference characters",
            "hallucinated_text_rate": "extra prediction characters / reference characters",
            "normalized_edit_distance": "mean character edit distance / longer string length",
            "missed_text_rate": "character deletions / reference characters",
            "substitution_rate": "character substitutions / reference characters",
            "failure_inclusive": "every eligible label remains in every recognition denominator",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-scale", type=float, default=1.0)
    return parser


def _load_labels(
    path: Path,
    label_scale: float,
) -> tuple[list[Label], int, Counter[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"reviewed labels were not found: {path}")
    labels: list[Label] = []
    reviewed = 0
    excluded: Counter[str] = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid label JSON on line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"label line {line_number} must be an object")
        page_id = row.get("page_id")
        if not isinstance(page_id, str) or not page_id.strip():
            raise ValueError(f"label line {line_number} lacks a page_id")
        raw_regions = row.get("regions")
        regions = raw_regions if isinstance(raw_regions, list) else [row]
        for raw_region in regions:
            reviewed += 1
            if not isinstance(raw_region, dict):
                raise ValueError(
                    f"label line {line_number} contains a non-object region"
                )
            legibility = raw_region.get("legibility", row.get("legibility"))
            if legibility != "legible":
                excluded["not_legible"] += 1
                continue
            region_type = raw_region.get("region_type", row.get("region_type"))
            if region_type not in ELIGIBLE_REGION_TYPES:
                excluded["unsupported_region_type"] += 1
                continue
            text = raw_region.get("text", raw_region.get("transcription"))
            box_value = raw_region.get("bbox", raw_region.get("crop_bbox"))
            if not isinstance(text, str) or not _normalize(text):
                raise ValueError(f"eligible label on line {line_number} lacks text")
            box = _box(box_value, line_number)
            scaled_box = tuple(coordinate * label_scale for coordinate in box)
            labels.append(Label(page_id.strip(), scaled_box, text))
    if not labels:
        raise ValueError("reviewed labels contain no eligible handwriting")
    return labels, reviewed, excluded


def _load_runs(root: Path) -> dict[str, Run | None]:
    if not root.is_dir():
        raise FileNotFoundError(f"model output root was not found: {root}")
    runs: dict[str, Run | None] = {}
    for path in sorted(root.rglob("*.json")):
        run = _load_run(path)
        aliases = {path.stem}
        if run.payload is not None:
            payload = run.payload
            for value in (
                payload.get("case_id"),
                _filename_stem(payload.get("filename")),
                _mapping(payload.get("result")).get("document_id"),
            ):
                if isinstance(value, str) and value.strip():
                    aliases.add(value.strip())
        for alias in aliases:
            runs[alias] = run if alias not in runs else Run(path, "invalid_run", None)
    return runs


def _load_run(path: Path) -> Run:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return Run(path, "invalid_run", None)
    if not isinstance(payload, dict):
        return Run(path, "invalid_run", None)
    if payload.get("request_status") == "failed":
        return Run(path, "failed_request", payload)
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("pages"), list):
        return Run(path, "invalid_run", payload)
    if result.get("status") not in (None, "success"):
        return Run(path, "failed_pipeline", payload)
    return Run(path, "success", payload, tuple(_regions(result["pages"])))


def _regions(pages: list[object]) -> list[Region]:
    regions = []
    for page_index, raw_page in enumerate(pages):
        page = _mapping(raw_page)
        page_text = _mapping(page.get("text"))
        evidence_ids = page_text.get("evidence_ids")
        evidence_order = (
            {
                region_id: (page_index, index)
                for index, region_id in enumerate(evidence_ids)
                if isinstance(region_id, str)
            }
            if isinstance(evidence_ids, list)
            else {}
        )
        raw_regions = page.get("regions")
        if not isinstance(raw_regions, list):
            continue
        for region_index, raw_region in enumerate(raw_regions):
            region = _mapping(raw_region)
            if region.get("kind") not in {"text", "word"}:
                continue
            text = region.get("text")
            box = _optional_box(region.get("bounding_box"))
            if box is None:
                continue
            text = text if isinstance(text, str) else ""
            provider = region.get("provider")
            provider = provider.strip() if isinstance(provider, str) else "unknown"
            provider = provider or "unknown"
            reading_order = region.get("reading_order")
            if _number(reading_order):
                order: tuple[float | str, ...] = (
                    float(page_index),
                    0.0,
                    float(reading_order),
                    float(region_index),
                )
            else:
                order = (
                    float(page_index),
                    1.0,
                    box[1],
                    box[0],
                    box[3],
                    box[2],
                    str(region.get("id", "")),
                    float(region_index),
                )
            region_id = region.get("id")
            rendered_order = evidence_order.get(region_id)
            resolution = region.get("resolution", "resolved")
            regions.append(
                Region(
                    region["kind"],
                    text,
                    provider,
                    box,
                    order,
                    rendered_order,
                    resolution if isinstance(resolution, str) else "",
                )
            )
    return regions


def _primary_prediction(label: Label, proposals: Sequence[Region]) -> str:
    rendered = [region for region in proposals if region.rendered_order is not None]
    candidates = _provider_predictions(label, rendered)
    if candidates:
        return min(
            candidates,
            key=lambda candidate: candidate.rendered_order or (math.inf, math.inf),
        ).text
    candidates = _provider_predictions(label, proposals)
    if not candidates:
        return ""
    return min(
        candidates, key=lambda candidate: (candidate.order, candidate.provider)
    ).text


def _provider_predictions(label: Label, proposals: Sequence[Region]) -> list[Candidate]:
    by_provider: defaultdict[str, list[Region]] = defaultdict(list)
    for region in proposals:
        if region.resolution == "resolved":
            by_provider[region.provider].append(region)
    return [
        _select_provider_prediction(label, provider, regions)
        for provider, regions in sorted(by_provider.items())
    ]


def _provider_prediction(label: Label, proposals: Sequence[Region]) -> str:
    candidates = _provider_predictions(label, proposals)
    return candidates[0].text if candidates else ""


def _select_provider_prediction(
    label: Label,
    provider: str,
    regions: Sequence[Region],
) -> Candidate:
    centered_words = [
        region
        for region in regions
        if region.kind == "word" and _center_inside(region.box, label.box)
    ]
    if centered_words:
        ordered = sorted(centered_words, key=lambda region: region.order)
        rendered = [
            region.rendered_order
            for region in centered_words
            if region.rendered_order is not None
        ]
        return Candidate(
            " ".join(region.text.strip() for region in ordered),
            provider,
            ordered[0].order,
            min(rendered) if rendered else None,
        )

    selected = min(
        regions,
        key=lambda region: (
            -_label_coverage(label.box, region.box),
            -_tightness(label.box, region.box),
            region.order,
        ),
    )
    return Candidate(
        selected.text,
        provider,
        selected.order,
        selected.rendered_order,
    )


def _recognition_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    exact = edits = insertions = deletions = substitutions = reference_characters = 0
    distances = []
    for reference, prediction in pairs:
        normalized_reference = _normalize(reference)
        normalized_prediction = _normalize(prediction)
        exact += normalized_prediction == normalized_reference
        counts = edit_counts(normalized_prediction, normalized_reference)
        edits += counts.edits
        insertions += counts.insertions
        deletions += counts.deletions
        substitutions += counts.substitutions
        reference_characters += len(normalized_reference)
        distances.append(
            normalized_edit_distance(normalized_prediction, normalized_reference)
        )
    return {
        "count": len(pairs),
        "exact_match": _ratio(exact, len(pairs)),
        "cer": _rate(edits, reference_characters),
        "hallucinated_text_rate": _rate(insertions, reference_characters),
        "normalized_edit_distance": _mean(distances),
        "missed_text_rate": _rate(deletions, reference_characters),
        "substitution_rate": _rate(substitutions, reference_characters),
    }


def _latency_metrics(runs: Sequence[Run]) -> dict[str, float | int | None]:
    values = []
    invalid = 0
    for run in runs:
        if run.payload is None:
            continue
        timing = _mapping(run.payload.get("timing"))
        value = timing.get("total_seconds", run.payload.get("elapsed_seconds"))
        if _number(value) and float(value) >= 0:
            values.append(float(value))
        elif value is not None:
            invalid += 1
    return {
        "count": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "invalid": invalid,
    }


def _box(value: object, line_number: int) -> tuple[float, float, float, float]:
    box = _optional_box(value)
    if box is None:
        raise ValueError(f"eligible label on line {line_number} has an invalid bbox")
    return box


def _optional_box(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        values = [value.get(key) for key in ("left", "top", "right", "bottom")]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        values = list(value)
    else:
        return None
    if not all(_number(item) for item in values):
        return None
    left, top, right, bottom = (float(item) for item in values)
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _proposal(
    reference: tuple[float, float, float, float],
    prediction: tuple[float, float, float, float],
) -> bool:
    return _intersection_area(reference, prediction) > 0 or _center_inside(
        prediction, reference
    )


def _label_coverage(
    label: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    return _intersection_area(label, region) / _area(label)


def _tightness(
    label: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    return _intersection_area(label, region) / _area(region)


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _center_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _filename_stem(value: object) -> str | None:
    return Path(value).stem if isinstance(value, str) else None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _ratio(numerator: int, denominator: int) -> dict[str, float | int | None]:
    return {
        "matched": numerator,
        "total": denominator,
        "rate": _rate(numerator, denominator),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


if __name__ == "__main__":
    raise SystemExit(main())
