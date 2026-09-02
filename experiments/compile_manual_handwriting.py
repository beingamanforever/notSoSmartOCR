"""Compile two independent box reviews into handwriting training crops."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError

from experiments.annotate_handwriting_pages import NEGATIVE_TYPES
from experiments.compile_handwriting_crops import normalize_text


LEGIBILITY = {"legible", "ambiguous", "unreadable"}
TRAINING_TYPES = {"field", "line"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    summary = compile_manual_handwriting(
        args.pages,
        args.primary,
        args.secondary,
        args.output,
        minimum_iou=args.minimum_iou,
        minimum_containment_overlap=args.minimum_containment_overlap,
        primary_scale=args.primary_scale,
        secondary_scale=args.secondary_scale,
        exclude_pages=args.exclude_page,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def compile_manual_handwriting(
    page_root: Path,
    primary_path: Path,
    secondary_path: Path,
    output_root: Path,
    *,
    minimum_iou: float = 0.5,
    minimum_containment_overlap: float = 0.8,
    primary_scale: float = 1.0,
    secondary_scale: float = 1.0,
    exclude_pages: Sequence[str] = (),
) -> dict[str, Any]:
    if not page_root.is_dir():
        raise FileNotFoundError(f"page root was not found: {page_root}")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not 0.0 < minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be between zero and one")
    if not 0.0 < minimum_containment_overlap <= 1.0:
        raise ValueError("minimum_containment_overlap must be between zero and one")
    if primary_scale <= 0.0 or secondary_scale <= 0.0:
        raise ValueError("coordinate scales must be positive")

    excluded = frozenset(exclude_pages)
    primary = [
        row
        for row in _load_review(primary_path, coordinate_scale=primary_scale)
        if row["page_id"] not in excluded
    ]
    secondary = [
        row
        for row in _load_review(secondary_path, coordinate_scale=secondary_scale)
        if row["page_id"] not in excluded
    ]
    pages = _index_pages(page_root)
    primary_reviewers = {row["reviewer_id"] for row in primary}
    secondary_reviewers = {row["reviewer_id"] for row in secondary}
    if primary_reviewers & secondary_reviewers:
        raise ValueError("reviews must use an independent reviewer")

    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    used_secondary: set[int] = set()
    field_counts: Counter[str] = Counter()
    for first in primary:
        match_index, iou, containment_overlap = _best_match(
            first,
            secondary,
            used_secondary,
            minimum_iou,
            minimum_containment_overlap,
        )
        if match_index is None:
            review.append(_review_record(first, "independent_region_missing"))
            continue
        used_secondary.add(match_index)
        second = secondary[match_index]
        region_type = first["region_type"]
        if region_type not in TRAINING_TYPES | NEGATIVE_TYPES:
            review.append(
                _review_record(
                    first,
                    "non_training_region",
                    second,
                    iou,
                    containment_overlap,
                )
            )
            continue
        if region_type in NEGATIVE_TYPES:
            target_state = "absent"
            transcription = ""
        elif first["legibility"] == second["legibility"] == "unreadable":
            target_state = "unreadable"
            transcription = ""
        elif first["legibility"] != "legible" or second["legibility"] != "legible":
            review.append(
                _review_record(
                    first,
                    "legibility_not_legible",
                    second,
                    iou,
                    containment_overlap,
                )
            )
            continue
        elif normalize_text(first["transcription"]) != normalize_text(
            second["transcription"]
        ):
            review.append(
                _review_record(
                    first,
                    "transcription_disagreement",
                    second,
                    iou,
                    containment_overlap,
                )
            )
            continue
        else:
            target_state = "resolved"
            transcription = first["transcription"].strip()
        page_id = first["page_id"]
        page_path = pages.get(page_id)
        if page_path is None:
            raise FileNotFoundError(f"reviewed page was not found: {page_id}")
        image_size = _image_size(page_path)
        crop_bbox = _union_box(first["bbox"], second["bbox"])
        _validate_box(crop_bbox, image_size, f"agreed bbox for {page_id}")
        padded_bbox = _padded_box(crop_bbox, image_size)
        field_counts[page_id] += 1
        field_id = f"{page_id}-HM{field_counts[page_id]:03d}"
        record = {
            "field_id": field_id,
            "page_id": page_id,
            "source_component_id": _source_component(page_id),
            "source_path": str(page_path.resolve()),
            "crop_bbox": crop_bbox,
            "padded_bbox": padded_bbox,
            "crop_path": f"crops/{field_id}.png",
            "tight_crop_path": f"crops/{field_id}.png",
            "padded_crop_path": f"crops/padded/{field_id}.png",
            "transcription": transcription,
            "reference": transcription,
            "target_state": target_state,
            "legibility": first["legibility"],
            "region_type": region_type,
            "reviewer_ids": sorted({first["reviewer_id"], second["reviewer_id"]}),
            "independent_reviews": [
                _review_evidence(first),
                _review_evidence(second),
            ],
            "review_iou": round(iou, 6),
            "review_containment_overlap": round(containment_overlap, 6),
        }
        if region_type in NEGATIVE_TYPES:
            record["negative_subtype"] = region_type
        accepted.append(record)

    for index, row in enumerate(secondary):
        if index not in used_secondary:
            review.append(_review_record(row, "unmatched_secondary"))

    summary = {
        "accepted_fields": len(accepted),
        "review_needed_fields": len(review),
        "source_components": len(
            {record["source_component_id"] for record in accepted}
        ),
        "review_reasons": dict(
            sorted(Counter(row["review_reason"] for row in review).items())
        ),
        "coordinate_scales": {
            "primary": round(primary_scale, 6),
            "secondary": round(secondary_scale, 6),
        },
        "matching_thresholds": {
            "minimum_iou": round(minimum_iou, 6),
            "minimum_containment_overlap": round(minimum_containment_overlap, 6),
        },
        "excluded_pages": sorted(excluded),
    }
    _publish(output_root, accepted, review, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.5)
    parser.add_argument("--minimum-containment-overlap", type=float, default=0.8)
    parser.add_argument("--primary-scale", type=float, default=1.0)
    parser.add_argument("--secondary-scale", type=float, default=1.0)
    parser.add_argument("--exclude-page", action="append", default=[])
    return parser


def _load_review(
    path: Path,
    *,
    coordinate_scale: float = 1.0,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"review was not found: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {path}") from error
        if isinstance(row, dict) and isinstance(row.get("regions"), list):
            reviewer_id = row.get("reviewer_id")
            page_id = row.get("page_id")
            for region in row["regions"]:
                if not isinstance(region, dict):
                    raise ValueError(f"invalid region on line {line_number}: {path}")
                rows.append(
                    _validate_row(
                        {
                            **region,
                            "page_id": page_id,
                            "reviewer_id": reviewer_id,
                            "transcription": region.get("text"),
                        },
                        path,
                        line_number,
                        coordinate_scale,
                    )
                )
        else:
            rows.append(_validate_row(row, path, line_number, coordinate_scale))
    if not rows:
        raise ValueError(f"review is empty: {path}")
    return rows


def _validate_row(
    value: object,
    path: Path,
    line_number: int,
    coordinate_scale: float,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"review row must be an object on line {line_number}: {path}")
    row = dict(value)
    for key in ("page_id", "transcription", "reviewer_id", "region_type"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"invalid {key} on line {line_number}: {path}")
    if not row["reviewer_id"].strip():
        raise ValueError(f"missing reviewer on line {line_number}: {path}")
    if row.get("legibility") not in LEGIBILITY:
        raise ValueError(f"invalid legibility on line {line_number}: {path}")
    region_type = row["region_type"]
    transcription = row["transcription"]
    if region_type in NEGATIVE_TYPES:
        if transcription.strip() or row["legibility"] != "legible":
            raise ValueError(
                f"invalid confident negative on line {line_number}: {path}"
            )
    elif region_type in TRAINING_TYPES and row["legibility"] == "unreadable":
        if transcription.strip():
            raise ValueError(f"unreadable row has text on line {line_number}: {path}")
    elif region_type in TRAINING_TYPES and not transcription.strip():
        raise ValueError(f"missing transcription on line {line_number}: {path}")
    _validate_box(row.get("bbox"), None, f"bbox on line {line_number}: {path}")
    row["bbox"] = [round(value * coordinate_scale) for value in row["bbox"]]
    return row


def _validate_box(
    value: object,
    image_size: tuple[int, int] | None,
    label: str,
) -> None:
    valid = (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )
    if not valid:
        raise ValueError(f"invalid {label}")
    left, top, right, bottom = value
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"invalid {label}")
    if image_size is not None and (right > image_size[0] or bottom > image_size[1]):
        raise ValueError(f"invalid {label}")


def _best_match(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    used: set[int],
    minimum_iou: float,
    minimum_containment_overlap: float,
) -> tuple[int | None, float, float]:
    matches = []
    transcription = normalize_text(row["transcription"])
    for index, candidate in enumerate(candidates):
        if (
            index in used
            or candidate["page_id"] != row["page_id"]
            or candidate["region_type"] != row["region_type"]
        ):
            continue
        iou, containment_overlap = _overlap_metrics(row["bbox"], candidate["bbox"])
        if iou < minimum_iou and containment_overlap < minimum_containment_overlap:
            continue
        same_text = transcription == normalize_text(candidate["transcription"])
        # Text may choose among geometry-qualified candidates, never create a match.
        matches.append((same_text, containment_overlap, iou, index))
    if not matches:
        return None, 0.0, 0.0
    _, containment_overlap, iou, index = max(
        matches, key=lambda item: (*item[:-1], -item[-1])
    )
    return index, iou, containment_overlap


def _overlap_metrics(first: list[int], second: list[int]) -> tuple[float, float]:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0, 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    iou = intersection / (first_area + second_area - intersection)
    containment_overlap = intersection / min(first_area, second_area)
    return iou, containment_overlap


def _union_box(first: list[int], second: list[int]) -> list[int]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def _padded_box(box: list[int], image_size: tuple[int, int]) -> list[int]:
    padding = max(8, round((box[3] - box[1]) * 0.5))
    return [
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(image_size[0], box[2] + padding),
        min(image_size[1], box[3] + padding),
    ]


def _source_component(page_id: str) -> str:
    marker = page_id.rfind("-P")
    return page_id[:marker] if marker > 0 else page_id


def _index_pages(root: Path) -> dict[str, Path]:
    pages: dict[str, Path] = {}
    for path in sorted(root.rglob("*.png"), key=lambda item: item.as_posix()):
        previous = pages.get(path.stem)
        if previous is not None:
            raise ValueError(
                f"duplicate reviewed page for {path.stem}: {previous} and {path}"
            )
        pages[path.stem] = path
    return pages


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not read page image: {path}") from error


def _review_record(
    row: dict[str, Any],
    reason: str,
    other: dict[str, Any] | None = None,
    iou: float | None = None,
    containment_overlap: float | None = None,
) -> dict[str, Any]:
    record = dict(row)
    record["review_reason"] = reason
    if other is not None:
        record["independent_review"] = other
    if iou is not None:
        record["review_iou"] = round(iou, 6)
    if containment_overlap is not None:
        record["review_containment_overlap"] = round(containment_overlap, 6)
    return record


def _review_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewer_id": row["reviewer_id"],
        "bbox": row["bbox"],
        "transcription": row["transcription"],
        "legibility": row["legibility"],
        "region_type": row["region_type"],
    }


def _publish(
    output_root: Path,
    accepted: list[dict[str, Any]],
    review: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    try:
        for row in accepted:
            with Image.open(row["source_path"]) as image:
                for path_key, box_key in (
                    ("tight_crop_path", "crop_bbox"),
                    ("padded_crop_path", "padded_bbox"),
                ):
                    destination = temporary / row[path_key]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(tuple(row[box_key])).save(destination, format="PNG")
        _write_jsonl(temporary / "accepted.jsonl", accepted)
        _write_jsonl(temporary / "review_needed.jsonl", review)
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
