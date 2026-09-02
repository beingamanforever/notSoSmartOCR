"""Compile reviewed handwriting annotations into geometry-backed crop candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
from random import Random
import re
import shutil
import tempfile
from typing import Any
import unicodedata

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REGION_KINDS = {"text", "word"}
FAMILY_PATTERN = re.compile(r"^(?P<family>.+)-P\d+$")


@dataclass(frozen=True)
class Annotation:
    case_id: str
    category_id: str
    field_id: str
    family_id: str
    reference: str
    legibility: str
    annotation_path: Path


@dataclass(frozen=True)
class Region:
    region_id: str
    text: str
    normalized_text: str
    provider: str
    bounding_box: tuple[int, int, int, int]
    output_path: Path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = compile_handwriting_crops(
        args.annotations,
        args.sources,
        args.model_output,
        args.output,
        fuzzy_threshold=args.fuzzy_threshold,
        ambiguity_margin=args.ambiguity_margin,
        dev_fraction=args.dev_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def compile_handwriting_crops(
    annotation_root: Path,
    source_root: Path,
    model_output_root: Path,
    output_root: Path,
    *,
    fuzzy_threshold: float = 0.88,
    ambiguity_margin: float = 0.08,
    dev_fraction: float = 0.2,
    split_seed: int = 17,
) -> dict[str, Any]:
    """Build a non-C14 candidate dataset without altering source material."""
    _validate_inputs(
        annotation_root,
        source_root,
        model_output_root,
        output_root,
        fuzzy_threshold,
        ambiguity_margin,
        dev_fraction,
    )
    annotations, excluded_c14 = _load_annotations(annotation_root)
    sources = _index_files(source_root, IMAGE_SUFFIXES, "source image")
    model_outputs = _index_files(model_output_root, {".json"}, "model output")

    page_cache: dict[str, tuple[Path, tuple[int, int], list[Region]] | str] = {}
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for annotation in annotations:
        if annotation.legibility != "legible":
            review.append(_review_record(annotation, "legibility_not_legible"))
            continue
        page = _load_page(
            annotation.case_id,
            sources,
            model_outputs,
            page_cache,
        )
        if isinstance(page, str):
            review.append(_review_record(annotation, page))
            continue
        source_path, image_size, regions = page
        match, review_reason, top_matches = _match_region(
            annotation.reference,
            regions,
            fuzzy_threshold=fuzzy_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        if match is None:
            review.append(
                _review_record(
                    annotation,
                    review_reason or "no_conservative_match",
                    top_matches=top_matches,
                    source_path=source_path,
                )
            )
            continue
        accepted.append(
            {
                "field_id": annotation.field_id,
                "case_id": annotation.case_id,
                "category_id": annotation.category_id,
                "family_id": annotation.family_id,
                "reference": annotation.reference,
                "legibility": annotation.legibility,
                "annotation_path": str(annotation.annotation_path.resolve()),
                "source_path": str(source_path.resolve()),
                **match,
                "padded_bbox": list(
                    _padded_bbox(tuple(match["region_bbox"]), image_size)
                ),
            }
        )

    accepted, collision_reviews = _reject_reused_geometry(accepted)
    review.extend(collision_reviews)
    _add_review_context(review, page_cache)
    family_splits = _split_families(
        {record["family_id"] for record in accepted},
        dev_fraction,
        split_seed,
    )
    for record in accepted:
        record["split"] = family_splits[record["family_id"]]
        record["tight_crop_path"] = (
            Path("crops") / record["split"] / "tight" / f"{record['field_id']}.png"
        ).as_posix()
        record["padded_crop_path"] = (
            Path("crops") / record["split"] / "padded" / f"{record['field_id']}.png"
        ).as_posix()
        record["crop_path"] = record["tight_crop_path"]

    summary = _summary(accepted, review, excluded_c14, family_splits)
    _publish(output_root, accepted, review, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--ambiguity-margin", type=float, default=0.08)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    return parser


def _validate_inputs(
    annotation_root: Path,
    source_root: Path,
    model_output_root: Path,
    output_root: Path,
    fuzzy_threshold: float,
    ambiguity_margin: float,
    dev_fraction: float,
) -> None:
    for label, root in (
        ("Annotation", annotation_root),
        ("Source", source_root),
        ("Model output", model_output_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{label} root not found: {root}")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be between zero and one")
    if not 0.0 <= ambiguity_margin <= 1.0:
        raise ValueError("ambiguity_margin must be between zero and one")
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")


def _load_annotations(root: Path) -> tuple[list[Annotation], int]:
    annotations: list[Annotation] = []
    excluded_c14 = 0
    for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
        payload = _read_json(path)
        case_id = payload.get("case_id")
        category_id = payload.get("category_id")
        handwriting = payload.get("handwriting")
        if not isinstance(case_id, str) or not isinstance(category_id, str):
            raise ValueError(f"Annotation lacks case/category identifiers: {path}")
        if not isinstance(handwriting, list):
            raise ValueError(f"Annotation handwriting must be a list: {path}")
        if category_id == "C14" or case_id.startswith("C14-"):
            excluded_c14 += len(handwriting)
            continue
        family_id = _family_id(case_id)
        for index, span in enumerate(handwriting, start=1):
            if not isinstance(span, dict):
                raise ValueError(f"Invalid handwriting span {index}: {path}")
            text = span.get("text")
            legibility = span.get("legibility")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Handwriting span {index} lacks text: {path}")
            if not isinstance(legibility, str) or not legibility:
                raise ValueError(f"Handwriting span {index} lacks legibility: {path}")
            annotations.append(
                Annotation(
                    case_id=case_id,
                    category_id=category_id,
                    field_id=f"{case_id}-H{index:03d}",
                    family_id=family_id,
                    reference=text,
                    legibility=legibility,
                    annotation_path=path,
                )
            )
    return annotations, excluded_c14


def _index_files(root: Path, suffixes: set[str], label: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem.startswith("C14-"):
            continue
        previous = index.get(path.stem)
        if previous is not None:
            raise ValueError(
                f"Duplicate {label} for {path.stem}: {previous} and {path}"
            )
        index[path.stem] = path
    return index


def _load_page(
    case_id: str,
    sources: dict[str, Path],
    model_outputs: dict[str, Path],
    cache: dict[str, tuple[Path, tuple[int, int], list[Region]] | str],
) -> tuple[Path, tuple[int, int], list[Region]] | str:
    if case_id in cache:
        return cache[case_id]
    source_path = sources.get(case_id)
    if source_path is None:
        cache[case_id] = "source_missing"
        return cache[case_id]
    model_output_path = model_outputs.get(case_id)
    if model_output_path is None:
        cache[case_id] = "model_output_missing"
        return cache[case_id]
    try:
        with Image.open(source_path) as image:
            image_size = image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read source image: {source_path}") from error
    payload = _read_json(model_output_path)
    result = payload.get("result")
    pages = result.get("pages") if isinstance(result, dict) else None
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        cache[case_id] = "model_page_missing"
        return cache[case_id]
    page = pages[0]
    model_size = (page.get("width"), page.get("height"))
    if model_size != image_size:
        cache[case_id] = "geometry_size_mismatch"
        return cache[case_id]
    raw_regions = page.get("regions")
    if not isinstance(raw_regions, list):
        cache[case_id] = "model_regions_missing"
        return cache[case_id]
    regions = [
        region
        for value in raw_regions
        if (region := _region(value, model_output_path, image_size)) is not None
    ]
    if not regions:
        cache[case_id] = "no_valid_geometry"
        return cache[case_id]
    cache[case_id] = source_path, image_size, regions
    return cache[case_id]


def _region(
    value: object,
    output_path: Path,
    image_size: tuple[int, int],
) -> Region | None:
    if not isinstance(value, dict) or value.get("kind") not in REGION_KINDS:
        return None
    region_id = value.get("id")
    text = value.get("text")
    provider = value.get("provider")
    if not all(
        isinstance(item, str) and item.strip() for item in (region_id, text, provider)
    ):
        return None
    bounding_box = _bounding_box(value.get("bounding_box"), image_size)
    if bounding_box is None:
        return None
    normalized_text = normalize_text(text)
    if not normalized_text:
        return None
    return Region(
        region_id=region_id,
        text=text,
        normalized_text=normalized_text,
        provider=provider,
        bounding_box=bounding_box,
        output_path=output_path,
    )


def _bounding_box(
    value: object, image_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    coordinates = [value.get(name) for name in ("left", "top", "right", "bottom")]
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in coordinates
    ):
        return None
    if any(not math.isfinite(float(item)) or int(item) != item for item in coordinates):
        return None
    left, top, right, bottom = (int(item) for item in coordinates)
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _match_region(
    reference: str,
    regions: list[Region],
    *,
    fuzzy_threshold: float,
    ambiguity_margin: float,
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    normalized_reference = normalize_text(reference)
    if not normalized_reference:
        return None, "no_conservative_match", []
    scored = sorted(
        (
            (_match_score(normalized_reference, region.normalized_text), region)
            for region in regions
        ),
        key=lambda item: (-item[0], item[1].provider, item[1].region_id),
    )
    best_score, best = scored[0]
    exact = normalized_reference == best.normalized_text
    eligible = exact or (
        len(normalized_reference) >= 4 and best_score >= fuzzy_threshold
    )
    if not eligible:
        return (
            None,
            "no_conservative_match",
            [_match_evidence(score, region) for score, region in scored[:3]],
        )
    competing = [
        (score, region)
        for score, region in scored[1:]
        if score > best_score - ambiguity_margin
        and region.bounding_box != best.bounding_box
    ]
    if competing:
        matches = [(best_score, best), *competing]
        return (
            None,
            "ambiguous_match",
            [_match_evidence(score, region) for score, region in matches[:3]],
        )
    return (
        {
            "match_method": "normalized_exact" if exact else "fuzzy",
            "match_score": round(best_score, 6),
            "matched_text": best.text,
            "provider": best.provider,
            "region_ids": [best.region_id],
            "region_bbox": list(best.bounding_box),
            "model_output_path": str(best.output_path.resolve()),
        },
        None,
        [_match_evidence(best_score, best)],
    )


def _match_score(reference: str, candidate: str) -> float:
    if reference == candidate:
        return 1.0
    return SequenceMatcher(None, reference, candidate, autojunk=False).ratio()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _match_evidence(score: float, region: Region) -> dict[str, Any]:
    return {
        "match_score": round(score, 6),
        "matched_text": region.text,
        "provider": region.provider,
        "region_ids": [region.region_id],
        "region_bbox": list(region.bounding_box),
        "model_output_path": str(region.output_path.resolve()),
    }


def _reject_reused_geometry(
    accepted: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_geometry: dict[tuple[str, str, tuple[int, ...]], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for record in accepted:
        key = (
            record["case_id"],
            record["model_output_path"],
            tuple(record["region_bbox"]),
        )
        by_geometry[key].append(record)
    collisions = {
        record["field_id"]
        for records in by_geometry.values()
        if len(records) > 1
        for record in records
    }
    kept = [record for record in accepted if record["field_id"] not in collisions]
    review = []
    for record in accepted:
        if record["field_id"] not in collisions:
            continue
        review.append(
            {
                key: record[key]
                for key in (
                    "field_id",
                    "case_id",
                    "category_id",
                    "family_id",
                    "reference",
                    "legibility",
                    "annotation_path",
                )
            }
            | {
                "review_reason": "geometry_reused",
                "top_matches": [
                    {
                        key: record[key]
                        for key in (
                            "match_score",
                            "matched_text",
                            "provider",
                            "region_ids",
                            "region_bbox",
                            "model_output_path",
                        )
                    }
                ],
            }
        )
    return kept, review


def _review_record(
    annotation: Annotation,
    reason: str,
    *,
    top_matches: list[dict[str, Any]] | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    record = {
        "field_id": annotation.field_id,
        "case_id": annotation.case_id,
        "category_id": annotation.category_id,
        "family_id": annotation.family_id,
        "reference": annotation.reference,
        "legibility": annotation.legibility,
        "annotation_path": str(annotation.annotation_path.resolve()),
        "review_reason": reason,
        "top_matches": top_matches or [],
    }
    if source_path is not None:
        record["source_path"] = str(source_path.resolve())
    return record


def _add_review_context(
    review: list[dict[str, Any]],
    page_cache: dict[str, tuple[Path, tuple[int, int], list[Region]] | str],
) -> None:
    for record in review:
        if record["review_reason"] != "no_conservative_match":
            continue
        top_matches = record["top_matches"]
        page = page_cache.get(record["case_id"])
        if not top_matches or page is None or isinstance(page, str):
            continue
        _, image_size, _ = page
        record["candidate_status"] = "review_only"
        record["review_bbox"] = list(
            _context_bbox(tuple(top_matches[0]["region_bbox"]), image_size)
        )
        record["review_crop_path"] = (
            Path("review") / f"{record['field_id']}.png"
        ).as_posix()


def _context_bbox(
    bounding_box: tuple[int, ...], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    _, top, _, bottom = bounding_box
    padding = max(24, 2 * (bottom - top))
    return _expand_bbox(bounding_box, image_size, padding)


def _padded_bbox(
    bounding_box: tuple[int, ...], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    _, top, _, bottom = bounding_box
    padding = max(8, (bottom - top) // 2)
    return _expand_bbox(bounding_box, image_size, padding)


def _expand_bbox(
    bounding_box: tuple[int, ...],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounding_box
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _split_families(
    families: set[str], dev_fraction: float, seed: int
) -> dict[str, str]:
    ordered = sorted(families)
    Random(seed).shuffle(ordered)
    dev_count = 0
    if len(ordered) > 1:
        dev_count = min(len(ordered) - 1, max(1, round(len(ordered) * dev_fraction)))
    dev = set(ordered[:dev_count])
    return {family: "dev" if family in dev else "train" for family in ordered}


def _summary(
    accepted: list[dict[str, Any]],
    review: list[dict[str, Any]],
    excluded_c14: int,
    family_splits: dict[str, str],
) -> dict[str, Any]:
    return {
        "accepted_fields": len(accepted),
        "review_needed_fields": len(review),
        "excluded_c14_fields": excluded_c14,
        "accepted_families": len(family_splits),
        "accepted_crop_files": 2 * len(accepted),
        "review_context_crops": sum("review_crop_path" in item for item in review),
        "split_fields": dict(
            sorted(Counter(item["split"] for item in accepted).items())
        ),
        "split_families": dict(sorted(Counter(family_splits.values()).items())),
        "review_reasons": dict(
            sorted(Counter(item["review_reason"] for item in review).items())
        ),
        "categories": dict(
            sorted(Counter(item["category_id"] for item in accepted).items())
        ),
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
        for split in ("train", "dev"):
            split_records = sorted(
                (item for item in accepted if item["split"] == split),
                key=lambda item: item["field_id"],
            )
            crop_root = temporary / "crops" / split
            crop_root.mkdir(parents=True, exist_ok=True)
            for record in split_records:
                _write_crop(
                    Path(record["source_path"]),
                    record["region_bbox"],
                    temporary / record["tight_crop_path"],
                )
                _write_crop(
                    Path(record["source_path"]),
                    record["padded_bbox"],
                    temporary / record["padded_crop_path"],
                )
            _write_jsonl(temporary / f"{split}.jsonl", split_records)
        (temporary / "review").mkdir()
        for record in review:
            if "review_crop_path" not in record:
                continue
            _write_crop(
                Path(record["source_path"]),
                record["review_bbox"],
                temporary / record["review_crop_path"],
            )
        _write_jsonl(
            temporary / "review_needed.jsonl",
            sorted(review, key=lambda item: item["field_id"]),
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_crop(
    source_path: Path,
    bounding_box: list[int],
    destination: Path,
) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            image.crop(tuple(bounding_box)).save(destination, format="PNG")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not crop source image: {source_path}") from error


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _family_id(case_id: str) -> str:
    match = FAMILY_PATTERN.fullmatch(case_id)
    return match.group("family") if match is not None else case_id


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
