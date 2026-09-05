"""Compose reviewed clinical crops with public handwriting train data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
import shutil
import tempfile
from typing import Any, Sequence
import unicodedata

TARGET_STATES = {"resolved", "absent", "unreadable"}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = compose_dataset(
        args.manual,
        args.public,
        args.output,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        exclude_pages=args.exclude_page,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manual", type=Path, action="append", required=True)
    parser.add_argument("--public", type=Path, action="append", default=[])
    parser.add_argument("--exclude-page", action="append", default=[])
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def compose_dataset(
    manual_roots: Sequence[Path],
    public_roots: Sequence[Path],
    output: Path,
    *,
    dev_fraction: float = 0.2,
    seed: int = 17,
    exclude_pages: Sequence[str] = (),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev fraction must be between zero and one")
    excluded = frozenset(exclude_pages)
    private = []
    literal_disagreements = 0
    for root in manual_roots:
        records, disagreements = _manual_records(root, excluded)
        private.extend(records)
        literal_disagreements += disagreements
    public = [record for root in public_roots for record in _public_records(root)]
    if not private:
        raise ValueError("no reviewed clinical crops remain")

    family_splits = _split_families(private, dev_fraction, seed)
    for record in private:
        record["split"] = family_splits[record["family_id"]]
    records = private + public
    _validate_records(records)
    if len(family_splits) < 2:
        raise ValueError("at least two private families are required")
    for record in records:
        split = record["split"]
        field_id = record["field_id"]
        record["tight_crop_path"] = f"crops/{split}/tight/{field_id}.png"
        record["padded_crop_path"] = f"crops/{split}/padded/{field_id}.png"
        record["crop_path"] = record["tight_crop_path"]

    summary = {
        "fields": len(records),
        "private_fields": len(private),
        "public_fields": len(public),
        "private_families": len(family_splits),
        "split_fields": dict(sorted(Counter(row["split"] for row in records).items())),
        "split_families": dict(sorted(Counter(family_splits.values()).items())),
        "target_states": dict(
            sorted(Counter(row["target_state"] for row in records).items())
        ),
        "excluded_pages": sorted(excluded),
        "excluded_literal_disagreements": literal_disagreements,
        "dev_real_only": all(
            row["data_origin"] == "private" for row in records if row["split"] == "dev"
        ),
        "seed": seed,
    }
    _publish(output, records, summary)
    return summary


def _manual_records(
    root: Path, excluded: frozenset[str]
) -> tuple[list[dict[str, Any]], int]:
    rows = _read_jsonl(root / "accepted.jsonl")
    records = []
    literal_disagreements = 0
    for row in rows:
        page_id = _text(row, "page_id", root)
        if page_id in excluded:
            continue
        if page_id.startswith("C14-"):
            raise ValueError(f"held-out C14 page is forbidden: {page_id}")
        target_state = _target_state(row, root)
        reference = _reference(row, target_state, root)
        reviewers, literal_agreement = _reviewers(row, target_state, reference, root)
        if not literal_agreement:
            literal_disagreements += 1
            continue
        field_id = _identifier(row, "field_id", root)
        family_id = _text(row, "source_component_id", root)
        record = {
            "field_id": field_id,
            "case_id": page_id,
            "category_id": page_id.split("-", 1)[0],
            "family_id": family_id,
            "split_group_id": _lineage_group(row, family_id, root),
            "reference": reference,
            "target_state": target_state,
            "data_origin": "private",
            "reviewer_ids": list(reviewers),
            "independent_reviews": row["independent_reviews"],
            "source_path": row.get("source_path"),
            "_tight_source": _crop_source(root, row.get("tight_crop_path")),
            "_padded_source": _crop_source(root, row.get("padded_crop_path")),
        }
        if target_state == "absent":
            record["abstention_subtype"] = _text(row, "negative_subtype", root)
        elif target_state == "unreadable":
            record["abstention_subtype"] = "unreadable"
        records.append(record)
    return records, literal_disagreements


def _public_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for row in _read_jsonl(root / "train.jsonl"):
        if row.get("data_origin") != "public" or row.get("split") != "train":
            raise ValueError(f"public source must be train-only public data: {root}")
        target_state = _target_state(row, root)
        field_id = _identifier(row, "field_id", root)
        family_id = _text(row, "family_id", root)
        records.append(
            {
                "field_id": field_id,
                "case_id": _text(row, "case_id", root),
                "category_id": _text(row, "category_id", root),
                "family_id": family_id,
                "split_group_id": _lineage_group(row, family_id, root),
                "reference": _reference(row, target_state, root),
                "target_state": target_state,
                "data_origin": "public",
                "split": "train",
                "dataset": row.get("dataset"),
                "license": row.get("license"),
                "_tight_source": _crop_source(root, row.get("tight_crop_path")),
                "_padded_source": _crop_source(root, row.get("padded_crop_path")),
            }
        )
    return records


def _reviewers(
    row: dict[str, Any], target_state: str, reference: str, root: Path
) -> tuple[tuple[str, ...], bool]:
    values = row.get("reviewer_ids")
    reviews = row.get("independent_reviews")
    if (
        not isinstance(values, list)
        or not isinstance(reviews, list)
        or len(values) != 2
        or len(reviews) != 2
        or not all(isinstance(review, dict) for review in reviews)
    ):
        raise ValueError(f"manual row lacks independent review evidence: {root}")
    reviewers = tuple(sorted({_nonempty(value) for value in values}))
    evidence = {_nonempty(review.get("reviewer_id")) for review in reviews}
    if len(reviewers) != 2 or evidence != set(reviewers):
        raise ValueError(f"manual row lacks two independent reviewers: {root}")
    literal_agreement = True
    for review in reviews:
        transcription = review.get("transcription")
        legibility = review.get("legibility")
        if not isinstance(transcription, str):
            raise ValueError(f"manual review has invalid transcription: {root}")
        if target_state == "resolved":
            if legibility != "legible":
                raise ValueError(f"manual resolved review is not legible: {root}")
            literal_agreement &= _literal_text(transcription) == _literal_text(
                reference
            )
        if target_state == "absent" and (
            transcription.strip()
            or review.get("region_type") != row.get("negative_subtype")
        ):
            raise ValueError(f"manual negative review is inconsistent: {root}")
        if target_state == "unreadable" and (
            transcription.strip() or legibility != "unreadable"
        ):
            raise ValueError(f"manual unreadable review is inconsistent: {root}")
    return reviewers, literal_agreement


def _literal_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _validate_records(records: list[dict[str, Any]]) -> None:
    field_ids = [record["field_id"] for record in records]
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("input datasets contain duplicate field ids")
    train_families = {
        row["family_id"]
        for row in records
        if row["split"] == "train" and row["data_origin"] == "private"
    }
    dev_families = {row["family_id"] for row in records if row["split"] == "dev"}
    if train_families & dev_families:
        raise ValueError("train and dev contain overlapping clinical families")
    group_lineage: dict[str, tuple[str, str]] = {}
    for row in records:
        lineage = (row["family_id"], row["split"])
        previous = group_lineage.setdefault(row["split_group_id"], lineage)
        if previous[0] != lineage[0]:
            raise ValueError("split group belongs to multiple families")
        if previous[1] != lineage[1]:
            raise ValueError("split group belongs to multiple splits")


def _lineage_group(row: dict[str, Any], family_id: str, root: Path) -> str:
    value = row.get("split_group_id", family_id)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid split_group_id in {root}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset rows were not found: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"row must be an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def _target_state(row: dict[str, Any], root: Path) -> str:
    value = row.get("target_state", "resolved")
    if value not in TARGET_STATES:
        raise ValueError(f"invalid target state in {root}")
    return value


def _reference(row: dict[str, Any], target_state: str, root: Path) -> str:
    value = row.get("reference", "")
    if not isinstance(value, str):
        raise ValueError(f"invalid reference in {root}")
    if target_state == "resolved" and not value.strip():
        raise ValueError(f"resolved row lacks a literal in {root}")
    if target_state != "resolved" and value.strip():
        raise ValueError(f"abstention row contains a literal in {root}")
    return value


def _identifier(row: dict[str, Any], key: str, root: Path) -> str:
    value = _text(row, key, root)
    if Path(value).name != value:
        raise ValueError(f"unsafe {key} in {root}")
    return value


def _text(row: dict[str, Any], key: str, root: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {key} in {root}")
    return value


def _nonempty(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewer id must be non-empty")
    return value


def _crop_source(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"crop path is missing in {root}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"crop path is unsafe in {root}")
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(f"crop was not found: {source}")
    return source


def _split_families(
    records: list[dict[str, Any]], fraction: float, seed: int
) -> dict[str, str]:
    counts = Counter(record["family_id"] for record in records)
    ordered = sorted(counts)
    Random(seed).shuffle(ordered)
    target = max(1, round(len(records) * fraction))
    dev = set()
    fields = 0
    for family in ordered[:-1]:
        dev.add(family)
        fields += counts[family]
        if fields >= target:
            break
    return {family: "dev" if family in dev else "train" for family in ordered}


def _publish(
    output: Path, records: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for record in records:
            for source_key, path_key in (
                ("_tight_source", "tight_crop_path"),
                ("_padded_source", "padded_crop_path"),
            ):
                destination = temporary / record[path_key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(record[source_key], destination)
        for split in ("train", "dev"):
            rows = [
                {key: value for key, value in record.items() if not key.startswith("_")}
                for record in records
                if record["split"] == split
            ]
            (temporary / f"{split}.jsonl").write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
