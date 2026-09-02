from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.compose_handwriting_dataset import compose_dataset


def test_composes_family_split_private_and_train_only_public(tmp_path: Path) -> None:
    manual = tmp_path / "manual"
    for index in range(3):
        _manual_row(manual, f"C08-D00{index + 1}-P001", f"literal {index}")
    excluded = "C08-D004-P001"
    _manual_row(manual, excluded, "duplicate")
    public = tmp_path / "public"
    _public_row(public)

    output = tmp_path / "composed"
    summary = compose_dataset(
        [manual],
        [public],
        output,
        dev_fraction=0.34,
        exclude_pages=[excluded],
    )
    train = _rows(output / "train.jsonl")
    dev = _rows(output / "dev.jsonl")

    assert summary["fields"] == 4
    assert summary["private_fields"] == 3
    assert summary["public_fields"] == 1
    assert summary["dev_real_only"] is True
    assert summary["excluded_pages"] == [excluded]
    assert summary["excluded_literal_disagreements"] == 0
    assert {row["data_origin"] for row in dev} == {"private"}
    assert {
        row["family_id"] for row in train if row["data_origin"] == "private"
    }.isdisjoint({row["family_id"] for row in dev})
    assert (
        next(row for row in train if row["data_origin"] == "public")["split"] == "train"
    )
    for row in train + dev:
        assert (output / row["tight_crop_path"]).is_file()
        assert (output / row["padded_crop_path"]).is_file()


def test_rejects_duplicate_fields_bad_review_and_existing_output(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _manual_row(first, "C08-D001-P001", "literal")
    _manual_row(second, "C08-D001-P001", "literal")
    with pytest.raises(ValueError, match="duplicate field"):
        compose_dataset([first, second], [], tmp_path / "duplicate")

    row = _rows(first / "accepted.jsonl")[0]
    row["reviewer_ids"] = ["one"]
    (first / "accepted.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="independent review"):
        compose_dataset([first], [], tmp_path / "bad")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        compose_dataset([first], [], existing)


def test_excludes_strict_literal_disagreement_for_re_review(tmp_path: Path) -> None:
    manual = tmp_path / "manual"
    for index in range(3):
        _manual_row(manual, f"C08-D00{index + 1}-P001", f"literal {index}")
    disputed_page = "C08-D004-P001"
    _manual_row(manual, disputed_page, "take 1-2")
    rows = _rows(manual / "accepted.jsonl")
    rows[-1]["independent_reviews"][1]["transcription"] = "take 1 2"
    (manual / "accepted.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "composed"
    summary = compose_dataset([manual], [], output)
    composed = _rows(output / "train.jsonl") + _rows(output / "dev.jsonl")

    assert summary["private_fields"] == 3
    assert summary["excluded_literal_disagreements"] == 1
    assert disputed_page not in {row["case_id"] for row in composed}


def test_rejects_single_private_family_without_publishing_output(
    tmp_path: Path,
) -> None:
    manual = tmp_path / "manual"
    _manual_row(manual, "C08-D001-P001", "first")
    _manual_row(manual, "C08-D001-P002", "second")
    output = tmp_path / "composed"

    with pytest.raises(ValueError, match="at least two private families"):
        compose_dataset([manual], [], output)

    assert not output.exists()


def _manual_row(root: Path, page_id: str, reference: str) -> None:
    field_id = f"{page_id}-HM001"
    tight = root / "crops" / f"{field_id}.png"
    padded = root / "crops" / "padded" / f"{field_id}.png"
    tight.parent.mkdir(parents=True, exist_ok=True)
    padded.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), "white").save(tight)
    Image.new("RGB", (30, 20), "white").save(padded)

    def review(reviewer: str) -> dict[str, object]:
        return {
            "reviewer_id": reviewer,
            "bbox": [1, 1, 10, 8],
            "transcription": reference,
            "legibility": "legible",
            "region_type": "field",
        }

    row = {
        "field_id": field_id,
        "page_id": page_id,
        "source_component_id": page_id.rsplit("-P", 1)[0],
        "reference": reference,
        "target_state": "resolved",
        "tight_crop_path": tight.relative_to(root).as_posix(),
        "padded_crop_path": padded.relative_to(root).as_posix(),
        "reviewer_ids": ["one", "two"],
        "independent_reviews": [review("one"), review("two")],
    }
    with (root / "accepted.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _public_row(root: Path) -> None:
    tight = root / "crops" / "tight" / "READ-L001.png"
    padded = root / "crops" / "padded" / "READ-L001.png"
    tight.parent.mkdir(parents=True, exist_ok=True)
    padded.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), "white").save(tight)
    Image.new("RGB", (30, 20), "white").save(padded)
    row = {
        "field_id": "READ-L001",
        "case_id": "READ-1",
        "category_id": "READ",
        "family_id": "READ-1",
        "reference": "public literal",
        "target_state": "resolved",
        "data_origin": "public",
        "split": "train",
        "tight_crop_path": tight.relative_to(root).as_posix(),
        "padded_crop_path": padded.relative_to(root).as_posix(),
    }
    (root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
