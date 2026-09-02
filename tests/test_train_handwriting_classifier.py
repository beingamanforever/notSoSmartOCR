from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from experiments.train_handwriting_classifier import load_samples

SCRIPT = Path(__file__).parents[1] / "experiments" / "train_handwriting_classifier.py"


def test_builds_family_separated_dual_view_samples_without_copying_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _row(root, "train", "train-pos", "family-a", "resolved")
    _row(root, "train", "train-neg", "family-b", "absent")
    _row(root, "dev", "dev-pos", "family-c", "unreadable")
    _row(root, "dev", "dev-neg", "family-d", "absent")

    train, dev = load_samples(root)

    assert [(item.field_id, item.view, item.label) for item in train] == [
        ("train-pos", "tight", 1),
        ("train-pos", "context", 1),
        ("train-neg", "tight", 0),
        ("train-neg", "context", 0),
    ]
    assert {item.family_id for item in train}.isdisjoint(
        {item.family_id for item in dev}
    )
    assert all(item.crop_path.is_relative_to(root) for item in train + dev)
    assert sorted(path.name for path in root.rglob("*.png")) == [
        "context.png",
        "context.png",
        "context.png",
        "context.png",
        "tight.png",
        "tight.png",
        "tight.png",
        "tight.png",
    ]


def test_validate_only_reports_pairs_without_importing_training_stack(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _row(root, "train", "train-pos", "family-a", "resolved")
    _row(root, "train", "train-neg", "family-b", "absent")
    _row(root, "dev", "dev-pos", "family-c", "resolved")
    _row(root, "dev", "dev-neg", "family-d", "absent")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(root),
            str(tmp_path / "unused.pt"),
            "--validate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "dev_families": 2,
        "dev_negative": 2,
        "dev_positive": 2,
        "dev_samples": 4,
        "train_families": 2,
        "train_negative": 2,
        "train_positive": 2,
        "train_samples": 4,
    }
    assert not (tmp_path / "unused.pt").exists()


def test_rejects_family_leakage_and_missing_hard_negative_metadata(
    tmp_path: Path,
) -> None:
    overlap = tmp_path / "overlap"
    _row(overlap, "train", "train-pos", "same", "resolved")
    _row(overlap, "train", "train-neg", "other", "absent")
    _row(overlap, "dev", "dev-pos", "same", "resolved")
    _row(overlap, "dev", "dev-neg", "dev-other", "absent")
    with pytest.raises(ValueError, match="overlapping families"):
        load_samples(overlap)

    missing = tmp_path / "missing"
    _row(missing, "train", "train-pos", "family-a", "resolved")
    negative = _row(missing, "train", "train-neg", "family-b", "absent")
    payload = json.loads(negative)
    del payload["abstention_subtype"]
    (missing / "train.jsonl").write_text(
        (missing / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        + "\n"
        + json.dumps(payload)
        + "\n",
        encoding="utf-8",
    )
    _row(missing, "dev", "dev-pos", "family-c", "resolved")
    _row(missing, "dev", "dev-neg", "family-d", "absent")
    with pytest.raises(ValueError, match="hard negative lacks subtype"):
        load_samples(missing)


def _row(
    root: Path,
    split: str,
    field_id: str,
    family_id: str,
    state: str,
) -> str:
    crop_root = root / "crops" / split / field_id
    crop_root.mkdir(parents=True, exist_ok=True)
    for name in ("tight.png", "context.png"):
        Image.new("RGB", (20, 10), "white").save(crop_root / name)
    row = {
        "field_id": field_id,
        "case_id": f"C08-{field_id}",
        "category_id": "C08",
        "family_id": family_id,
        "split": split,
        "target_state": state,
        "tight_crop_path": (crop_root / "tight.png").relative_to(root).as_posix(),
        "padded_crop_path": (crop_root / "context.png").relative_to(root).as_posix(),
    }
    if state == "absent":
        row["abstention_subtype"] = "printed_only"
    path = root / f"{split}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return json.dumps(row)
