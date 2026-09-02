from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "experiments" / "compile_handwriting_crops.py"


def test_compiles_unique_non_c14_matches_and_routes_uncertainty(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    sources = tmp_path / "sources"
    model_output = tmp_path / "model-output"
    output = tmp_path / "compiled"
    for root in (annotations, sources, model_output):
        root.mkdir()

    _annotation(
        annotations,
        "C08-D001-P001",
        "C08",
        [
            ("10 ㎎", "legible"),
            ("aspirin daily", "legible"),
        ],
    )
    _annotation(
        annotations,
        "C08-D002-P001",
        "C08",
        [("Follow-up", "legible")],
    )
    _annotation(
        annotations,
        "C13-D003-P002",
        "C13",
        [
            ("Dose", "legible"),
            ("uncertain name", "partial"),
            ("missing phrase", "legible"),
        ],
    )
    _annotation(
        annotations,
        "C14-D001-P001",
        "C14",
        [("must be excluded", "legible")],
    )
    _page(sources, "C08-D001-P001", (120, 80), (20, 30, 40))
    _page(sources, "C08-D002-P001", (120, 80), (50, 60, 70))
    _page(sources, "C13-D003-P002", (120, 80), (80, 90, 100))
    _page(sources, "C14-D001-P001", (120, 80), (110, 120, 130))
    _model_result(
        model_output,
        "C08-D001-P001",
        [
            _region("exact", "10 mg", "nemotron", [10, 10, 30, 30]),
            _region("fuzzy", "aspirin daliy", "nemotron", [40, 10, 90, 30]),
        ],
    )
    _model_result(
        model_output,
        "C08-D002-P001",
        [_region("punctuation", "follow up", "reader-b", [20, 20, 70, 40])],
    )
    _model_result(
        model_output,
        "C13-D003-P002",
        [
            _region("dose-a", "Dose", "reader-a", [10, 40, 35, 60]),
            _region("dose-b", "Dose", "reader-b", [60, 40, 85, 60]),
        ],
    )
    _model_result(
        model_output,
        "C14-D001-P001",
        [_region("excluded", "must be excluded", "reader-c", [5, 5, 50, 20])],
    )
    original_annotations = {
        path: path.read_bytes() for path in annotations.rglob("*.json")
    }
    original_sources = {path: path.read_bytes() for path in sources.rglob("*.png")}

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--annotations",
            str(annotations),
            "--sources",
            str(sources),
            "--model-output",
            str(model_output),
            "--output",
            str(output),
            "--fuzzy-threshold",
            "0.85",
            "--dev-fraction",
            "0.5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["accepted_fields"] == 3
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "accepted_families": 2,
        "accepted_crop_files": 6,
        "accepted_fields": 3,
        "categories": {"C08": 3},
        "excluded_c14_fields": 1,
        "review_needed_fields": 3,
        "review_context_crops": 1,
        "review_reasons": {
            "ambiguous_match": 1,
            "legibility_not_legible": 1,
            "no_conservative_match": 1,
        },
        "split_families": {"dev": 1, "train": 1},
        "split_fields": {"dev": 2, "train": 1},
    }
    train = _jsonl(output / "train.jsonl")
    dev = _jsonl(output / "dev.jsonl")
    accepted = train + dev
    assert {item["match_method"] for item in accepted} == {
        "normalized_exact",
        "fuzzy",
    }
    assert {item["provider"] for item in accepted} == {"nemotron", "reader-b"}
    fuzzy = next(item for item in accepted if item["match_method"] == "fuzzy")
    assert fuzzy["match_score"] >= 0.85
    assert fuzzy["matched_text"] == "aspirin daliy"
    assert {item["family_id"] for item in train}.isdisjoint(
        {item["family_id"] for item in dev}
    )
    for item in accepted:
        assert item["crop_path"] == item["tight_crop_path"]
        assert item["tight_crop_path"].startswith(f"crops/{item['split']}/tight/")
        assert item["padded_crop_path"].startswith(f"crops/{item['split']}/padded/")
        left, top, right, bottom = item["region_bbox"]
        padded_left, padded_top, padded_right, padded_bottom = item["padded_bbox"]
        assert padded_left <= left < right <= padded_right
        assert padded_top <= top < bottom <= padded_bottom
        with Image.open(output / item["tight_crop_path"]) as tight:
            assert tight.size == (right - left, bottom - top)
            with Image.open(item["source_path"]) as source:
                assert tight.getpixel((0, 0)) == source.getpixel((left, top))
        with Image.open(output / item["padded_crop_path"]) as padded:
            assert padded.size == (
                padded_right - padded_left,
                padded_bottom - padded_top,
            )
    review = _jsonl(output / "review_needed.jsonl")
    ambiguous = next(
        item for item in review if item["review_reason"] == "ambiguous_match"
    )
    assert [item["provider"] for item in ambiguous["top_matches"]] == [
        "reader-a",
        "reader-b",
    ]
    unmatched = next(
        item for item in review if item["review_reason"] == "no_conservative_match"
    )
    assert unmatched["candidate_status"] == "review_only"
    assert unmatched["review_crop_path"].startswith("review/")
    assert unmatched["review_bbox"] == [0, 0, 75, 80]
    assert "split" not in unmatched
    assert all(
        "review_crop_path" not in item for item in review if item is not unmatched
    )
    with Image.open(output / unmatched["review_crop_path"]) as context:
        assert context.size == (75, 80)
        with Image.open(unmatched["source_path"]) as source:
            assert context.getpixel((0, 0)) == source.getpixel((0, 0))
    assert not {item["field_id"] for item in accepted} & {
        item["field_id"] for item in review
    }
    assert all(not item["case_id"].startswith("C14-") for item in accepted + review)
    assert original_annotations == {
        path: path.read_bytes() for path in annotations.rglob("*.json")
    }
    assert original_sources == {
        path: path.read_bytes() for path in sources.rglob("*.png")
    }


def test_rejects_invalid_or_reused_geometry_instead_of_guessing(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    sources = tmp_path / "sources"
    model_output = tmp_path / "model-output"
    output = tmp_path / "compiled"
    for root in (annotations, sources, model_output):
        root.mkdir()
    _annotation(
        annotations,
        "C08-D004-P001",
        "C08",
        [("Same", "legible"), ("same", "legible")],
    )
    _annotation(
        annotations,
        "C08-D005-P001",
        "C08",
        [("Outside", "legible")],
    )
    _page(sources, "C08-D004-P001", (80, 60), (10, 20, 30))
    _page(sources, "C08-D005-P001", (80, 60), (10, 20, 30))
    _model_result(
        model_output,
        "C08-D004-P001",
        [_region("one", "same", "reader", [10, 10, 40, 30])],
        size=(80, 60),
    )
    _model_result(
        model_output,
        "C08-D005-P001",
        [_region("bad", "Outside", "reader", [10, 10, 100, 30])],
        size=(80, 60),
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--annotations",
            str(annotations),
            "--sources",
            str(sources),
            "--model-output",
            str(model_output),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert _jsonl(output / "train.jsonl") == []
    assert _jsonl(output / "dev.jsonl") == []
    review = _jsonl(output / "review_needed.jsonl")
    assert [item["review_reason"] for item in review] == [
        "geometry_reused",
        "geometry_reused",
        "no_valid_geometry",
    ]
    assert (
        json.loads((output / "summary.json").read_text())["review_context_crops"] == 0
    )
    assert json.loads((output / "summary.json").read_text())["accepted_crop_files"] == 0
    assert not list((output / "review").glob("*.png"))
    assert not list((output / "crops").rglob("*.png"))


def _annotation(
    root: Path,
    case_id: str,
    category_id: str,
    handwriting: list[tuple[str, str]],
) -> None:
    path = root / category_id / f"{case_id}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "category_id": category_id,
                "handwriting": [
                    {"text": text, "legibility": legibility}
                    for text, legibility in handwriting
                ],
            }
        ),
        encoding="utf-8",
    )


def _page(
    root: Path, case_id: str, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    Image.new("RGB", size, color).save(root / f"{case_id}.png")


def _model_result(
    root: Path,
    case_id: str,
    regions: list[dict[str, object]],
    *,
    size: tuple[int, int] = (120, 80),
) -> None:
    width, height = size
    (root / f"{case_id}.json").write_text(
        json.dumps(
            {
                "result": {
                    "document_id": case_id,
                    "pages": [
                        {
                            "page_number": 1,
                            "width": width,
                            "height": height,
                            "regions": regions,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def _region(
    region_id: str,
    text: str,
    provider: str,
    bbox: list[int],
) -> dict[str, object]:
    left, top, right, bottom = bbox
    return {
        "id": region_id,
        "kind": "text",
        "text": text,
        "provider": provider,
        "bounding_box": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
