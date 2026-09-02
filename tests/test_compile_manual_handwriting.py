from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.compile_manual_handwriting import compile_manual_handwriting


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(
    *,
    bbox: list[int],
    text: str,
    reviewer: str,
    legibility: str = "legible",
    region_type: str = "line",
) -> dict[str, object]:
    return {
        "page_id": "C08-D011-P001",
        "bbox": bbox,
        "transcription": text,
        "legibility": legibility,
        "reviewer_id": reviewer,
        "region_type": region_type,
    }


def test_compiles_only_independently_agreed_legible_text(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (200, 120), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "reviews" / "primary.jsonl"
    secondary = tmp_path / "reviews" / "secondary.jsonl"
    _write_jsonl(
        primary,
        [
            _row(bbox=[10, 10, 90, 35], text="Dose 5 mg", reviewer="one"),
            _row(bbox=[10, 45, 90, 70], text="unclear", reviewer="one"),
            _row(
                bbox=[110, 10, 180, 35],
                text="",
                reviewer="one",
                legibility="unreadable",
                region_type="signature",
            ),
            _row(
                bbox=[110, 75, 180, 95],
                text="",
                reviewer="one",
                region_type="control_mark",
            ),
        ],
    )
    _write_jsonl(
        secondary,
        [
            _row(bbox=[12, 11, 92, 36], text="dose 5 mg", reviewer="two"),
            _row(bbox=[11, 46, 91, 71], text="different", reviewer="two"),
            _row(
                bbox=[111, 11, 181, 36],
                text="",
                reviewer="two",
                legibility="unreadable",
                region_type="signature",
            ),
            _row(
                bbox=[111, 76, 181, 96],
                text="",
                reviewer="two",
                region_type="control_mark",
            ),
        ],
    )

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)

    accepted = [
        json.loads(line)
        for line in (output / "accepted.jsonl").read_text().splitlines()
    ]
    review = [
        json.loads(line)
        for line in (output / "review_needed.jsonl").read_text().splitlines()
    ]
    assert summary["accepted_fields"] == 1
    assert summary["review_needed_fields"] == 3
    assert summary["coordinate_scales"] == {"primary": 1.0, "secondary": 1.0}
    assert summary["matching_thresholds"] == {
        "minimum_iou": 0.5,
        "minimum_containment_overlap": 0.8,
    }
    assert summary["excluded_pages"] == []
    assert accepted[0]["transcription"] == "Dose 5 mg"
    assert accepted[0]["source_component_id"] == "C08-D011"
    assert accepted[0]["reviewer_ids"] == ["one", "two"]
    assert (output / accepted[0]["crop_path"]).is_file()
    assert {row["review_reason"] for row in review} == {
        "non_training_region",
        "transcription_disagreement",
    }


def test_rejects_same_reviewer_invalid_box_and_existing_output(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (50, 50), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(primary, [_row(bbox=[1, 1, 20, 20], text="x", reviewer="same")])
    _write_jsonl(secondary, [_row(bbox=[1, 1, 20, 20], text="x", reviewer="same")])

    with pytest.raises(ValueError, match="independent reviewer"):
        compile_manual_handwriting(pages, primary, secondary, tmp_path / "same")

    _write_jsonl(
        secondary,
        [_row(bbox=[10, 10, 8, 20], text="x", reviewer="different")],
    )
    with pytest.raises(ValueError, match="invalid bbox"):
        compile_manual_handwriting(pages, primary, secondary, tmp_path / "bad")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        compile_manual_handwriting(pages, primary, secondary, existing)

    with pytest.raises(ValueError, match="minimum_containment_overlap"):
        compile_manual_handwriting(
            pages,
            primary,
            secondary,
            tmp_path / "threshold",
            minimum_containment_overlap=0.0,
        )


def test_accepts_page_annotations_from_the_ui(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (80, 50), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    first = {
        "page_id": "C08-D011-P001",
        "reviewer_id": "one",
        "regions": [
            {
                "bbox": [5, 5, 60, 30],
                "text": "Take daily",
                "legibility": "legible",
                "region_type": "field",
            }
        ],
    }
    second = {
        **first,
        "reviewer_id": "two",
        "regions": [{**first["regions"][0], "bbox": [6, 6, 61, 31]}],
    }
    _write_jsonl(primary, [first])
    _write_jsonl(secondary, [second])

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)

    assert summary["accepted_fields"] == 1


def test_finds_reviewed_pages_in_category_folders(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    category = pages / "handwritten"
    category.mkdir(parents=True)
    Image.new("RGB", (80, 50), "white").save(category / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [_row(bbox=[5, 5, 60, 30], text="Take daily", reviewer="one")],
    )
    _write_jsonl(
        secondary,
        [_row(bbox=[6, 6, 61, 31], text="Take daily", reviewer="two")],
    )

    summary = compile_manual_handwriting(
        pages,
        primary,
        secondary,
        tmp_path / "compiled",
    )

    assert summary["accepted_fields"] == 1


def test_excludes_disclosed_page_from_both_reviews(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (80, 50), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [_row(bbox=[5, 5, 60, 30], text="Take daily", reviewer="one")],
    )
    _write_jsonl(
        secondary,
        [_row(bbox=[6, 6, 61, 31], text="Take daily", reviewer="two")],
    )

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(
        pages,
        primary,
        secondary,
        output,
        exclude_pages=["C08-D011-P001"],
    )

    assert summary["accepted_fields"] == 0
    assert summary["review_needed_fields"] == 0
    assert summary["excluded_pages"] == ["C08-D011-P001"]
    assert (output / "accepted.jsonl").read_text() == ""


def test_scales_preview_coordinates_before_matching(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (100, 100), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [_row(bbox=[5, 5, 25, 15], text="Daily", reviewer="one")],
    )
    _write_jsonl(
        secondary,
        [_row(bbox=[10, 10, 50, 30], text="Daily", reviewer="two")],
    )

    output = tmp_path / "scaled"
    summary = compile_manual_handwriting(
        pages,
        primary,
        secondary,
        output,
        primary_scale=2.0,
    )

    assert summary["accepted_fields"] == 1
    assert summary["coordinate_scales"] == {"primary": 2.0, "secondary": 1.0}


def test_compiles_resolved_absent_and_unreadable_targets_with_both_crops(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    page = Image.new("RGB", (160, 100), "white")
    page.paste("black", (10, 10, 40, 30))
    page.save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    first = [
        _row(bbox=[10, 10, 40, 30], text="Dose 5 mg", reviewer="one"),
        _row(
            bbox=[60, 10, 90, 30],
            text="",
            reviewer="one",
            region_type="printed_only",
        ),
        _row(
            bbox=[110, 10, 140, 30],
            text="",
            reviewer="one",
            legibility="unreadable",
            region_type="field",
        ),
    ]
    second = [
        {
            **row,
            "reviewer_id": "two",
            "bbox": [row["bbox"][0] + 1, 11, row["bbox"][2] + 1, 31],
        }
        for row in first
    ]
    _write_jsonl(primary, first)
    _write_jsonl(secondary, second)

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)
    accepted = [
        json.loads(line)
        for line in (output / "accepted.jsonl").read_text().splitlines()
    ]

    assert summary["accepted_fields"] == 3
    assert [row["target_state"] for row in accepted] == [
        "resolved",
        "absent",
        "unreadable",
    ]
    assert accepted[0]["reference"] == accepted[0]["transcription"] == "Dose 5 mg"
    assert accepted[1]["reference"] == accepted[1]["transcription"] == ""
    assert accepted[1]["negative_subtype"] == "printed_only"
    assert accepted[2]["reference"] == accepted[2]["transcription"] == ""
    assert accepted[0]["source_component_id"] == "C08-D011"
    assert [review["reviewer_id"] for review in accepted[0]["independent_reviews"]] == [
        "one",
        "two",
    ]
    for row in accepted:
        assert row["crop_path"] == row["tight_crop_path"]
        tight_path = output / row["tight_crop_path"]
        padded_path = output / row["padded_crop_path"]
        assert tight_path.is_file()
        assert padded_path.is_file()
        with Image.open(tight_path) as tight, Image.open(padded_path) as padded:
            assert padded.width > tight.width
            assert padded.height > tight.height


def test_negative_subtype_disagreement_stays_in_review(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (80, 50), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [_row(bbox=[5, 5, 30, 25], text="", reviewer="one", region_type="blank")],
    )
    _write_jsonl(
        secondary,
        [
            _row(
                bbox=[5, 5, 30, 25],
                text="",
                reviewer="two",
                region_type="stray_mark",
            )
        ],
    )

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)

    assert summary["accepted_fields"] == 0
    assert summary["review_reasons"] == {
        "independent_region_missing": 1,
        "unmatched_secondary": 1,
    }


def test_accepts_context_box_by_containment_and_records_both_metrics(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (160, 100), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [_row(bbox=[40, 40, 80, 60], text="Same field", reviewer="one")],
    )
    _write_jsonl(
        secondary,
        [_row(bbox=[20, 20, 120, 80], text="same field", reviewer="two")],
    )

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)
    accepted = json.loads((output / "accepted.jsonl").read_text())

    assert summary["accepted_fields"] == 1
    assert accepted["review_iou"] == pytest.approx(0.133333, abs=1e-6)
    assert accepted["review_containment_overlap"] == 1.0


def test_exact_text_breaks_tie_only_after_spatial_filter(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (180, 120), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [_row(bbox=[40, 40, 80, 60], text="Target", reviewer="one")],
    )
    _write_jsonl(
        secondary,
        [
            _row(bbox=[38, 38, 82, 62], text="Other", reviewer="two"),
            _row(bbox=[20, 20, 120, 80], text="target", reviewer="two"),
        ],
    )

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)
    accepted = json.loads((output / "accepted.jsonl").read_text())

    assert summary["accepted_fields"] == 1
    assert summary["review_reasons"] == {"unmatched_secondary": 1}
    assert accepted["independent_reviews"][1]["bbox"] == [20, 20, 120, 80]


def test_exact_text_cannot_override_geometry_or_legibility(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (180, 120), "white").save(pages / "C08-D011-P001.png")
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [
            _row(bbox=[10, 10, 40, 30], text="Target", reviewer="one"),
            _row(
                bbox=[40, 50, 80, 70],
                text="Ambiguous",
                reviewer="one",
                legibility="ambiguous",
            ),
        ],
    )
    _write_jsonl(
        secondary,
        [
            _row(bbox=[11, 11, 41, 31], text="Other", reviewer="two"),
            _row(bbox=[110, 10, 140, 30], text="Target", reviewer="two"),
            _row(
                bbox=[20, 40, 100, 80],
                text="Ambiguous",
                reviewer="two",
                legibility="ambiguous",
            ),
        ],
    )

    output = tmp_path / "compiled"
    summary = compile_manual_handwriting(pages, primary, secondary, output)
    review = [
        json.loads(line)
        for line in (output / "review_needed.jsonl").read_text().splitlines()
    ]

    assert summary["accepted_fields"] == 0
    assert summary["review_reasons"] == {
        "legibility_not_legible": 1,
        "transcription_disagreement": 1,
        "unmatched_secondary": 1,
    }
    ambiguous = next(
        row for row in review if row["review_reason"] == "legibility_not_legible"
    )
    assert ambiguous["review_iou"] < 0.5
    assert ambiguous["review_containment_overlap"] == 1.0
