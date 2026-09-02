from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.annotate_handwriting_pages import AnnotationStore
from experiments import rank_handwriting_review as ranker


def test_rank_is_blind_deterministic_capped_and_failure_inclusive(
    tmp_path: Path,
) -> None:
    cases = [
        ("F1", "P1", "A", 0.1, "a", "b", "success"),
        ("F2", "P1", "B", 0.2, "a", "b", "success"),
        ("F3", "P2", "A", 0.3, "a", "b", "success"),
        ("F4", "P3", "A", 0.4, "a", "b", "success"),
        ("F5", "P4", "C", 0.5, "a", "", "failed"),
        ("F6", "P5", "D", 0.9, "a", "b", "success"),
    ]
    review_queue, phi4_results = _write_inputs(tmp_path, cases)

    summary = ranker.rank_review_queue(
        review_queue,
        phi4_results,
        tmp_path / "ranked",
        limit=3,
        max_per_page=1,
        max_per_family=2,
    )

    result_path = tmp_path / "ranked" / ranker.RESULT_FILE
    queue_path = tmp_path / "ranked" / ranker.QUEUE_FILE
    page_queue_path = tmp_path / "ranked" / ranker.PAGE_QUEUE_FILE
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert [row["field_id"] for row in rows] == ["F1", "F2", "F3", "F4", "F6", "F5"]
    assert [row["field_id"] for row in rows if row["selected"]] == ["F1", "F3", "F6"]
    assert rows[0]["ranking_components"] == {
        "model_disagreement": 1.0,
        "low_nemotron_confidence": 0.9,
    }
    assert rows[0]["priority_score"] == 0.9
    assert rows[1]["selection_reasons"] == ["page_cap"]
    assert rows[3]["selection_reasons"] == ["family_cap"]
    assert rows[-1]["phi4"]["status"] == "failed"
    assert rows[-1]["priority_score"] is None
    assert summary["unresolved_fields"] == 6
    assert summary["selected_fields"] == 3
    assert summary["selected_source_pages"] == 3
    assert summary["phi4_failures_or_missing"] == 1
    assert summary["failures_remain_in_denominator"] is True
    assert payload["ranking"]["clinical_criticality_used"] is False
    assert payload["privacy"]["model_requests_made"] is False

    serialized = result_path.read_text(encoding="utf-8") + queue_path.read_text(
        encoding="utf-8"
    )
    assert "SECRET_GROUND_TRUTH" not in serialized
    assert '"reference"' not in serialized
    assert len(queue_path.read_text(encoding="utf-8").splitlines()) == 6
    page_queue = [
        json.loads(line)
        for line in page_queue_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [page["page_id"] for page in page_queue] == ["P1", "P2", "P5"]
    assert all(set(page) == {"page_id", "image_path"} for page in page_queue)
    assert all(Path(page["image_path"]).is_absolute() for page in page_queue)
    forbidden = {
        "reference",
        "prediction",
        "candidate_text",
        "confidence",
        "model_output_path",
    }
    assert not any(forbidden & set(page) for page in page_queue)
    store = AnnotationStore(page_queue_path, tmp_path / "manual-review.jsonl")
    assert [page.page_id for page in store.pages] == ["P1", "P2", "P5"]
    assert not any(tmp_path.glob(".ranked-*"))


def test_missing_phi4_and_broken_nemotron_evidence_are_retained(
    tmp_path: Path,
) -> None:
    review_queue, phi4_results = _write_inputs(
        tmp_path,
        [("F1", "P1", "A", 0.4, "a", "b", "success")],
    )
    phi4_results.write_text(json.dumps({"rows": []}), encoding="utf-8")
    review = json.loads(review_queue.read_text(encoding="utf-8"))
    review["top_matches"][0]["region_ids"] = ["missing"]
    review_queue.write_text(json.dumps(review) + "\n", encoding="utf-8")

    summary = ranker.rank_review_queue(
        review_queue,
        phi4_results,
        tmp_path / "ranked",
        limit=1,
        max_per_page=1,
        max_per_family=1,
    )

    payload = json.loads(
        (tmp_path / "ranked" / ranker.RESULT_FILE).read_text(encoding="utf-8")
    )
    row = payload["rows"][0]
    assert row["phi4"]["status"] == "missing"
    assert row["nemotron"]["status"] == "failed"
    assert row["nemotron"]["error"] == "region_not_found"
    assert row["priority_score"] is None
    assert row["selected"] is True
    assert summary["unscored_fields"] == 1
    assert summary["phi4_failures_or_missing"] == 1
    assert summary["nemotron_failures"] == 1


def test_duplicate_phi4_results_are_rejected(tmp_path: Path) -> None:
    review_queue, phi4_results = _write_inputs(
        tmp_path,
        [("F1", "P1", "A", 0.4, "a", "b", "success")],
    )
    row = {"field_id": "F1", "status": "success", "prediction": "b"}
    phi4_results.write_text(json.dumps({"rows": [row, row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate Phi-4 field_id"):
        ranker.rank_review_queue(
            review_queue,
            phi4_results,
            tmp_path / "ranked",
        )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("Same TEXT", "same text", 0.0), ("abc", "axc", 1 / 3), ("", "x", 1.0)],
)
def test_normalized_edit_distance(left: str, right: str, expected: float) -> None:
    assert ranker.normalized_edit_distance(left, right) == pytest.approx(
        expected, abs=1e-6
    )


def _write_inputs(
    root: Path,
    cases: list[tuple[str, str, str, float, str, str, str]],
) -> tuple[Path, Path]:
    review_dir = root / "review"
    model_dir = root / "models"
    source_dir = root / "source"
    review_dir.mkdir()
    model_dir.mkdir()
    source_dir.mkdir()
    review_rows = []
    phi4_rows: list[dict[str, Any]] = []
    for (
        field_id,
        case_id,
        family_id,
        confidence,
        candidate,
        prediction,
        status,
    ) in cases:
        crop = review_dir / f"{field_id}.png"
        Image.new("RGB", (12, 8), "white").save(crop)
        source = source_dir / f"{case_id}.png"
        if not source.exists():
            Image.new("RGB", (40, 30), "white").save(source)
        model_output = model_dir / f"{field_id}.json"
        model_output.write_text(
            json.dumps(
                {
                    "result": {
                        "pages": [
                            {
                                "regions": [
                                    {
                                        "id": f"region-{field_id}",
                                        "text": candidate,
                                        "confidence": confidence,
                                        "provider": "nemotron",
                                    }
                                ]
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        review_rows.append(
            {
                "field_id": field_id,
                "case_id": case_id,
                "family_id": family_id,
                "category_id": "C08",
                "review_reason": ranker.REVIEW_REASON,
                "review_crop_path": str(crop.relative_to(root)),
                "source_path": str(source),
                "reference": f"SECRET_GROUND_TRUTH_{field_id}",
                "top_matches": [
                    {
                        "model_output_path": str(model_output),
                        "region_ids": [f"region-{field_id}"],
                        "matched_text": candidate,
                    }
                ],
            }
        )
        phi4_rows.append(
            {
                "field_id": field_id,
                "status": status,
                "prediction": prediction,
                "error_type": "RuntimeError" if status == "failed" else None,
                "error": "fixture failure" if status == "failed" else None,
            }
        )
    review_queue = root / "review_needed.jsonl"
    review_queue.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(review_rows)),
        encoding="utf-8",
    )
    phi4_results = root / "phi4.json"
    phi4_results.write_text(
        json.dumps({"rows": list(reversed(phi4_rows))}), encoding="utf-8"
    )
    return review_queue, phi4_results
