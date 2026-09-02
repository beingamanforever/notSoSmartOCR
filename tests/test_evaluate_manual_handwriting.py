from __future__ import annotations

import json
from pathlib import Path

from experiments.evaluate_manual_handwriting import evaluate_manual_handwriting, main


def test_scores_localized_handwriting_failure_inclusively(tmp_path: Path) -> None:
    labels = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        labels,
        [
            _review("split", "Dose 5 mg", [0, 0, 100, 30], "field"),
            _review("no-proposal", "Follow up", [0, 0, 80, 30], "line"),
            _review("failed", "Failed page", [0, 0, 80, 30], "field"),
            _review("missing", "Missing run", [0, 0, 80, 30], "line"),
            _review("invalid", "Invalid run", [0, 0, 80, 30], "field"),
            _review("split", "not scored", [0, 40, 80, 60], "signature"),
            {
                **_review("split", "ambiguous", [0, 70, 80, 90], "line"),
                "legibility": "ambiguous",
            },
        ],
    )
    runs = tmp_path / "runs"
    _write_json(
        runs / "split.json",
        _run(
            1.0,
            [
                _region("one", "Dose", [2, 2, 32, 25], order=1),
                _region("two", "5", [35, 2, 45, 25], order=2),
                _region("three", "mg", [50, 2, 75, 25], order=3),
                _region("other", "wrong", [120, 2, 160, 25], order=4),
            ],
        ),
    )
    _write_json(
        runs / "no-proposal.json",
        _run(3.0, [_region("outside", "Follow up", [100, 0, 180, 30])]),
    )
    _write_json(
        runs / "failed.json",
        {"request_status": "failed", "elapsed_seconds": 5.0, "error": "private"},
    )
    (runs / "invalid.json").write_text("not json", encoding="utf-8")

    report = evaluate_manual_handwriting(labels, runs)

    assert report["labels"] == {
        "reviewed": 7,
        "eligible": 5,
        "excluded": {"not_legible": 1, "unsupported_region_type": 1},
    }
    assert report["proposal_recall"] == {"matched": 1, "total": 5, "rate": 0.2}
    assert report["recognition"] == {
        "count": 5,
        "exact_match": {"matched": 1, "total": 5, "rate": 0.2},
        "cer": 0.823529,
        "hallucinated_text_rate": 0.0,
        "normalized_edit_distance": 0.8,
        "missed_text_rate": 0.823529,
        "substitution_rate": 0.0,
    }
    assert report["providers"]["tesseract"]["proposal_recall"]["matched"] == 1
    assert (
        report["providers"]["tesseract"]["recognition"]["exact_match"]["matched"] == 1
    )
    assert report["run_failures"] == {
        "missing_run": {"pages": 1, "labels": 1},
        "invalid_run": {"pages": 1, "labels": 1},
        "failed_request": {"pages": 1, "labels": 1},
        "failed_pipeline": {"pages": 0, "labels": 0},
    }
    assert report["latency_seconds"] == {
        "count": 3,
        "p50": 3.0,
        "p95": 4.8,
        "invalid": 0,
    }
    serialized = json.dumps(report)
    for private_text in (
        "Dose 5 mg",
        "Follow up",
        "Failed page",
        "Missing run",
        "Invalid run",
        "private",
    ):
        assert private_text not in serialized


def test_supports_page_reviews_provider_candidates_and_cli(tmp_path: Path) -> None:
    labels = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        labels,
        [
            {
                "page_id": "page-one",
                "reviewer_id": "reviewer",
                "regions": [
                    {
                        "bbox": [10, 10, 90, 40],
                        "text": "beta blocker",
                        "legibility": "legible",
                        "region_type": "line",
                    }
                ],
            }
        ],
    )
    runs = tmp_path / "runs"
    _write_json(
        runs / "renamed.json",
        {
            "filename": "page-one.png",
            **_run(
                2.0,
                [
                    _region("second", "blocker", [45, 12, 85, 35], order=2),
                    _region("first", "beta", [12, 12, 42, 35], order=1),
                    _region(
                        "alternate",
                        "wrong",
                        [10, 10, 90, 40],
                        provider="alternate",
                    ),
                ],
                evidence_ids=["first", "second"],
            ),
        },
    )
    output = tmp_path / "report.json"

    assert (
        main(
            [
                "--labels",
                str(labels),
                "--model-output",
                str(runs),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metadata"] == {"label_scale": 1.0}
    assert report["recognition"]["exact_match"]["rate"] == 1.0
    assert report["providers"]["tesseract"]["recognition"]["exact_match"]["rate"] == 1.0
    assert report["providers"]["alternate"]["recognition"]["exact_match"]["rate"] == 0.0
    assert report["latency_seconds"]["p95"] == 2.0


def test_reference_text_cannot_choose_an_alternate_prediction(tmp_path: Path) -> None:
    labels = tmp_path / "reviewed.jsonl"
    primary_text = "private primary value"
    alternate_text = "private alternate value"
    _write_jsonl(
        labels,
        [
            _review("page", primary_text, [0, 0, 100, 30], "field"),
            _review("page", alternate_text, [0, 0, 100, 30], "field"),
        ],
    )
    runs = tmp_path / "runs"
    _write_json(
        runs / "page.json",
        _run(
            1.0,
            [
                _region(
                    "primary",
                    primary_text,
                    [0, 0, 100, 30],
                    provider="rendered-provider",
                ),
                _region(
                    "alternate",
                    alternate_text,
                    [0, 0, 100, 30],
                    provider="alternate-provider",
                ),
            ],
            evidence_ids=["primary"],
        ),
    )

    report = evaluate_manual_handwriting(labels, runs)

    assert report["recognition"]["exact_match"] == {
        "matched": 1,
        "total": 2,
        "rate": 0.5,
    }
    assert (
        report["providers"]["alternate-provider"]["recognition"]["exact_match"][
            "matched"
        ]
        == 1
    )
    serialized = json.dumps(report)
    assert primary_text not in serialized
    assert alternate_text not in serialized


def test_text_region_fallback_uses_coverage_then_tightness(tmp_path: Path) -> None:
    labels = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        labels,
        [_review("page", "tight prediction", [20, 10, 80, 30], "line")],
    )
    runs = tmp_path / "runs"
    _write_json(
        runs / "page.json",
        _run(
            1.0,
            [
                _region(
                    "wide",
                    "wide prediction",
                    [0, 0, 100, 40],
                    kind="text",
                ),
                _region(
                    "tight",
                    "tight prediction",
                    [20, 10, 80, 30],
                    kind="text",
                ),
            ],
            evidence_ids=[],
        ),
    )

    report = evaluate_manual_handwriting(labels, runs)

    assert report["recognition"]["exact_match"]["rate"] == 1.0


def test_scales_1800_high_labels_to_2200_high_model_geometry(tmp_path: Path) -> None:
    labels = tmp_path / "reviewed.jsonl"
    private_text = "private scaled handwriting"
    _write_jsonl(
        labels,
        [_review("page", private_text, [100, 900, 300, 1000], "field")],
    )
    runs = tmp_path / "runs"
    _write_json(
        runs / "page.json",
        _run(
            1.0,
            [_region("scaled", private_text, [125, 1105, 360, 1215])],
        ),
    )

    unscaled = evaluate_manual_handwriting(labels, runs)
    output = tmp_path / "scaled-report.json"
    scale = 2200 / 1800
    assert (
        main(
            [
                "--labels",
                str(labels),
                "--model-output",
                str(runs),
                "--output",
                str(output),
                "--label-scale",
                str(scale),
            ]
        )
        == 0
    )
    scaled = json.loads(output.read_text(encoding="utf-8"))

    assert unscaled["proposal_recall"]["rate"] == 0.0
    assert scaled["proposal_recall"]["rate"] == 1.0
    assert scaled["recognition"]["exact_match"]["rate"] == 1.0
    assert scaled["metadata"] == {"label_scale": scale}
    assert private_text not in json.dumps(scaled)


def _review(
    page_id: str,
    text: str,
    bbox: list[int],
    region_type: str,
) -> dict[str, object]:
    return {
        "page_id": page_id,
        "bbox": bbox,
        "transcription": text,
        "legibility": "legible",
        "region_type": region_type,
    }


def _run(
    seconds: float,
    regions: list[dict[str, object]],
    *,
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    if evidence_ids is None:
        evidence_ids = [
            region_id
            for region in regions
            if isinstance((region_id := region.get("id")), str)
        ]
    return {
        "timing": {"total_seconds": seconds},
        "result": {
            "status": "success",
            "pages": [
                {
                    "text": {
                        "value": "private page text",
                        "evidence_ids": evidence_ids,
                    },
                    "regions": regions,
                }
            ],
        },
    }


def _region(
    region_id: str,
    text: str,
    bbox: list[int],
    *,
    order: int | None = None,
    provider: str = "tesseract",
    kind: str = "word",
) -> dict[str, object]:
    left, top, right, bottom = bbox
    region: dict[str, object] = {
        "id": region_id,
        "kind": kind,
        "text": text,
        "provider": provider,
        "resolution": "resolved",
        "bounding_box": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
    }
    if order is not None:
        region["reading_order"] = order
    return region


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
