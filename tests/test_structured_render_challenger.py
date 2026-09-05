from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.openrouter import OpenRouterResult

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import structured_render_challenger as challenger  # noqa: E402


def test_three_inputs_use_strict_evidence_and_only_one_sends_image(
    tmp_path: Path,
) -> None:
    manifest = _write_case(tmp_path)
    calls = []

    def fake_call(mode, image_path, prompt, schema, provider_slug, max_tokens):
        calls.append((mode, image_path, prompt, schema, provider_slug, max_tokens))
        return _result(
            _valid_prediction(
                challenger.PAGE_LITERAL_ID if mode == "flat_text" else "e1"
            )
        )

    report = challenger.run_benchmark(
        manifest,
        provider_slug="pinned-provider",
        max_tokens=512,
        challenger=fake_call,
    )

    assert [call[0] for call in calls] == list(challenger.MODES)
    assert calls[0][1] == tmp_path / "page.png"
    assert calls[1][1] is None
    assert calls[2][1] is None
    assert all(call[3] == challenger.OUTPUT_SCHEMA for call in calls)
    assert all(call[4:] == ("pinned-provider", 512) for call in calls)
    assert "page-literal" in calls[2][2]
    assert "Canonical evidence JSON" not in calls[2][2]
    assert "Flat literal OCR" in calls[2][2]
    assert report["challenger"] == "qwen/qwen3.7-flash"
    assert report["eligible_runtime"] is False
    assert report["cases"][0]["provenance"] == {"generator": "tests.synthetic_page"}
    assert all(
        report["summary"][mode]["acceptance_rate"] == 1.0 for mode in challenger.MODES
    )


def test_strict_validation_rejects_unknown_literals_and_invalid_boxes() -> None:
    source = _source()
    prediction = _valid_prediction("e1")
    prediction["page"]["regions"][0]["text"] = "invented diagnosis"
    prediction["page"]["regions"][0]["bounding_box"]["right"] = 101

    errors = challenger.validate_output(prediction, source)

    assert "page.regions[0].text contains unsupported literals" in errors
    assert "page.regions[0].bounding_box is outside the page" in errors


@pytest.mark.parametrize(
    ("source_text", "candidate"),
    [
        ("Dose 1 5 mg", "Dose 1.5 mg"),
        ("diabetes no", "no diabetes"),
        ("10 20 mg", "10-20 mg"),
    ],
)
def test_strict_validation_preserves_literal_order_and_punctuation(
    source_text: str, candidate: str
) -> None:
    source, prediction = _text_case(source_text, region_text=candidate)

    errors = challenger.validate_output(prediction, source)

    assert "page.regions[0].text contains unsupported literals" in errors


@pytest.mark.parametrize(
    ("source_text", "markdown"),
    [
        ("Dose 1 5 mg", "Dose 1.5 mg"),
        ("diabetes no", "no diabetes"),
        ("10 20 mg", "10-20 mg"),
    ],
)
def test_rendered_markdown_preserves_literal_order_and_punctuation(
    source_text: str, markdown: str
) -> None:
    source, prediction = _text_case(source_text, markdown=markdown)

    errors = challenger.validate_output(prediction, source)

    assert "rendered Markdown contains unsupported literals" in errors


@pytest.mark.parametrize(
    ("source_text", "markdown"),
    [
        ("Dose: 1.5 mg", "## **Dose:** 1.5 mg"),
        ("Follow-up", "- *Follow-up*"),
        ("Dose 10-20 mg", "| Dose |\n| --- |\n| 10-20 mg |"),
    ],
)
def test_rendered_markdown_allows_explicit_formatting(
    source_text: str, markdown: str
) -> None:
    source, prediction = _text_case(source_text, markdown=markdown)

    assert challenger.validate_output(prediction, source) == []


def test_strict_validation_rejects_shape_errors_before_semantic_checks() -> None:
    prediction = _valid_prediction("e1")
    prediction["page"]["regions"][0]["unexpected"] = True

    assert challenger.validate_output(prediction, _source()) == [
        "schema mismatch: $.page.regions[0] has unexpected unexpected"
    ]


def test_table_and_control_proposals_require_cited_evidence() -> None:
    source = {
        "width": 100,
        "height": 80,
        "evidence": [
            {
                "id": "cell",
                "kind": "text",
                "text": "Value",
                "reading_order": 0,
                "bounding_box": {"left": 10, "top": 10, "right": 40, "bottom": 20},
            },
            {
                "id": "label",
                "kind": "text",
                "text": "Agree",
                "reading_order": 1,
                "bounding_box": {"left": 10, "top": 30, "right": 45, "bottom": 40},
            },
        ],
    }
    prediction = {
        "page": {
            "width": 100,
            "height": 80,
            "regions": [
                {
                    "id": "table-1",
                    "kind": "table",
                    "text": "",
                    "evidence_ids": ["cell"],
                    "bounding_box": {"left": 5, "top": 5, "right": 50, "bottom": 25},
                    "reading_order": 0,
                    "structure": {
                        "type": "table",
                        "row_count": 1,
                        "column_count": 1,
                        "cells": [
                            {
                                "row": 0,
                                "column": 0,
                                "text": "Value",
                                "evidence_ids": ["cell"],
                                "bounding_box": {
                                    "left": 10,
                                    "top": 10,
                                    "right": 40,
                                    "bottom": 20,
                                },
                            }
                        ],
                        "state": "none",
                        "label_evidence_ids": [],
                    },
                },
                {
                    "id": "control-1",
                    "kind": "control",
                    "text": "Agree",
                    "evidence_ids": ["label"],
                    "bounding_box": {"left": 5, "top": 28, "right": 50, "bottom": 42},
                    "reading_order": 1,
                    "structure": {
                        "type": "control",
                        "row_count": 0,
                        "column_count": 0,
                        "cells": [],
                        "state": "checked",
                        "label_evidence_ids": ["label"],
                    },
                },
            ],
            "rendered_markdown": "| Value |\n| --- |\n\n- [x] Agree",
        }
    }

    assert challenger.validate_output(prediction, source) == []
    prediction["page"]["regions"][0]["structure"]["cells"][0]["evidence_ids"] = [
        "missing"
    ]
    assert any(
        "evidence_ids are invalid" in error
        for error in challenger.validate_output(prediction, source)
    )


def test_cli_requires_explicit_public_data_confirmation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        challenger.main(
            [
                str(tmp_path / "cases.json"),
                str(tmp_path / "out.json"),
                "--provider",
                "pinned",
            ]
        )


def test_cli_writes_failure_inclusive_three_mode_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_case(tmp_path)

    def fake_call(mode, image_path, prompt, schema, provider_slug, max_tokens):
        evidence_id = challenger.PAGE_LITERAL_ID if mode == "flat_text" else "e1"
        return _result(_valid_prediction(evidence_id))

    monkeypatch.setattr(challenger, "_openrouter_challenger", fake_call)
    output = tmp_path / "report.json"

    assert (
        challenger.main(
            [
                str(manifest),
                str(output),
                "--provider",
                "pinned-provider",
                "--confirm-public-data",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["cases"]) == 3
    assert {record["status"] for record in report["cases"]} == {"accepted"}
    assert all(report["summary"][mode]["abstained"] == 0 for mode in challenger.MODES)


def test_manifest_rejects_private_or_unproven_cases(tmp_path: Path) -> None:
    manifest = _write_case(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][0] = {
        **payload["cases"][0],
        "classification": "private",
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="classified public or synthetic"):
        challenger.run_benchmark(
            manifest, provider_slug="pinned", challenger=lambda *args: _result({})
        )


def _write_case(root: Path) -> Path:
    Image.new("RGB", (100, 80), "white").save(root / "page.png")
    canonical = {
        "schema_version": 2,
        "pages": [
            {
                "page_number": 1,
                "width": 100,
                "height": 80,
                "regions": _source()["evidence"],
            }
        ],
    }
    (root / "page.json").write_text(json.dumps(canonical), encoding="utf-8")
    manifest = root / "cases.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "synthetic-1",
                        "classification": "synthetic",
                        "generator": "tests.synthetic_page",
                        "image": "page.png",
                        "canonical": "page.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _source() -> dict:
    return {
        "width": 100,
        "height": 80,
        "evidence": [
            {
                "id": "e1",
                "kind": "text",
                "text": "Patient Name",
                "reading_order": 0,
                "bounding_box": {"left": 10, "top": 10, "right": 90, "bottom": 25},
            }
        ],
    }


def _text_case(
    source_text: str,
    *,
    region_text: str | None = None,
    markdown: str | None = None,
) -> tuple[dict, dict]:
    source = _source()
    source["evidence"][0]["text"] = source_text
    prediction = _valid_prediction("e1")
    prediction["page"]["regions"][0]["text"] = (
        source_text if region_text is None else region_text
    )
    prediction["page"]["rendered_markdown"] = (
        source_text if markdown is None else markdown
    )
    return source, prediction


def _valid_prediction(evidence_id: str) -> dict:
    return {
        "page": {
            "width": 100,
            "height": 80,
            "regions": [
                {
                    "id": "out-1",
                    "kind": "field",
                    "text": "Patient Name",
                    "evidence_ids": [evidence_id],
                    "bounding_box": {"left": 10, "top": 10, "right": 90, "bottom": 25},
                    "reading_order": 0,
                    "structure": {
                        "type": "none",
                        "row_count": 0,
                        "column_count": 0,
                        "cells": [],
                        "state": "none",
                        "label_evidence_ids": [],
                    },
                }
            ],
            "rendered_markdown": "**Patient Name**",
        }
    }


def _result(content: dict) -> OpenRouterResult:
    return OpenRouterResult(
        content=content,
        model=challenger.QWEN_37_FLASH_MODEL,
        provider="Pinned Provider",
        usage={"cost": 0.001},
        cost=0.001,
        latency_ms=10,
        attempts=1,
    )
