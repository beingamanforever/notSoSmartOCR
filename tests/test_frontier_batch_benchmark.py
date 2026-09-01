from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from urllib.request import Request

import pytest
from PIL import Image

from ocr_pipeline import openrouter_batch
from ocr_pipeline.openrouter import GEMINI_37_FLASH_BATCH_MODEL, GEMINI_37_FLASH_MODEL
from ocr_pipeline.openrouter_batch import BatchItemResult, OpenRouterBatchResult

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import frontier_batch_benchmark  # noqa: E402


def test_batch_benchmark_scores_unordered_partial_results_failure_inclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clinocr_root(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[Request] = []

    def transport(request: Request, timeout: float):
        calls.append(request)
        if request.method == "POST":
            return 200, {}, _json({"id": "batch-123", "status": "validating"})
        return (
            200,
            {},
            _json(
                {
                    "id": "batch-123",
                    "status": "completed",
                    "results": [
                        {
                            "custom_id": "poor/template_2_sample_1_poor",
                            "response": None,
                            "error": {
                                "code": "provider_error",
                                "message": "provider unavailable",
                                "status_code": 503,
                            },
                        },
                        {
                            "custom_id": "normal/template_1_sample_2_normal",
                            "response": {
                                "model": GEMINI_37_FLASH_MODEL,
                                "openrouter_metadata": {
                                    "endpoints": {
                                        "available": [
                                            {
                                                "provider": "Google Vertex",
                                                "selected": True,
                                            }
                                        ]
                                    }
                                },
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps(
                                                {
                                                    "text": "<table><tr><td>HELLO</td>"
                                                    "<td>WORLD</td></tr></table>"
                                                }
                                            )
                                        },
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {"total_tokens": 22, "cost": 0.0002},
                                "latency_ms": 12.5,
                            },
                            "error": None,
                        },
                    ],
                }
            ),
        )

    def fake_transport_batch(*args, **kwargs):
        return openrouter_batch.repair_images_batch(
            *args,
            **kwargs,
            poll_interval_seconds=0,
            transport=transport,
            sleeper=lambda seconds: None,
        )

    monkeypatch.setattr(
        frontier_batch_benchmark, "repair_images_batch", fake_transport_batch
    )
    payload = frontier_batch_benchmark.run_benchmark(
        "clinocr",
        root,
        provider_slug="google-vertex",
        clinocr_role="eval",
        selected_subsets={"normal", "poor"},
        limit_per_subset=1,
        max_tokens=512,
    )

    submitted = json.loads(calls[0].data)
    assert [request["custom_id"] for request in submitted["requests"]] == [
        "normal/template_1_sample_2_normal",
        "poor/template_2_sample_1_poor",
    ]
    first_body = submitted["requests"][0]["body"]
    assert first_body["messages"][0]["content"][0]["text"] == (
        frontier_batch_benchmark.PROMPT
    )
    assert first_body["response_format"]["json_schema"]["schema"] == (
        frontier_batch_benchmark.TEXT_SCHEMA
    )
    assert first_body["max_tokens"] == 512
    assert payload["run_config"] == {
        "reader": "openrouter-batch",
        "requested_model": GEMINI_37_FLASH_BATCH_MODEL,
        "submitted_model": GEMINI_37_FLASH_MODEL,
        "provider_slug": "google-vertex",
        "batch_id": None,
        "clinocr_role": "eval",
        "selected_subsets": ["normal", "poor"],
        "limit_per_subset": 1,
        "prompt_version": "literal-transcription-v1",
        "max_tokens": 512,
    }
    assert payload["batch"] == {
        "id": "batch-123",
        "resumed": False,
        "submission_config_verified": True,
        "status": "completed",
        "polls": 1,
        "latency_ms": payload["batch"]["latency_ms"],
        "wall_latency_ms": payload["batch"]["wall_latency_ms"],
        "retention": {
            "days": 30,
            "scope": "inputs_and_results",
            "caveat": (
                "OpenRouter retains batch inputs and results for 30 days; "
                "submit public benchmark data only."
            ),
        },
    }
    assert payload["summary"]["cases"] == 2
    assert payload["summary"]["covered_cases"] == 1
    assert payload["summary"]["coverage"] == 0.5
    assert payload["summary"]["failed_cases"] == 1
    assert payload["summary"]["failure_rate"] == 0.5
    assert payload["summary"]["abstained_cases"] == 0
    assert payload["summary"]["abstention_rate"] == 0.0
    assert payload["summary"]["failure_codes"] == {"provider_error": 1}
    assert payload["summary"]["cer"]["case_mean"] == 0.5
    assert payload["summary"]["wer"]["case_mean"] == 0.5
    assert payload["summary"]["latency_ms"] == {
        "basis": "provider_api_latency_ms",
        "observed_cases": 1,
        "missing_cases": 1,
        "p50": 12.5,
        "p95": 12.5,
    }
    assert payload["summary"]["reported_cost"] == 0.0002
    assert payload["summary"]["cost_per_page"] is None
    assert payload["summary"]["cost_per_reported_page"] == 0.0002
    assert payload["summary"]["unreported_cost_cases"] == 1
    assert payload["summary"]["wall_latency_ms"] == payload["batch"]["wall_latency_ms"]
    assert payload["summary"]["pages_per_second"] > 0
    assert payload["summary"]["throughput_basis"] == "submission_to_completed_wall"

    success, failure = payload["cases"]
    assert success["prediction"] == "HELLO WORLD"
    assert success["metrics"]["cer"]["rate"] == 0.0
    assert success["model"] == GEMINI_37_FLASH_MODEL
    assert success["provider"] == "Google Vertex"
    assert success["usage"] == {"total_tokens": 22, "cost": 0.0002}
    assert success["cost"] == 0.0002
    assert success["api_latency_ms"] == 12.5
    assert success["latency_ms"] == 12.5
    assert failure["prediction"] == ""
    assert failure["metrics"]["cer"]["rate"] == 1.0
    assert failure["provider_error"] == {
        "code": "provider_error",
        "status_code": 503,
        "attempts": None,
        "latency_ms": None,
    }
    assert failure["latency_ms"] == payload["batch"]["latency_ms"]


def test_cli_writes_result_only_after_batch_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested" / "result.json"
    calls = []

    def fake_run(dataset, root, model, **options):
        assert not output.exists()
        calls.append((dataset, root, model, options))
        return {"batch": {"status": "completed"}}

    monkeypatch.setattr(frontier_batch_benchmark, "run_benchmark", fake_run)
    assert (
        frontier_batch_benchmark.main(
            [
                "funsd",
                str(tmp_path),
                str(output),
                "--provider",
                "google-vertex",
                "--batch-id",
                "batch-existing",
                "--max-tokens",
                "64",
            ]
        )
        == 0
    )
    assert calls[0][2] == GEMINI_37_FLASH_BATCH_MODEL
    assert calls[0][3]["provider_slug"] == "google-vertex"
    assert calls[0][3]["batch_id"] == "batch-existing"
    assert json.loads(output.read_text()) == {"batch": {"status": "completed"}}


def test_runner_marks_resumed_submission_config_as_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clinocr_root(tmp_path)

    def fake_resume(images, prompt, schema, **options):
        assert options["batch_id"] == "batch-existing"
        return OpenRouterBatchResult(
            batch_id="batch-existing",
            status="completed",
            items=(
                BatchItemResult(
                    custom_id="normal/template_1_sample_2_normal",
                    content={"text": ""},
                    model=GEMINI_37_FLASH_MODEL,
                    provider="Google Vertex",
                    usage={},
                    finish_reason="stop",
                ),
            ),
            latency_ms=5.0,
            polls=1,
        )

    monkeypatch.setattr(frontier_batch_benchmark, "repair_images_batch", fake_resume)
    payload = frontier_batch_benchmark.run_benchmark(
        "clinocr",
        root,
        provider_slug="google-vertex",
        batch_id="batch-existing",
        clinocr_role="eval",
        selected_subsets={"normal"},
        limit_per_subset=1,
    )

    assert payload["run_config"]["batch_id"] == "batch-existing"
    assert payload["batch"]["id"] == "batch-existing"
    assert payload["batch"]["resumed"] is True
    assert payload["batch"]["submission_config_verified"] is False
    assert payload["summary"]["failed_cases"] == 0
    assert payload["summary"]["failure_rate"] == 0.0
    assert payload["summary"]["abstained_cases"] == 1
    assert payload["summary"]["abstention_rate"] == 1.0
    assert payload["summary"]["reported_cost"] is None
    assert payload["summary"]["cost_per_page"] is None
    assert payload["summary"]["cost_per_reported_page"] is None
    assert payload["summary"]["unreported_cost_cases"] == 1
    assert payload["summary"]["wall_latency_ms"] is None
    assert payload["summary"]["pages_per_second"] is None
    assert payload["summary"]["throughput_basis"] == ("unavailable_for_resumed_batch")
    assert payload["cases"][0]["status"] == "success"
    assert payload["cases"][0]["failures"] == []


def test_batch_runner_rejects_other_models_and_requires_provider(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unsupported frontier batch model"):
        frontier_batch_benchmark.run_benchmark(
            "clinocr",
            tmp_path,
            GEMINI_37_FLASH_MODEL,
            provider_slug="google-vertex",
        )
    with pytest.raises(ValueError, match="Provider slug is required"):
        frontier_batch_benchmark.run_benchmark("clinocr", tmp_path)


def _clinocr_root(tmp_path: Path) -> Path:
    root = tmp_path / "ClinOCR-Bench"
    root.mkdir()
    rows = [
        ("normal", "1", "1", "exemplar", "EXEMPLAR"),
        ("normal", "1", "2", "eval", "Hello World"),
        ("poor", "2", "1", "eval", "Missing Text"),
    ]
    with (root / "oneshot_lookup.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["subset", "template", "sample", "role"])
        for subset, template, sample, role, reference in rows:
            writer.writerow([subset, template, sample, role])
            stem = f"template_{template}_sample_{sample}_{subset}"
            scans = root / "scans" / subset
            ground_truth = root / "ground_truth" / subset
            scans.mkdir(parents=True, exist_ok=True)
            ground_truth.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 40), "white").save(scans / f"{stem}.png")
            (ground_truth / f"{stem}.txt").write_text(reference, encoding="utf-8")
    return root


def _json(value: object) -> bytes:
    return json.dumps(value).encode()
