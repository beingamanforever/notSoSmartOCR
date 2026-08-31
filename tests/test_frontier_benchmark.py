from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

from ocr_pipeline.openrouter import OpenRouterError, OpenRouterResult

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import frontier_benchmark  # noqa: E402


def test_frontier_cli_keeps_provider_failures_in_denominator(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "ClinOCR-Bench"
    dataset.mkdir()
    rows = [
        ("normal", "1", "1", "exemplar", "EXEMPLAR"),
        ("normal", "1", "2", "eval", "Hello World"),
        ("normal", "1", "3", "eval", "LIMITED OUT"),
        ("poor", "2", "1", "eval", "Missing Text"),
    ]
    with (dataset / "oneshot_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["subset", "template", "sample", "role"])
        for subset, template, sample, role, reference in rows:
            writer.writerow([subset, template, sample, role])
            stem = f"template_{template}_sample_{sample}_{subset}"
            scans = dataset / "scans" / subset
            ground_truth = dataset / "ground_truth" / subset
            scans.mkdir(parents=True, exist_ok=True)
            ground_truth.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 40), "white").save(scans / f"{stem}.png")
            (ground_truth / f"{stem}.txt").write_text(reference, encoding="utf-8")

    calls: list[tuple[str, str | None, int]] = []

    def fake_repair(image_path, prompt, schema, *, model, max_tokens, provider_slug):
        assert prompt == frontier_benchmark.PROMPT
        assert schema == frontier_benchmark.TEXT_SCHEMA
        calls.append((model, provider_slug, max_tokens))
        if "poor" in str(image_path):
            raise OpenRouterError("provider unavailable")
        return OpenRouterResult(
            content={"text": "<table><tr><td>HELLO</td><td>WORLD</td></tr></table>"},
            model=model,
            provider="Pinned Provider",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
                "cost": 0.0002,
            },
            cost=0.0002,
            latency_ms=12.5,
            attempts=1,
        )

    monkeypatch.setattr(frontier_benchmark, "repair_image", fake_repair)
    output = tmp_path / "result.json"
    exit_code = frontier_benchmark.main(
        [
            "clinocr",
            str(dataset),
            str(output),
            "--model",
            "meta/muse-glimmer-30b",
            "--provider",
            "pinned-provider",
            "--subset",
            "normal",
            "--subset",
            "poor",
            "--limit-per-subset",
            "1",
            "--max-tokens",
            "512",
        ]
    )

    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [
        ("meta/muse-glimmer-30b", "pinned-provider", 512),
        ("meta/muse-glimmer-30b", "pinned-provider", 512),
    ]
    assert result["requested_model"] == "meta/muse-glimmer-30b"
    assert result["provider_slug"] == "pinned-provider"
    assert result["selected_subsets"] == ["normal", "poor"]
    assert result["prompt_version"] == "literal-transcription-v1"
    assert result["run_config"] == {
        "reader": "openrouter-direct",
        "requested_model": "meta/muse-glimmer-30b",
        "provider_slug": "pinned-provider",
        "selected_subsets": ["normal", "poor"],
        "limit_per_subset": 1,
        "prompt_version": "literal-transcription-v1",
        "max_tokens": 512,
    }
    assert result["summary"]["cases"] == 2
    assert result["summary"]["covered_cases"] == 1
    assert result["summary"]["coverage"] == 0.5
    assert result["summary"]["failed_cases"] == 1
    assert result["summary"]["pipeline_failures"] == 1
    assert result["summary"]["status_counts"] == {"failed": 1, "success": 1}
    assert result["summary"]["failure_codes"] == {"openrouter_error": 1}
    assert result["summary"]["cer"]["case_mean"] == 0.5
    assert result["summary"]["wer"]["case_mean"] == 0.5

    success, failure = result["cases"]
    assert success["id"].startswith("normal/")
    assert success["prediction"] == "HELLO WORLD"
    assert success["reference"] == "Hello World"
    assert success["metrics"]["cer"]["rate"] == 0.0
    assert success["model"] == "meta/muse-glimmer-30b"
    assert success["provider"] == "Pinned Provider"
    assert success["usage"]["total_tokens"] == 22
    assert success["cost"] == 0.0002
    assert success["api_latency_ms"] == 12.5
    assert success["finish_reason"] == "stop"
    assert success["provider_error"] is None
    assert success["wall_latency_ms"] >= 0
    assert success["latency_ms"] == success["wall_latency_ms"]

    assert failure["id"].startswith("poor/")
    assert failure["prediction"] == ""
    assert failure["status"] == "failed"
    assert failure["metrics"]["cer"]["rate"] == 1.0
    assert failure["metrics"]["wer"]["rate"] == 1.0
    assert failure["model"] is None
    assert failure["provider"] is None
    assert failure["usage"] == {}
    assert failure["cost"] is None
    assert failure["api_latency_ms"] is None
    assert failure["finish_reason"] is None
    assert failure["provider_error"] == {
        "code": "openrouter_error",
        "status_code": None,
        "attempts": None,
        "latency_ms": None,
    }
    assert failure["wall_latency_ms"] >= 0
    assert failure["failures"] == [
        {
            "stage": "provider",
            "code": "openrouter_error",
            "message": "provider unavailable",
        }
    ]
    assert all("sample_1_normal" not in case["id"] for case in result["cases"])


def test_run_benchmark_rejects_unsupported_model(tmp_path: Path) -> None:
    try:
        frontier_benchmark.run_benchmark("clinocr", tmp_path, "qwen/qwen3-32b")
    except ValueError as error:
        assert str(error) == "Unsupported frontier model: qwen/qwen3-32b"
    else:
        raise AssertionError("text-only model must be rejected")
