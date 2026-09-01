from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.openrouter import OpenRouterError, OpenRouterResult

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import frontier_benchmark  # noqa: E402


@pytest.mark.parametrize(
    "model",
    [
        "qwen/qwen3.7-flash",
        "qwen/qwen3.8-flash",
        "qwen/qwen2.5-vl-72b-instruct",
        "z-ai/glm-5.3-flash",
        "google/gemini-3.7-flash",
        "deepseek/deepseek-v4-flash-vision-exp",
        "meta/muse-glimmer-30b",
        "anthropic/claude-opus-5",
    ],
)
def test_frontier_cli_accepts_current_public_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    calls = []

    def fake_run(dataset, root, requested_model, **options):
        calls.append((dataset, root, requested_model, options["provider_slug"]))
        return {"requested_model": requested_model}

    monkeypatch.setattr(frontier_benchmark, "run_benchmark", fake_run)
    output = tmp_path / "result.json"

    assert (
        frontier_benchmark.main(
            [
                "clinocr",
                str(tmp_path),
                str(output),
                "--model",
                model,
                "--provider",
                "pinned-provider",
            ]
        )
        == 0
    )
    assert calls == [("clinocr", tmp_path, model, "pinned-provider")]
    assert json.loads(output.read_text(encoding="utf-8")) == {"requested_model": model}


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

    def fake_repair(
        image_path,
        prompt,
        schema,
        *,
        model,
        max_tokens,
        provider_slug,
        public_benchmark,
    ):
        assert prompt == frontier_benchmark.PROMPT
        assert schema == frontier_benchmark.TEXT_SCHEMA
        assert public_benchmark is True
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
            "--clinocr-role",
            "eval",
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
        "clinocr_role": "eval",
        "selected_subsets": ["normal", "poor"],
        "limit_per_subset": 1,
        "prompt_version": "literal-transcription-v1",
        "max_tokens": 512,
    }
    assert result["summary"]["cases"] == 2
    assert result["clinocr_role"] == "eval"
    assert result["summary"]["covered_cases"] == 1
    assert result["summary"]["coverage"] == 0.5
    assert result["summary"]["failed_cases"] == 1
    assert result["summary"]["failure_rate"] == 0.5
    assert result["summary"]["abstained_cases"] == 0
    assert result["summary"]["abstention_rate"] == 0.0
    assert result["summary"]["pipeline_failures"] == 1
    assert result["summary"]["status_counts"] == {"failed": 1, "success": 1}
    assert result["summary"]["failure_codes"] == {"openrouter_error": 1}
    assert result["summary"]["cer"]["case_mean"] == 0.5
    assert result["summary"]["wer"]["case_mean"] == 0.5
    assert result["summary"]["reported_cost"] == 0.0002
    assert result["summary"]["cost_per_page"] is None
    assert result["summary"]["cost_per_reported_page"] == 0.0002
    assert result["summary"]["unreported_cost_cases"] == 1
    assert result["summary"]["wall_latency_ms"] >= 0
    assert result["summary"]["pages_per_second"] > 0

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


def test_successful_empty_frontier_response_is_an_abstention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    case = frontier_benchmark.BenchmarkCase(
        id="normal/blank",
        cluster_id="blank",
        subset="normal",
        image_path=image_path,
        reference="Expected text",
    )
    monkeypatch.setattr(
        frontier_benchmark,
        "repair_image",
        lambda *args, **kwargs: OpenRouterResult(
            content={"text": ""},
            model="meta/muse-glimmer-30b",
            provider="Pinned Provider",
            usage={},
            cost=None,
            latency_ms=4.0,
            attempts=1,
        ),
    )

    record = frontier_benchmark._evaluate_case(
        case,
        tmp_path,
        "meta/muse-glimmer-30b",
        "pinned-provider",
        512,
    )
    summary = frontier_benchmark._summarize([record])

    assert record["status"] == "success"
    assert record["failures"] == []
    assert summary["failed_cases"] == 0
    assert summary["abstained_cases"] == 1
    assert summary["abstention_rate"] == 1.0
    assert summary["reported_cost"] is None
    assert summary["cost_per_page"] is None
    assert summary["cost_per_reported_page"] is None
    assert summary["unreported_cost_cases"] == 1


def test_frontier_benchmark_requires_a_pinned_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Provider slug is required"):
        frontier_benchmark.run_benchmark(
            "clinocr",
            tmp_path,
            "meta/muse-glimmer-30b",
        )
