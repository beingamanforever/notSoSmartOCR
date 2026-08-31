from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

from ocr_pipeline.contracts import BoundingBox, TextRegion
from ocr_pipeline.openrouter import OpenRouterError, OpenRouterResult
from ocr_pipeline.providers import GLMOCRDirectReader, GLMOCRReader

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import cascade_benchmark  # noqa: E402


class FakeReader:
    name = "fake-local"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return [
            TextRegion(
                id="stable",
                kind="word",
                text="BASE",
                confidence=0.99,
                bounding_box=BoundingBox(0, 0, 30, 20),
                reading_order=1,
                provider=self.name,
            ),
            TextRegion(
                id="risky",
                kind="word",
                text="",
                confidence=0.1,
                bounding_box=BoundingBox(30, 0, 60, 20),
                reading_order=2,
                provider=self.name,
            ),
        ]


def test_cascade_cli_compares_local_and_repaired_without_gold_routing(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _clinocr_dataset(tmp_path)
    calls: list[tuple[str, str | None, int]] = []
    primary_calls = 0

    def fake_repair(image_path, prompt, schema, *, model, provider_slug, max_tokens):
        nonlocal primary_calls
        assert "RECOVERED" not in prompt
        assert "normal" not in prompt
        assert "poor" not in prompt
        region_id = schema["properties"]["id"]["enum"][0]
        calls.append((model, provider_slug, max_tokens))
        if model == "meta/muse-glimmer-30b":
            primary_calls += 1
        if primary_calls == 2 and model == "meta/muse-glimmer-30b":
            raise OpenRouterError("provider unavailable")
        return OpenRouterResult(
            content={"id": region_id, "text": "RECOVERED"},
            model=model,
            provider="Pinned Provider",
            usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            cost=0.0002,
            latency_ms=10.0,
            attempts=1,
        )

    captured_args = None

    def fake_reader_from_args(args):
        nonlocal captured_args
        captured_args = args
        return FakeReader()

    monkeypatch.setattr(cascade_benchmark, "repair_image", fake_repair)
    monkeypatch.setattr(cascade_benchmark, "_reader_from_args", fake_reader_from_args)
    output = tmp_path / "result.json"
    exit_code = cascade_benchmark.main(
        [
            "clinocr",
            str(dataset),
            str(output),
            "--reader",
            "paddleocr-vl",
            "--model",
            "meta/muse-glimmer-30b",
            "--provider",
            "pinned-provider",
            "--verifier-provider",
            "verifier-provider",
            "--limit-per-subset",
            "1",
            "--max-tokens",
            "512",
            "--backend",
            "vllm-server",
            "--device",
            "gpu:0",
            "--use-doc-orientation-classify",
            "--no-use-doc-unwarping",
        ]
    )

    assert exit_code == 0
    assert captured_args.reader == "paddleocr-vl"
    assert captured_args.backend == "vllm-server"
    assert captured_args.device == "gpu:0"
    assert captured_args.use_doc_orientation_classify is True
    assert captured_args.use_doc_unwarping is False
    assert calls == [
        ("meta/muse-glimmer-30b", "pinned-provider", 512),
        ("qwen/qwen3.8-flash", "verifier-provider", 512),
        ("meta/muse-glimmer-30b", "pinned-provider", 512),
    ]

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["repair_model"] == "meta/muse-glimmer-30b"
    assert result["verifier_model"] == "qwen/qwen3.8-flash"
    assert result["run_config"] == {
        "reader": "fake-local",
        "reader_options": {},
        "repair_model": "meta/muse-glimmer-30b",
        "provider_slug": "pinned-provider",
        "verifier_model": "qwen/qwen3.8-flash",
        "verifier_provider_slug": "verifier-provider",
        "selected_subset": None,
        "limit_per_subset": 1,
        "max_tokens": 512,
        "repair_protocol_version": "different-model-literal-agreement-v2",
    }
    assert result["repair_protocol_version"] == ("different-model-literal-agreement-v2")
    assert result["summary"]["local"]["cases"] == 2
    assert result["summary"]["repaired"]["cases"] == 2
    assert result["summary"]["local"]["wer"]["case_mean"] == 0.5
    assert result["summary"]["repaired"]["wer"]["case_mean"] == 0.25
    assert result["summary"]["repair_calls"] == 3
    assert result["summary"]["reported_repair_cost"] == 0.0004
    assert result["summary"]["unreported_cost_calls"] == 1
    assert result["summary"]["wall_latency_ms"] >= 0

    success, abstained = result["cases"]
    assert success["local"]["prediction"] == "BASE "
    assert success["repaired"]["prediction"] == "BASE RECOVERED"
    assert success["local"]["metrics"]["wer"]["rate"] == 0.5
    assert success["repaired"]["metrics"]["wer"]["rate"] == 0.0
    assert success["repair"]["calls"] == 2
    assert success["repair"]["primary_calls"] == 1
    assert success["repair"]["verifier_calls"] == 1
    assert success["repair"]["accepted"] == 1
    assert success["repair"]["reported_cost"] == 0.0004
    assert success["repair"]["unreported_cost_calls"] == 0
    assert success["repair"]["records"][0]["primary"]["model"] == (
        "meta/muse-glimmer-30b"
    )
    assert success["repair"]["records"][0]["verifier"]["model"] == (
        "qwen/qwen3.8-flash"
    )
    assert success["wall_latency_ms"] >= success["local_wall_latency_ms"]

    assert abstained["local"]["metrics"]["wer"]["rate"] == 0.5
    assert abstained["repaired"]["metrics"]["wer"]["rate"] == 0.5
    assert abstained["repaired"]["status"] == "partial"
    assert abstained["repair"]["calls"] == 1
    assert abstained["repair"]["primary_calls"] == 1
    assert abstained["repair"]["verifier_calls"] == 0
    assert abstained["repair"]["accepted"] == 0
    assert abstained["repair"]["abstained"] == 1
    assert abstained["repair"]["unreported_cost_calls"] == 1
    assert abstained["repair"]["records"][0]["reason"] == ("primary_openrouter_error")
    assert result["summary"]["repaired"]["failed_cases"] == 1
    assert result["summary"]["repaired"]["failure_codes"] == {
        "primary_openrouter_error": 1
    }
    assert all("sample_1_normal" not in case["id"] for case in result["cases"])


def test_subset_selection_never_reaches_repair_policy(tmp_path: Path) -> None:
    dataset = _clinocr_dataset(tmp_path)
    primary_prompts: list[str] = []
    verifier_prompts: list[str] = []

    def repair(image_path: Path, prompt: str, schema):
        primary_prompts.append(prompt)
        region_id = schema["properties"]["id"]["enum"][0]
        return OpenRouterResult(
            content={"id": region_id, "text": "RECOVERED"},
            model="qwen/qwen3.8-flash",
            provider=None,
            usage={},
            cost=None,
            latency_ms=1.0,
            attempts=1,
        )

    def verifier(image_path: Path, prompt: str, schema):
        verifier_prompts.append(prompt)
        assert "RECOVERED" not in prompt
        assert "reference" not in prompt.casefold()
        assert "subset" not in prompt.casefold()
        region_id = schema["properties"]["id"]["enum"][0]
        return OpenRouterResult(
            content={"id": region_id, "text": "RECOVERED"},
            model="meta/muse-glimmer-30b",
            provider=None,
            usage={},
            cost=None,
            latency_ms=1.0,
            attempts=1,
        )

    result = cascade_benchmark.run_benchmark(
        "clinocr",
        dataset,
        FakeReader(),
        "qwen/qwen3.8-flash",
        subset="poor",
        repair_call=repair,
        verifier_call=verifier,
    )

    assert result["summary"]["local"]["cases"] == 1
    assert result["cases"][0]["subset"] == "poor"
    assert len(primary_prompts) == 1
    assert len(verifier_prompts) == 1
    assert "poor" not in primary_prompts[0]
    assert "RECOVERED" not in primary_prompts[0]


def test_cascade_benchmark_selects_glm_readers(tmp_path: Path, monkeypatch) -> None:
    captured_readers = []

    def fake_run_benchmark(dataset, root, reader, model, **options):
        captured_readers.append(reader)
        return {"reader": reader.name}

    monkeypatch.setattr(cascade_benchmark, "run_benchmark", fake_run_benchmark)
    common = ["clinocr", str(tmp_path), str(tmp_path / "result.json")]

    assert (
        cascade_benchmark.main(
            [
                *common,
                "--reader",
                "glm-ocr",
                "--ocr-api-host",
                "localhost",
                "--ocr-api-port",
                "9000",
                "--layout-device",
                "cuda:0",
            ]
        )
        == 0
    )
    assert (
        cascade_benchmark.main(
            [*common, "--reader", "glm-ocr-direct", "--max-new-tokens", "256"]
        )
        == 0
    )

    sdk_reader, direct_reader = captured_readers
    assert isinstance(sdk_reader, GLMOCRReader)
    assert sdk_reader.ocr_api_host == "localhost"
    assert sdk_reader.ocr_api_port == 9000
    assert sdk_reader.layout_device == "cuda:0"
    assert isinstance(direct_reader, GLMOCRDirectReader)
    assert direct_reader.max_new_tokens == 256


def test_same_model_verifier_is_rejected_even_with_different_providers(
    tmp_path: Path,
) -> None:
    dataset = _clinocr_dataset(tmp_path)

    try:
        cascade_benchmark.run_benchmark(
            "clinocr",
            dataset,
            FakeReader(),
            "qwen/qwen3.8-flash",
            provider_slug="primary-provider",
            verifier_model="qwen/qwen3.8-flash",
            verifier_provider_slug="other-provider",
        )
    except ValueError as error:
        assert str(error) == "Verifier must use a different model"
    else:
        raise AssertionError("same-model agreement must not be called independent")


def _clinocr_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ClinOCR-Bench"
    dataset.mkdir()
    rows = [
        ("normal", "1", "1", "exemplar"),
        ("normal", "1", "2", "eval"),
        ("normal", "1", "3", "eval"),
        ("poor", "2", "1", "eval"),
    ]
    with (dataset / "oneshot_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["subset", "template", "sample", "role"])
        for subset, template, sample, role in rows:
            writer.writerow([subset, template, sample, role])
            stem = f"template_{template}_sample_{sample}_{subset}"
            scans = dataset / "scans" / subset
            ground_truth = dataset / "ground_truth" / subset
            scans.mkdir(parents=True, exist_ok=True)
            ground_truth.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 40), "white").save(scans / f"{stem}.png")
            reference = "EXEMPLAR" if role == "exemplar" else "BASE RECOVERED"
            (ground_truth / f"{stem}.txt").write_text(reference, encoding="utf-8")
    return dataset
