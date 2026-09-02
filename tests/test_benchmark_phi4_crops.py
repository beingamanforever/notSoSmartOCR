from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import benchmark_phi4 as phi4
from experiments import benchmark_phi4_crops as benchmark


class FakeTensor:
    def __init__(self, values: list[int], shape: tuple[int, ...]) -> None:
        self.values = values
        self.shape = shape
        self.dtype = "fake-float"

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def reshape(self, *shape: int) -> FakeTensor:
        assert shape == (-1,)
        return self

    def tolist(self) -> list[int]:
        return self.values


class FakeInputs(dict[str, Any]):
    def to(self, device: str) -> FakeInputs:
        assert device == "cpu"
        return self


class FakeProcessor:
    def __init__(self, prediction: str = "Exact literal") -> None:
        self.prediction = prediction
        self.prompts: list[str] = []
        self.image_sizes: list[tuple[int, int]] = []
        self.image_processor = SimpleNamespace(dynamic_hd=True, max_num_crops=16)

    def __call__(self, **options: Any) -> FakeInputs:
        prompt = options["text"]
        image = options["images"]
        assert prompt.startswith("<|user|><|image_1|>")
        assert prompt.endswith("<|end|><|assistant|>")
        assert options["return_tensors"] == "pt"
        assert isinstance(image, Image.Image)
        self.prompts.append(prompt)
        self.image_sizes.append(image.size)
        return FakeInputs(
            input_ids=FakeTensor([1, 2, 3], (1, 3)),
            pixel_values=FakeTensor(list(range(12)), (1, 3, 2, 2)),
        )

    def decode(self, tokens: list[int], **options: Any) -> str:
        assert options == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return self.prediction


class FakeModel:
    device = "cpu"

    def __init__(self, generated_tokens: int = 2) -> None:
        self.generated_tokens = generated_tokens
        self.calls = 0
        self.config = SimpleNamespace(
            _name_or_path=phi4.MODEL_ID,
            _commit_hash=phi4.MODEL_REVISION,
        )

    def generate(self, **inputs: Any) -> list[list[int]]:
        self.calls += 1
        assert inputs["max_new_tokens"] == 32
        assert inputs["do_sample"] is False
        assert inputs["num_beams"] == 1
        assert inputs["generation_config"] == "fixed-generation-config"
        return [[1, 2, 3, *range(100, 100 + self.generated_tokens)]]


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    __version__ = "fixture"
    version = SimpleNamespace(cuda=None)
    cuda = FakeCuda()

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


def test_runs_native_and_scaled_crops_with_local_evidence(tmp_path: Path) -> None:
    run_root = _write_crop_panel(tmp_path)
    output = tmp_path / "phi4-crops.json"
    processor = FakeProcessor()
    model = FakeModel()

    payload = benchmark.run_benchmark(
        run_root,
        output,
        max_new_tokens=32,
        device="cpu",
        warmup=False,
        processor=processor,
        model=model,
        generation_config="fixed-generation-config",
        torch_module=FakeTorch(),
    )

    assert model.calls == 88
    assert processor.image_sizes[:44] == [(8, 8)] * 44
    assert processor.image_sizes[44:] == [(24, 24)] * 44
    assert payload["model"]["requested_revision"] == (
        "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
    )
    assert payload["privacy"] == {
        "execution": "local_only",
        "private_uploads": False,
        "ground_truth_in_prompt": False,
        "model_downloads": False,
    }
    assert payload["evidence"] == {
        "paired_variants_complete": True,
        "failures_remain_in_denominators": True,
        "predictions_persisted_locally": True,
    }
    for variant in benchmark.VARIANTS:
        run = payload["runs"][variant]
        assert len(run["rows"]) == 44
        assert run["summary"]["normalized_exact"] == 44
        assert run["summary"]["character_insertions"] == 0
        assert run["summary"]["critical_substitutions"] == 0
        assert run["summary"]["latency_ms"].keys() == {"p50", "p95", "max"}
        assert run["rows"][0]["bbox"] == [0, 0, 8, 8]
        assert run["rows"][0]["source_file"] == (
            f"crops/C14-D001-P001-H001-{variant}.png"
        )
    assert payload["operations"]["cuda_memory_mib"]["available"] is False
    assert all(phi4.CROP_PROMPT in prompt for prompt in processor.prompts)
    assert all("Exact literal" not in prompt for prompt in processor.prompts)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"


def test_token_limit_is_failure_inclusive(tmp_path: Path) -> None:
    run_root = _write_crop_panel(tmp_path, symbolic_nulls=True)
    processor = FakeProcessor("Wrong literal")
    model = FakeModel(generated_tokens=32)

    payload = benchmark.run_benchmark(
        run_root,
        tmp_path / "failed.json",
        max_new_tokens=32,
        device="cpu",
        warmup=False,
        processor=processor,
        model=model,
        generation_config="fixed-generation-config",
        torch_module=FakeTorch(),
    )

    for variant in benchmark.VARIANTS:
        run = payload["runs"][variant]
        assert run["summary"]["fields"] == 44
        assert run["summary"]["failed_fields"] == 44
        assert run["summary"]["failure_codes"] == {"phi4_token_limit": 44}
        assert run["summary"]["critical_substitutions"] == 39
        assert all(row["status"] == "failed" for row in run["rows"])
        assert all(row["hit_token_limit"] for row in run["rows"])


def test_repetition_below_token_limit_is_failure_inclusive(tmp_path: Path) -> None:
    run_root = _write_crop_panel(tmp_path)
    processor = FakeProcessor("loop phrase " * 10)

    payload = benchmark.run_benchmark(
        run_root,
        tmp_path / "repetition.json",
        max_new_tokens=32,
        device="cpu",
        warmup=False,
        processor=processor,
        model=FakeModel(generated_tokens=2),
        generation_config="fixed-generation-config",
        torch_module=FakeTorch(),
    )

    for variant in benchmark.VARIANTS:
        run = payload["runs"][variant]
        assert run["summary"]["fields"] == 44
        assert run["summary"]["failed_fields"] == 44
        assert run["summary"]["failure_codes"] == {"phi4_repetition": 44}
        assert run["summary"]["character_insertions"] > 0
        assert all(row["status"] == "failed" for row in run["rows"])
        assert all(not row["hit_token_limit"] for row in run["rows"])
        assert all(row["repetition_detected"] for row in run["rows"])


def test_criticality_uses_annotation_content_not_field_ids() -> None:
    assert benchmark._is_critical_reference("Ø") is False
    assert benchmark._is_critical_reference(" ∅ ") is False
    assert benchmark._is_critical_reference("0") is True
    assert benchmark._is_critical_reference("Ø 97.6") is True
    assert benchmark._is_critical_reference("clinical literal") is True


def test_cli_reports_validation_errors_without_attribute_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as raised:
        benchmark.main(
            [
                str(tmp_path / "result.json"),
                "--run-root",
                str(tmp_path / "missing"),
            ]
        )

    assert raised.value.code == 2


def _write_crop_panel(tmp_path: Path, *, symbolic_nulls: bool = False) -> Path:
    run_root = tmp_path / "ministral-c14-crops-source-only"
    crop_root = run_root / "crops"
    crop_root.mkdir(parents=True)
    records = []
    for index in range(1, 45):
        case_id = "C14-D001-P001" if index <= 30 else "C14-D002-P001"
        field_index = index if index <= 30 else index - 30
        field_id = f"{case_id}-H{field_index:03d}"
        native = f"{field_id}-native.png"
        scaled = f"{field_id}-scaled.png"
        Image.new("RGB", (8, 8), "white").save(crop_root / native)
        Image.new("RGB", (24, 24), "white").save(crop_root / scaled)
        records.append(
            {
                "case_id": case_id,
                "field_id": field_id,
                "bbox": [0, 0, 8, 8],
                "reference": (
                    "Ø"
                    if symbolic_nulls
                    and case_id == "C14-D001-P001"
                    and 9 <= index <= 13
                    else "Exact literal"
                ),
                "native": native,
                "scaled": scaled,
            }
        )
    (run_root / "ground_truth.json").write_text(json.dumps(records), encoding="utf-8")
    return run_root
