from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import benchmark_phi4 as benchmark


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
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_processor = SimpleNamespace(
            dynamic_hd=True,
            max_num_crops=16,
            image_size=[448, 448],
        )

    def __call__(self, **options: Any) -> FakeInputs:
        prompt = options["text"]
        assert prompt.startswith("<|user|><|image_1|>")
        assert prompt.endswith("<|end|><|assistant|>")
        assert options["return_tensors"] == "pt"
        assert isinstance(options["images"], Image.Image)
        self.prompts.append(prompt)
        return FakeInputs(
            input_ids=FakeTensor([1, 2, 3], (1, 3)),
            pixel_values=FakeTensor(list(range(12)), (1, 3, 2, 2)),
            image_sizes=FakeTensor([8, 8], (1, 2)),
            num_crops=FakeTensor([1], (1,)),
        )

    def decode(self, tokens: list[int], **options: Any) -> str:
        assert tokens == [101, 102]
        assert options == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return "Exact literal"


class FakeModel:
    device = "cpu"

    def __init__(self, fail_at: set[int] | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at or set()
        self.config = SimpleNamespace(
            _name_or_path=benchmark.MODEL_ID,
            _commit_hash="a" * 40,
            model_type="phi4mm",
            architectures=["Phi4MMForCausalLM"],
        )

    def generate(self, **inputs: Any) -> list[list[int]]:
        self.calls += 1
        assert inputs["max_new_tokens"] == 32
        assert inputs["do_sample"] is False
        assert inputs["num_beams"] == 1
        assert inputs["generation_config"] == "fixed-generation-config"
        if self.calls in self.fail_at:
            raise RuntimeError("fixture inference failure")
        return [[1, 2, 3, 101, 102]]


class LimitProcessor(FakeProcessor):
    def decode(self, tokens: list[int], **options: Any) -> str:
        assert len(tokens) == 32
        return "repeated output"


class LimitModel(FakeModel):
    def generate(self, **inputs: Any) -> list[list[int]]:
        self.calls += 1
        return [[1, 2, 3, *range(32)]]


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


def test_triage_is_local_failure_inclusive_and_reject_only(tmp_path: Path) -> None:
    challenge = tmp_path / "challenge"
    for case_id in benchmark.DEFAULT_TRIAGE_IDS:
        _write_page(challenge, case_id)

    output = tmp_path / "triage.json"
    model = FakeModel(fail_at={3})
    payload = benchmark.run_benchmark(
        challenge,
        output,
        model_name=benchmark.MODEL_ID,
        revision="a" * 40,
        track="triage",
        max_new_tokens=32,
        device="cpu",
        warmup=False,
        processor=FakeProcessor(),
        model=model,
        generation_config="fixed-generation-config",
        torch_module=FakeTorch(),
    )

    records = payload["tracks"]["full_pages"]["cases"]
    assert len(records) == 5
    assert model.calls == 5
    assert payload["privacy"] == {
        "execution": "local_only",
        "private_uploads": False,
        "ground_truth_in_prompt": False,
    }
    assert payload["evidence"]["scope"] == "reject_only"
    assert payload["evidence"]["promotion_evidence_complete"] is False
    assert payload["tracks"]["full_pages"]["summary"]["failed_cases"] == 1
    assert payload["tracks"]["full_pages"]["handwriting_exact_recovery"] == {
        "eligible": 5,
        "recovered": 4,
        "rate": 0.8,
        "matching": "casefolded whitespace-normalized nonoverlapping exact spans",
    }
    assert records[2]["status"] == "failed"
    assert records[2]["metrics"]["cer"]["rate"] == 1.0
    assert records[0]["processor_metadata"]["crop_tile_fields"] == {
        "image_sizes": {"shape": [1, 2], "values": [8, 8]},
        "num_crops": {"shape": [1], "values": [1]},
    }
    assert len(payload["tracks"]["full_pages"]["manual_review"]) == 5
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"


def test_full_protocol_runs_22_pages_and_both_44_crop_arms(tmp_path: Path) -> None:
    challenge = tmp_path / "challenge"
    for index in range(1, 3):
        _write_page(challenge, f"C14-D{index:03d}-P001")
    for index in range(1, 21):
        _write_page(challenge, f"C08-D{index:03d}-P001")
    _write_crops(challenge)

    processor = FakeProcessor()
    model = FakeModel()
    payload = benchmark.run_benchmark(
        challenge,
        tmp_path / "full.json",
        model_name=benchmark.MODEL_ID,
        revision="a" * 40,
        track="full",
        max_new_tokens=32,
        device="cpu",
        warmup=False,
        processor=processor,
        model=model,
        generation_config="fixed-generation-config",
        torch_module=FakeTorch(),
    )

    assert model.calls == 22 + 44 + 44
    assert len(payload["tracks"]["full_pages"]["cases"]) == 22
    assert len(payload["tracks"]["c14_crops_native"]["cases"]) == 44
    assert len(payload["tracks"]["c14_crops_scaled"]["cases"]) == 44
    assert payload["tracks"]["c14_crops_native"]["summary"]["cer"]["micro"] == 0
    assert payload["tracks"]["c14_crops_scaled"]["summary"]["cer"]["micro"] == 0
    assert payload["evidence"]["scope"] == "promotion_evaluation"
    assert payload["evidence"]["promotion_evidence_complete"] is True
    assert all("Exact literal" not in prompt for prompt in processor.prompts)
    assert all(
        benchmark.PAGE_PROMPT in prompt or benchmark.CROP_PROMPT in prompt
        for prompt in processor.prompts
    )


def test_token_limit_is_a_failure_not_successful_coverage(tmp_path: Path) -> None:
    challenge = tmp_path / "challenge"
    for case_id in benchmark.DEFAULT_TRIAGE_IDS:
        _write_page(challenge, case_id)

    payload = benchmark.run_benchmark(
        challenge,
        tmp_path / "triage.json",
        model_name=benchmark.MODEL_ID,
        revision="a" * 40,
        track="triage",
        max_new_tokens=32,
        device="cpu",
        warmup=False,
        processor=LimitProcessor(),
        model=LimitModel(),
        generation_config="fixed-generation-config",
        torch_module=FakeTorch(),
    )

    summary = payload["tracks"]["full_pages"]["summary"]
    assert summary["failed_cases"] == 5
    assert summary["failure_codes"] == {"phi4_token_limit": 5}
    assert all(
        record["status"] == "failed" and record["hit_token_limit"]
        for record in payload["tracks"]["full_pages"]["cases"]
    )


def _write_page(challenge: Path, case_id: str) -> None:
    category = case_id.split("-", 1)[0]
    annotation = challenge / "annotations" / "primary" / category / f"{case_id}.json"
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "source_only": True,
                "page_legibility": "complete",
                "challenges": ["fixture challenge"],
                "handwriting": [{"text": "Exact literal", "legibility": "legible"}],
                "transcription": {
                    "reading_order_text": "Exact literal",
                    "unresolved_spans": [],
                },
            }
        ),
        encoding="utf-8",
    )
    source = (
        challenge / "sources" / benchmark.CATEGORY_SOURCES[category] / f"{case_id}.png"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(source)


def _write_crops(challenge: Path) -> None:
    run_root = challenge / "runs" / "ministral-c14-crops-source-only"
    crop_root = run_root / "crops"
    crop_root.mkdir(parents=True)
    records = []
    for index in range(1, 45):
        field_id = f"C14-D001-P001-H{index:03d}"
        native = f"{field_id}-native.png"
        scaled = f"{field_id}-scaled.png"
        Image.new("RGB", (8, 8), "white").save(crop_root / native)
        Image.new("RGB", (16, 16), "white").save(crop_root / scaled)
        records.append(
            {
                "case_id": "C14-D001-P001",
                "field_id": field_id,
                "bbox": [0, 0, 8, 8],
                "reference": "Exact literal",
                "native": native,
                "scaled": scaled,
            }
        )
    (run_root / "ground_truth.json").write_text(json.dumps(records), encoding="utf-8")
