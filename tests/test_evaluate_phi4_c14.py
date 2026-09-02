from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from experiments import evaluate_phi4_c14 as evaluate


class FakeIds:
    def size(self, dimension: int) -> int:
        assert dimension == 1
        return 2


class FakeInputs(dict[str, Any]):
    input_ids = FakeIds()

    def to(self, device: str) -> FakeInputs:
        assert device == "cuda:0"
        return self


class FakeTokenizer:
    def apply_chat_template(self, messages: Any, **options: Any) -> str:
        assert evaluate.evaluate.finetune.PROMPT in messages[0]["content"]
        assert options == {"tokenize": False, "add_generation_prompt": True}
        return "prompt"


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, prompt: str, **options: Any) -> FakeInputs:
        assert prompt == "prompt"
        assert options["return_tensors"] == "pt"
        return FakeInputs(input_ids=FakeIds())

    def decode(self, generated: list[int], **options: Any) -> str:
        assert options == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return {10: "wrong private text", 20: "private literal"}[generated[0]]


class FakeModel:
    def __init__(self) -> None:
        self.adapted = False
        self.calls = 0
        self.config = SimpleNamespace(use_cache=False)

    def eval(self) -> None:
        return None

    def generate(self, **options: Any) -> list[list[int]]:
        assert options["max_new_tokens"] == 128
        assert options["do_sample"] is False
        assert options["num_beams"] == 1
        self.calls += 1
        if not self.adapted and self.calls == 2:
            raise RuntimeError("secret inference detail")
        return [[0, 1, 20 if self.adapted else 10]]


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class FakeClock:
    def __init__(self) -> None:
        self.value = -0.01

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def test_compares_fixed_pairs_without_persisting_private_text(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_root = _write_panel(tmp_path)
    adapter = tmp_path / "vision_decoder_lora.pt"
    adapter.write_bytes(b"fixture")
    output = tmp_path / "c14-evaluation.json"
    model = FakeModel()
    monkeypatch.setattr(
        evaluate.evaluate.finetune, "enable_vision_decoder_lora", lambda model: {}
    )
    monkeypatch.setattr(
        evaluate.evaluate.finetune,
        "load_trainable_state",
        _load_adapter,
    )

    payload = evaluate.run_evaluation(
        run_root,
        adapter,
        output,
        processor=FakeProcessor(),
        model=model,
        torch_module=FakeTorch(),
        clock=FakeClock(),
        warmup=False,
    )

    stock = payload["arms"]["stock"]
    adapted = payload["arms"]["adapter"]
    assert model.calls == 88
    assert stock["metrics"]["fields"] == 88
    assert stock["metrics"]["failed_fields"] == 1
    assert stock["metrics"]["failure_codes"] == {"phi4_inference_failed": 1}
    assert stock["metrics"]["exact_rate"] == 0.0
    assert stock["metrics"]["cer"] > 0
    assert stock["metrics"]["missed_character_rate"] > 0
    assert stock["metrics"]["hallucinated_character_rate"] > 0
    assert stock["metrics"]["abstention_fields"] == 0
    assert stock["metrics"]["abstention_accuracy"] is None
    assert stock["metrics"]["critical_fields"] == 88
    assert stock["metrics"]["critical_substitutions"] == 87
    assert stock["views"]["native"]["fields"] == 44
    assert stock["views"]["scaled"]["fields"] == 44
    assert adapted["metrics"]["exact_rate"] == 1.0
    assert adapted["metrics"]["cer"] == 0.0
    assert adapted["metrics"]["critical_substitutions"] == 0
    assert payload["evidence"] == {
        "paired_fields_complete": True,
        "identical_pairs": True,
        "failures_remain_in_denominators": True,
        "private_text_persisted": False,
    }
    assert all(
        {"pair_id", "field_id", "case_id", "view", "bbox", "status"} <= row.keys()
        and "prediction" not in row
        and "reference" not in row
        for arm in payload["arms"].values()
        for row in arm["rows"]
    )
    persisted = output.read_text(encoding="utf-8")
    assert "private literal" not in persisted
    assert "wrong private text" not in persisted
    assert "secret inference detail" not in persisted


def _write_panel(tmp_path: Path) -> Path:
    run_root = tmp_path / "c14"
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
                "reference": "private literal",
                "native": native,
                "scaled": scaled,
            }
        )
    (run_root / "ground_truth.json").write_text(json.dumps(records), encoding="utf-8")
    return run_root


def _load_adapter(
    model: FakeModel,
    path: Path,
    torch: Any,
    *,
    evaluation_family_ids: set[str],
) -> None:
    assert evaluation_family_ids == {"C14-D001-P001", "C14-D002-P001"}
    model.adapted = True
    model.calls = 0
