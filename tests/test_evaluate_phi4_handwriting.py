from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import evaluate_phi4_handwriting as evaluate


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
        assert evaluate.finetune.PROMPT in messages[0]["content"]
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
        return {
            10: "axc",
            11: "copied print",
            20: "abc",
            21: "123",
            22: evaluate.finetune.NO_HANDWRITING,
        }[generated[0]]


class FakeModel:
    def __init__(self) -> None:
        self.adapted = False
        self.calls = 0

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def generate(self, **options: Any) -> list[list[int]]:
        assert options["max_new_tokens"] == 128
        assert options["do_sample"] is False
        assert options["num_beams"] == 1
        self.calls += 1
        if self.adapted:
            return [[0, 1, 19 + self.calls]]
        if self.calls == 2:
            raise RuntimeError("private text must not escape")
        return [[0, 1, 10 if self.calls == 1 else 11]]


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


def test_compares_stock_and_adapter_without_persisting_private_text(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = tmp_path / "composed-v4"
    _write_record(dataset, "train", "train", "train-family", "train literal")
    _write_record(dataset, "dev", "one", "dev-family-a", "abc")
    _write_record(dataset, "dev", "two", "dev-family-a", "123")
    _write_record(
        dataset,
        "dev",
        "blank",
        "dev-family-b",
        "",
        target_state="absent",
        abstention_subtype="blank",
    )
    adapter = tmp_path / "vision_decoder_lora.pt"
    adapter.write_bytes(b"fixture")
    output = tmp_path / "evaluation.json"
    model = FakeModel()
    monkeypatch.setattr(
        evaluate.finetune, "enable_vision_decoder_lora", lambda model: {}
    )
    monkeypatch.setattr(
        evaluate.finetune,
        "load_trainable_state",
        _load_adapter,
    )

    payload = evaluate.run_evaluation(
        dataset,
        adapter,
        output,
        processor=FakeProcessor(),
        model=model,
        torch_module=FakeTorch(),
        clock=FakeClock(),
        warmup=False,
    )

    stock = payload["arms"]["stock"]["metrics"]
    adapted = payload["arms"]["adapter"]["metrics"]
    assert stock == {
        "fields": 3,
        "failed_fields": 1,
        "failure_codes": {"phi4_inference_failed": 1},
        "exact": 0,
        "exact_rate": 0.0,
        "cer": 0.666667,
        "missed_character_rate": 0.5,
        "hallucinated_character_rate": 0.0,
        "abstention_fields": 1,
        "abstention_accuracy": 0.0,
        "critical_substitutions": 1,
        "latency_ms": {"p50": 10.0, "p95": 10.0},
    }
    assert adapted["exact_rate"] == 1.0
    assert adapted["cer"] == 0.0
    assert adapted["abstention_accuracy"] == 1.0
    assert adapted["critical_substitutions"] == 0
    assert payload["dataset"]["family_disjoint"] is True
    assert payload["evidence"]["failures_remain_in_denominators"] is True
    persisted = output.read_text(encoding="utf-8")
    assert "train literal" not in persisted
    assert "train-family" not in persisted
    assert "dev-family-a" not in persisted
    assert '"field_id"' not in persisted
    assert '"family_id"' not in persisted
    assert "copied print" not in persisted
    assert "private text must not escape" not in persisted
    assert all(
        "prediction" not in row and "reference" not in row
        for arm in payload["arms"].values()
        for row in arm["rows"]
    )


def _write_record(
    root: Path,
    split: str,
    field_id: str,
    family_id: str,
    reference: str,
    *,
    target_state: str = "resolved",
    abstention_subtype: str | None = None,
) -> None:
    crop = root / "crops" / split / f"{field_id}.png"
    crop.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), "white").save(crop)
    reviews = [
        {
            "reviewer_id": reviewer,
            "transcription": reference if target_state == "resolved" else "",
            "legibility": "legible" if target_state == "resolved" else "not_applicable",
            "region_type": abstention_subtype or "field",
        }
        for reviewer in ("one", "two")
    ]
    record = {
        "field_id": field_id,
        "case_id": field_id,
        "category_id": "C08",
        "family_id": family_id,
        "reference": reference,
        "crop_path": crop.relative_to(root).as_posix(),
        "split": split,
        "data_origin": "private",
        "target_state": target_state,
        "reviewer_ids": ["one", "two"],
        "independent_reviews": reviews,
    }
    if abstention_subtype is not None:
        record["abstention_subtype"] = abstention_subtype
    with (root / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _load_adapter(
    model: FakeModel,
    path: Path,
    torch: Any,
    *,
    evaluation_family_ids: set[str],
) -> None:
    assert evaluation_family_ids == {"dev-family-a", "dev-family-b"}
    model.adapted = True
    model.calls = 0
