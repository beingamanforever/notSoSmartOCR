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
from experiments import benchmark_got as benchmark


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


class FakeModel:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            _name_or_path=benchmark.MODEL_ID,
            _commit_hash=benchmark.MODEL_REVISION,
            model_type="GOT",
            architectures=["GOTQwenForCausalLM"],
        )

    def chat_crop(self, tokenizer: Any, image: Image.Image, **options: Any) -> str:
        return self._respond("multi_crop", tokenizer, image, options)

    def chat(self, tokenizer: Any, image: Image.Image, **options: Any) -> str:
        return self._respond("single", tokenizer, image, options)

    def _respond(
        self,
        method: str,
        tokenizer: Any,
        image: Image.Image,
        options: dict[str, Any],
    ) -> str:
        assert isinstance(tokenizer, FakeTokenizer)
        assert options == {
            "ocr_type": "ocr",
            "render": False,
            "save_render_file": None,
            "print_prompt": False,
            "gradio_input": True,
            "stream_flag": False,
        }
        self.calls.append({"method": method, "size": image.size})
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return output


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


def test_five_page_triage_is_failure_inclusive_and_reject_only(
    tmp_path: Path,
) -> None:
    challenge = _write_challenge(tmp_path)
    output = tmp_path / "got.json"
    model = FakeModel(
        [
            "Exact literal",
            "Exact literal",
            RuntimeError("fixture inference failure"),
            "Exact literal",
            "Exact literal",
        ]
    )

    payload = benchmark.run_benchmark(
        challenge,
        output,
        method="multi_crop",
        orientation_policy="source",
        device="cpu",
        warmup=False,
        tokenizer=FakeTokenizer(),
        model=model,
        torch_module=FakeTorch(),
    )

    assert len(model.calls) == 5
    assert {record["id"] for record in payload["cases"]} == set(benchmark.TRIAGE_IDS)
    assert payload["summary"]["failed_cases"] == 1
    assert payload["summary"]["coverage"] == 0.8
    assert payload["cases"][2]["metrics"]["cer"]["rate"] == 1.0
    assert payload["handwriting_unlocalized_phrase_recovery"] == {
        "eligible": 5,
        "recovered": 4,
        "rate": 0.8,
        "excluded_partial_or_illegible": 0,
        "matching": "casefolded whitespace-normalized nonoverlapping exact spans",
        "interpretation": (
            "Upper-bound unlocalized phrase presence. GOT emits no geometry, so a "
            "phrase duplicated in printed text can count as present."
        ),
        "handwriting_specific": False,
    }
    assert payload["evidence"]["scope"] == "reject_only"
    assert payload["evidence"]["promotion_evidence_complete"] is False
    assert payload["evidence"]["failures_remain_in_denominators"] is True
    assert payload["evidence"]["default_backend_eligible"] is False
    assert len(payload["manual_review"]) == 5
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"


def test_rotation_sweep_retains_candidates_and_never_sees_reference(
    tmp_path: Path,
) -> None:
    challenge = _write_challenge(tmp_path)
    per_page = ["x", "Exact literal", "x\nx\nx", ""]
    model = FakeModel(per_page * len(benchmark.TRIAGE_IDS))

    payload = benchmark.run_benchmark(
        challenge,
        tmp_path / "sweep.json",
        method="multi_crop",
        orientation_policy="sweep",
        device="cpu",
        warmup=False,
        tokenizer=FakeTokenizer(),
        model=model,
        torch_module=FakeTorch(),
    )

    assert len(model.calls) == 20
    assert all(
        record["selected_rotation_degrees_ccw"] == 90 for record in payload["cases"]
    )
    assert all(
        len(record["orientation_candidates"]) == 4 for record in payload["cases"]
    )
    assert payload["summary"]["cer"]["micro"] == 0
    assert payload["evidence"]["ground_truth_used_for_selection"] is False
    assert model.calls[:4] == [
        {"method": "multi_crop", "size": (12, 8)},
        {"method": "multi_crop", "size": (8, 12)},
        {"method": "multi_crop", "size": (12, 8)},
        {"method": "multi_crop", "size": (8, 12)},
    ]
    first_candidates = payload["cases"][0]["orientation_candidates"]
    assert first_candidates[1]["generated_tokens_after_decode"] == 2
    assert first_candidates[2]["selection_signal"] == {
        "score": 1,
        "alphanumeric_characters": 3,
        "repeated_line_alphanumeric_characters": 2,
    }
    assert first_candidates[3]["status"] == "failed"
    assert first_candidates[3]["failure"]["code"] == "got_inference_failed"


def test_remote_revision_must_be_immutable_and_match_loaded_model(
    tmp_path: Path,
) -> None:
    challenge = _write_challenge(tmp_path)
    model = FakeModel(["Exact literal"] * 5)
    common = {
        "challenge_root": challenge,
        "output": tmp_path / "got.json",
        "method": "multi_crop",
        "orientation_policy": "source",
        "device": "cpu",
        "warmup": False,
        "tokenizer": FakeTokenizer(),
        "model": model,
        "torch_module": FakeTorch(),
    }

    with pytest.raises(ValueError, match="full commit SHA"):
        benchmark.run_benchmark(revision="main", **common)

    model.config._commit_hash = "0" * 40
    with pytest.raises(RuntimeError, match="does not match"):
        benchmark.run_benchmark(**common)


def test_token_limited_pages_fail_without_discarding_diagnostics(
    tmp_path: Path,
) -> None:
    challenge = _write_challenge(tmp_path)
    output = " ".join(["loop"] * (benchmark.OFFICIAL_MAX_NEW_TOKENS - 1))

    payload = benchmark.run_benchmark(
        challenge,
        tmp_path / "token-limit.json",
        method="multi_crop",
        orientation_policy="source",
        device="cpu",
        warmup=False,
        tokenizer=FakeTokenizer(),
        model=FakeModel([output] * len(benchmark.TRIAGE_IDS)),
        torch_module=FakeTorch(),
    )

    assert payload["summary"]["failed_cases"] == len(benchmark.TRIAGE_IDS)
    assert payload["summary"]["coverage"] == 0
    assert all(record["prediction"] == "" for record in payload["cases"])
    candidate = payload["cases"][0]["orientation_candidates"][0]
    assert candidate["status"] == "failed"
    assert candidate["possible_token_limit"] is True
    assert candidate["failure"]["code"] == "got_token_limit"
    assert candidate["prediction"] == output


def _write_challenge(tmp_path: Path) -> Path:
    challenge = tmp_path / "challenge"
    for case_id in benchmark.TRIAGE_IDS:
        category = case_id.split("-", 1)[0]
        annotation = (
            challenge / "annotations" / "primary" / category / f"{case_id}.json"
        )
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "source_only": True,
                    "page_legibility": "complete",
                    "challenges": ["fixture hard page"],
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
            challenge
            / "sources"
            / benchmark.CATEGORY_SOURCES[category]
            / f"{case_id}.png"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8), "white").save(source)
    return challenge
