from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import benchmark_phi4 as phi4
from experiments import benchmark_phi4_review_queue as benchmark


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    __version__ = "fixture"
    version = SimpleNamespace(cuda=None)
    cuda = FakeCuda()


class FakeModel:
    config = SimpleNamespace(
        _name_or_path=phi4.MODEL_ID,
        _commit_hash=benchmark.MODEL_REVISION,
    )


def test_queue_is_deterministic_failure_inclusive_and_reference_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _write_queue(tmp_path, count=4)
    calls: list[dict[str, Any]] = []
    load_calls: list[dict[str, Any]] = []

    def fake_load_model(*args: Any, **options: Any) -> tuple[Any, ...]:
        load_calls.append(options)
        return object(), FakeModel(), "fixed-config", FakeTorch()

    def fake_run_input(**options: Any) -> dict[str, Any]:
        calls.append(options)
        failed = options["id"] == "F002"
        return {
            "status": "failed" if failed else "success",
            "prediction": "" if failed else f"prediction-{options['id']}",
            "failures": (
                [{"type": "RuntimeError", "message": "local failure"}] if failed else []
            ),
            "latency_ms": 2.5,
            "generated_tokens": 0 if failed else 2,
            "hit_token_limit": False,
        }

    monkeypatch.setattr(benchmark.phi4, "load_model", fake_load_model)
    monkeypatch.setattr(benchmark.phi4, "_run_input", fake_run_input)

    payload = benchmark.run_benchmark(
        queue,
        tmp_path / "output",
        max_new_tokens=32,
        device="cpu",
    )

    assert load_calls == [
        {
            "device": "cpu",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "local_files_only": True,
            "seed": 0,
        }
    ]
    assert [call["id"] for call in calls] == ["F001", "F002", "F003", "F004"]
    assert all(call["prompt"] == phi4.CROP_PROMPT for call in calls)
    assert all(call["reference"] is None for call in calls)
    assert payload["summary"]["attempted"] == 4
    assert payload["summary"]["succeeded"] == 3
    assert payload["summary"]["failed"] == 1
    assert payload["summary"]["failure_types"] == {"RuntimeError": 1}
    assert len(payload["rows"]) == 4
    assert payload["rows"][1]["status"] == "failed"
    assert payload["rows"][1]["error"] == "local failure"

    output = tmp_path / "output" / benchmark.RESULT_FILE
    serialized = output.read_text(encoding="utf-8")
    assert "SECRET_REFERENCE" not in serialized
    assert '"reference"' not in serialized
    assert not output.with_suffix(output.suffix + ".tmp").exists()
    assert json.loads(serialized) == payload


@pytest.mark.parametrize(("count", "limit", "expected"), [(1, None, 1), (5, 2, 2)])
def test_variable_queue_counts_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    limit: int | None,
    expected: int,
) -> None:
    queue = _write_queue(tmp_path, count=count)

    def fake_run_input(**options: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "prediction": options["id"],
            "failures": [],
            "latency_ms": 1.0,
            "generated_tokens": 1,
            "hit_token_limit": False,
        }

    monkeypatch.setattr(benchmark.phi4, "_run_input", fake_run_input)
    payload = benchmark.run_benchmark(
        queue,
        tmp_path / "output",
        device="cpu",
        limit=limit,
        processor=object(),
        model=FakeModel(),
        generation_config="fixed-config",
        torch_module=FakeTorch(),
    )

    assert payload["panel"]["available"] == count
    assert payload["panel"]["selected"] == expected
    assert payload["summary"]["attempted"] == expected
    assert [row["field_id"] for row in payload["rows"]] == [
        f"F{index:03d}" for index in range(1, expected + 1)
    ]


def test_missing_or_escaping_crop_is_rejected(tmp_path: Path) -> None:
    queue = tmp_path / "review_needed.jsonl"
    queue.write_text(
        json.dumps(
            {
                "review_reason": benchmark.REVIEW_REASON,
                "field_id": "F001",
                "case_id": "C001",
                "family_id": "D001",
                "category_id": "CAT",
                "review_crop_path": "../outside.png",
                "reference": "SECRET_REFERENCE",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="review crop was not found"):
        benchmark.load_review_cases(queue)


def _write_queue(tmp_path: Path, *, count: int) -> Path:
    review = tmp_path / "review"
    review.mkdir()
    records = [
        {
            "review_reason": "legibility_not_legible",
            "field_id": "IGNORED",
            "case_id": "C000",
            "family_id": "D000",
            "category_id": "CAT",
            "review_crop_path": "review/ignored.png",
            "reference": "SECRET_REFERENCE_IGNORED",
        }
    ]
    for index in reversed(range(1, count + 1)):
        field_id = f"F{index:03d}"
        Image.new("RGB", (8, 8), "white").save(review / f"{field_id}.png")
        records.append(
            {
                "review_reason": benchmark.REVIEW_REASON,
                "field_id": field_id,
                "case_id": f"C{index:03d}",
                "family_id": f"D{index:03d}",
                "category_id": "CAT",
                "review_crop_path": f"review/{field_id}.png",
                "reference": f"SECRET_REFERENCE_{index}",
                "top_matches": [],
            }
        )
    queue = tmp_path / "review_needed.jsonl"
    queue.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return queue
