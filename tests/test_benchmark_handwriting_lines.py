from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
from transformers import ViTImageProcessor

from experiments import benchmark_handwriting_lines as benchmark


def test_line_benchmark_scores_literal_text_and_keeps_failures_in_denominator(
    tmp_path: Path,
) -> None:
    cases = [
        {
            "id": "one",
            "image": "one.png",
            "source_dimensions": [100, 128],
            "presented_dimensions": [100, 128],
            "reference": "Été ici",
        },
        {
            "id": "two",
            "image": "two.png",
            "source_dimensions": [100, 128],
            "presented_dimensions": [100, 128],
            "reference": "abc",
        },
    ]
    rows = [
        benchmark._row(
            cases[0],
            {
                "status": "success",
                "prediction": "  Été   ici ",
                "provider_output": "  Été   ici ",
                "model_input_dimensions": {"reader_input": [100, 128]},
                "failure": None,
            },
            10,
            1,
        ),
        benchmark._row(
            cases[1],
            {
                "status": "failed",
                "prediction": "",
                "provider_output": None,
                "model_input_dimensions": None,
                "failure": {"code": "failed"},
            },
            20,
            1,
        ),
    ]
    args = Namespace(
        dataset="iam",
        reader="trocr",
        limit=2,
        device="cuda:0",
        batch_size=1,
        max_new_tokens=128,
        model_path=None,
        adapter_path=None,
    )

    payload = benchmark._summary_payload(
        args,
        benchmark.DATASETS["iam"],
        object(),
        rows,
        tmp_path / "rows.jsonl",
        100,
        {
            "sample_id": "one",
            "status": "success",
            "failure": None,
            "latency_ms": 10,
            "sample_is_scored_in_benchmark": True,
        },
        1,
        status="complete",
        total_cases=2,
    )

    assert payload["metrics"]["literal"]["exact"] == 1
    assert payload["metrics"]["literal"]["cer"] == 0.3
    assert payload["metrics"]["literal"]["wer"] == pytest.approx(1 / 3)
    assert payload["metrics"]["failed_lines"] == 1
    assert payload["metrics"]["failures_remain_in_denominators"] is True
    assert rows[0]["metrics"]["literal"]["prediction"] == "Été ici"
    assert rows[0]["reader_prediction"] == "  Été   ici "


def test_prepare_replaces_existing_pixels_from_the_pinned_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [{"image": Image.new("RGB", (2, 1), "red"), "text": "first"}]
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **options: records  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(
        benchmark.DATASETS,
        "fixture",
        benchmark.DatasetSpec("fixture/source", "pinned", "test", 1, "English"),
    )

    benchmark.prepare_dataset("fixture", tmp_path)
    records[0] = {"image": Image.new("RGB", (2, 1), "blue"), "text": "second"}
    benchmark.prepare_dataset("fixture", tmp_path)

    image_path = tmp_path / "fixture" / "pinned" / "test" / "000000.png"
    with Image.open(image_path) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    ground_truth = image_path.with_name("ground_truth.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"reference": "second"' in ground_truth


def test_failed_batch_is_not_silently_retried_as_single_items(tmp_path: Path) -> None:
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        Image.new("RGB", (2, 1), "white").save(path)

    class BrokenReader:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def transcribe_batch(self, images: list[Image.Image]) -> list[str]:
            self.calls.append(len(images))
            raise RuntimeError("batch failed")

    reader = BrokenReader()
    predictions = benchmark._predict(
        "phi4",
        reader,
        [{"image_path": path} for path in paths],
    )

    assert reader.calls == [2]
    assert [prediction["status"] for prediction in predictions] == [
        "failed",
        "failed",
    ]


def test_missing_trocr_crop_size_does_not_block_inference(tmp_path: Path) -> None:
    image_path = tmp_path / "line.png"
    Image.new("RGB", (100, 128), "white").save(image_path)

    class Reader:
        _processor = SimpleNamespace(image_processor=ViTImageProcessor(size=384))

        def transcribe_batch(self, images: list[Image.Image]) -> list[str]:
            return ["read"] * len(images)

    predictions = benchmark._predict("trocr", Reader(), [{"image_path": image_path}])

    assert predictions[0]["status"] == "success"
    assert predictions[0]["model_input_dimensions"] == {
        "reader_input": [100, 128],
        "processor_size": {"height": 384, "width": 384},
        "processor_crop_size": None,
    }
    benchmark.json.dumps(predictions[0])


def test_arbitrary_falcon_directory_is_not_attributed_to_pinned_weights(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "snapshots" / "42ec56b72a23984ac059e7c8a6d397a8529423fe"
    model_path.mkdir(parents=True)
    args = Namespace(reader="falcon", model_path=model_path)
    reader = SimpleNamespace(
        provenance={
            "id": "tiiuae/Falcon-OCR",
            "revision": "pinned",
            "identity_verified": True,
        }
    )

    provenance = benchmark._reader_provenance(args, reader)

    assert provenance["revision"] == "unverified"
    assert provenance["identity_verified"] is False


def test_warmup_failure_does_not_remove_test_lines_from_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = benchmark.DatasetSpec("fixture/source", "pinned", "test", 2, "English")
    monkeypatch.setitem(benchmark.DATASETS, "fixture", spec)
    root = tmp_path / "data" / "fixture" / "pinned" / "test"
    root.mkdir(parents=True)
    records = []
    for index, reference in enumerate(("one", "two")):
        image_name = f"{index:06d}.png"
        Image.new("RGB", (2, 1), "white").save(root / image_name)
        records.append(
            {
                "id": f"fixture-test-{index:06d}",
                "image": image_name,
                "reference": reference,
                "source_dimensions": [2, 1],
                "source_mode": "RGB",
                "presented_dimensions": [2, 1],
                "presented_mode": "RGB",
            }
        )
    (root / "ground_truth.jsonl").write_text(
        "".join(f"{benchmark.json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    calls = 0

    def predict(
        reader_name: str, reader: object, cases: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [benchmark._failure(RuntimeError("warmup failed"))]
        return [
            {
                "status": "success",
                "prediction": case["reference"],
                "provider_output": case["reference"],
                "model_input_dimensions": {"reader_input": [2, 1]},
                "failure": None,
            }
            for case in cases
        ]

    monkeypatch.setattr(benchmark, "_load_reader", lambda args: object())
    monkeypatch.setattr(benchmark, "_predict", predict)
    args = Namespace(
        dataset="fixture",
        reader="trocr",
        data_root=tmp_path / "data",
        output=tmp_path / "result.json",
        limit=None,
        device="cuda:0",
        batch_size=1,
        max_new_tokens=128,
        model_path=None,
        adapter_path=None,
    )

    payload = benchmark.run_benchmark(args)

    assert payload["status"] == "complete"
    assert payload["progress"] == {"completed": 2, "expected": 2}
    assert payload["latency"]["warmup"]["status"] == "failed"
    assert payload["metrics"]["failed_lines"] == 0
