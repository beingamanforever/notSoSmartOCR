from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ocr_pipeline.pipeline import process_document
from ocr_pipeline import cli as ocr_cli
from ocr_pipeline.contracts import DocumentResult
from ocr_pipeline.pilot import (
    PILOT_CHECKPOINT,
    PILOT_MODEL_ID,
    PILOT_ORIGIN,
    PILOT_REPOSITORY,
    PILOT_SOURCE_LICENSE,
    PILOT_SOURCE_REVISION,
    PILOT_WEIGHTS_DOI,
    PILOT_WEIGHTS_LICENSE,
    PilotOCRReader,
)


class FakeTensor:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: object) -> FakeTensor:
        self.device = device
        return self


class FakeDevice:
    type = "cuda"


class FakeModel:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, batch: dict[str, object], **options: object) -> dict[str, object]:
        assert set(batch) == {"imgs", "token_prompt"}
        assert options == {"use_amp": True, "num_beams": 1, "max_length": 2048}
        return {"str_pred": [self.prediction]}


class FakeRuntime:
    def __init__(self, prediction: str) -> None:
        self.model = FakeModel(prediction)
        self.tokenizer = object()
        self.device = FakeDevice()
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            device=self._device,
        )
        self.load_calls: list[dict[str, object]] = []

    def _device(self, name: str) -> FakeDevice:
        assert name == "cuda"
        return self.device

    def load_pilot_model(self, **options: object) -> tuple[object, object, object]:
        self.load_calls.append(options)
        return (
            self.model,
            self.tokenizer,
            {
                "name": "pilot_generic",
                "supported_tasks": [
                    "ocr_with_boxes",
                    "ocr",
                    "find_it",
                    "ocr_on_box",
                ],
                "preprocessing": {
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                    "coord_bin_size": 10,
                },
                "decoder": {"max_length": 2048},
            },
        )

    def prepare_batch_for_inference(self, **options: object) -> dict[str, FakeTensor]:
        image = options["image"]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert options["tokenizer"] is self.tokenizer
        assert options["prompt"] == "<ocr_with_boxes>"
        assert options["mean"] == [123.675, 116.28, 103.53]
        assert options["std"] == [58.395, 57.12, 57.375]
        return {"imgs": FakeTensor(), "token_prompt": FakeTensor()}


def make_local_release(tmp_path: Path) -> Path:
    root = tmp_path / "PILOT"
    for path in (
        root / "run_pilot.py",
        root / "pilot" / "__init__.py",
        root / "configs" / "pilot_generic.json",
        root / "checkpoints" / "pilot_generic.pt",
        root / "checkpoints" / "tokenizer" / "tokenizer.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_pilot_reader_emits_paired_lines_with_exact_provenance(tmp_path: Path) -> None:
    root = make_local_release(tmp_path)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    runtime = FakeRuntime(
        "<x_1><y_2> First line<x_8><y_4><sep/><x_2><y_5> Second line<x_12><y_8>"
    )

    result = process_document(image_path, PilotOCRReader(root, runtime=runtime))

    assert result.status == "success"
    assert result.pages[0].reader == "pilot-ocr"
    assert result.pages[0].text.value == "First line Second line"
    assert [region.text for region in result.pages[0].regions] == [
        "First line",
        "Second line",
    ]
    assert [region.bounding_box.__dict__ for region in result.pages[0].regions] == [
        {"left": 10, "top": 20, "right": 80, "bottom": 40},
        {"left": 20, "top": 50, "right": 120, "bottom": 80},
    ]
    assert [region.reading_order for region in result.pages[0].regions] == [1, 2]
    assert all(region.kind == "line" for region in result.pages[0].regions)
    assert result.pages[0].regions[0].text_provenance == {
        "method": "pilot_ocr_with_boxes",
        "model": {
            "id": PILOT_MODEL_ID,
            "repository": PILOT_REPOSITORY,
            "source_revision": PILOT_SOURCE_REVISION,
            "source_license": PILOT_SOURCE_LICENSE,
            "weights_doi": PILOT_WEIGHTS_DOI,
            "weights_license": PILOT_WEIGHTS_LICENSE,
            "checkpoint": PILOT_CHECKPOINT,
            "origin": PILOT_ORIGIN,
            "local_files_only": True,
        },
        "prompt": "<ocr_with_boxes>",
        "generation": {
            "max_length": 2048,
            "num_beams": 1,
            "do_sample": False,
        },
        "geometry": {"coord_bin_size": 10},
    }
    assert len(runtime.load_calls) == 1
    assert runtime.load_calls[0] == {
        "config_path": root / "configs" / "pilot_generic.json",
        "device": runtime.device,
        "checkpoint_path": root / "checkpoints" / "pilot_generic.pt",
        "tokenizer_path": root / "checkpoints" / "tokenizer",
    }


@pytest.mark.parametrize(
    "prediction",
    [
        "plain ungrounded text",
        "<x_1><y_2><x_8><y_4>",
        "<x_8><y_2> reversed<x_1><y_4>",
        "<x_1><y_2> out of bounds<x_30><y_4>",
        "<x_1><y_2> valid<x_8><y_4><sep/>",
        "<x_1><y_2> missing bottom right",
        "<x_1><y_2> extra<x_3><y_3><x_8><y_4>",
    ],
)
def test_pilot_reader_fails_closed_on_malformed_output(
    tmp_path: Path,
    prediction: str,
) -> None:
    root = make_local_release(tmp_path)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    result = process_document(
        image_path,
        PilotOCRReader(root, runtime=FakeRuntime(prediction)),
    )

    assert result.status == "failed"
    assert result.pages[0].regions == []
    assert result.failures[0].code == "pilot_output_failed"


def test_pilot_reader_requires_complete_local_release(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PILOT local release is incomplete"):
        PilotOCRReader(tmp_path)


def test_pilot_reader_rejects_non_generic_config(tmp_path: Path) -> None:
    root = make_local_release(tmp_path)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    runtime = FakeRuntime("<x_0><y_0> text<x_1><y_1>")
    load_generic_model = runtime.load_pilot_model

    def load_wrong_model(**options: object) -> tuple[object, object, object]:
        model, tokenizer, config = load_generic_model(**options)
        config["name"] = "pilot_iam"
        return model, tokenizer, config

    runtime.load_pilot_model = load_wrong_model  # type: ignore[method-assign]

    result = process_document(image_path, PilotOCRReader(root, runtime=runtime))

    assert result.status == "failed"
    assert result.failures[0].code == "pilot_init_failed"


def test_cli_selects_pilot_only_as_review_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_local_release(tmp_path)
    captured: list[object] = []

    def fake_process_document(
        source: Path, reader: object, *, pdf_dpi: int
    ) -> DocumentResult:
        captured.append(reader)
        return DocumentResult(
            document_id="test",
            source={"name": source.name, "kind": "image"},
            status="success",
        )

    monkeypatch.setattr(ocr_cli, "process_document", fake_process_document)

    assert (
        ocr_cli.main(
            [
                str(tmp_path / "page.png"),
                "--reader",
                "pilot-ocr",
                "--review-challenger",
                "--pilot-root",
                str(root),
                "--pilot-device",
                "cuda:0",
                "--output",
                str(tmp_path / "output.json"),
            ]
        )
        == 0
    )
    assert len(captured) == 1
    reader = captured[0]
    assert isinstance(reader, PilotOCRReader)
    assert reader.pilot_root == root.resolve()
    assert reader.device_name == "cuda:0"


def test_cli_rejects_pilot_without_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        ocr_cli.main(
            [
                str(tmp_path / "page.png"),
                "--reader",
                "pilot-ocr",
                "--review-challenger",
            ]
        )
