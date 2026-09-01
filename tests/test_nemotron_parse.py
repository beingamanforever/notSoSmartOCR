from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import ocr_pipeline.nemotron_parse as nemotron_parse
from ocr_pipeline.nemotron_parse import DEFAULT_PROMPT, NemotronParseReader
from ocr_pipeline.pipeline import process_document


def test_parse_reader_preserves_contract_geometry_and_provenance(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (832, 1024), "white").save(image_path)

    def generate(path: Path, prompt: str) -> str:
        assert path == image_path
        assert prompt == DEFAULT_PROMPT
        return (
            "</s><s>"
            "<x_0.25><y_0.25>First line with literal <x_0.25>\nSecond line"
            "<x_0.5><y_0.5><class_Text>"
            "<x_0.5><y_0.5>\\begin{tabular}A & B\\end{tabular}"
            "<x_0.75><y_0.75><class_Table>"
            "</s>"
        )

    result = process_document(image_path, NemotronParseReader(generator=generate))

    assert result.status == "success"
    assert result.pages[0].reader == "nemotron-parse-2.0"
    first, second = result.pages[0].regions
    assert first.kind == "Text"
    assert first.text == "First line with literal <x_0.25>\nSecond line"
    assert first.confidence is None
    assert first.reading_order == 1
    assert first.bounding_box.left == 0
    assert first.bounding_box.top == 0
    assert first.bounding_box.right == 416
    assert first.bounding_box.bottom == 512
    assert second.kind == "Table"
    assert second.text == "\\begin{tabular}A & B\\end{tabular}"
    assert second.bounding_box.left == 416
    assert second.bounding_box.top == 512
    assert second.bounding_box.right == 832
    assert second.bounding_box.bottom == 1024
    assert second.text_provenance is not None
    assert second.text_provenance["model"]["id"] == ("nvidia/NVIDIA-Nemotron-Parse-2.0")
    assert second.text_provenance["vision_encoder"]["origin"] == "NVIDIA"
    assert second.text_provenance["decoder"]["origin"] == (
        "Facebook AI Research (Meta)"
    )
    assert second.structure == {
        "semantic_class": "Table",
        "normalized_bbox": [0.5, 0.5, 0.75, 0.75],
    }


def test_parse_reader_reports_truncated_output_at_caller_boundary(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    result = process_document(
        image_path,
        NemotronParseReader(
            generator=lambda path, prompt: (
                "<x_0.1><y_0.1>complete<x_0.4><y_0.4><class_Text>"
                "<x_0.5><y_0.5>truncated<x_0.9><y_0.9>"
            )
        ),
    )

    assert result.status == "failed"
    assert result.failures[0].code == "invalid_reader_output"
    assert "outside complete regions" in result.failures[0].message


def test_parse_reader_reports_prediction_and_invalid_box_failures(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def fail(path: Path, prompt: str) -> str:
        raise RuntimeError("generation stopped")

    predict_result = process_document(
        image_path,
        NemotronParseReader(generator=fail),
    )
    box_result = process_document(
        image_path,
        NemotronParseReader(
            generator=lambda path, prompt: "<x_0.9><y_0.1>bad<x_0.2><y_0.8><class_Text>"
        ),
    )

    assert predict_result.failures[0].code == "nemotron_parse_predict_failed"
    assert box_result.failures[0].code == "invalid_reader_output"
    assert "positive area" in box_result.failures[0].message


def test_parse_reader_returns_pipeline_no_text_failure_for_blank_page(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    result = process_document(
        image_path,
        NemotronParseReader(generator=lambda path, prompt: "</s><s></s>"),
    )

    assert result.status == "failed"
    assert result.failures[0].code == "no_text_detected"


def test_parse_reader_transforms_large_landscape_page(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("L", (2496, 1536), "white").save(image_path)
    output = "<x_0.25><y_0.25>text<x_0.75><y_0.75><class_Text>"

    result = process_document(
        image_path,
        NemotronParseReader(generator=lambda path, prompt: output),
    )

    box = result.pages[0].regions[0].bounding_box
    assert (box.left, box.top, box.right, box.bottom) == (624, 0, 1872, 1536)


def test_parse_reader_pins_model_and_reports_missing_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    def unavailable(**options: object) -> object:
        raise OSError("weights are absent")

    monkeypatch.setattr(nemotron_parse, "_TransformersGenerator", unavailable)
    result = process_document(image_path, NemotronParseReader())

    assert result.failures[0].code == "nemotron_parse_model_unavailable"
    with pytest.raises(TypeError):
        NemotronParseReader(model_path="custom/checkpoint")  # type: ignore[call-arg]
