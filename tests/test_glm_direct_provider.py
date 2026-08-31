from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import GLMOCRDirectReader


class FakeTensor:
    shape = (1, 3)


class FakeInputs(dict):
    def to(self, device: str) -> FakeInputs:
        assert device == "cuda:0"
        return self


class FakeProcessor:
    def apply_chat_template(self, messages: object, **options: object) -> FakeInputs:
        content = messages[0]["content"]  # type: ignore[index]
        assert content[1] == {"type": "text", "text": "Text Recognition:"}
        assert Path(content[0]["url"]).is_file()
        assert options == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return FakeInputs(input_ids=FakeTensor(), token_type_ids=object())

    def decode(self, token_ids: list[int], **options: object) -> str:
        assert token_ids == [101, 102]
        assert options == {"skip_special_tokens": True}
        return "  Exact text  "


class FakeModel:
    device = "cuda:0"

    def generate(self, **inputs: object) -> list[list[int]]:
        assert "token_type_ids" not in inputs
        assert inputs["max_new_tokens"] == 256
        assert inputs["do_sample"] is False
        return [[1, 2, 3, 101, 102]]


def test_glm_direct_reader_processes_model_only_output(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    result = process_document(
        image_path,
        GLMOCRDirectReader(
            max_new_tokens=256,
            processor=FakeProcessor(),
            model=FakeModel(),
        ),
    )

    assert result.status == "success"
    assert result.pages[0].reader == "glm-ocr-direct"
    assert result.pages[0].text.value == "Exact text"
    assert len(result.pages[0].regions) == 1
    region = result.pages[0].regions[0]
    assert region.kind == "page_text"
    assert region.confidence is None
    assert region.reading_order == 1
    assert region.bounding_box.left == 0
    assert region.bounding_box.top == 0
    assert region.bounding_box.right == 200
    assert region.bounding_box.bottom == 100


def test_glm_direct_reader_reports_predict_and_output_failures(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class BrokenModel(FakeModel):
        def generate(self, **inputs: object) -> list[list[int]]:
            raise RuntimeError("generation failed")

    class EmptyProcessor(FakeProcessor):
        def decode(self, token_ids: list[int], **options: object) -> str:
            return ""

    predict_result = process_document(
        image_path,
        GLMOCRDirectReader(processor=FakeProcessor(), model=BrokenModel()),
    )
    output_result = process_document(
        image_path,
        GLMOCRDirectReader(
            max_new_tokens=256, processor=EmptyProcessor(), model=FakeModel()
        ),
    )

    assert predict_result.failures[0].code == "glm_direct_predict_failed"
    assert output_result.failures[0].code == "glm_direct_output_failed"


def test_glm_direct_reader_reports_import_and_init_failures(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    missing_module = ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", missing_module)
    import_result = process_document(image_path, GLMOCRDirectReader())
    assert import_result.failures[0].code == "glm_direct_import_failed"

    failing_module = ModuleType("transformers")

    class FailingLoader:
        @classmethod
        def from_pretrained(cls, *args: object, **options: object) -> object:
            raise RuntimeError("model unavailable")

    failing_module.AutoProcessor = FailingLoader  # type: ignore[attr-defined]
    failing_module.AutoModelForImageTextToText = FailingLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", failing_module)
    init_result = process_document(image_path, GLMOCRDirectReader())
    assert init_result.failures[0].code == "glm_direct_init_failed"
