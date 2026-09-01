from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import GraniteDoclingReader, _load_doctags_converter


class FakeTensor:
    shape = (1, 3)


class FakeInputs(dict):
    def to(self, device: str) -> FakeInputs:
        assert device == "cuda:0"
        return self


class FakeProcessor:
    def apply_chat_template(self, messages: object, **options: object) -> str:
        content = messages[0]["content"]  # type: ignore[index]
        assert content == [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."},
        ]
        assert options == {"add_generation_prompt": True}
        return "prompt"

    def __call__(self, **options: object) -> FakeInputs:
        assert options["text"] == "prompt"
        images = options["images"]
        assert len(images) == 1  # type: ignore[arg-type]
        assert options["return_tensors"] == "pt"
        return FakeInputs(input_ids=FakeTensor())

    def decode(self, token_ids: list[int], **options: object) -> str:
        assert token_ids == [101, 102]
        assert options == {"skip_special_tokens": False}
        return "  <text>Exact text</text>"


class FakeModel:
    device = "cuda:0"

    def generate(self, **inputs: object) -> list[list[int]]:
        assert inputs["max_new_tokens"] == 256
        assert inputs["do_sample"] is False
        return [[1, 2, 3, 101, 102]]


def test_granite_reader_processes_structured_page(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    result = process_document(
        image_path,
        GraniteDoclingReader(
            max_new_tokens=256,
            processor=FakeProcessor(),
            model=FakeModel(),
            converter=lambda doctags, image: (
                "Heading\nCell A\nCell B" if image.size == (200, 100) else ""
            ),
        ),
    )

    assert result.status == "success"
    assert result.pages[0].reader == "granite-docling"
    assert result.pages[0].text.value == "Heading\nCell A\nCell B"
    region = result.pages[0].regions[0]
    assert region.provider == "granite-docling"
    assert region.kind == "page_text"
    assert region.bounding_box.right == 200
    assert region.bounding_box.bottom == 100


def test_granite_reader_reports_predict_and_output_failures(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class BrokenModel(FakeModel):
        def generate(self, **inputs: object) -> list[list[int]]:
            raise RuntimeError("generation failed")

    predict_result = process_document(
        image_path,
        GraniteDoclingReader(
            processor=FakeProcessor(),
            model=BrokenModel(),
            converter=lambda doctags, image: "text",
        ),
    )
    output_result = process_document(
        image_path,
        GraniteDoclingReader(
            max_new_tokens=256,
            processor=FakeProcessor(),
            model=FakeModel(),
            converter=lambda doctags, image: "",
        ),
    )

    assert predict_result.failures[0].code == "granite_predict_failed"
    assert output_result.failures[0].code == "granite_output_failed"


def test_docling_converter_exports_visible_text_and_markdown(monkeypatch) -> None:
    docling_core = ModuleType("docling_core")
    types_module = ModuleType("docling_core.types")
    doc_module = ModuleType("docling_core.types.doc")
    document_module = ModuleType("docling_core.types.doc.document")

    class FakeDocTagsDocument:
        @classmethod
        def from_doctags_and_image_pairs(
            cls, doctags: list[str], images: list[Image.Image]
        ) -> str:
            assert doctags == ["<doctags>"]
            assert images[0].size == (20, 10)
            return "tagged"

    class LoadedDocument:
        def export_to_text(self) -> str:
            return "Heading\nCell A\nCell B"

        def export_to_markdown(self) -> str:
            return "# Heading\n\n| A | B |\n| - | - |"

    class FakeDoclingDocument:
        def __init__(self, *, name: str) -> None:
            assert name == "Document"

        def load_from_doctags(self, tagged: str) -> LoadedDocument:
            assert tagged == "tagged"
            self.export_to_text = LoadedDocument().export_to_text
            self.export_to_markdown = LoadedDocument().export_to_markdown
            return self

    doc_module.DoclingDocument = FakeDoclingDocument  # type: ignore[attr-defined]
    document_module.DocTagsDocument = FakeDocTagsDocument  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "docling_core", docling_core)
    monkeypatch.setitem(sys.modules, "docling_core.types", types_module)
    monkeypatch.setitem(sys.modules, "docling_core.types.doc", doc_module)
    monkeypatch.setitem(sys.modules, "docling_core.types.doc.document", document_module)

    converter = _load_doctags_converter()

    assert converter("<doctags>", Image.new("RGB", (20, 10))) == (
        "Heading\nCell A\nCell B"
    )
    markdown_converter = _load_doctags_converter("markdown")
    assert markdown_converter("<doctags>", Image.new("RGB", (20, 10))) == (
        "# Heading\n\n| A | B |\n| - | - |"
    )


def test_granite_markdown_output_is_explicit_and_validated(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    result = process_document(
        image_path,
        GraniteDoclingReader(
            output_format="markdown",
            max_new_tokens=256,
            processor=FakeProcessor(),
            model=FakeModel(),
            converter=lambda doctags, image: "# Heading\n\n| A | B |",
        ),
    )

    assert result.pages[0].regions[0].kind == "page_markdown"
    assert result.pages[0].text.value.startswith("# Heading")

    with pytest.raises(ValueError, match="output format"):
        GraniteDoclingReader(output_format="html")


def test_granite_reader_reports_import_and_init_failures(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    missing_module = ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", missing_module)
    import_result = process_document(
        image_path,
        GraniteDoclingReader(converter=lambda doctags, image: "text"),
    )
    assert import_result.failures[0].code == "granite_import_failed"

    failing_module = ModuleType("transformers")

    class FailingLoader:
        @classmethod
        def from_pretrained(cls, *args: object, **options: object) -> object:
            raise RuntimeError("model unavailable")

    failing_module.AutoProcessor = FailingLoader  # type: ignore[attr-defined]
    failing_module.AutoModelForMultimodalLM = FailingLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", failing_module)
    init_result = process_document(
        image_path,
        GraniteDoclingReader(converter=lambda doctags, image: "text"),
    )
    assert init_result.failures[0].code == "granite_init_failed"


def test_granite_reader_checks_docling_before_inference(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    missing_package = ModuleType("docling_core")
    monkeypatch.setitem(sys.modules, "docling_core", missing_package)
    monkeypatch.delitem(sys.modules, "docling_core.types", raising=False)
    monkeypatch.delitem(sys.modules, "docling_core.types.doc", raising=False)

    class UnusedModel(FakeModel):
        def generate(self, **inputs: object) -> list[list[int]]:
            raise AssertionError("inference must not run without Docling Core")

    result = process_document(
        image_path,
        GraniteDoclingReader(processor=FakeProcessor(), model=UnusedModel()),
    )

    assert result.failures[0].code == "granite_import_failed"
