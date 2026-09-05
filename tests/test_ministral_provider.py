from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from ocr_pipeline import cli as ocr_cli
from ocr_pipeline.contracts import DocumentResult
from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import (
    MINISTRAL_MODEL_ID,
    MINISTRAL_MODEL_LICENSE,
    MINISTRAL_MODEL_ORIGIN,
    MINISTRAL_MODEL_REVISION,
    MinistralOCRReader,
    MinistralStructuredImageCall,
    ReaderError,
)
from ocr_pipeline.openrouter import OpenRouterError


class FakeTensor:
    shape = (1, 3)


class FakeInputs(dict):
    def to(self, device: str) -> FakeInputs:
        assert device == "cuda:0"
        return self


class FakeProcessor:
    def apply_chat_template(self, messages: object, **options: object) -> FakeInputs:
        content = messages[0]["content"]  # type: ignore[index]
        assert content[0]["type"] == "image"
        assert Path(content[0]["url"]).is_file()
        assert content[1] == {"type": "text", "text": "Exact OCR prompt"}
        assert options == {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return FakeInputs(input_ids=FakeTensor())

    def decode(self, token_ids: list[int], **options: object) -> str:
        assert token_ids == [101, 102]
        assert options == {"skip_special_tokens": True}
        return "  Exact text  "


class FakeModel:
    device = "cuda:0"

    def generate(self, **inputs: object) -> list[list[int]]:
        assert inputs["max_new_tokens"] == 256
        assert inputs["do_sample"] is False
        return [[1, 2, 3, 101, 102]]


def test_ministral_reader_returns_full_page_text_with_provenance(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    result = process_document(
        image_path,
        MinistralOCRReader(
            model_name="mistralai/test-model",
            model_revision="a" * 40,
            prompt="Exact OCR prompt",
            max_new_tokens=256,
            processor=FakeProcessor(),
            model=FakeModel(),
        ),
    )

    assert result.status == "success"
    assert result.pages[0].reader == "ministral-ocr"
    assert result.pages[0].text.value == "Exact text"
    region = result.pages[0].regions[0]
    assert region.kind == "page_text"
    assert region.bounding_box.right == 200
    assert region.bounding_box.bottom == 100
    assert region.text_provenance == {
        "method": "ministral_full_page_generation",
        "provider": "ministral-ocr",
        "model": {
            "id": None,
            "loaded_from": "mistralai/test-model",
            "revision": "a" * 40,
            "origin": None,
            "license": None,
            "identity_verified": False,
            "local_files_only": True,
        },
        "prompt": "Exact OCR prompt",
        "generation": {"max_new_tokens": 256, "do_sample": False},
        "processor_input": "chat_template",
    }


def test_ministral_base_reader_uses_plain_image_text_input(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (20, 10), "white").save(image_path)

    class BaseProcessor(FakeProcessor):
        def apply_chat_template(
            self, messages: object, **options: object
        ) -> FakeInputs:
            raise ValueError("this processor does not have a chat template")

        def __call__(self, **inputs: object) -> FakeInputs:
            image = inputs["images"]
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            assert inputs["text"] == "<s>[INST][IMG]Exact OCR prompt[/INST]"
            assert inputs["return_tensors"] == "pt"
            return FakeInputs(input_ids=FakeTensor())

    result = process_document(
        image_path,
        MinistralOCRReader(
            prompt="Exact OCR prompt",
            max_new_tokens=256,
            processor=BaseProcessor(),
            model=FakeModel(),
        ),
    )

    assert result.status == "success"
    assert result.pages[0].regions[0].text_provenance is not None
    assert (
        result.pages[0].regions[0].text_provenance["processor_input"]
        == "base_image_text"
    )


def test_ministral_reader_reports_predict_and_output_failures(tmp_path: Path) -> None:
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
        MinistralOCRReader(
            prompt="Exact OCR prompt",
            processor=FakeProcessor(),
            model=BrokenModel(),
        ),
    )
    output_result = process_document(
        image_path,
        MinistralOCRReader(
            prompt="Exact OCR prompt",
            max_new_tokens=256,
            processor=EmptyProcessor(),
            model=FakeModel(),
        ),
    )

    assert predict_result.failures[0].code == "ministral_predict_failed"
    assert output_result.failures[0].code == "ministral_output_failed"


def test_ministral_reader_reports_import_init_and_image_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    import_result = process_document(image_path, MinistralOCRReader())
    assert import_result.failures[0].code == "ministral_import_failed"

    failing_module = ModuleType("transformers")

    class FailingLoader:
        @classmethod
        def from_pretrained(cls, *args: object, **options: object) -> object:
            raise RuntimeError("model unavailable")

    failing_module.AutoProcessor = FailingLoader  # type: ignore[attr-defined]
    failing_module.FineGrainedFP8Config = object  # type: ignore[attr-defined]
    failing_module.Mistral3ForConditionalGeneration = FailingLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", failing_module)
    init_result = process_document(image_path, MinistralOCRReader())
    assert init_result.failures[0].code == "ministral_init_failed"

    invalid_image = tmp_path / "invalid.png"
    invalid_image.write_text("not an image", encoding="utf-8")
    with pytest.raises(ReaderError, match="cannot identify image file") as error:
        MinistralOCRReader().read(invalid_image, 1)
    assert error.value.code == "ministral_image_failed"


def test_ministral_loaders_use_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    module = ModuleType("transformers")
    quantization_config = object()

    class QuantizationConfig:
        def __new__(cls, *, dequantize: bool) -> object:
            assert dequantize is True
            return quantization_config

    class ProcessorLoader:
        @classmethod
        def from_pretrained(cls, model_name: str, **options: object) -> object:
            calls.append((model_name, options))
            return object()

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, model_name: str, **options: object) -> object:
            calls.append((model_name, options))
            return object()

    module.AutoProcessor = ProcessorLoader  # type: ignore[attr-defined]
    module.FineGrainedFP8Config = QuantizationConfig  # type: ignore[attr-defined]
    module.Mistral3ForConditionalGeneration = ModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)

    MinistralOCRReader()._initialize_components()

    assert calls == [
        (
            MINISTRAL_MODEL_ID,
            {
                "revision": MINISTRAL_MODEL_REVISION,
                "fix_mistral_regex": True,
                "local_files_only": True,
            },
        ),
        (
            MINISTRAL_MODEL_ID,
            {
                "revision": MINISTRAL_MODEL_REVISION,
                "device_map": "cuda:0",
                "local_files_only": True,
                "quantization_config": quantization_config,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"prompt": "  "}, "prompt must not be empty"),
        ({"max_new_tokens": 0}, "max_new_tokens must be positive"),
        ({"device_map": " "}, "device_map must not be empty"),
        ({"model_revision": "main"}, "model revision must be an immutable commit"),
        ({"model_revision": "a" * 40}, "must use the pinned model revision"),
    ],
)
def test_ministral_reader_rejects_invalid_configuration(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MinistralOCRReader(**options)


def test_ministral_model_identity_is_pinned_and_eligible() -> None:
    reader = MinistralOCRReader(processor=FakeProcessor(), model=FakeModel())

    assert reader.model_name == "mistralai/Ministral-3-3B-Instruct-2512"
    assert reader.model_revision == "b35d4dfe56c142746f54dbd64f579faab2744308"
    assert reader.local_files_only is True
    assert MINISTRAL_MODEL_ORIGIN == "Mistral AI, France"
    assert MINISTRAL_MODEL_LICENSE == "Apache-2.0"
    assert reader.provenance["identity_verified"] is True


def test_local_ministral_directory_is_not_mislabeled_as_verified() -> None:
    reader = MinistralOCRReader(
        model_name="/models/unknown-copy",
        model_revision="a" * 40,
        processor=FakeProcessor(),
        model=FakeModel(),
    )

    assert reader.provenance == {
        "id": None,
        "loaded_from": "/models/unknown-copy",
        "revision": "a" * 40,
        "origin": None,
        "license": None,
        "identity_verified": False,
        "local_files_only": True,
    }


def test_ministral_structured_call_reuses_local_reader_and_parses_exact_json(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sheet.png"
    Image.new("RGB", (20, 10), "white").save(image_path)

    class StructuredProcessor(FakeProcessor):
        def apply_chat_template(
            self, messages: object, **options: object
        ) -> FakeInputs:
            prompt = messages[0]["content"][1]["text"]  # type: ignore[index]
            assert prompt.startswith("Choose evidence")
            assert '"required":["decision"]' in prompt
            return FakeInputs(input_ids=FakeTensor())

        def decode(self, token_ids: list[int], **options: object) -> str:
            return '{"decision":"candidate:1"}'

    reader = MinistralOCRReader(
        max_new_tokens=256,
        processor=StructuredProcessor(),
        model=FakeModel(),
    )
    result = MinistralStructuredImageCall(reader)(
        image_path,
        "Choose evidence",
        {
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
        },
    )

    assert result.content == {"decision": "candidate:1"}
    assert result.model == MINISTRAL_MODEL_ID
    assert result.provider == "local"
    assert result.attempts == 1
    assert result.cost is None


def test_ministral_structured_call_rejects_non_json(tmp_path: Path) -> None:
    image_path = tmp_path / "sheet.png"
    Image.new("RGB", (20, 10), "white").save(image_path)

    class InvalidProcessor(FakeProcessor):
        def apply_chat_template(
            self, messages: object, **options: object
        ) -> FakeInputs:
            return FakeInputs(input_ids=FakeTensor())

        def decode(self, token_ids: list[int], **options: object) -> str:
            return "not json"

    call = MinistralStructuredImageCall(
        MinistralOCRReader(
            max_new_tokens=256,
            processor=InvalidProcessor(),
            model=FakeModel(),
        )
    )

    with pytest.raises(OpenRouterError) as error:
        call(image_path, "Choose evidence", {"type": "object"})
    assert error.value.code == "ministral_structured_output_failed"


def test_ministral_structured_call_keeps_schema_in_base_processor_prompt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sheet.png"
    Image.new("RGB", (20, 10), "white").save(image_path)

    class BaseStructuredProcessor(FakeProcessor):
        def apply_chat_template(
            self, messages: object, **options: object
        ) -> FakeInputs:
            raise ValueError("this processor does not have a chat template")

        def __call__(self, **inputs: object) -> FakeInputs:
            prompt = str(inputs["text"])
            assert prompt.startswith("<s>[INST][IMG]Choose evidence")
            assert '"required":["decision"]' in prompt
            return FakeInputs(input_ids=FakeTensor())

        def decode(self, token_ids: list[int], **options: object) -> str:
            return '{"decision":"candidate:1"}'

    result = MinistralStructuredImageCall(
        MinistralOCRReader(
            max_new_tokens=256,
            processor=BaseStructuredProcessor(),
            model=FakeModel(),
        )
    )(
        image_path,
        "Choose evidence",
        {"type": "object", "required": ["decision"]},
    )

    assert result.content == {"decision": "candidate:1"}


def test_cli_selects_local_ministral_only_as_review_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
                "ministral-ocr",
                "--review-challenger",
                "--max-new-tokens",
                "256",
                "--ministral-model",
                "/models/ministral",
                "--ministral-model-revision",
                "a" * 40,
                "--output",
                str(tmp_path / "output.json"),
            ]
        )
        == 0
    )
    assert len(captured) == 1
    reader = captured[0]
    assert isinstance(reader, MinistralOCRReader)
    assert reader.model_name == "/models/ministral"
    assert reader.model_revision == "a" * 40
    assert reader.max_new_tokens == 256
    assert reader.local_files_only is True


def test_cli_rejects_unmeasured_ministral_without_review_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        ocr_cli.main(
            [
                str(tmp_path / "page.png"),
                "--reader",
                "ministral-ocr",
            ]
        )
