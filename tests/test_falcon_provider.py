from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from PIL import Image
import pytest

from ocr_pipeline.falcon import (
    FALCON_INFERENCE_REPOSITORY_REVISION,
    FALCON_MODEL_ID,
    FALCON_MODEL_LICENSE,
    FALCON_MODEL_ORIGIN,
    FALCON_MODEL_REVISION,
    FALCON_OCR_CATEGORIES,
    FalconOCRReader,
)
from ocr_pipeline import cli as ocr_cli
from ocr_pipeline.contracts import DocumentResult
from ocr_pipeline.pipeline import process_document


class FakeModel:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs or ["  Literal text\n"]
        self.calls: list[tuple[list[Image.Image], dict[str, object]]] = []

    def generate(self, images: list[Image.Image], **options: object) -> list[str]:
        self.calls.append((images, options))
        return self.outputs


def test_reader_returns_raw_full_page_text_and_verified_provenance(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    model = FakeModel()
    reader = FalconOCRReader(model=model)

    regions = reader.read(image_path, 3)

    assert len(regions) == 1
    region = regions[0]
    assert region.id == "p3-page-1"
    assert region.kind == "page_text"
    assert region.text == "  Literal text\n"
    assert region.bounding_box.right == 200
    assert region.bounding_box.bottom == 100
    assert region.text_provenance == {
        "method": "falcon_core_full_page_generation",
        "provider": "falcon-ocr",
        "model": reader.provenance,
        "category": "plain",
        "generation": {
            "category": "plain",
            "max_new_tokens": 3072,
            "effective_max_new_tokens": 1536,
            "temperature": 0.0,
            "max_dimension": 1024,
            "compile": False,
        },
        "output_validation": {
            "raw_response_preserved": True,
            "nonempty": True,
            "exact_repetition_loop": False,
            "termination_observable": False,
            "truncation_observable": False,
        },
    }
    assert model.calls[0][1] == {
        "category": ["plain"],
        "max_new_tokens": 1536,
        "temperature": 0.0,
        "max_dimension": 1024,
        "compile": False,
    }
    assert "layout_model" not in model.calls[0][1]


def test_crop_transcription_routes_only_explicit_official_categories() -> None:
    class CategoryModel(FakeModel):
        def generate(self, images: list[Image.Image], **options: object) -> list[str]:
            self.calls.append((images, options))
            assert options["category"] == ["text", "table"]
            return ["name", "<table><tr><td>1</td></tr></table>"]

    model = CategoryModel()
    reader = FalconOCRReader(model=model)
    crops = [Image.new("RGB", (20, 20)), Image.new("RGB", (21, 20))]

    output = reader.transcribe_crops(crops, ["text", "table"])

    assert output == ["name", "<table><tr><td>1</td></tr></table>"]
    assert model.calls == [
        (
            crops,
            {
                **reader.generation_config.__dict__,
                "category": ["text", "table"],
                "max_new_tokens": 1024,
            },
        )
    ]
    with pytest.raises(ValueError, match="Unsupported Falcon-OCR category"):
        reader.transcribe_crops(crops[:1], ["handwriting"])
    with pytest.raises(ValueError, match="equal lengths"):
        reader.transcribe_crops(crops, ["text"])
    assert FALCON_OCR_CATEGORIES == (
        "plain",
        "text",
        "table",
        "formula",
        "caption",
        "footnote",
        "list-item",
        "page-footer",
        "page-header",
        "section-header",
        "title",
    )


def test_crop_transcription_batches_mixed_categories_in_one_model_call() -> None:
    class MixedCategoryModel(FakeModel):
        def generate(self, images: list[Image.Image], **options: object) -> list[str]:
            self.calls.append((images, options))
            categories = options["category"]
            assert isinstance(categories, list)
            return [
                f"{category}-{image.width}"
                for category, image in zip(categories, images, strict=True)
            ]

    model = MixedCategoryModel()
    reader = FalconOCRReader(model=model)
    crops = [
        Image.new("RGB", (20, 20)),
        Image.new("RGB", (21, 20)),
        Image.new("RGB", (22, 20)),
        Image.new("RGB", (23, 20)),
    ]

    output = reader.transcribe_crops(crops, ["table", "text", "table", "text"])

    assert output == ["table-20", "text-21", "table-22", "text-23"]
    assert len(model.calls) == 1
    assert model.calls[0][0] == crops
    assert model.calls[0][1]["category"] == ["table", "text", "table", "text"]


def test_crop_transcription_pads_thin_inputs_before_falcon_resize() -> None:
    class SizeModel(FakeModel):
        def generate(self, images: list[Image.Image], **options: object) -> list[str]:
            self.calls.append((images, options))
            assert images[0].size == (1536, 16)
            assert images[1].size == (2000, 21)
            return ["first", "second"]

    model = SizeModel()
    reader = FalconOCRReader(model=model, max_dimension=1536)
    thin = Image.new("RGB", (1536, 15), "black")
    extreme = Image.new("RGB", (2000, 16), "black")

    assert reader.transcribe_crops([thin, extreme], ["text", "text"]) == [
        "first",
        "second",
    ]
    assert thin.size == (1536, 15)
    assert extreme.size == (2000, 16)


@pytest.mark.parametrize("output", [[""], [" \n"], ["loop\n" * 4]])
def test_reader_rejects_empty_and_exact_repetition_outputs(
    tmp_path: Path, output: list[str]
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    result = process_document(image_path, FalconOCRReader(model=FakeModel(output)))

    assert result.status == "failed"
    assert result.failures[0].code in {
        "falcon_output_failed",
        "falcon_repetition_loop",
    }


def test_reader_preserves_nonloop_repetition_and_reports_generation_failure(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    text = "No No No\nDose: 5 mg\nDose: 5 mg\n"
    assert FalconOCRReader(model=FakeModel([text])).read(image_path, 1)[0].text == text

    class BrokenModel:
        def generate(self, images: object, **options: object) -> object:
            raise RuntimeError("generation failed")

    result = process_document(image_path, FalconOCRReader(model=BrokenModel()))
    assert result.failures[0].code == "falcon_predict_failed"


def test_official_loader_is_local_pinned_bf16_and_uses_remote_model_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    torch_module = ModuleType("torch")
    bfloat16 = object()
    torch_module.bfloat16 = bfloat16  # type: ignore[attr-defined]
    torch_module._dynamo = SimpleNamespace(  # type: ignore[attr-defined]
        config=SimpleNamespace(cache_size_limit=8, accumulated_recompile_limit=256)
    )
    transformers_module = ModuleType("transformers")

    class Loader:
        @classmethod
        def from_pretrained(cls, model_name: str, **options: object) -> object:
            calls.append((model_name, options))
            return object()

    transformers_module.AutoModelForCausalLM = Loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    FalconOCRReader()._initialize_model()

    assert calls == [
        (
            FALCON_MODEL_ID,
            {
                "trust_remote_code": True,
                "torch_dtype": bfloat16,
                "device_map": "cuda:0",
                "local_files_only": True,
                "revision": FALCON_MODEL_REVISION,
            },
        )
    ]
    assert torch_module._dynamo.config.cache_size_limit == 256
    assert torch_module._dynamo.config.accumulated_recompile_limit == 2048


def test_local_directory_has_unverified_identity_and_no_revision_claim(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "unknown-falcon-copy"
    model_dir.mkdir()
    reader = FalconOCRReader(model_name_or_path=model_dir, model=FakeModel())

    assert reader.provenance == {
        "id": None,
        "loaded_from": str(model_dir),
        "revision": "unverified",
        "origin": "unverified",
        "license": "unverified",
        "identity_verified": False,
        "local_files_only": True,
        "inference_repository_audit_reference": {
            "url": "https://github.com/tiiuae/Falcon-Perception",
            "revision": FALCON_INFERENCE_REPOSITORY_REVISION,
            "loaded_code_match": "unverified",
        },
    }


def test_official_model_identity_is_not_proven_by_local_storage_path(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "falcon-ocr"
    model_dir.mkdir()
    reader = FalconOCRReader(
        model_name_or_path=FALCON_MODEL_ID,
        local_model_path=model_dir,
        model=FakeModel(),
    )

    assert reader.provenance["id"] == FALCON_MODEL_ID
    assert reader.provenance["loaded_from"] == str(model_dir)
    assert reader.provenance["revision"] == "unverified"
    assert reader.provenance["identity_verified"] is False


def test_official_model_can_load_from_local_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    model_dir = tmp_path / "falcon-ocr"
    model_dir.mkdir()
    bfloat16 = object()
    torch_module = ModuleType("torch")
    torch_module.bfloat16 = bfloat16  # type: ignore[attr-defined]
    torch_module._dynamo = SimpleNamespace(  # type: ignore[attr-defined]
        config=SimpleNamespace(cache_size_limit=8, accumulated_recompile_limit=256)
    )
    transformers_module = ModuleType("transformers")

    class Loader:
        @staticmethod
        def from_pretrained(source: str, **options: object) -> FakeModel:
            calls.append((source, options))
            return FakeModel()

    transformers_module.AutoModelForCausalLM = Loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    FalconOCRReader(
        model_name_or_path=FALCON_MODEL_ID,
        local_model_path=model_dir,
    )._initialize_model()

    assert calls == [
        (
            str(model_dir),
            {
                "trust_remote_code": True,
                "torch_dtype": bfloat16,
                "device_map": "cuda:0",
                "local_files_only": True,
            },
        )
    ]


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"local_files_only": False}, ValueError, "requires local_files_only"),
        ({"model_revision": "main"}, ValueError, "pinned model revision"),
        ({"category": "layout"}, ValueError, "Unsupported Falcon-OCR category"),
        ({"device_map": "cpu"}, ValueError, "must select CUDA"),
        ({"max_new_tokens": 0}, ValueError, "from 1 to 3072"),
        ({"max_new_tokens": 3073}, ValueError, "from 1 to 3072"),
        ({"temperature": -0.1}, ValueError, "must not be negative"),
        ({"temperature": float("inf")}, ValueError, "infinite"),
        ({"max_dimension": 0}, ValueError, "positive multiple of 16"),
        ({"max_dimension": 1540}, ValueError, "positive multiple of 16"),
        (
            {"model_name_or_path": "/missing/local/falcon"},
            FileNotFoundError,
            "model directory was not found",
        ),
    ],
)
def test_reader_rejects_unsafe_or_invalid_configuration(
    options: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        FalconOCRReader(**options)


def test_model_identity_and_generation_config_are_immutable() -> None:
    reader = FalconOCRReader(model=FakeModel())

    assert reader.model_name_or_path == "tiiuae/Falcon-OCR"
    assert reader.model_revision == "42ec56b72a23984ac059e7c8a6d397a8529423fe"
    assert FALCON_MODEL_ORIGIN == "TII, UAE"
    assert FALCON_MODEL_LICENSE == "Apache-2.0"
    assert reader.provenance["identity_verified"] is True
    with pytest.raises(FrozenInstanceError):
        reader.generation_config.max_new_tokens = 1  # type: ignore[misc]


def test_cli_selects_falcon_only_as_review_challenger(
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
    output = tmp_path / "output.json"

    assert (
        ocr_cli.main(
            [
                str(tmp_path / "page.png"),
                "--reader",
                "falcon-ocr",
                "--review-challenger",
                "--falcon-category",
                "table",
                "--falcon-max-new-tokens",
                "512",
                "--falcon-max-dimension",
                "1536",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(captured) == 1
    reader = captured[0]
    assert isinstance(reader, FalconOCRReader)
    assert reader.category == "table"
    assert reader.generation_config.max_new_tokens == 512
    assert reader.generation_config.max_dimension == 1536


def test_cli_rejects_falcon_without_review_flag(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        ocr_cli.main([str(tmp_path / "page.png"), "--reader", "falcon-ocr"])


def _serve_layout_elements(elements: list[dict[str, object]]):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps({"elements": elements, "model": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_looped_element_becomes_unreadable_with_read_terminated_provenance() -> None:
    """A read that self-terminated inside its loop is not `truncated`, only `looped`."""
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    server = _serve_layout_elements(
        [
            {
                "category": "table",
                "bbox": [0, 0, 600, 40],
                "score": 0.9,
                "text": "$1,659 $546 " * 24,
                "truncated": False,
                "looped": True,
            }
        ]
    )
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_address[1]}")
        region = reader.read(Path(__file__), 1)[0]
    finally:
        server.shutdown()
        server.server_close()

    assert region.resolution == "unreadable"
    assert (
        region.text_provenance["read_terminated"]
        == "repetition loop the retry ladder did not recover"
    )


def test_retried_element_carries_read_retries_provenance() -> None:
    """A retry that recovered clean text is still recorded, for observability."""
    from ocr_pipeline.falcon_layout import FalconLayoutReader

    server = _serve_layout_elements(
        [
            {
                "category": "text",
                "bbox": [0, 0, 600, 40],
                "score": 0.9,
                "text": "Patient Name: Hugh Brown",
                "truncated": False,
                "retries": 2,
            }
        ]
    )
    try:
        reader = FalconLayoutReader(f"http://127.0.0.1:{server.server_address[1]}")
        region = reader.read(Path(__file__), 1)[0]
    finally:
        server.shutdown()
        server.server_close()

    assert region.resolution == "resolved"
    assert region.text_provenance["read_retries"] == 2
    assert "read_terminated" not in region.text_provenance
