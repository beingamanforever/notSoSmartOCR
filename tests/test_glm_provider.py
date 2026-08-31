from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import GLMOCRReader


class FakeResult:
    def __init__(self, output: object) -> None:
        self.output = output

    def to_dict(self) -> object:
        return self.output


class FakePipeline:
    def __init__(self, output: object) -> None:
        self.output = output

    def parse(self, image_path: str, **options: object) -> FakeResult:
        assert Path(image_path).is_file()
        assert options == {
            "preserve_order": True,
            "save_layout_visualization": False,
        }
        return FakeResult(self.output)


def test_glm_reader_processes_official_structured_output(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    pipeline = FakePipeline(
        {
            "json_result": [
                [
                    {
                        "index": 8,
                        "label": "title",
                        "content": "Report",
                        "bbox_2d": [100, 100, 900, 300],
                    },
                    {
                        "index": 1,
                        "label": "text",
                        "content": "Body",
                        "bbox_2d": [100, 400, 900, 800],
                    },
                ]
            ],
            "markdown_result": "# Report\n\nBody",
        }
    )

    result = process_document(image_path, GLMOCRReader(pipeline=pipeline))

    assert result.status == "success"
    assert result.pages[0].reader == "glm-ocr"
    assert result.pages[0].text.value == "Report Body"
    assert [region.reading_order for region in result.pages[0].regions] == [1, 2]
    assert [region.confidence for region in result.pages[0].regions] == [None, None]
    assert result.pages[0].regions[0].bounding_box.left == 20
    assert result.pages[0].regions[0].bounding_box.top == 10
    assert result.pages[0].regions[1].bounding_box.right == 180
    assert result.pages[0].regions[1].bounding_box.bottom == 80


def test_glm_reader_reports_predict_and_output_failures(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class BrokenPipeline:
        def parse(self, image_path: str, **options: object) -> FakeResult:
            raise RuntimeError("service unavailable")

    predict_result = process_document(
        image_path, GLMOCRReader(pipeline=BrokenPipeline())
    )
    invalid_result = process_document(
        image_path, GLMOCRReader(pipeline=FakePipeline({"json_result": []}))
    )
    reported_error = process_document(
        image_path, GLMOCRReader(pipeline=FakePipeline({"error": "request failed"}))
    )

    assert predict_result.failures[0].code == "glm_predict_failed"
    assert reported_error.failures[0].code == "glm_predict_failed"
    assert invalid_result.failures[0].code == "invalid_reader_output"


def test_glm_reader_reports_import_and_init_failures(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    missing_module = ModuleType("glmocr")
    monkeypatch.setitem(sys.modules, "glmocr", missing_module)
    import_result = process_document(image_path, GLMOCRReader())
    assert import_result.failures[0].code == "glm_import_failed"

    failing_module = ModuleType("glmocr")

    def fail_init(**options: object) -> object:
        assert options == {
            "mode": "selfhosted",
            "ocr_api_host": "127.0.0.1",
            "ocr_api_port": 8080,
            "layout_device": "cuda:0",
        }
        raise RuntimeError("layout model unavailable")

    failing_module.GlmOcr = fail_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "glmocr", failing_module)
    init_result = process_document(
        image_path,
        GLMOCRReader(
            ocr_api_host="127.0.0.1",
            ocr_api_port=8080,
            layout_device="cuda:0",
        ),
    )
    assert init_result.failures[0].code == "glm_init_failed"
