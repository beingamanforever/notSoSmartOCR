from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from ocr_pipeline.pipeline import process_document
from ocr_pipeline.providers import PaddleOCRVLReader


class FakePipeline:
    def __init__(self, result: object) -> None:
        self.result = result

    def predict(self, image_path: str) -> list[object]:
        assert Path(image_path).is_file()
        return [self.result]


def test_paddle_reader_processes_image_with_layout_evidence(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    pipeline = FakePipeline(
        {
            "parsing_res_list": [
                {
                    "label": "header",
                    "bbox": [10, 5, 290, 18],
                    "content": "Header",
                },
                {"label": "table", "bbox": [120, 80, 280, 180], "content": "A | B"},
                {"label": "text", "bbox": [10, 20, 110, 60], "content": "Hello"},
            ],
            "layout_det_res": {
                "boxes": [
                    {
                        "label": "header",
                        "score": 0.88,
                        "coordinate": [10, 5, 290, 18],
                        "order": None,
                    },
                    {
                        "label": "text",
                        "score": 0.97,
                        "coordinate": [10, 20, 110, 60],
                        "order": 1,
                    },
                    {
                        "label": "table",
                        "score": 0.91,
                        "coordinate": [120, 80, 280, 180],
                        "order": 1,
                    },
                ]
            },
        }
    )

    result = process_document(image_path, PaddleOCRVLReader(pipeline=pipeline))

    assert result.status == "success"
    assert result.pages[0].reader == "paddleocr-vl-1.6"
    assert result.pages[0].text.value == "Header A | B Hello"
    assert result.pages[0].text.evidence_ids == [
        "p1-block-1",
        "p1-block-2",
        "p1-block-3",
    ]
    assert [region.kind for region in result.pages[0].regions] == [
        "header",
        "table",
        "text",
    ]
    assert [region.reading_order for region in result.pages[0].regions] == [1, 2, 3]
    assert [region.confidence for region in result.pages[0].regions] == [
        0.88,
        0.91,
        0.97,
    ]
    assert result.pages[0].regions[2].bounding_box.left == 10


def test_paddle_reader_reports_predict_and_output_failures(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class BrokenPipeline:
        def predict(self, image_path: str) -> list[object]:
            raise RuntimeError("GPU unavailable")

    predict_result = process_document(
        image_path, PaddleOCRVLReader(pipeline=BrokenPipeline())
    )
    invalid_result = process_document(
        image_path, PaddleOCRVLReader(pipeline=FakePipeline({"wrong": []}))
    )

    assert predict_result.failures[0].code == "paddle_predict_failed"
    assert invalid_result.failures[0].code == "invalid_reader_output"


def test_paddle_reader_reports_import_and_init_failures(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    missing_module = ModuleType("paddleocr")
    monkeypatch.setitem(sys.modules, "paddleocr", missing_module)
    import_result = process_document(image_path, PaddleOCRVLReader())
    assert import_result.failures[0].code == "paddle_import_failed"

    failing_module = ModuleType("paddleocr")

    def fail_init(**options: object) -> object:
        assert options == {
            "pipeline_version": "v1.6",
            "vl_rec_backend": "native",
            "device": "gpu:0",
        }
        raise RuntimeError("model unavailable")

    failing_module.PaddleOCRVL = fail_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", failing_module)
    init_result = process_document(image_path, PaddleOCRVLReader(device="gpu:0"))
    assert init_result.failures[0].code == "paddle_init_failed"

    def fail_preprocessed_init(**options: object) -> object:
        assert options == {
            "pipeline_version": "v1.6",
            "vl_rec_backend": "native",
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": False,
        }
        raise RuntimeError("model unavailable")

    failing_module.PaddleOCRVL = fail_preprocessed_init  # type: ignore[attr-defined]
    preprocessed_result = process_document(
        image_path,
        PaddleOCRVLReader(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
        ),
    )
    assert preprocessed_result.failures[0].code == "paddle_init_failed"
