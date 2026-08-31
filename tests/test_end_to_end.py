from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ocr_pipeline import (
    GLMOCRDirectReader,
    GLMOCRReader,
    PaddleOCRVLReader,
    TesseractReader,
)
from ocr_pipeline import cli as ocr_cli
from ocr_pipeline.contracts import BoundingBox, DocumentResult, EvidenceText, TextRegion
from ocr_pipeline.pipeline import process_document


def test_cli_preserves_page_order_evidence_and_failures(tmp_path: Path) -> None:
    tesseract = shutil.which("tesseract")
    assert tesseract, "Tesseract is required for the end-to-end test"
    assert shutil.which("pdftoppm"), "pdftoppm is required for the end-to-end test"

    source = tmp_path / "two-pages.pdf"
    _write_pdf(source)

    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "tesseract"
    wrapper.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  *page-2.png) echo 'controlled page failure' >&2; exit 9 ;;\n"
        "esac\n"
        f'exec {shlex.quote(tesseract)} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    output = tmp_path / "result.json"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["PATH"] = f"{wrapper_dir}{os.pathsep}{environment['PATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ocr_pipeline.cli",
            str(source),
            "--output",
            str(output),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "partial"
    assert [page["page_number"] for page in result["pages"]] == [1, 2, 3]

    first_page, second_page, third_page = result["pages"]
    assert first_page["width"] > 0 and first_page["height"] > 0
    assert first_page["regions"]
    region_ids = {region["id"] for region in first_page["regions"]}
    assert set(first_page["text"]["evidence_ids"]) == region_ids
    assert "FIRST" in first_page["text"]["value"].upper()
    assert all(region["provider"] == "tesseract" for region in first_page["regions"])
    assert all(
        region["bounding_box"]["right"] > region["bounding_box"]["left"]
        and region["bounding_box"]["bottom"] > region["bounding_box"]["top"]
        for region in first_page["regions"]
    )

    assert second_page["width"] > 0 and second_page["height"] > 0
    assert second_page["route"] == "review"
    assert second_page["regions"] == []
    assert second_page["failure_ids"] == ["failure-1"]
    assert third_page["route"] == "review"
    assert len(third_page["regions"]) == 1
    empty_region = third_page["regions"][0]
    assert empty_region["id"] == "p3-empty-page-1"
    assert empty_region["kind"] == "page_text"
    assert empty_region["text"] == ""
    assert empty_region["confidence"] is None
    assert empty_region["bounding_box"] == {
        "left": 0,
        "top": 0,
        "right": third_page["width"],
        "bottom": third_page["height"],
    }
    assert empty_region["reading_order"] == 1
    assert empty_region["provider"] == "tesseract"
    assert third_page["text"] == {
        "value": "",
        "evidence_ids": ["p3-empty-page-1"],
    }
    assert third_page["failure_ids"] == ["failure-2"]
    assert result["failures"] == [
        {
            "id": "failure-1",
            "stage": "ocr",
            "code": "reader_failed",
            "message": "controlled page failure",
            "page_number": 2,
        },
        {
            "id": "failure-2",
            "stage": "ocr",
            "code": "no_text_detected",
            "message": "The reader returned no text regions",
            "page_number": 3,
        },
    ]


def test_multi_page_tiff_ingests_every_frame_in_order(tmp_path: Path) -> None:
    tesseract = shutil.which("tesseract")
    assert tesseract, "Tesseract is required for the end-to-end test"

    source = tmp_path / "two-frames.tiff"
    first = _text_image("FIRST TIFF")
    second = _text_image("SECOND TIFF")
    first.save(source, format="TIFF", save_all=True, append_images=[second])

    result = process_document(source, TesseractReader(executable=tesseract))

    assert result.status == "success"
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "FIRST" in result.pages[0].text.value.upper()
    assert "SECOND" in result.pages[1].text.value.upper()


def test_empty_reader_result_creates_full_page_repair_target(tmp_path: Path) -> None:
    source = tmp_path / "blank.png"
    Image.new("RGB", (80, 60), "white").save(source)

    result = process_document(source, _EmptyReader())

    assert result.status == "failed"
    assert [failure.code for failure in result.failures] == ["no_text_detected"]
    page = result.pages[0]
    assert page.route == "review"
    assert page.failure_ids == [result.failures[0].id]
    assert page.text == EvidenceText("", ["p1-empty-page-1"])
    assert page.regions == [
        TextRegion(
            id="p1-empty-page-1",
            kind="page_text",
            text="",
            confidence=None,
            bounding_box=BoundingBox(0, 0, 80, 60),
            reading_order=1,
            provider="empty",
        )
    ]


def test_image_ingestion_applies_exif_orientation_before_ocr(tmp_path: Path) -> None:
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[274] = 8
    image.save(source, exif=exif)

    reader = _RecordingReader()
    result = process_document(source, reader)

    assert result.status == "success"
    assert (result.pages[0].width, result.pages[0].height) == (20, 40)
    assert reader.size == (20, 40)
    assert reader.suffix == ".png"
    with Image.open(source) as original:
        assert original.size == (40, 20)
        assert original.getexif()[274] == 8


def test_cli_selects_model_readers_and_passes_reader_options(
    tmp_path: Path, monkeypatch
) -> None:
    captured_readers = []

    def fake_process_document(source, reader, *, pdf_dpi):
        captured_readers.append((reader, pdf_dpi))
        return DocumentResult(
            document_id="test",
            source={"name": "test.png", "kind": "image"},
            status="success",
        )

    monkeypatch.setattr(ocr_cli, "process_document", fake_process_document)
    source = tmp_path / "test.png"

    assert (
        ocr_cli.main(
            [
                str(source),
                "--reader",
                "paddleocr-vl",
                "--backend",
                "native",
                "--device",
                "gpu:0",
                "--use-doc-orientation-classify",
                "--no-use-doc-unwarping",
                "--output",
                str(tmp_path / "paddle.json"),
            ]
        )
        == 0
    )
    assert (
        ocr_cli.main(
            [
                str(source),
                "--reader",
                "glm-ocr",
                "--ocr-api-host",
                "127.0.0.1",
                "--ocr-api-port",
                "8080",
                "--layout-device",
                "cuda:1",
                "--dpi",
                "200",
                "--output",
                str(tmp_path / "sdk.json"),
            ]
        )
        == 0
    )
    assert (
        ocr_cli.main(
            [
                str(source),
                "--reader",
                "glm-ocr-direct",
                "--max-new-tokens",
                "512",
                "--output",
                str(tmp_path / "direct.json"),
            ]
        )
        == 0
    )

    paddle_reader, _ = captured_readers[0]
    assert isinstance(paddle_reader, PaddleOCRVLReader)
    assert paddle_reader.backend == "native"
    assert paddle_reader.device == "gpu:0"
    assert paddle_reader.use_doc_orientation_classify is True
    assert paddle_reader.use_doc_unwarping is False
    sdk_reader, dpi = captured_readers[1]
    assert isinstance(sdk_reader, GLMOCRReader)
    assert sdk_reader.ocr_api_host == "127.0.0.1"
    assert sdk_reader.ocr_api_port == 8080
    assert sdk_reader.layout_device == "cuda:1"
    assert dpi == 200
    direct_reader, _ = captured_readers[2]
    assert isinstance(direct_reader, GLMOCRDirectReader)
    assert direct_reader.max_new_tokens == 512


def _write_pdf(path: Path) -> None:
    font = ImageFont.load_default(size=72)
    pages = []
    for text in ("FIRST PAGE", "SECOND PAGE", ""):
        page = Image.new("RGB", (1000, 600), "white")
        ImageDraw.Draw(page).text((80, 220), text, fill="black", font=font)
        pages.append(page)
    pages[0].save(
        path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )


def _text_image(text: str) -> Image.Image:
    image = Image.new("RGB", (1000, 600), "white")
    ImageDraw.Draw(image).text(
        (80, 220), text, fill="black", font=ImageFont.load_default(size=72)
    )
    return image


class _EmptyReader:
    name = "empty"

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        return []


class _RecordingReader:
    name = "recording"

    def __init__(self) -> None:
        self.size: tuple[int, int] | None = None
        self.suffix = ""

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        with Image.open(image_path) as image:
            self.size = image.size
        self.suffix = image_path.suffix
        assert self.size is not None
        width, height = self.size
        return [
            TextRegion(
                id=f"p{page_number}-block-1",
                kind="text",
                text="visible",
                confidence=1.0,
                bounding_box=BoundingBox(0, 0, width, height),
                reading_order=1,
                provider=self.name,
            )
        ]
