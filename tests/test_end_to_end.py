from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr_pipeline import (
    GLMOCRDirectReader,
    GLMOCRReader,
    GraniteDoclingReader,
    NemotronOCRV2Reader,
    PaddleOCRVLReader,
    TesseractReader,
)
from ocr_pipeline import cli as ocr_cli
from ocr_pipeline import providers as ocr_providers
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


def test_tesseract_explicit_thresholding_is_provenanced(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "faint.png"
    Image.new("RGB", (80, 40), "white").save(source)
    calls = []

    def fake_run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
                "left\ttop\twidth\theight\tconf\ttext\n"
                "5\t1\t1\t1\t1\t1\t4\t5\t20\t10\t98\tFAINT\n"
            ),
        )

    monkeypatch.setattr(ocr_providers.subprocess, "run", fake_run)
    reader = TesseractReader(page_segmentation_mode=3, thresholding_method=1)

    result = process_document(source, reader)

    assert calls[0][0][-5:] == ["--psm", "3", "-c", "thresholding_method=1", "tsv"]
    assert result.pages[0].regions[0].text_provenance == {
        "method": "tesseract_tsv",
        "block_num": 1,
        "paragraph_num": 1,
        "line_num": 1,
        "word_num": 1,
        "tesseract_config": {
            "page_segmentation_mode": 3,
            "thresholding_method": 1,
        },
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"page_segmentation_mode": 14}, "page segmentation"),
        ({"thresholding_method": 3}, "thresholding"),
    ],
)
def test_tesseract_rejects_invalid_configuration(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        TesseractReader(**options)


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
                "--public-comparator",
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
                "nemotron-ocr-v2",
                "--nemotron-language",
                "en",
                "--nemotron-merge-level",
                "sentence",
                "--batch-size",
                "3",
                "--output",
                str(tmp_path / "nemotron.json"),
            ]
        )
        == 0
    )
    assert (
        ocr_cli.main(
            [
                str(source),
                "--reader",
                "granite-docling",
                "--max-new-tokens",
                "1024",
                "--granite-output-format",
                "markdown",
                "--output",
                str(tmp_path / "granite.json"),
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
                "--public-comparator",
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
                "--public-comparator",
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
    nemotron_reader, _ = captured_readers[1]
    assert isinstance(nemotron_reader, NemotronOCRV2Reader)
    assert nemotron_reader.language == "en"
    assert nemotron_reader.merge_level == "sentence"
    assert nemotron_reader.batch_size == 3
    granite_reader, _ = captured_readers[2]
    assert isinstance(granite_reader, GraniteDoclingReader)
    assert granite_reader.max_new_tokens == 1024
    assert granite_reader.output_format == "markdown"
    sdk_reader, dpi = captured_readers[3]
    assert isinstance(sdk_reader, GLMOCRReader)
    assert sdk_reader.ocr_api_host == "127.0.0.1"
    assert sdk_reader.ocr_api_port == 8080
    assert sdk_reader.layout_device == "cuda:1"
    assert dpi == 200
    direct_reader, _ = captured_readers[4]
    assert isinstance(direct_reader, GLMOCRDirectReader)
    assert direct_reader.max_new_tokens == 512


def test_cli_requires_explicit_public_comparator_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        ocr_cli.main(
            [
                str(tmp_path / "page.png"),
                "--reader",
                "paddleocr-vl",
            ]
        )

    assert raised.value.code == 2


@pytest.mark.parametrize("source_kind", ["pdf", "tiff"])
def test_nemotron_native_batch_preserves_multi_page_source_order(
    tmp_path: Path,
    source_kind: str,
) -> None:
    source = tmp_path / f"ordered.{source_kind}"
    pages = [
        Image.new("RGB", (60, 40), color) for color in ((30, 30, 30), (220, 220, 220))
    ]
    pages[0].save(
        source,
        format=source_kind.upper(),
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )

    class OrderedBatchPipeline:
        def __call__(
            self, image_paths: list[str], *, merge_level: str
        ) -> list[list[dict[str, object]]]:
            results = []
            for image_path in image_paths:
                with Image.open(image_path) as image:
                    page_label = (
                        "dark" if image.convert("L").getpixel((0, 0)) < 128 else "light"
                    )
                results.append(
                    [
                        {
                            "text": page_label,
                            "confidence": 0.9,
                            "left": 0.0,
                            "lower": 0.0,
                            "right": 1.0,
                            "upper": 1.0,
                        }
                    ]
                )
            return results

    result = process_document(
        source,
        NemotronOCRV2Reader(batch_size=2, pipeline=OrderedBatchPipeline()),
    )

    assert result.status == "success"
    assert [page.page_number for page in result.pages] == [1, 2]
    assert [page.text.value for page in result.pages] == ["dark", "light"]


def test_invalid_batch_contract_fails_each_page_without_single_page_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "two-pages.tiff"
    page = Image.new("RGB", (20, 10), "white")
    page.save(source, format="TIFF", save_all=True, append_images=[page])

    class InvalidBatchReader:
        name = "invalid-batch"
        batch_size = 2

        def read(self, image_path: Path, page_number: int):
            raise AssertionError("invalid native batches must not retry per page")

        def read_batch(self, image_paths: list[Path], page_numbers: list[int]):
            return [[]]

    result = process_document(source, InvalidBatchReader())

    assert result.status == "failed"
    assert len(result.pages) == 2
    assert [failure.code for failure in result.failures] == [
        "invalid_batch_output",
        "invalid_batch_output",
    ]


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
