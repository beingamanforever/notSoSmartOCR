from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.contracts import BoundingBox, EvidenceText, TextRegion
from ocr_pipeline.providers import GraniteDoclingReader, NemotronOCRV2Reader

from experiments import omnidocbench_export


def test_cli_exports_markdown_and_keeps_failures_in_denominator(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("readable.jpg", "reader-failure.jpg", "not-selected.jpg"):
        Image.new("RGB", (80, 40), "white").save(image_root / name)

    annotations = tmp_path / "OmniDocBench.json"
    records = [
        _record("readable.jpg", "english"),
        _record("reader-failure.jpg", "english"),
        _record("missing.jpg", "english"),
        _record("not-selected.jpg", "simplified_chinese"),
    ]
    annotations.write_text(json.dumps(records), encoding="utf-8")

    tesseract = tmp_path / "tesseract"
    tesseract.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  *reader-failure.jpg) echo 'controlled failure' >&2; exit 9 ;;\n"
        "esac\n"
        "printf 'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n'\n"
        "printf '5\\t1\\t1\\t1\\t1\\t1\\t1\\t2\\t20\\t10\\t99\\tHeading\\n'\n"
        "printf '5\\t1\\t1\\t1\\t2\\t1\\t1\\t14\\t20\\t10\\t99\\t<table><tr><td>A</td></tr></table>\\n'\n"
        "printf '5\\t1\\t1\\t1\\t3\\t1\\t1\\t26\\t20\\t10\\t99\\t$$x^2$$\\n'\n",
        encoding="utf-8",
    )
    tesseract.chmod(0o755)

    output_dir = tmp_path / "predictions"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "experiments" / "omnidocbench_export.py"),
            str(annotations),
            str(image_root),
            str(output_dir),
            "--dataset-revision",
            "hf-revision-abc123",
            "--attribute",
            "language=english",
            "--limit",
            "3",
            "--tesseract",
            str(tesseract),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    assert (output_dir / "readable.md").read_text(encoding="utf-8") == (
        "Heading <table><tr><td>A</td></tr></table> $$x^2$$\n"
    )
    assert (output_dir / "reader-failure.md").read_text(encoding="utf-8") == ""
    assert (output_dir / "missing.md").read_text(encoding="utf-8") == ""
    assert not (output_dir / "not-selected.md").exists()

    readable = json.loads(
        (output_dir / "structured" / "readable.json").read_text(encoding="utf-8")
    )
    assert readable["case_id"] == "readable.jpg"
    assert readable["geometry_availability"] == "available"
    assert readable["pages"][0]["width"] == 80
    assert readable["pages"][0]["height"] == 40
    assert readable["pages"][0]["route"] == "accept_local"
    assert readable["pages"][0]["regions"][0] == {
        "id": "p1-word-1",
        "kind": "word",
        "text": "Heading",
        "confidence": 0.99,
        "bounding_box": {"left": 1, "top": 2, "right": 21, "bottom": 12},
        "reading_order": 1,
        "provider": "tesseract",
        "text_provenance": {
            "method": "tesseract_tsv",
            "block_num": 1,
            "paragraph_num": 1,
            "line_num": 1,
            "word_num": 1,
        },
        "resolution": "resolved",
        "alternatives": [],
        "structure": None,
        "geometry_availability": "available",
    }

    failed = json.loads(
        (output_dir / "structured" / "reader-failure.json").read_text(encoding="utf-8")
    )
    missing = json.loads(
        (output_dir / "structured" / "missing.json").read_text(encoding="utf-8")
    )
    assert failed["failures"][0]["code"] == "reader_failed"
    assert failed["pages"][0]["failure_ids"] == ["failure-1"]
    assert missing["failures"][0]["code"] == "source_not_found"
    assert missing["pages"] == []
    assert missing["geometry_availability"] == "unavailable"

    report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["dataset_revision"] == "hf-revision-abc123"
    assert report["reader"] == "tesseract"
    assert report["run_config"] == {
        "reader": "tesseract",
        "reader_options": {
            "language": "eng",
            "executable": str(tesseract),
            "timeout_seconds": 120,
            "page_segmentation_mode": None,
            "thresholding_method": None,
        },
        "markdown_assembly": "tesseract_tsv_paragraphs",
        "workers": 1,
        "limit": 3,
        "attributes": {"language": "english"},
    }
    assert {
        name: report[name]
        for name in (
            "attempted",
            "covered",
            "failed",
            "abstained",
            "success",
            "partial",
        )
    } == {
        "attempted": 3,
        "covered": 1,
        "failed": 2,
        "abstained": 2,
        "success": 1,
        "partial": 0,
    }
    assert report["case_ids"] == [
        "readable.jpg",
        "reader-failure.jpg",
        "missing.jpg",
    ]
    assert report["markdown_output_dir"] == str(output_dir)
    assert report["structured_output_dir"] == str(output_dir / "structured")
    assert set(report["latency_ms"]) == {"p50", "p95"}
    assert report["wall_latency_ms"] >= 0
    assert report["pages_per_second"] >= 0
    assert [page["status"] for page in report["pages"]] == [
        "success",
        "failed",
        "failed",
    ]
    assert report["pages"][1]["failures"][0]["code"] == "reader_failed"
    assert report["pages"][2]["failures"][0]["code"] == "source_not_found"
    assert all(page["latency_ms"] >= 0 for page in report["pages"])


def test_export_preserves_provenance_geometry_and_reading_order(
    tmp_path: Path, monkeypatch
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 60), "white").save(image_root / "page.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([_record("page.png", "english")]))

    class OrderedReader:
        name = "ordered"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="second",
                    kind="text",
                    text="Second",
                    confidence=0.8,
                    bounding_box=BoundingBox(10, 30, 90, 50),
                    reading_order=2,
                    provider=self.name,
                    text_provenance={"method": "native", "source": "line-2"},
                ),
                TextRegion(
                    id="first",
                    kind="title",
                    text="First",
                    confidence=0.9,
                    bounding_box=BoundingBox(5, 5, 95, 25),
                    reading_order=1,
                    provider=self.name,
                    text_provenance={"method": "native", "source": "line-1"},
                ),
            ]

    calls = 0
    process_document = omnidocbench_export.process_document

    def count_process_document(source: Path, reader: OrderedReader):
        nonlocal calls
        calls += 1
        return process_document(source, reader)

    monkeypatch.setattr(
        omnidocbench_export,
        "process_document",
        count_process_document,
    )
    output_dir = tmp_path / "run"
    report = omnidocbench_export.export_predictions(
        annotations,
        image_root,
        output_dir,
        OrderedReader(),
        dataset_revision="revision-exact",
    )

    assert calls == 1
    assert (output_dir / "page.md").read_text(encoding="utf-8") == ("First\n\nSecond\n")
    structured = json.loads(
        (output_dir / "structured" / "page.json").read_text(encoding="utf-8")
    )
    regions = structured["pages"][0]["regions"]
    assert [region["id"] for region in regions] == ["first", "second"]
    assert regions[0]["text_provenance"] == {
        "method": "native",
        "source": "line-1",
    }
    assert regions[0]["bounding_box"] == {
        "left": 5,
        "top": 5,
        "right": 95,
        "bottom": 25,
    }
    assert report["dataset_revision"] == "revision-exact"


def test_markdown_groups_tesseract_words_by_exact_tsv_line() -> None:
    page = omnidocbench_export.PageResult(
        page_number=1,
        width=200,
        height=80,
        reader="tesseract",
        route="accept_local",
        text=EvidenceText("Alpha beta Gamma", []),
        regions=[
            _tsv_word("alpha", "Alpha", order=1, line=1, word=1),
            _tsv_word("beta", "beta", order=2, line=1, word=2),
            _tsv_word("gamma", "Gamma", order=3, line=2, word=1),
            _tsv_word(
                "delta",
                "Delta",
                order=4,
                line=1,
                word=1,
                paragraph=2,
            ),
        ],
    )
    result = omnidocbench_export.DocumentResult(
        document_id="page",
        source={"name": "page.png", "kind": "image"},
        status="success",
        pages=[page],
    )

    assert omnidocbench_export._markdown(result) == "Alpha beta Gamma\n\nDelta\n"


def test_single_tesseract_paragraph_stays_one_official_text_block() -> None:
    regions = [
        _tsv_word("alpha", "Alpha", order=1, line=1, word=1),
        _tsv_word("beta", "beta", order=2, line=1, word=2),
        _tsv_word("gamma", "Gamma", order=3, line=2, word=1),
    ]

    markdown = omnidocbench_export._tesseract_markdown(regions)
    blocks = markdown.split("\n\n")
    if len(blocks) == 1:
        blocks = markdown.split("\n")

    assert blocks == ["Alpha beta Gamma"]


def test_export_labels_full_page_text_geometry_as_page_only(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (70, 30), "white").save(image_root / "page.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([_record("page.png", "english")]))

    class FullPageReader:
        name = "full-page"

        def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
            return [
                TextRegion(
                    id="page-text",
                    kind="page_text",
                    text="Only page-level text",
                    confidence=None,
                    bounding_box=BoundingBox(0, 0, 70, 30),
                    reading_order=1,
                    provider=self.name,
                    text_provenance={"method": "full_page_generation"},
                )
            ]

    output_dir = tmp_path / "run"
    omnidocbench_export.export_predictions(
        annotations,
        image_root,
        output_dir,
        FullPageReader(),
        dataset_revision="revision",
    )

    structured = json.loads(
        (output_dir / "structured" / "page.json").read_text(encoding="utf-8")
    )
    page = structured["pages"][0]
    assert structured["geometry_availability"] == "page_only"
    assert page["geometry_availability"] == "page_only"
    assert page["regions"][0]["geometry_availability"] == "page_only"
    assert page["regions"][0]["bounding_box"] == {
        "left": 0,
        "top": 0,
        "right": 70,
        "bottom": 30,
    }


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_export_does_not_overwrite_existing_run_path(
    tmp_path: Path, existing_kind: str
) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([_record("page.png", "english")]))
    output = tmp_path / "run"
    if existing_kind == "directory":
        output.mkdir()
        marker = output / "existing.txt"
        marker.write_text("keep", encoding="utf-8")
    else:
        output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        omnidocbench_export.export_predictions(
            annotations,
            tmp_path,
            output,
            object(),
            dataset_revision="revision",
        )

    if existing_kind == "directory":
        assert marker.read_text(encoding="utf-8") == "keep"
        assert list(output.iterdir()) == [marker]
    else:
        assert output.read_text(encoding="utf-8") == "keep"


def test_cli_selects_eligible_model_readers(tmp_path: Path, monkeypatch) -> None:
    readers = []

    def fake_export_predictions(annotations, image_root, output_dir, reader, **options):
        readers.append(reader)
        return {"failed": 0, "partial": 0}

    monkeypatch.setattr(
        omnidocbench_export,
        "export_predictions",
        fake_export_predictions,
    )
    common = [
        str(tmp_path / "annotations.json"),
        str(tmp_path / "images"),
        str(tmp_path / "output"),
        "--dataset-revision",
        "revision",
    ]

    assert (
        omnidocbench_export.main(
            [*common, "--reader", "granite-docling", "--max-new-tokens", "512"]
        )
        == 0
    )
    assert (
        omnidocbench_export.main(
            [
                *common,
                "--reader",
                "nemotron-ocr-v2",
                "--nemotron-language",
                "en",
                "--nemotron-merge-level",
                "sentence",
            ]
        )
        == 0
    )

    granite, nemotron = readers
    assert isinstance(granite, GraniteDoclingReader)
    assert granite.max_new_tokens == 512
    assert granite.output_format == "markdown"
    assert isinstance(nemotron, NemotronOCRV2Reader)
    assert nemotron.language == "en"
    assert nemotron.merge_level == "sentence"


def test_concurrent_export_is_tesseract_only(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([_record("page.png", "english")]))

    with pytest.raises(ValueError, match="supports Tesseract only"):
        omnidocbench_export.export_predictions(
            annotations,
            tmp_path,
            tmp_path / "run",
            object(),
            dataset_revision="revision",
            workers=2,
        )


def _record(image_path: str, language: str) -> dict[str, object]:
    return {
        "layout_dets": [{"text": "ground truth must not control selection"}],
        "page_info": {
            "image_path": image_path,
            "page_attribute": {"language": language},
        },
    }


def _tsv_word(
    region_id: str,
    text: str,
    *,
    order: int,
    line: int,
    word: int,
    paragraph: int = 1,
) -> TextRegion:
    return TextRegion(
        id=region_id,
        kind="word",
        text=text,
        confidence=0.99,
        bounding_box=BoundingBox(10 * word, 10 * line, 10 * word + 8, 10 * line + 8),
        reading_order=order,
        provider="tesseract",
        text_provenance={
            "method": "tesseract_tsv",
            "block_num": 1,
            "paragraph_num": paragraph,
            "line_num": line,
            "word_num": word,
        },
    )
