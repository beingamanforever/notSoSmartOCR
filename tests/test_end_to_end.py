from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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
    assert third_page["regions"] == []
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
