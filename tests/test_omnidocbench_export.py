from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


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
        "Heading\n\n<table><tr><td>A</td></tr></table>\n\n$$x^2$$\n"
    )
    assert (output_dir / "reader-failure.md").read_text(encoding="utf-8") == ""
    assert (output_dir / "missing.md").read_text(encoding="utf-8") == ""
    assert not (output_dir / "not-selected.md").exists()

    report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["dataset_revision"] == "hf-revision-abc123"
    assert report["reader"] == "tesseract"
    assert report["run_config"] == {
        "reader": "tesseract",
        "reader_options": {
            "language": "eng",
            "executable": str(tesseract),
            "timeout_seconds": 120,
        },
        "limit": 3,
        "attributes": {"language": "english"},
    }
    assert {
        name: report[name] for name in ("attempted", "success", "partial", "failed")
    } == {"attempted": 3, "success": 1, "partial": 0, "failed": 2}
    assert [page["status"] for page in report["pages"]] == [
        "success",
        "failed",
        "failed",
    ]
    assert report["pages"][1]["failures"][0]["code"] == "reader_failed"
    assert report["pages"][2]["failures"][0]["code"] == "source_not_found"
    assert all(page["latency_ms"] >= 0 for page in report["pages"])


def _record(image_path: str, language: str) -> dict[str, object]:
    return {
        "layout_dets": [{"text": "ground truth must not control selection"}],
        "page_info": {
            "image_path": image_path,
            "page_attribute": {"language": language},
        },
    }
