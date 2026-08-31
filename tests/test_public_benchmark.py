from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.providers import GLMOCRDirectReader, GLMOCRReader, PaddleOCRVLReader

from experiments import public_benchmark


def test_clinocr_cli_excludes_exemplar_and_keeps_failures_in_metrics(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "ClinOCR-Bench"
    scans = dataset / "scans" / "normal"
    references = dataset / "ground_truth" / "normal"
    scans.mkdir(parents=True)
    references.mkdir(parents=True)

    rows = [
        ("normal", "1", "1", "exemplar"),
        ("normal", "1", "2", "eval"),
        ("normal", "1", "3", "eval"),
    ]
    with (dataset / "oneshot_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "subset",
                "template",
                "sample",
                "role",
                "homo_template",
                "homo_sample",
                "hetero_template",
                "hetero_sample",
            ]
        )
        for subset, template, sample, role in rows:
            writer.writerow([subset, template, sample, role, "", "", "", ""])
            stem = f"template_{template}_sample_{sample}_{subset}"
            Image.new("RGB", (80, 40), "white").save(scans / f"{stem}.png")
            reference = "EXEMPLAR" if role == "exemplar" else '"HELLO WORLD'
            if sample == "3":
                reference = "MISSING TEXT"
            (references / f"{stem}.txt").write_text(reference, encoding="utf-8")

    tesseract = tmp_path / "tesseract"
    tesseract.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  *sample_1*) echo 'exemplar must not run' >&2; exit 9 ;;\n"
        "esac\n"
        "printf 'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n'\n"
        'case "$1" in\n'
        "  *sample_2*) printf '5\\t1\\t1\\t1\\t1\\t1\\t1\\t2\\t20\\t10\\t99\\t\"HELLO\\n' ; "
        "printf '5\\t1\\t1\\t1\\t1\\t2\\t22\\t2\\t20\\t10\\t99\\tWORLD\\n' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    tesseract.chmod(0o755)

    output = tmp_path / "result.json"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "experiments" / "public_benchmark.py"),
            "clinocr",
            str(dataset),
            str(output),
            "--workers",
            "2",
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

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    summary = result["summary"]
    assert summary["cases"] == 2
    assert result["run_config"] == {
        "reader": "tesseract",
        "reader_options": {
            "language": "eng",
            "executable": str(tesseract),
            "timeout_seconds": 120,
        },
        "workers": 2,
        "selected_subsets": None,
        "limit_per_subset": None,
        "pdf_dpi": 300,
    }
    assert summary["covered_cases"] == 1
    assert summary["coverage"] == 0.5
    assert summary["failed_cases"] == 1
    assert summary["pipeline_failures"] == 1
    assert summary["status_counts"] == {"failed": 1, "success": 1}
    assert summary["failure_codes"] == {"no_text_detected": 1}
    assert summary["cer"] == {
        "case_mean": 0.5,
        "micro": round(12 / 24, 6),
        "edits": 12,
        "reference_units": 24,
    }
    assert summary["wer"] == {
        "case_mean": 0.5,
        "micro": round(2 / 4, 6),
        "edits": 2,
        "reference_units": 4,
    }
    assert result["subsets"]["normal"] == summary
    assert all("sample_1" not in case["id"] for case in result["cases"])
    assert {case["cluster_id"] for case in result["cases"]} == {"1"}

    readable, blank = result["cases"]
    assert readable["prediction"] == readable["reference"] == '"HELLO WORLD'
    assert readable["status"] == "success"
    assert readable["failures"] == []
    assert blank["prediction"] == ""
    assert blank["reference"] == "MISSING TEXT"
    assert blank["status"] == "failed"
    assert blank["metrics"]["cer"]["rate"] == 1.0
    assert blank["metrics"]["wer"]["rate"] == 1.0
    assert blank["failures"][0]["code"] == "no_text_detected"
    assert all(case["latency_ms"] >= 0 for case in result["cases"])


def test_benchmark_scores_table_html_as_visible_text(tmp_path: Path) -> None:
    dataset = tmp_path / "ClinOCR-Bench"
    scans = dataset / "scans" / "tables"
    references = dataset / "ground_truth" / "tables"
    scans.mkdir(parents=True)
    references.mkdir(parents=True)
    (dataset / "oneshot_lookup.csv").write_text(
        "subset,template,sample,role\ntables,9,2,eval\n",
        encoding="utf-8",
    )
    stem = "template_9_sample_2_tables"
    Image.new("RGB", (80, 40), "white").save(scans / f"{stem}.png")
    (references / f"{stem}.txt").write_text("A B", encoding="utf-8")

    tesseract = tmp_path / "tesseract"
    tesseract.write_text(
        "#!/bin/sh\n"
        "printf 'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n'\n"
        "printf '5\\t1\\t1\\t1\\t1\\t1\\t1\\t2\\t20\\t10\\t99\\t<table><tr><td>A</td><td>B</td></tr></table>\\n'\n",
        encoding="utf-8",
    )
    tesseract.chmod(0o755)

    output = tmp_path / "result.json"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "experiments" / "public_benchmark.py"),
            "clinocr",
            str(dataset),
            str(output),
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

    assert completed.returncode == 0, completed.stderr
    case = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
    assert case["prediction"] == "A B"
    assert case["structured_prediction"].startswith("<table>")
    assert case["metrics"]["cer"]["rate"] == 0.0


def test_visible_text_keeps_literal_angle_brackets() -> None:
    assert public_benchmark._transcription_text("x < y > z") == "x < y > z"


def test_public_benchmark_selects_glm_readers_sequentially(
    tmp_path: Path, monkeypatch
) -> None:
    captured_readers = []

    def fake_run_benchmark(dataset, root, workers, reader, **options):
        captured_readers.append((reader, workers))
        return {"reader": reader.name}

    monkeypatch.setattr(public_benchmark, "run_benchmark", fake_run_benchmark)
    common = ["clinocr", str(tmp_path), str(tmp_path / "result.json")]

    assert (
        public_benchmark.main(
            [
                *common,
                "--reader",
                "glm-ocr",
                "--ocr-api-host",
                "localhost",
                "--ocr-api-port",
                "9000",
                "--layout-device",
                "cuda:0",
            ]
        )
        == 0
    )
    assert (
        public_benchmark.main(
            [*common, "--reader", "glm-ocr-direct", "--max-new-tokens", "256"]
        )
        == 0
    )

    sdk_reader, sdk_workers = captured_readers[0]
    assert isinstance(sdk_reader, GLMOCRReader)
    assert sdk_reader.ocr_api_host == "localhost"
    assert sdk_reader.ocr_api_port == 9000
    assert sdk_reader.layout_device == "cuda:0"
    assert sdk_workers == 1
    direct_reader, direct_workers = captured_readers[1]
    assert isinstance(direct_reader, GLMOCRDirectReader)
    assert direct_reader.max_new_tokens == 256
    assert direct_workers == 1

    with pytest.raises(SystemExit) as error:
        public_benchmark.main([*common, "--reader", "glm-ocr", "--workers", "2"])
    assert error.value.code == 2


def test_paddle_run_config_distinguishes_base_and_unwarping() -> None:
    base = public_benchmark._run_config(
        PaddleOCRVLReader(device="gpu:0"), 1, {"rotated"}, 3, 300
    )
    unwarping = public_benchmark._run_config(
        PaddleOCRVLReader(device="gpu:0", use_doc_unwarping=True),
        1,
        {"rotated"},
        3,
        300,
    )

    assert base != unwarping
    assert base["reader_options"]["use_doc_unwarping"] is None
    assert unwarping["reader_options"]["use_doc_unwarping"] is True
