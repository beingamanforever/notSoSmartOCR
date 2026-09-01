from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from ocr_pipeline.providers import (
    GLMOCRDirectReader,
    GLMOCRReader,
    GraniteDoclingReader,
    NemotronOCRV2Reader,
    PaddleOCRVLReader,
    TesseractReader,
)

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
    assert result["dataset_revision"] == "v1.0"
    assert result["case_ids"] == [
        "normal/template_1_sample_2_normal",
        "normal/template_1_sample_3_normal",
    ]
    assert result["run_config"] == {
        "reader": "tesseract",
        "reader_options": {
            "language": "eng",
            "executable": str(tesseract),
            "timeout_seconds": 120,
            "page_segmentation_mode": None,
            "thresholding_method": None,
        },
        "workers": 2,
        "selected_subsets": None,
        "limit_per_subset": None,
        "pdf_dpi": 300,
        "clinocr_role": "eval",
    }
    assert summary["covered_cases"] == 1
    assert summary["coverage"] == 0.5
    assert summary["failed_cases"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["abstained_cases"] == 0
    assert summary["abstention_rate"] == 0.0
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
    assert summary["wall_latency_ms"] >= 0
    assert summary["pages_per_second"] > 0
    assert summary["gpu_memory"] == {
        "basis": "pytorch_cuda_peak_allocated_bytes",
        "peak_bytes": None,
        "available": False,
        "scope": "whole benchmark process",
    }
    assert "reported_cost" not in summary
    assert "cost_per_page" not in summary
    assert "unreported_cost_cases" not in summary
    subset_summary = result["subsets"]["normal"]
    assert all(summary[key] == value for key, value in subset_summary.items())
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

    exemplars = public_benchmark.discover_cases(
        "clinocr",
        dataset,
        clinocr_role="exemplar",
    )
    assert [case.id for case in exemplars] == ["normal/template_1_sample_1_normal"]


def test_abstention_excludes_failed_empty_predictions() -> None:
    records = [
        {
            "prediction": "",
            "status": "success",
            "failures": [],
            "metrics": public_benchmark._score("", "text"),
            "latency_ms": 1.0,
        },
        {
            "prediction": "",
            "status": "failed",
            "failures": [
                {
                    "stage": "reader",
                    "code": "reader_error",
                    "message": "reader failed",
                }
            ],
            "metrics": public_benchmark._score("", "text"),
            "latency_ms": 2.0,
        },
    ]

    summary = public_benchmark._summarize(records)

    assert summary["failed_cases"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["abstained_cases"] == 1
    assert summary["abstention_rate"] == 0.5


def test_summary_marks_unobserved_latency_as_missing() -> None:
    record = {
        "prediction": "",
        "status": "failed",
        "failures": [{"code": "provider_error"}],
        "metrics": public_benchmark._score("", "text"),
        "api_latency_ms": None,
        "cost": None,
    }

    summary = public_benchmark._summarize([record], latency_field="api_latency_ms")

    assert summary["latency_ms"] == {
        "basis": "per_case_wall_latency_ms",
        "observed_cases": 0,
        "missing_cases": 1,
        "p50": None,
        "p95": None,
    }
    assert summary["cost_per_page"] is None
    assert summary["cost_per_reported_page"] is None


def test_transcription_metrics_cover_edit_types_and_failure_denominators_end_to_end(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "ClinOCR-Bench"
    scans = dataset / "scans" / "normal"
    references = dataset / "ground_truth" / "normal"
    scans.mkdir(parents=True)
    references.mkdir(parents=True)
    reference_texts = (
        "abc",
        "alpha beta",
        "alpha",
        "cat",
        "",
        "lost",
    )
    with (dataset / "oneshot_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["subset", "template", "sample", "role"])
        for sample, reference in enumerate(reference_texts, start=1):
            writer.writerow(["normal", sample, sample, "eval"])
            stem = f"template_{sample}_sample_{sample}_normal"
            Image.new("RGB", (80, 40), "white").save(scans / f"{stem}.png")
            (references / f"{stem}.txt").write_text(reference, encoding="utf-8")

    tesseract = tmp_path / "tesseract"
    tesseract.write_text(
        "#!/bin/sh\n"
        "printf 'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n'\n"
        'case "$1" in\n'
        "  *sample_1*) text='abc' ;;\n"
        "  *sample_2*) text='alpha' ;;\n"
        "  *sample_3*) text='alpha beta' ;;\n"
        "  *sample_4*) text='cut' ;;\n"
        "  *sample_5*) text='extra' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
        "printf '5\\t1\\t1\\t1\\t1\\t1\\t1\\t2\\t20\\t10\\t99\\t%s\\n' \"$text\"\n",
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
    cases = result["cases"]
    assert [case["status"] for case in cases] == [
        "success",
        "success",
        "success",
        "success",
        "success",
        "failed",
    ]
    assert cases[0]["metrics"]["normalized_edit_distance"] == {
        "edits": 0,
        "longer_text_characters": 3,
        "rate": 0.0,
    }
    assert cases[0]["metrics"]["character_edit_counts"] == {
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
    }
    assert cases[0]["metrics"]["word_edit_counts"] == {
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
    }
    assert cases[0]["metrics"]["missed_text_rate"]["rate"] == 0.0
    assert cases[0]["metrics"]["hallucinated_text_rate"]["rate"] == 0.0
    assert cases[1]["metrics"]["character_edit_counts"] == {
        "substitutions": 0,
        "insertions": 0,
        "deletions": 5,
    }
    assert cases[1]["metrics"]["word_edit_counts"]["deletions"] == 1
    assert cases[2]["metrics"]["character_edit_counts"]["insertions"] == 5
    assert cases[2]["metrics"]["word_edit_counts"]["insertions"] == 1
    assert cases[3]["metrics"]["character_edit_counts"]["substitutions"] == 1
    assert cases[3]["metrics"]["word_edit_counts"]["substitutions"] == 1
    assert cases[4]["metrics"]["cer"]["rate"] == 5.0
    assert cases[4]["metrics"]["missed_text_rate"] == {
        "character_deletions": 0,
        "normalized_reference_characters": 0,
        "rate": 0.0,
    }
    assert cases[4]["metrics"]["hallucinated_text_rate"] == {
        "character_insertions": 5,
        "normalized_reference_characters": 0,
        "rate": 5.0,
    }
    assert cases[5]["prediction"] == ""
    assert cases[5]["metrics"]["character_edit_counts"]["deletions"] == 4
    assert cases[5]["metrics"]["missed_text_rate"]["rate"] == 1.0

    summary = result["summary"]
    assert summary["normalized_edit_distance"] == {
        "case_mean": 0.555555,
        "micro": 0.571429,
        "edits": 20,
        "longer_text_characters": 35,
    }
    assert summary["character_edit_counts"] == {
        "substitutions": 1,
        "insertions": 10,
        "deletions": 9,
    }
    assert summary["word_edit_counts"] == {
        "substitutions": 1,
        "insertions": 2,
        "deletions": 2,
    }
    assert summary["missed_text_rate"] == {
        "case_mean": 0.25,
        "micro": 0.36,
        "character_deletions": 9,
        "normalized_reference_characters": 25,
    }
    assert summary["hallucinated_text_rate"] == {
        "case_mean": 1.0,
        "micro": 0.4,
        "character_insertions": 10,
        "normalized_reference_characters": 25,
    }


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


def test_public_benchmark_selects_model_readers_sequentially(
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
    assert (
        public_benchmark.main(
            [*common, "--reader", "granite-docling", "--max-new-tokens", "512"]
        )
        == 0
    )
    assert (
        public_benchmark.main(
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
    granite_reader, granite_workers = captured_readers[2]
    assert isinstance(granite_reader, GraniteDoclingReader)
    assert granite_reader.max_new_tokens == 512
    assert granite_reader.output_format == "text"
    assert granite_workers == 1
    nemotron_reader, nemotron_workers = captured_readers[3]
    assert isinstance(nemotron_reader, NemotronOCRV2Reader)
    assert nemotron_reader.language == "en"
    assert nemotron_reader.merge_level == "sentence"
    assert nemotron_workers == 1

    with pytest.raises(SystemExit) as error:
        public_benchmark.main([*common, "--reader", "glm-ocr", "--workers", "2"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "invalid_host",
    (
        "bench:SENTINEL_SECRET@localhost",
        "localhost?token=SENTINEL_SECRET",
        "https://bench:SENTINEL_SECRET@localhost/ocr?token=SENTINEL_SECRET",
    ),
    ids=("userinfo", "query", "url"),
)
def test_glm_cli_rejects_credentials_before_writing_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
    invalid_host: str,
) -> None:
    benchmark_called = False

    def fake_run_benchmark(*args, **kwargs):
        nonlocal benchmark_called
        benchmark_called = True
        return {}

    monkeypatch.setattr(public_benchmark, "run_benchmark", fake_run_benchmark)
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit) as error:
        public_benchmark.main(
            [
                "clinocr",
                str(tmp_path),
                str(output),
                "--reader",
                "glm-ocr",
                "--ocr-api-host",
                invalid_host,
            ]
        )

    assert error.value.code == 2
    assert benchmark_called is False
    assert not output.exists()
    stderr = capsys.readouterr().err
    assert invalid_host not in stderr
    assert "SENTINEL_SECRET" not in stderr


def test_glm_report_serializes_plain_host_and_rejects_bypasses(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_run_benchmark(dataset, root, workers, reader, **options):
        return {
            "run_config": public_benchmark._run_config(
                reader,
                workers,
                options["selected_subsets"],
                options["limit_per_subset"],
                options["pdf_dpi"],
                options["clinocr_role"],
            )
        }

    monkeypatch.setattr(public_benchmark, "run_benchmark", fake_run_benchmark)
    output = tmp_path / "result.json"

    assert (
        public_benchmark.main(
            [
                "clinocr",
                str(tmp_path),
                str(output),
                "--reader",
                "glm-ocr",
                "--ocr-api-host",
                "127.0.0.1",
            ]
        )
        == 0
    )
    report = output.read_text(encoding="utf-8")
    assert json.loads(report)["run_config"]["reader_options"]["ocr_api_host"] == (
        "127.0.0.1"
    )

    sentinel = "SENTINEL_SECRET"
    with pytest.raises(ValueError) as error:
        public_benchmark.reader_config(
            GLMOCRReader(ocr_api_host=f"user:{sentinel}@localhost")
        )
    assert sentinel not in report
    assert sentinel not in str(error.value)


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


def test_tesseract_run_config_records_segmentation_and_thresholding() -> None:
    config = public_benchmark.reader_config(
        TesseractReader(page_segmentation_mode=3, thresholding_method=1)
    )

    assert config["reader_options"]["page_segmentation_mode"] == 3
    assert config["reader_options"]["thresholding_method"] == 1
