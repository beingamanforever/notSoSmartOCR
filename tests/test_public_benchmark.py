from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


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
