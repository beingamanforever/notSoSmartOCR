from __future__ import annotations

import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from experiments.prepare_handwriting_pages import main, prepare_pages


def _pdf(path: Path, color: str) -> None:
    Image.new("RGB", (80, 120), color).save(path, resolution=72)


def test_prepares_only_non_heldout_document_families(tmp_path: Path) -> None:
    source = tmp_path / "source"
    forms = source / "Forms"
    forms.mkdir(parents=True)
    for name, color in (("a.png", "red"), ("b.png", "blue"), ("c.png", "green")):
        Image.new("RGB", (80, 120), color).save(forms / name)
    shutil.copyfile(forms / "b.png", forms / "renamed-copy.png")

    heldout = tmp_path / "heldout" / "sources" / "C01-forms"
    heldout.mkdir(parents=True)
    Image.new("RGB", (80, 120), "blue").save(heldout / "C01-D002-P001.png")

    output = tmp_path / "prepared"
    summary = prepare_pages(
        source,
        tmp_path / "heldout",
        output,
        categories=["Forms"],
        dpi=72,
    )

    records = [
        json.loads(line)
        for line in (output / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["candidate_documents"] == 2
    assert summary["candidate_pages"] == 2
    assert summary["excluded_heldout_or_name_match"] == 2
    assert {record["family_id"] for record in records} == {"C01-D001", "C01-D003"}
    assert all(record["state"] == "pending" for record in records)
    assert all((output / record["image_path"]).is_file() for record in records)
    assert (output / "contact-sheets" / "sheet-001.png").is_file()

    selected = tmp_path / "selected"
    selected_summary = prepare_pages(
        source,
        tmp_path / "heldout",
        selected,
        categories=["C01"],
        families=["C01-D003"],
        dpi=72,
    )
    assert selected_summary["candidate_documents"] == 1


def test_rejects_unknown_category_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "Forms").mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(source / "Forms" / "a.png")
    heldout = tmp_path / "heldout"
    (heldout / "sources").mkdir(parents=True)

    with pytest.raises(ValueError, match="unknown categories"):
        prepare_pages(
            source,
            heldout,
            tmp_path / "unknown",
            categories=["Missing"],
        )

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        prepare_pages(source, heldout, output, categories=["Forms"])


def test_cli_prioritizes_pdf_pages_with_less_embedded_text(tmp_path: Path) -> None:
    source = tmp_path / "source"
    forms = source / "Forms"
    forms.mkdir(parents=True)
    _pdf(forms / "a-dense.pdf", "white")
    _pdf(forms / "b-scan.pdf", "black")
    heldout = tmp_path / "heldout"
    (heldout / "sources").mkdir(parents=True)

    pdftotext = tmp_path / "pdftotext"
    pdftotext.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "if 'dense' in Path(sys.argv[-2]).name:\n"
        "    print('embedded text')\n",
        encoding="utf-8",
    )
    pdftotext.chmod(0o755)

    output = tmp_path / "prioritized"
    result = main(
        [
            str(source),
            str(heldout),
            str(output),
            "--category",
            "Forms",
            "--dpi",
            "72",
            "--prioritize-low-text",
            "--pdftotext",
            str(pdftotext),
        ]
    )

    assert result == 0
    records = [
        json.loads(line)
        for line in (output / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["family_id"] for record in records] == [
        "C01-D002",
        "C01-D001",
    ]
    assert [record["embedded_text_characters"] for record in records] == [0, 12]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["prioritized_low_text"] is True
