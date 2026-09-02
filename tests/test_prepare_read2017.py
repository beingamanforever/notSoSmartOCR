from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.prepare_read2017 import prepare_read2017


PAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="page.jpg" imageWidth="100" imageHeight="60">
    <TextRegion id="r1">
      <TextLine id="l1"><Coords points="5,5 45,5 45,20 5,20"/><TextEquiv><Unicode>First & line</Unicode></TextEquiv></TextLine>
      <TextLine id="l2"><Coords points="10,25 60,25 60,40 10,40"/><TextEquiv><Unicode>Second line</Unicode></TextEquiv></TextLine>
      <TextLine id="l3"><Coords points="20,42 90,42 90,55 20,55"/><TextEquiv><Unicode>Third line</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def test_prepares_evenly_sampled_read_line_crops(tmp_path: Path) -> None:
    source = tmp_path / "read"
    source.mkdir()
    Image.new("RGB", (100, 60), "white").save(source / "page.jpg")
    (source / "page.xml").write_text(PAGE_XML, encoding="utf-8")
    output = tmp_path / "prepared"

    summary = prepare_read2017(source, output, max_lines_per_page=2)

    rows = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert summary == {
        "dataset": "READ 2017 Train-A",
        "license": "CC BY 4.0",
        "pages": 1,
        "lines": 2,
        "max_lines_per_page": 2,
        "split": "train_only",
    }
    assert [row["reference"] for row in rows] == ["First & line", "Third line"]
    assert {row["data_origin"] for row in rows} == {"public"}
    assert {row["family_id"] for row in rows} == {"READ2017-page"}
    for row in rows:
        tight = output / row["tight_crop_path"]
        padded = output / row["padded_crop_path"]
        assert tight.is_file()
        assert padded.is_file()
        with Image.open(tight) as tight_image, Image.open(padded) as padded_image:
            assert padded_image.width > tight_image.width
            assert padded_image.height > tight_image.height


def test_rejects_missing_input_existing_output_and_unsafe_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source"):
        prepare_read2017(tmp_path / "missing", tmp_path / "output")

    source = tmp_path / "read"
    source.mkdir()
    (source / "page.xml").write_text(
        PAGE_XML.replace('imageFilename="page.jpg"', 'imageFilename="../page.jpg"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        prepare_read2017(source, tmp_path / "output")

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_read2017(source, output)
