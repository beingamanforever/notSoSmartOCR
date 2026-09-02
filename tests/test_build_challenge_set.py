from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from experiments.build_challenge_set import build_challenge_set


def _pdf(path: Path, pages: int) -> None:
    images = [
        Image.new("RGB", (80, 120), color=(index, 20, 30)) for index in range(pages)
    ]
    images[0].save(path, save_all=True, append_images=images[1:], resolution=72)


def test_builds_page_level_panel_once_with_sidecars_excluded(tmp_path: Path) -> None:
    source = tmp_path / "source"
    first = source / "A forms"
    second = source / "B empty"
    first.mkdir(parents=True)
    second.mkdir()
    _pdf(first / "one.pdf", 5)
    _pdf(first / "two.pdf", 3)
    (first / "derived.txt").write_text("not an OCR input", encoding="utf-8")
    Image.new("RGB", (30, 40), "white").save(first / "three.jpg")
    output = tmp_path / "panel"

    summary = build_challenge_set(
        source,
        output,
        pages_per_category=6,
        dpi=72,
    )

    assert summary["selected_pages"] == 6
    assert summary["categories"] == [
        {
            "category_id": "C01",
            "category": "A forms",
            "visual_sources": 3,
            "available_pages": 9,
            "selected_pages": 6,
        },
        {
            "category_id": "C02",
            "category": "B empty",
            "visual_sources": 0,
            "available_pages": 0,
            "selected_pages": 0,
        },
    ]
    pages = sorted((output / "sources" / "C01-a-forms").glob("*.png"))
    assert [path.name for path in pages] == [
        "C01-D001-P001.png",
        "C01-D001-P003.png",
        "C01-D001-P005.png",
        "C01-D002-P001.png",
        "C01-D003-P001.png",
        "C01-D003-P003.png",
    ]
    for page in pages:
        with Image.open(page) as image:
            image.verify()

    with pytest.raises(FileExistsError):
        build_challenge_set(source, output, pages_per_category=6, dpi=72)
