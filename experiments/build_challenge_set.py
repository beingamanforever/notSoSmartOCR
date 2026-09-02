"""Build a page-level local challenge panel from document folders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VISUAL_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}


@dataclass(frozen=True)
class Source:
    path: Path
    number: int
    pages: int


@dataclass(frozen=True)
class Pick:
    source: Source
    page: int


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_challenge_set(
        args.source,
        args.output,
        pages_per_category=args.pages_per_category,
        dpi=args.dpi,
        pdfinfo=args.pdfinfo,
        pdftoppm=args.pdftoppm,
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_challenge_set(
    source_root: Path,
    output_root: Path,
    *,
    pages_per_category: int = 20,
    dpi: int = 300,
    pdfinfo: str = "pdfinfo",
    pdftoppm: str = "pdftoppm",
) -> dict[str, Any]:
    if pages_per_category < 1:
        raise ValueError("pages_per_category must be positive")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Challenge source not found: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Challenge output already exists: {output_root}")

    categories = sorted(
        (path for path in source_root.iterdir() if path.is_dir()),
        key=_path_key,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}-",
            dir=output_root.parent,
        )
    )
    summaries: list[dict[str, Any]] = []
    try:
        sources_root = temporary / "sources"
        for category_number, category in enumerate(categories, start=1):
            category_id = f"C{category_number:02d}"
            sources = _sources(category, pdfinfo)
            picks = select_pages(sources, pages_per_category)
            category_dir = sources_root / f"{category_id}-{_slug(category.name)}"
            if picks:
                category_dir.mkdir(parents=True)
            for pick in picks:
                case_id = f"{category_id}-D{pick.source.number:03d}-P{pick.page:03d}"
                _render_page(
                    pick,
                    category_dir / f"{case_id}.png",
                    dpi=dpi,
                    pdftoppm=pdftoppm,
                )
            summaries.append(
                {
                    "category_id": category_id,
                    "category": category.name,
                    "visual_sources": len(sources),
                    "available_pages": sum(item.pages for item in sources),
                    "selected_pages": len(picks),
                }
            )
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "source": str(source_root.resolve()),
        "output": str(output_root.resolve()),
        "dpi": dpi,
        "pages_per_category": pages_per_category,
        "selected_pages": sum(item["selected_pages"] for item in summaries),
        "categories": summaries,
    }


def select_pages(sources: list[Source], limit: int) -> list[Pick]:
    if not sources or limit < 1:
        return []
    available = sum(source.pages for source in sources)
    if available <= limit:
        return [
            Pick(source, page)
            for source in sources
            for page in range(1, source.pages + 1)
        ]
    if len(sources) >= limit:
        indices = _even_positions(len(sources), limit, zero_based=True)
        picks = []
        for rank, index in enumerate(indices):
            source = sources[index]
            page = _spread_page(source.pages, rank, len(indices))
            picks.append(Pick(source, page))
        return picks

    counts = [1] * len(sources)
    remaining = limit - len(sources)
    while remaining:
        eligible = [
            index
            for index, source in enumerate(sources)
            if counts[index] < source.pages
        ]
        if not eligible:
            break
        index = max(
            eligible,
            key=lambda item: (sources[item].pages / counts[item], -item),
        )
        counts[index] += 1
        remaining -= 1

    return [
        Pick(source, page)
        for source, count in zip(sources, counts, strict=True)
        for page in _even_positions(source.pages, count)
    ]


def _sources(category: Path, pdfinfo: str) -> list[Source]:
    paths = sorted(
        (
            path
            for path in category.rglob("*")
            if path.is_file() and path.suffix.lower() in VISUAL_SUFFIXES
        ),
        key=lambda path: _relative_key(path, category),
    )
    sources = []
    for number, path in enumerate(paths, start=1):
        pages = (
            _pdf_pages(path, pdfinfo)
            if path.suffix.lower() == ".pdf"
            else _image_pages(path)
        )
        sources.append(Source(path=path, number=number, pages=pages))
    return sources


def _pdf_pages(path: Path, pdfinfo: str) -> int:
    result = subprocess.run(
        [pdfinfo, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not read PDF page count: {path}")
    pages = int(match.group(1))
    if pages < 1:
        raise ValueError(f"PDF has no pages: {path}")
    return pages


def _image_pages(path: Path) -> int:
    try:
        with Image.open(path) as image:
            return int(getattr(image, "n_frames", 1))
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image: {path}") from error


def _render_page(
    pick: Pick,
    output: Path,
    *,
    dpi: int,
    pdftoppm: str,
) -> None:
    if pick.source.path.suffix.lower() == ".pdf":
        prefix = output.with_suffix("")
        subprocess.run(
            [
                pdftoppm,
                "-f",
                str(pick.page),
                "-l",
                str(pick.page),
                "-singlefile",
                "-r",
                str(dpi),
                "-png",
                str(pick.source.path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        try:
            with Image.open(pick.source.path) as image:
                image.seek(pick.page - 1)
                image.copy().save(output, format="PNG")
        except (OSError, EOFError, UnidentifiedImageError) as error:
            raise ValueError(
                f"Could not render image page: {pick.source.path}"
            ) from error
    _verify_png(output)


def _verify_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Rendered page is invalid: {path}") from error


def _spread_page(page_count: int, rank: int, total: int) -> int:
    if total == 1:
        return 1 + (page_count - 1) // 2
    return 1 + round(rank * (page_count - 1) / (total - 1))


def _even_positions(count: int, wanted: int, *, zero_based: bool = False) -> list[int]:
    if wanted < 1 or wanted > count:
        raise ValueError("wanted must be between one and count")
    offset = 0 if zero_based else 1
    if wanted == 1:
        return [offset + (count - 1) // 2]
    return [
        offset + round(index * (count - 1) / (wanted - 1)) for index in range(wanted)
    ]


def _path_key(path: Path) -> tuple[str, str]:
    return path.name.casefold(), path.name


def _relative_key(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    return relative.casefold(), relative


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "category"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pages-per-category", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdfinfo", default="pdfinfo")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
