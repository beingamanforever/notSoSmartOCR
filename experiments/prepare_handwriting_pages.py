"""Prepare non-held-out document pages for manual handwriting triage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import filecmp
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from experiments.build_challenge_set import (
    Pick,
    Source,
    _even_positions,
    _path_key,
    _render_page,
    _sources,
)


CASE_PATTERN = re.compile(r"^(?P<category>C\d+)-D(?P<document>\d+)-P\d+$")
SHEET_COLUMNS = 3
SHEET_ROWS = 3
CELL_WIDTH = 520
CELL_HEIGHT = 680
LABEL_HEIGHT = 32


@dataclass(frozen=True)
class Candidate:
    category_id: str
    category_name: str
    source: Source

    @property
    def family_id(self) -> str:
        return f"{self.category_id}-D{self.source.number:03d}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_pages(
        args.source,
        args.heldout,
        args.output,
        categories=args.category,
        families=args.family,
        pages_per_document=args.pages_per_document,
        max_documents=args.max_documents,
        dpi=args.dpi,
        pdfinfo=args.pdfinfo,
        pdftoppm=args.pdftoppm,
        prioritize_low_text=args.prioritize_low_text,
        pdftotext=args.pdftotext,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--category",
        action="append",
        required=True,
        help="top-level category name or category id; repeat as needed",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="candidate family id to retain; repeat as needed",
    )
    parser.add_argument("--pages-per-document", type=_positive_int, default=1)
    parser.add_argument("--max-documents", type=_positive_int)
    parser.add_argument("--dpi", type=_minimum_dpi, default=180)
    parser.add_argument("--pdfinfo", default="pdfinfo")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument(
        "--prioritize-low-text",
        action="store_true",
        help="order selected pages by increasing embedded PDF text",
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    return parser


def prepare_pages(
    source_root: Path,
    heldout_root: Path,
    output_root: Path,
    *,
    categories: Sequence[str],
    families: Sequence[str] = (),
    pages_per_document: int = 1,
    max_documents: int | None = None,
    dpi: int = 180,
    pdfinfo: str = "pdfinfo",
    pdftoppm: str = "pdftoppm",
    prioritize_low_text: bool = False,
    pdftotext: str = "pdftotext",
) -> dict[str, object]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root was not found: {source_root}")
    heldout_sources = heldout_root / "sources"
    if not heldout_sources.is_dir():
        raise FileNotFoundError(f"held-out sources were not found: {heldout_sources}")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if pages_per_document < 1:
        raise ValueError("pages per document must be positive")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max documents must be positive")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    category_dirs = sorted(
        (path for path in source_root.iterdir() if path.is_dir()), key=_path_key
    )
    indexed = [
        (f"C{index:02d}", path, _sources(path, pdfinfo))
        for index, path in enumerate(category_dirs, start=1)
    ]
    requested = {value.casefold() for value in categories}
    available = {
        value.casefold()
        for category_id, path, _ in indexed
        for value in (category_id, path.name)
    }
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"unknown categories: {', '.join(missing)}")

    heldout_numbers = _heldout_numbers(heldout_sources)
    heldout_paths = _heldout_paths(indexed, heldout_numbers)
    heldout_names = {_name_key(path) for path in heldout_paths}
    heldout_by_size = _paths_by_size(heldout_paths)
    candidates = []
    excluded_heldout = 0
    excluded_duplicate = 0
    seen_names: set[str] = set()
    seen_by_size: dict[int, list[Path]] = {}
    for category_id, category, sources in indexed:
        if not ({category_id.casefold(), category.name.casefold()} & requested):
            continue
        for source in sources:
            name_key = _name_key(source.path)
            if (
                source.number in heldout_numbers.get(category_id, set())
                or name_key in heldout_names
            ):
                excluded_heldout += 1
                continue
            size = source.path.stat().st_size
            if _matches_any(source.path, heldout_by_size.get(size, [])):
                excluded_heldout += 1
                continue
            if name_key in seen_names or _matches_any(
                source.path, seen_by_size.get(size, [])
            ):
                excluded_duplicate += 1
                continue
            seen_names.add(name_key)
            seen_by_size.setdefault(size, []).append(source.path)
            candidates.append(Candidate(category_id, category.name, source))

    if families:
        requested_families = set(families)
        available_families = {candidate.family_id for candidate in candidates}
        missing_families = sorted(requested_families - available_families)
        if missing_families:
            raise ValueError(
                f"unknown candidate families: {', '.join(missing_families)}"
            )
        candidates = [
            candidate
            for candidate in candidates
            if candidate.family_id in requested_families
        ]

    candidates = _limit_candidates(candidates, max_documents)
    selected_pages: list[tuple[Candidate, int, int | None]] = [
        (candidate, page, None)
        for candidate in candidates
        for page in _selected_pages(candidate.source, pages_per_document)
    ]
    if prioritize_low_text:
        selected_pages = sorted(
            (
                (
                    candidate,
                    page,
                    _embedded_text_characters(candidate.source, page, pdftotext),
                )
                for candidate, page, _ in selected_pages
            ),
            key=lambda item: (
                item[2],
                item[0].category_id,
                item[0].source.number,
                item[1],
            ),
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    records: list[dict[str, object]] = []
    try:
        pages_root = temporary / "pages"
        pages_root.mkdir()
        for candidate, page, embedded_text_characters in selected_pages:
            page_id = f"{candidate.family_id}-P{page:03d}"
            image_path = pages_root / f"{page_id}.png"
            _render_page(
                Pick(candidate.source, page),
                image_path,
                dpi=dpi,
                pdftoppm=pdftoppm,
            )
            record = {
                "page_id": page_id,
                "family_id": candidate.family_id,
                "category_id": candidate.category_id,
                "source_path": str(candidate.source.path.resolve()),
                "source_page": page,
                "image_path": image_path.relative_to(temporary).as_posix(),
                "state": "pending",
            }
            if embedded_text_characters is not None:
                record["embedded_text_characters"] = embedded_text_characters
            records.append(record)
        _write_jsonl(temporary / "queue.jsonl", records)
        sheets = _write_sheets(temporary, records)
        summary = {
            "candidate_documents": len(candidates),
            "candidate_pages": len(records),
            "contact_sheets": sheets,
            "excluded_heldout_or_name_match": excluded_heldout,
            "excluded_duplicate_names": excluded_duplicate,
            "dpi": dpi,
            "pages_per_document": pages_per_document,
            "prioritized_low_text": prioritize_low_text,
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _heldout_numbers(root: Path) -> dict[str, set[int]]:
    values: dict[str, set[int]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        match = CASE_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        category_id = match.group("category")
        values.setdefault(category_id, set()).add(int(match.group("document")))
    return values


def _heldout_paths(
    indexed: list[tuple[str, Path, list[Source]]],
    heldout_numbers: dict[str, set[int]],
) -> list[Path]:
    paths = []
    for category_id, _, sources in indexed:
        numbers = heldout_numbers.get(category_id, set())
        by_number = {source.number: source for source in sources}
        missing = sorted(numbers - by_number.keys())
        if missing:
            raise ValueError(
                f"held-out document numbers no longer map to {category_id}: {missing}"
            )
        paths.extend(by_number[number].path for number in numbers)
    return paths


def _paths_by_size(paths: Sequence[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.stat().st_size, []).append(path)
    return grouped


def _matches_any(path: Path, candidates: Sequence[Path]) -> bool:
    return any(filecmp.cmp(path, candidate, shallow=False) for candidate in candidates)


def _limit_candidates(
    candidates: list[Candidate], max_documents: int | None
) -> list[Candidate]:
    if max_documents is None or len(candidates) <= max_documents:
        return candidates
    return [
        candidates[index]
        for index in _even_positions(len(candidates), max_documents, zero_based=True)
    ]


def _selected_pages(source: Source, count: int) -> list[int]:
    wanted = min(count, source.pages)
    return _even_positions(source.pages, wanted)


def _embedded_text_characters(source: Source, page: int, pdftotext: str) -> int:
    if source.path.suffix.lower() != ".pdf":
        return 0
    result = subprocess.run(
        [
            pdftotext,
            "-f",
            str(page),
            "-l",
            str(page),
            str(source.path),
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(not character.isspace() for character in result.stdout)


def _name_key(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", " ", path.stem.casefold()).strip()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_sheets(root: Path, records: list[dict[str, object]]) -> int:
    sheets_root = root / "contact-sheets"
    sheets_root.mkdir()
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    font = ImageFont.load_default()
    for sheet_index, start in enumerate(range(0, len(records), per_sheet), start=1):
        sheet = Image.new(
            "RGB",
            (SHEET_COLUMNS * CELL_WIDTH, SHEET_ROWS * CELL_HEIGHT),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(records[start : start + per_sheet]):
            row, column = divmod(offset, SHEET_COLUMNS)
            left = column * CELL_WIDTH
            top = row * CELL_HEIGHT
            image_path = root / str(record["image_path"])
            with Image.open(image_path) as image:
                preview = ImageOps.contain(
                    image.convert("RGB"),
                    (CELL_WIDTH - 16, CELL_HEIGHT - LABEL_HEIGHT - 16),
                )
            preview_left = left + (CELL_WIDTH - preview.width) // 2
            preview_top = top + LABEL_HEIGHT + 8
            sheet.paste(preview, (preview_left, preview_top))
            draw.text(
                (left + 8, top + 8), str(record["page_id"]), fill="black", font=font
            )
            draw.rectangle(
                (left, top, left + CELL_WIDTH - 1, top + CELL_HEIGHT - 1),
                outline="#9ca3af",
            )
        sheet.save(sheets_root / f"sheet-{sheet_index:03d}.png", format="PNG")
    return (len(records) + per_sheet - 1) // per_sheet


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _minimum_dpi(value: str) -> int:
    parsed = int(value)
    if parsed < 72:
        raise argparse.ArgumentTypeError("must be at least 72")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
