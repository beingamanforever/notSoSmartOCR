"""Programmatic text of born-digital PDF pages, read with Docling's PDFium backend.

A born-digital page carries its exact text and positions, including checkbox glyphs
that OCR decoders drop. Docling reads such pages from their cells and OCRs only when
bitmaps cover more than `bitmap_area_threshold` of the page; the same rule is used
here, so scans (and scans with an embedded OCR layer) still go to the decoder.
"""

from __future__ import annotations

import ctypes
import json
import unicodedata
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

# Docling's OcrOptions default: OCR a page only if bitmaps cover more than this share.
BITMAP_AREA_THRESHOLD = 0.05
# Wingdings draws checkboxes with Latin-1 code points; these are its box glyphs.
WINGDINGS = {
    "¨": "☐",
    "o": "□",
    "q": "☐",
    "n": "■",
    "þ": "☑",
    "ý": "☒",
    "ü": "✓",
    "û": "✗",
    "x": "☒",
}
SIDECAR_SUFFIX = ".cells.json"


class Char(NamedTuple):
    """A PDFium character: its box in top-left display points and a Wingdings flag."""

    text: str
    left: float
    top: float
    right: float
    bottom: float
    symbol: bool

    @property
    def x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def y(self) -> float:
        return (self.top + self.bottom) / 2


def write_cell_sidecars(pdf: Path, page_images: list[Path]) -> None:
    """Write each programmatic page's cells, in its image's pixels, next to the image.

    A page whose text runs sideways or upside down is turned upright first: its text
    layer states the reading direction exactly, so no classifier is needed.
    """
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument
    from PIL import Image

    document = InputDocument(
        path_or_stream=pdf, format=InputFormat.PDF, backend=PyPdfiumDocumentBackend
    )
    try:
        for index, image_path in enumerate(page_images):
            page = document._backend.load_page(index)
            try:
                size = page.get_size()
                with Image.open(image_path) as image:
                    scale = image.width / size.width, image.height / size.height
                    image_size = image.size
                chars = _chars(page, size)
                turns = _upright_turns(chars)
                cells = _page_cells(page, size, scale, chars, READING_DIRECTION[turns])
            finally:
                page.unload()
            if cells is None:
                continue
            if turns:
                with Image.open(image_path) as image:
                    image.rotate(90 * turns, expand=True).save(image_path)
                cells = [
                    {**cell, "bbox": _turn_box(cell["bbox"], image_size, turns)}
                    for cell in cells
                ]
            sidecar(image_path).write_text(json.dumps(cells))
    finally:
        document._backend.unload()


def read_cells(image_path: Path) -> list[dict[str, Any]] | None:
    path = sidecar(image_path)
    return json.loads(path.read_text()) if path.is_file() else None


def sidecar(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + SIDECAR_SUFFIX)


def _page_cells(page, size, scale, chars, direction) -> list[dict[str, Any]] | None:
    bitmaps = sum(rect.area() for rect in page.get_bitmap_rects())
    if bitmaps > BITMAP_AREA_THRESHOLD * size.width * size.height:
        return None
    import numpy as np

    has_symbols = any(char.symbol for char in chars)
    x = np.array([char.x for char in chars])
    y = np.array([char.y for char in chars])
    rotation = page._ppage.get_rotation()
    origin = page._ppage.get_bbox()[:2]
    zero = _display_point(rotation, size)(0, 0)
    shifted = _display_point(rotation, size, origin)(0, 0)
    dx, dy = shifted[0] - zero[0], shifted[1] - zero[1]
    cells = []
    for cell in page.get_text_cells():
        if cell.from_ocr:
            continue
        box = cell.rect.to_top_left_origin(size.height).to_bounding_box()
        # Docling rotates PDFium rectangles but does not subtract the page origin.
        box = (box.l + dx, box.t + dy, box.r + dx, box.b + dy)
        if box[2] <= 0 or box[3] <= 0 or box[0] >= size.width or box[1] >= size.height:
            continue
        inside = [
            chars[index]
            for index in np.flatnonzero(
                (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
            )
        ]
        lines = _lines(inside, direction)
        drawn = _visible("".join(char.text for line in lines for char in line))
        # PDFium can return one text rectangle for several lines run together
        # ("SQ CM2=LXW"), or repeat text a PDF overprints to fake bold; its own
        # characters give each visible line back once.
        pieces = (
            [(_line_text(line), _line_box(line)) for line in lines]
            if lines and (len(lines) > 1 or drawn != _visible(cell.text))
            else [(_unicode(cell.text, inside) if has_symbols else cell.text, box)]
        )
        cells += [
            {
                "text": text,
                "bbox": [
                    max(0, left) * scale[0],
                    max(0, top) * scale[1],
                    min(size.width, right) * scale[0],
                    min(size.height, bottom) * scale[1],
                ],
            }
            for text, (left, top, right, bottom) in pieces
            if text.strip()
        ]
    return cells or None


def _visible(text: str) -> str:
    # NFKC: PDFium's cell text spells ligatures out ("ff") where its characters keep "ﬀ".
    return "".join(unicodedata.normalize("NFKC", text).split())


def _lines(chars: list[Char], direction) -> list[list[Char]]:
    """A cell's characters, split where the stream steps back against the reading
    direction: the start of the next line."""
    lines: list[list[Char]] = []
    previous = None
    kept: list[Char] = []
    for char in chars:
        if char.text in "\r\n":
            continue
        size = max(char.right - char.left, char.bottom - char.top)
        # Fake-bold copies sit a fraction of a point off; a real double letter ("ll")
        # sits a whole glyph width away.
        overlap = min(char.right - char.left, char.bottom - char.top) / 4
        if not char.text.isspace() and any(
            other.text == char.text
            and abs(other.x - char.x) + abs(other.y - char.y) < overlap
            for other in kept
        ):
            continue  # the same glyph overprinted in place
        if not char.text.isspace():
            kept.append(char)
            if previous is None or (
                (char.x - previous.x) * direction[0]
                + (char.y - previous.y) * direction[1]
                < -size / 2
            ):
                lines.append([])
            previous = char
        if lines:
            lines[-1].append(char)
    return lines


def _line_text(line: list[Char]) -> str:
    return "".join(
        WINGDINGS.get(char.text, char.text) if char.symbol else char.text
        for char in line
    ).strip()


def _line_box(line: list[Char]) -> tuple[float, float, float, float]:
    visible = [char for char in line if not char.text.isspace()]
    return (
        min(char.left for char in visible),
        min(char.top for char in visible),
        max(char.right for char in visible),
        max(char.bottom for char in visible),
    )


def _chars(page, size) -> list[Char]:
    """Every character of the page, in PDFium's stream order."""
    import pypdfium2.raw as pdfium

    text_page = page._ppage.get_textpage()
    display = _display_point(
        page._ppage.get_rotation(), size, page._ppage.get_bbox()[:2]
    )
    name = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    box = [ctypes.c_double() for _ in range(4)]
    found = []
    try:
        for index in range(text_page.count_chars()):
            pdfium.FPDFText_GetFontInfo(
                text_page.raw, index, name, 128, ctypes.byref(flags)
            )
            pdfium.FPDFText_GetCharBox(
                text_page.raw, index, *(ctypes.byref(v) for v in box)
            )
            left, right, bottom, top = (v.value for v in box)
            (x0, y0), (x1, y1) = display(left, bottom), display(right, top)
            if (
                max(x0, x1) <= 0
                or max(y0, y1) <= 0
                or min(x0, x1) >= size.width
                or min(y0, y1) >= size.height
            ):
                continue
            found.append(
                Char(
                    chr(pdfium.FPDFText_GetUnicode(text_page.raw, index)),
                    min(x0, x1),
                    min(y0, y1),
                    max(x0, x1),
                    max(y0, y1),
                    b"wingdings" in name.value.lower(),
                )
            )
    finally:
        text_page.close()
    return found


def _display_point(rotation: int, size, origin=(0, 0)):
    """Map PDF user space (y up) to the displayed page, which /Rotate turns clockwise."""
    width, height = size.width, size.height  # displayed size
    transform = {
        0: lambda x, y: (x, height - y),
        90: lambda x, y: (y, x),
        180: lambda x, y: (width - x, y),
        270: lambda x, y: (width - y, height - x),
    }[rotation % 360]
    return lambda x, y: transform(x - origin[0], y - origin[1])


# Reading direction on the displayed page for each upright correction (_upright_turns).
READING_DIRECTION = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _upright_turns(chars) -> int:
    """Counterclockwise quarter turns that make the page's dominant reading direction rightward."""
    from collections import Counter

    steps = Counter()
    for a, b in pairwise(chars):
        dx, dy = b.x - a.x, b.y - a.y
        if a.text.isspace() or b.text.isspace() or not 0 < abs(dx) + abs(dy) < 30:
            continue
        steps[(0 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 3)] += 1
    return steps.most_common(1)[0][0] if steps else 0


def _turn_box(box, size, turns):
    """A box in an image of `size` after Pillow's counterclockwise `rotate(90 * turns, expand=True)`."""
    (left, top, right, bottom), (width, height) = box, size
    for _ in range(turns):
        left, top, right, bottom = top, width - right, bottom, width - left
        width, height = height, width
    return [left, top, right, bottom]


def _unicode(text: str, chars: list[Char]) -> str:
    """Rebuild the cell from its characters, mapping those drawn in Wingdings."""
    inside = [(char.text, char.symbol) for char in chars if not char.text.isspace()]
    if not any(symbol for _, symbol in inside):
        return text
    visible = [char for char in text if not char.isspace()]
    if visible != [char for char, _ in inside]:
        return text  # ponytail: cell and characters disagree, keep the backend text
    mapped = iter(
        WINGDINGS.get(char, char) if symbol else char for char, symbol in inside
    )
    return "".join(char if char.isspace() else next(mapped) for char in text)
