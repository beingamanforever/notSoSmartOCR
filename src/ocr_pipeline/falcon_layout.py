"""Read pages with the official Falcon-Perception layout-aware OCR engine.

The engine detects regions with PP-DocLayoutV3 and reads each one with a
category-specific prompt, so checkbox glyphs survive into the text instead of being
absorbed into a neighbouring word box. It runs behind loopback HTTP because its torch
pin differs from the pipeline's.

Regions are page areas, not words. `confidence` carries the layout detector's score,
which says how sure the detector is that a region is there - it is not a statement about
whether the characters were read correctly, and the provenance records that so a reviewer
is not misled by the confidence colouring.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError

PROVIDER = "falcon-perception"
HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
# Only paired asterisks: a lone one is a real footnote mark on many clinical forms.
EMPHASIS = re.compile(r"\*\*(.+?)\*\*")
# Markup truncated mid-tag ("</sup" with no ">") is handed back as text, not a tag.
PARTIAL_TAG = re.compile(r"</?[a-zA-Z][^<>]*$")
# PP-DocLayoutV3 categories that carry no text to read.
EMPTY_CATEGORIES = frozenset({"image", "picture", "figure", "chart", "seal"})
# Layout categories mapped onto the kinds the rest of the pipeline already understands.
KIND_BY_CATEGORY = {
    "table": "table",
    "formula": "formula",
    "doc_title": "title",
    "title": "title",
    "paragraph_title": "section-header",
    "section-header": "section-header",
    "page-header": "page-header",
    "page-footer": "page-footer",
    "header": "page-header",
    "footer": "page-footer",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "caption": "caption",
    "figure_title": "caption",
    "list-item": "list-item",
}


class FalconLayoutReader:
    """Page reader backed by the Falcon-Perception `ocr_layout` engine."""

    name = "falcon-perception-layout"

    def __init__(self, url: str, *, timeout_seconds: float = 600) -> None:
        if not url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("Falcon layout service must be reached on loopback")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._model: dict[str, Any] = {}

    def check_health(self) -> None:
        body = self._request(f"{self.url}/health", None)
        self._model = dict(body.get("model") or {})

    def read(self, image_path: Path, page_number: int) -> list[TextRegion]:
        payload = {
            "image": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "page_text": False,
        }
        body = self._request(f"{self.url}/read", payload)
        elements = body.get("elements")
        if not isinstance(elements, list):
            raise ReaderError("falcon_layout_failed", "malformed element payload")
        model = dict(body.get("model") or self._model)

        regions = []
        for index, element in enumerate(elements, start=1):
            category = str(element.get("category") or "text")
            text = str(element.get("text") or "").strip()
            if not text or category in EMPTY_CATEGORIES:
                continue
            box = _bounding_box(element.get("bbox"))
            if box is None:
                continue
            text, kind = _plain_text(text, KIND_BY_CATEGORY.get(category, "text"))
            if not text:
                continue
            structure = _table_structure(text) if kind == "table" else None
            if structure:
                text = _table_text(structure)
            terminated = _read_terminated(element)
            regions.append(
                TextRegion(
                    id=f"p{page_number}-falcon-{index}",
                    kind=kind,
                    text=text,
                    confidence=float(element.get("score") or 0.0),
                    bounding_box=box,
                    reading_order=index,
                    provider=PROVIDER,
                    text_provenance={
                        "method": "falcon_perception_layout_ocr",
                        "layout_category": category,
                        # Named so nobody reads this as a recognition confidence.
                        "layout_detection_score": element.get("score"),
                        "confidence_meaning": "layout detection, not recognition",
                        "model": model,
                        **({"read_terminated": terminated} if terminated else {}),
                        **(
                            {"read_retries": int(element["retries"])}
                            if element.get("retries") is not None
                            else {}
                        ),
                    },
                    # A read that never terminated, or looped past what the retry
                    # ladder could recover, is evidence of what the model produced and
                    # not a claim about the page. Left resolved it renders invented
                    # text as document content.
                    resolution="unreadable" if terminated else "resolved",
                    structure=structure or {},
                )
            )
        if not regions:
            # A page with no readable region is a failed read, not an empty document.
            # Reporting it as success renders a handful of geometric controls and calls
            # the result "Complete", which is how a warming service looked like a page
            # containing eleven characters.
            raise ReaderError(
                "falcon_layout_empty",
                f"the layout reader returned no readable region for page {page_number}",
            )
        return regions

    def _request(self, url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as reply:
                return json.loads(reply.read())
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            raise ReaderError("falcon_layout_failed", str(error)) from error


class _TableParser(HTMLParser):
    """Collect the model's HTML table as rows of (text, rowspan, colspan, header)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self._row: list[tuple[str, int, int, bool]] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._header = False
        self._in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self._in_head = True
        elif tag == "tr":
            # A row still open when the next one starts was never closed. The model's
            # table markup is often truncated mid-row, so flushing here rather than
            # resetting is what keeps that row's cells.
            self._close_row()
            self._row = []
        elif tag in {"td", "th"}:
            self._close_cell()
            values = dict(attrs)
            self._cell = []
            self._span = (_span(values.get("rowspan")), _span(values.get("colspan")))
            self._header = tag == "th" or self._in_head
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self._in_head = False
        elif tag == "tr":
            self._close_row()
        elif tag in {"td", "th"}:
            self._close_cell()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:
        super().close()
        # Truncated markup closes nothing: no </td>, no </tr>, no </table>.
        self._close_row()

    def _close_cell(self) -> None:
        if self._cell is None:
            return
        if self._row is None:
            self._row = []
        text = PARTIAL_TAG.sub("", "".join(self._cell)).strip()
        self._row.append((text, *self._span, self._header))
        self._cell = None

    def _close_row(self) -> None:
        self._close_cell()
        if self._row:
            self.rows.append(self._row)
        self._row = None


def _table_structure(markup: str) -> dict[str, Any] | None:
    """Place the model's HTML cells on the grid the rest of the pipeline renders.

    Without this the table's HTML source is the region's text, so the plain-text lane
    prints `<table><thead><tr><th>` at a reviewer instead of a table.
    """
    parser = _TableParser()
    parser.feed(markup)
    parser.close()
    if not parser.rows:
        return None

    occupied: dict[tuple[int, int], bool] = {}
    placed: list[tuple[list[int], list[int], str, bool]] = []
    column_count = 0
    for row_index, row in enumerate(parser.rows):
        column = 0
        for text, rowspan, colspan, header in row:
            while (row_index, column) in occupied:
                column += 1
            row_nums = list(range(row_index, row_index + rowspan))
            column_nums = list(range(column, column + colspan))
            for position in ((r, c) for r in row_nums for c in column_nums):
                occupied[position] = header
            placed.append((row_nums, column_nums, text, header))
            column += colspan
            column_count = max(column_count, column)
    if not occupied:
        return None

    # Rows and columns the markup implies but no cell writes into. The markup is often
    # ragged, and padding those out to a rectangle renders a grid of blank rows.
    keep_rows = sorted({r for rows, _, text, _ in placed if text for r in rows})
    keep_columns = sorted({c for _, cols, text, _ in placed if text for c in cols})
    if not keep_rows or not keep_columns:
        return None
    row_of = {row: index for index, row in enumerate(keep_rows)}
    column_of = {column: index for index, column in enumerate(keep_columns)}

    cells: list[dict[str, Any]] = []
    taken: set[tuple[int, int]] = set()
    for row_nums, column_nums, text, header in placed:
        rows = [row_of[r] for r in row_nums if r in row_of]
        columns = [column_of[c] for c in column_nums if c in column_of]
        if not rows or not columns:
            continue
        cells.append(_cell(len(cells), rows, columns, text, header))
        taken.update((r, c) for r in rows for c in columns)
    row_count = len(keep_rows)
    column_count = len(keep_columns)
    # A browser draws a short row as empty trailing cells, and the topology check needs
    # the grid to be a complete rectangle, so the same cells are made explicit here.
    for position in (
        (r, c)
        for r in range(row_count)
        for c in range(column_count)
        if (r, c) not in taken
    ):
        cells.append(_cell(len(cells), [position[0]], [position[1]], "", False))
    # Counted from the parsed rows, not the padded grid: a short header row gets filler
    # cells, and those are not headers.
    header_rows = 0
    for row in parser.rows[: len(keep_rows)]:
        if not row or not all(cell[3] for cell in row):
            break
        header_rows += 1
    return {
        "role": "table",
        "header_row_count": header_rows,
        "row_count": row_count,
        "column_count": column_count,
        "cells": cells,
    }


def _cell(
    index: int, row_nums: list[int], column_nums: list[int], text: str, header: bool
) -> dict[str, Any]:
    return {
        "id": f"falcon-cell-{index + 1}",
        "row_nums": row_nums,
        "column_nums": column_nums,
        "text": text,
        "resolution": "resolved",
        "column_header": header,
    }


def _table_text(structure: dict[str, Any]) -> str:
    grid = [
        ["" for _ in range(structure["column_count"])]
        for _ in range(structure["row_count"])
    ]
    for cell in structure["cells"]:
        grid[cell["row_nums"][0]][cell["column_nums"][0]] = cell["text"].replace(
            "\n", " "
        )
    return "\n".join("\t".join(row).rstrip() for row in grid).strip()


def _span(value: str | None) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return 1


def _plain_text(text: str, kind: str) -> tuple[str, str]:
    """Strip the model's markdown markers, promoting a marked heading to its kind.

    Falcon writes markdown even though the official prompts only ask for content, so a
    title arrives as "# Title". `TextRegion.text` is plain text and `kind` carries the
    structure, so leaving the marker in place double-encodes it: the markdown lane adds
    its own prefix ("### # Title") and the text lane shows the bare marker to a reviewer.
    Table and formula markup is the region's content, so it passes through untouched.
    """
    if kind in {"table", "formula"}:
        return text, kind
    level = 0
    lines = []
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            level = level or len(heading.group(1))
            line = heading.group(2)
        lines.append(EMPHASIS.sub(r"\1", line).rstrip())
    plain = "\n".join(lines).strip()
    if level and kind == "text":
        kind = "title" if level == 1 else "section-header"
    return plain, kind


def _read_terminated(element: dict[str, Any]) -> str | None:
    """Why a read's text is evidence rather than a claim, if it is at all."""
    if element.get("truncated"):
        return "spent its whole token budget"
    if element.get("looped"):
        return "repetition loop the retry ladder did not recover"
    return None


def _bounding_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    left, top, right, bottom = (int(round(float(part))) for part in value)
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)
