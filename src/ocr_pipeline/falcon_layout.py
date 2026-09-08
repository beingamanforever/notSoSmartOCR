"""Read learned layout regions with the official Falcon OCR decoder.

Decoder scores and layout detection scores have separate provenance. A region box
does not imply word locations, and token likelihood is not calibrated correctness.
"""

from __future__ import annotations

import base64
import json
import io
import math
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import BoundingBox, TextRegion
from .providers import ReaderError
from .falcon import FALCON_SERVICE_MAX_RESPONSE_BYTES

PROVIDER = "falcon-perception"
# Match the presentation HTML span limit and bound expanded grids before allocation.
MAX_TABLE_SPAN = 100
MAX_TABLE_GRID_CELLS = 10_000
HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
# Only paired asterisks: a lone one is a real footnote mark on many clinical forms.
EMPHASIS = re.compile(r"\*\*(.+?)\*\*")
# Markup truncated mid-tag ("</sup" with no ">") is handed back as text, not a tag.
# Layout categories that carry image pixels rather than text.
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
    "algorithm": "code",
    "code": "code",
}


class FalconLayoutReader:
    """Page reader backed by the Falcon-Perception `ocr_layout` engine."""

    name = "falcon-perception-layout"

    def __init__(self, url: str, *, timeout_seconds: float = 600) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Falcon layout service must be reached on loopback")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._model: dict[str, Any] = {}

    def check_health(self) -> None:
        body = self._request(f"{self.url}/health", None)
        self._model = dict(body.get("model") or {})

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._model)

    @property
    def generation(self) -> dict[str, Any]:
        return {"engine": "falcon-perception", "termination_observable": True}

    def transcribe_crops(self, images, categories) -> list[str]:
        from .falcon import FALCON_OCR_CATEGORIES

        if not images or len(images) != len(categories) or len(images) > 24:
            raise ValueError("Expected 1 to 24 crops with matching categories")
        if any(category not in FALCON_OCR_CATEGORIES for category in categories):
            raise ValueError("Unsupported crop category")
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        body = self._request(
            f"{self.url}/generate", {"images": encoded, "categories": list(categories)}
        )
        texts = body.get("texts")
        if (
            not isinstance(texts, list)
            or len(texts) != len(images)
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ReaderError("falcon_crop_failed", "malformed crop payload")
        return texts

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
            if not text and category not in EMPTY_CATEGORIES | {
                "checkbox-selected",
                "checkbox-unselected",
            }:
                continue
            box = _bounding_box(element.get("bbox"))
            if box is None:
                continue
            detection = {
                key: element[key]
                for key in ("query_id", "class_scores")
                if key in element
            }
            terminated = _read_terminated(element)
            attempts = element.get("read_attempts") or []
            read_history = {
                **({"read_attempts": attempts} if attempts else {}),
                **({"read_terminated": terminated} if terminated else {}),
                **(
                    {"read_retries": int(element["retries"])}
                    if element.get("retries") is not None
                    else {}
                ),
            }
            # A completed retry is an alternative, not independent confirmation.
            read_resolution = (
                "unreadable"
                if terminated
                else "conflicting"
                if attempts
                else "resolved"
            )
            if category in {"checkbox-selected", "checkbox-unselected"}:
                selected = category == "checkbox-selected"
                regions.append(
                    TextRegion(
                        id=f"p{page_number}-falcon-{index}",
                        kind="checkbox",
                        text=text,
                        confidence=(element.get("generation") or {}).get("token_score"),
                        bounding_box=box,
                        reading_order=index,
                        provider=PROVIDER,
                        text_provenance={
                            "method": "learned_checkbox_detection",
                            "reading_order_method": "detector_sequence",
                            "model": model,
                            "layout_detection": detection,
                            "generation": element.get("generation") or {},
                            "raw_text": text,
                            **read_history,
                            "confidence_meaning": "decoder token likelihood, uncalibrated",
                        },
                        resolution=read_resolution,
                        structure={
                            "state": "selected" if selected else "unselected",
                            "state_confidence": float(element["score"]),
                            "label_evidence_ids": [],
                            "label": text,
                        },
                    )
                )
                continue
            if category in EMPTY_CATEGORIES:
                with Image.open(image_path) as source:
                    box = BoundingBox(
                        max(0, box.left),
                        max(0, box.top),
                        min(source.width, box.right),
                        min(source.height, box.bottom),
                    )
                    if box.right <= box.left or box.bottom <= box.top:
                        continue
                    with source.crop(
                        (box.left, box.top, box.right, box.bottom)
                    ) as crop:
                        buffer = io.BytesIO()
                        crop.save(buffer, format="PNG")
                        asset = {
                            "media_type": "image/png",
                            "width": crop.width,
                            "height": crop.height,
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        }
                regions.append(
                    TextRegion(
                        id=f"p{page_number}-falcon-{index}",
                        kind="figure",
                        text="",
                        confidence=None,
                        bounding_box=box,
                        reading_order=index,
                        provider=PROVIDER,
                        text_provenance={
                            "method": "source_image_crop",
                            "reading_order_method": "detector_sequence",
                            "layout_detection": detection,
                            "layout_category": category,
                            "layout_detection_score": element.get("score"),
                            "model": model,
                        },
                        structure={"image": asset},
                    )
                )
                continue
            text, kind = _plain_text(text, KIND_BY_CATEGORY.get(category, "text"))
            if not text:
                continue
            structure = None
            structure_error = None
            if kind == "table":
                try:
                    structure = _table_structure(text, element.get("generation"))
                except ValueError as error:
                    structure_error = str(error)
            if structure:
                text = _table_text(structure)
            regions.append(
                TextRegion(
                    id=f"p{page_number}-falcon-{index}",
                    kind=kind,
                    text=text,
                    confidence=(element.get("generation") or {}).get("token_score"),
                    bounding_box=box,
                    reading_order=index,
                    provider=PROVIDER,
                    text_provenance={
                        "method": "falcon_perception_layout_ocr",
                        "reading_order_method": "detector_sequence",
                        "layout_detection": detection,
                        "generation": element.get("generation") or {},
                        "raw_text": element.get("text") or "",
                        **read_history,
                        **(
                            {"table_structure_error": structure_error}
                            if structure_error
                            else {}
                        ),
                        "layout_category": category,
                        # Named so nobody reads this as a recognition confidence.
                        "layout_detection_score": element.get("score"),
                        "confidence_meaning": "decoder token likelihood, uncalibrated"
                        if (element.get("generation") or {}).get("token_score")
                        is not None
                        else "recognition unavailable",
                        "model": model,
                    },
                    resolution="unreadable" if structure_error else read_resolution,
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
        _assign_image_owners(regions)
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
                body = reply.read(FALCON_SERVICE_MAX_RESPONSE_BYTES + 1)
            if len(body) > FALCON_SERVICE_MAX_RESPONSE_BYTES:
                raise ValueError("Falcon response exceeded its size limit")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("Falcon response must be an object")
            return result
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            raise ReaderError("falcon_layout_failed", str(error)) from error


def _assign_image_owners(regions: list[TextRegion]) -> None:
    # These assets are crops of the same source page: exact containment preserves
    # every child pixel without merging boxes or suppressing text detections.
    images = [
        region
        for region in regions
        if region.kind == "figure"
        and not region.text
        and (region.text_provenance or {}).get("method") == "source_image_crop"
        and (region.structure or {}).get("image")
    ]
    images.sort(
        key=lambda region: (
            -(
                (region.bounding_box.right - region.bounding_box.left)
                * (region.bounding_box.bottom - region.bounding_box.top)
            )
        )
    )
    parents: list[TextRegion] = []
    for region in images:
        box = region.bounding_box
        parent = next(
            (
                item
                for item in parents
                if item.bounding_box.left <= box.left
                and item.bounding_box.top <= box.top
                and item.bounding_box.right >= box.right
                and item.bounding_box.bottom >= box.bottom
            ),
            None,
        )
        if parent is None:
            parents.append(region)
            continue
        region.structure.update(role="figure_source", parent_region_id=parent.id)
        children = parent.structure.setdefault("child_evidence_ids", [])
        if region.id not in children:
            children.append(region.id)


class _TableParser(HTMLParser):
    """Collect the model's HTML table as rows of (text, rowspan, colspan, header)."""

    def __init__(self, markup: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.line_offsets = [
            0,
            *(index + 1 for index, char in enumerate(markup) if char == "\n"),
        ]
        self.ranges: list[tuple[int, int]] = []
        self._cell_start = 0
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
            self._cell_start = self._offset() + len(self.get_starttag_text())
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
        text = "".join(self._cell).strip()
        if len(self.ranges) >= MAX_TABLE_GRID_CELLS:
            raise ValueError("Table exceeds the 10000-cell resource limit")
        self._row.append((text, *self._span, self._header))
        self.ranges.append((self._cell_start, self._offset()))
        self._cell = None

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def _close_row(self) -> None:
        self._close_cell()
        if self._row:
            self.rows.append(self._row)
        self._row = None


def _table_structure(
    markup: str, generation: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Place the model's HTML cells on the grid the rest of the pipeline renders.

    Without this the table's HTML source is the region's text, so the plain-text lane
    prints `<table><thead><tr><th>` at a reviewer instead of a table.
    """
    parser = _TableParser(markup)
    parser.feed(markup)
    parser.close()
    if not parser.rows:
        return None

    occupied: dict[tuple[int, int], bool] = {}
    placed: list[tuple[list[int], list[int], str, bool]] = []
    column_count = 0
    row_count = 0
    for row_index, row in enumerate(parser.rows):
        column = 0
        for text, rowspan, colspan, header in row:
            while (row_index, column) in occupied:
                column += 1
            row_count = max(row_count, row_index + rowspan)
            column_count = max(column_count, column + colspan)
            if row_count * column_count > MAX_TABLE_GRID_CELLS:
                raise ValueError("Table exceeds the 10000-cell resource limit")
            row_nums = list(range(row_index, row_index + rowspan))
            column_nums = list(range(column, column + colspan))
            for position in ((r, c) for r in row_nums for c in column_nums):
                occupied[position] = header
            placed.append((row_nums, column_nums, text, header))
            column += colspan
    if not occupied:
        return None

    cells: list[dict[str, Any]] = []
    taken: set[tuple[int, int]] = set()
    # Empty cells are positions in the source table, not permission to shift values.
    for rows, columns, text, header in placed:
        cell = _cell(len(cells), rows, columns, text, header)
        start, end = parser.ranges[len(cells)]
        scores = [
            token["score"]
            for token in (generation or {}).get("tokens", [])
            if token["start"] < end and token["end"] > start
        ]
        if scores and text:
            cell["confidence"] = math.exp(
                sum(math.log(max(score, 1e-30)) for score in scores) / len(scores)
            )
            cell["confidence_kind"] = "Decoder token score"
            cell["source"] = PROVIDER
        cells.append(cell)
        taken.update((r, c) for r in rows for c in columns)
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
    for row in parser.rows:
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
        span = max(1, int(str(value)))
    except (TypeError, ValueError):
        if value is not None and value.strip().lstrip("+").isdecimal():
            raise ValueError("Table span exceeds the 100-span resource limit") from None
        return 1
    if span > MAX_TABLE_SPAN:
        raise ValueError("Table span exceeds the 100-span resource limit")
    return span


def _plain_text(text: str, kind: str) -> tuple[str, str]:
    """Strip the model's markdown markers, promoting a marked heading to its kind.

    Falcon writes markdown even though the official prompts only ask for content, so a
    title arrives as "# Title". `TextRegion.text` is plain text and `kind` carries the
    structure, so leaving the marker in place double-encodes it: the markdown lane adds
    its own prefix ("### # Title") and the text lane shows the bare marker to a reviewer.
    Table and formula markup is the region's content, so it passes through untouched.
    """
    if re.fullmatch(r"(`{3,}|~{3,})[^\n]*\n[\s\S]*\n\1\s*", text):
        return text, "code"
    if kind in {"table", "formula", "code"}:
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
