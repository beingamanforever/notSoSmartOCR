"""Split a detected text region that encloses several framed panels; re-read each panel.

The layout detector sometimes draws one box around separately framed panels (a DEA card's
registration, certificate, "report changes" and instructions panels). Falcon then reads
them as one block: fields drop out, the panels' order is lost, and the whole block looks
like one doubtful region. Panels are found by cutting the region's ink at blank rows, then
blank columns; the cut is kept only when every piece carries its own drawn frame, so a
paragraph (blank lines between text lines, no frame) is never split. Falcon then reads each
panel as one crop. Its field grids come back as tables and the text after them (a name and
address) stays with the last grid, as on any Falcon table read.

Falcon's reading of a crop depends on its framing: on the pipeline's render of the DEA card,
the tight registration panel read stopped after its first field grid, while the same panel
with a white margin read both grids and the address. A batch of the four tight panels also
cut the first one short; with margins, one batch read every word in half the time of four
requests (Falcon runs requests one at a time). So the panel reads are kept only when they
hold every word of the merged reading; otherwise they are read again, one request per
panel, then without the margin, and if words are still missing the merged region stays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .contracts import BoundingBox, TextRegion
from .falcon_layout import (
    CALIBRATED_CONFIDENCE,
    TABLE_START,
    UNCALIBRATED_CONFIDENCE,
    _table_parts,
    _table_structure,
    _table_text,
    _unread,
    calibrated_confidence,
)

LOGGER = logging.getLogger(__name__)
PANEL_MIN_PAGE_SHARE = 0.15  # only regions this large can hide several panels
PANEL_MIN_GAP = 8  # blank pixels between two frames at 300 dpi (DEA card: 10 px to its cut line)
RULE_WIDTH = 6  # a long piece thinner than any readable text line is a separator rule
FRAME_SHARE = 0.8  # share of each side of a panel that its frame line covers
FRAME_EDGE = 8  # pixels from a side where its frame line may run (allows slight skew)
FRAME_MIN_SIDE = 60  # a framed panel is taller and wider than one text line
INK_THRESHOLD = 160
SPLIT_KINDS = {"text", "list-item", "paragraph"}
# (white margin in pixels, all panels in one request): tried in order until no word is lost.
PANEL_READS = ((40, True), (40, False), (0, False))


class PanelSplitStage:
    name = "panel-split"

    def __init__(self, reader: Any) -> None:
        self.reader = reader  # the Falcon layout reader: read_crops(images, category)

    def apply(self, image_path: Path, page_number: int, regions: list[TextRegion]) -> list[TextRegion]:
        with Image.open(image_path) as source:
            page = source.convert("RGB")
        try:
            page_area = page.width * page.height
            result: list[TextRegion] = []
            for region in regions:
                panels = _panels_of(region, page, page_area)
                if len(panels) < 2:
                    result.append(region)
                    continue
                reads = self._complete_reads(region, page, panels)
                result.extend(_panel_regions(region, panels, reads) if reads else [region])
            return result
        finally:
            page.close()

    def _complete_reads(self, region: TextRegion, page: Image.Image, panels) -> list[dict[str, Any]] | None:
        """Panel reads that keep every word of the merged reading, or None (logged)."""
        for margin, batched in PANEL_READS:
            crops = [ImageOps.expand(page.crop(box), margin, "white") for box in panels]
            try:
                if batched:
                    reads = self.reader.read_crops(crops, "plain")
                else:
                    reads = [self.reader.read_crops([crop], "plain")[0] for crop in crops]
            except Exception:  # the merged reading stays; the failure is logged, not hidden
                LOGGER.exception("Panel re-read failed for %s", region.id)
                return None
            joined = "\n".join(read["text"] for read in reads)
            missing = [word for word in region.text.split() if _unread(word, joined)]
            if not missing and all(read["text"].strip() for read in reads):
                return reads
        LOGGER.warning("Kept %s whole: its panel reads miss %s", region.id, missing[:8])
        return None


def _panels_of(region: TextRegion, page: Image.Image, page_area: int) -> list[tuple[int, int, int, int]]:
    box = region.bounding_box
    area = (box.right - box.left) * (box.bottom - box.top)
    learned = bool((region.text_provenance or {}).get("layout_category"))
    if region.kind not in SPLIT_KINDS or not learned:
        return []
    if area <= PANEL_MIN_PAGE_SHARE * page_area:
        return []
    crop = page.crop((box.left, box.top, box.right, box.bottom)).convert("L")
    ink = np.asarray(crop) < INK_THRESHOLD
    crop.close()
    return [
        (box.left + left, box.top + top, box.left + right, box.top + bottom)
        for left, top, right, bottom in framed_panels(ink)
    ]


def framed_panels(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes (left, top, right, bottom) of framed panels in an ink mask, or [] when the ink
    does not split into two or more framed pieces. A framed piece holding framed pieces of
    its own (a card's outer border around two panels) gives its inner panels."""
    parts = [
        (left, top, right, bottom)
        for top, bottom in _runs(ink.any(axis=1))
        for left, right in _runs(ink[top:bottom].any(axis=0))
    ]
    parts = [part for part in parts if not _rule(part)]
    if not all(_framed(ink[top:bottom, left:right]) for left, top, right, bottom in parts):
        return []
    panels = []
    for left, top, right, bottom in parts:
        dx, dy, inside = _inside_frame(ink[top:bottom, left:right])
        nested = framed_panels(inside)
        panels.extend(
            [(left + dx + a, top + dy + b, left + dx + c, top + dy + d) for a, b, c, d in nested]
            or [(left, top, right, bottom)]
        )
    return panels if len(panels) >= 2 else []


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inked spans separated by at least PANEL_MIN_GAP blank pixels."""
    spans: list[tuple[int, int]] = []
    start = None
    blank = 0
    for index, inked in enumerate(mask):
        if inked:
            if start is None:
                start = index
            blank = 0
            end = index + 1
        elif start is not None:
            blank += 1
            if blank >= PANEL_MIN_GAP:
                spans.append((start, end))
                start = None
    if start is not None:
        spans.append((start, end))
    return spans


def _rule(box: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    thin, long = sorted((right - left, bottom - top))
    return thin <= RULE_WIDTH and long >= FRAME_MIN_SIDE


def _framed(piece: np.ndarray) -> bool:
    """A drawn frame: ink along nearly all of each of the piece's four sides."""
    height, width = piece.shape
    if min(height, width) < FRAME_MIN_SIDE:
        return False
    sides = (
        piece[:, :FRAME_EDGE].any(axis=1),
        piece[:, -FRAME_EDGE:].any(axis=1),
        piece[:FRAME_EDGE].any(axis=0),
        piece[-FRAME_EDGE:].any(axis=0),
    )
    return all(side.mean() >= FRAME_SHARE for side in sides)


def _inside_frame(piece: np.ndarray) -> tuple[int, int, np.ndarray]:
    """The piece without its frame lines: drop edge rows and columns that are mostly ink."""
    top, bottom = 0, piece.shape[0]
    while top < bottom and piece[top].mean() >= 0.5:
        top += 1
    while bottom > top and piece[bottom - 1].mean() >= 0.5:
        bottom -= 1
    rows = piece[top:bottom]
    left, right = 0, piece.shape[1]
    while left < right and rows[:, left].mean() >= 0.5:
        left += 1
    while right > left and rows[:, right - 1].mean() >= 0.5:
        right -= 1
    return left, top, rows[:, left:right]


def _panel_regions(
    region: TextRegion, panels: list[tuple[int, int, int, int]], reads: list[dict[str, Any]]
) -> list[TextRegion]:
    """One region per panel, or one per field grid when the panel holds several grids."""
    parent = region.text_provenance or {}
    layout_score = parent.get("layout_detection_score")
    split = []
    for panel_index, (box, read) in enumerate(zip(panels, reads, strict=True), start=1):
        category = "table" if TABLE_START.search(read["text"]) else "text"
        parts = _table_parts({"category": category, **read})
        for part_index, part in enumerate(parts, start=1):
            text, structure, kind = part["text"].strip(), None, category
            provenance = {
                "method": "falcon_perception_layout_ocr",
                "reading_order_method": "framed_panels",
                "generation": part["generation"],
                "raw_text": part["text"],
                "layout_category": category,
                "layout_detection_score": layout_score,
                "model": parent.get("model"),
                "split_from": region.id,
                "split_method": "framed_panels",
            }
            if category == "table":
                try:
                    structure = _table_structure(text, part["generation"])
                    text = _table_text(structure)
                except ValueError as error:  # the model's markup stays as text, never dropped
                    kind, provenance["table_structure_error"] = "text", str(error)
            token_score = part["generation"].get("token_score")
            calibrated = kind == "text" and isinstance(token_score, int | float) and isinstance(layout_score, int | float)
            provenance["confidence_meaning"] = CALIBRATED_CONFIDENCE if calibrated else UNCALIBRATED_CONFIDENCE
            split.append(
                TextRegion(
                    id=f"{region.id}-{panel_index}" + (f"-{part_index}" if len(parts) > 1 else ""),
                    kind=kind,
                    text=text,
                    confidence=calibrated_confidence(token_score, layout_score) if calibrated else token_score,
                    bounding_box=BoundingBox(*box),
                    reading_order=region.reading_order,
                    provider=region.provider,
                    text_provenance=provenance,
                    structure=structure,
                )
            )
    return split
