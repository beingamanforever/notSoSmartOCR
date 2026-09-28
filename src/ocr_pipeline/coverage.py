"""Docling's orphan clusters for scanned pages: detected text no layout region holds.

A ruled table covers only its grid: its read transcribes the grid, and the rulings are
found by morphological opening with long line kernels (the lattice step of Camelot and
pdfplumber).

Docling's layout postprocessor turns every text cell no cluster took into its own text
cluster (`create_orphan_clusters`), so nothing on a born-digital page is dropped. A scan
has no cells, so words from a text detector stand in for them: handwriting beside a
form's print and notes beside a ruled table were otherwise never read.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median
from typing import Any

from PIL import Image

Detection = dict[str, Any]
# Containers group other regions and are never decoded, so they read none of their words.
CONTAINER_LABELS = frozenset({"form", "key_value_region"})
# A ruling spans at least half of the table crop, as Camelot's line scale assumes.
RULING_SHARE = 0.5


def orphan_text_detections(
    detections: Sequence[Detection],
    words: Sequence[Sequence[float]],
    image: Image.Image,
) -> list[Detection]:
    """Text detections for lines of detected words whose centres no region covers.

    `words` are page-pixel boxes (left, top, right, bottom, score). A ruled table covers
    only its grid, as its read transcribes only the grid.
    """
    covers = [
        (ruled_extent(image, item["box"]) or item["box"])
        if item["label"] == "table"
        else item["box"]
        for item in detections
        if item["label"] not in CONTAINER_LABELS
    ]
    orphans = [
        word
        for word in words
        if not any(_holds(cover, word) for cover in covers)
    ]
    return [
        {
            "label": "text",
            "score": sum(word[4] for word in line) / len(line),
            "box": (
                min(word[0] for word in line),
                min(word[1] for word in line),
                max(word[2] for word in line),
                max(word[3] for word in line),
            ),
            "orphan_text": True,
        }
        for line in _lines(orphans)
    ]


def _holds(box: Sequence[float], word: Sequence[float]) -> bool:
    x, y = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _lines(words: Sequence[Sequence[float]]) -> list[list[Sequence[float]]]:
    """Group words into lines.

    Rows follow docTR's DocumentBuilder: a word joins a row when its y-centre is within
    half the median word height of the row's mean y-centre. A row then splits where the
    gap exceeds the median word height (docTR adapts this distance to page spacing).
    """
    if not words:
        return []
    height = median(word[3] - word[1] for word in words)
    rows: list[list[Sequence[float]]] = []
    centres: list[float] = []  # running sum of each row's word centres
    for word in sorted(words, key=lambda word: ((word[1] + word[3]) / 2, word[0])):
        centre = (word[1] + word[3]) / 2
        if rows and abs(centre - centres[-1] / len(rows[-1])) < height / 2:
            rows[-1].append(word)
            centres[-1] += centre
        else:
            rows.append([word])
            centres.append(centre)
    lines = []
    for row in rows:
        row.sort(key=lambda word: word[0])
        line = [row[0]]
        for word in row[1:]:
            if word[0] - line[-1][2] > height:
                lines.append(line)
                line = []
            line.append(word)
        lines.append(line)
    return lines


def ruled_extent(
    image: Image.Image, box: Sequence[float]
) -> tuple[float, float, float, float] | None:
    """Page box between the outermost rulings, or None when the table is not ruled."""
    import cv2
    import numpy as np

    # Clamped: PIL pads outside the image with black, which Otsu would call rulings.
    left, top = max(0, int(box[0])), max(0, int(box[1]))
    right, bottom = min(image.width, int(box[2])), min(image.height, int(box[3]))
    crop = np.asarray(image.convert("L").crop((left, top, right, bottom)))
    if crop.size == 0:
        return None
    ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rows = _rulings(ink, 0)
    columns = _rulings(ink, 1)
    if len(rows) < 3 or len(columns) < 2:
        return None
    return (left + columns[0], top + rows[0], left + columns[-1], top + rows[-1])


def _rulings(ink, axis: int) -> list[int]:
    """Positions of line runs covering RULING_SHARE of the crop along one axis."""
    import cv2

    length = max(1, int(ink.shape[1 - axis] * RULING_SHARE))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (length, 1) if axis == 0 else (1, length)
    )
    profile = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel).sum(axis=1 - axis) > 0
    positions, start = [], None
    for index, on in enumerate([*profile.tolist(), False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            positions.append((start + index) // 2)
            start = None
    return positions
