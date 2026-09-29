"""Docling's orphan clusters for scanned pages: detected text no layout region holds.

Docling's layout postprocessor turns every text cell no cluster took into its own text
cluster (`create_orphan_clusters`), so nothing on a born-digital page is dropped. A scan
has no cells, so words from a text detector stand in for them: handwriting beside a
form's print and notes beside a ruled table were otherwise never read.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median
from typing import Any


Detection = dict[str, Any]
# Containers group other regions and are never decoded, so they read none of their words.
CONTAINER_LABELS = frozenset({"form", "key_value_region"})
# Docling's `_assign_cells_to_clusters` min_overlap (intersection over the cell).
MIN_CELL_SHARE = 0.2


def grown_tables(
    detections: Sequence[Detection], words: Sequence[Sequence[float]]
) -> list[Detection]:
    """Each table grown to the detected words it holds, as Docling regrows a table.

    A word belongs to the table holding the largest share of it, above 0.2, and a table's
    box becomes the union with its words (`_assign_cells_to_clusters`,
    `_adjust_cluster_bboxes`). A roster's margin notes straddle Heron's table edge: the
    table crop cut them off, so they were read as loose lines under the table.
    """
    grown = [dict(item) for item in detections]
    tables = [item for item in grown if item["label"] == "table"]
    detected = [item["box"] for item in tables]  # words are assigned before regrowth
    for word in words:
        share, table = max(
            ((_share(word, box), table) for box, table in zip(detected, tables)),
            key=lambda pair: pair[0],
            default=(0.0, None),
        )
        if table is not None and share > MIN_CELL_SHARE:
            table["box"] = tuple(
                f(a, b) for f, a, b in zip((min, min, max, max), table["box"], word[:4])
            )
    return grown


def _share(word: Sequence[float], box: Sequence[float]) -> float:
    width = max(0.0, min(word[2], box[2]) - max(word[0], box[0]))
    height = max(0.0, min(word[3], box[3]) - max(word[1], box[1]))
    area = (word[2] - word[0]) * (word[3] - word[1])
    return width * height / area if area > 0 else 0.0


def orphan_text_detections(
    detections: Sequence[Detection],
    words: Sequence[Sequence[float]],
) -> list[Detection]:
    """Text detections for lines of detected words whose centres no region covers.

    `words` are page-pixel boxes (left, top, right, bottom, score). A table covers its
    whole box, grown to its words (`grown_tables`).
    """
    covers = [item["box"] for item in detections if item["label"] not in CONTAINER_LABELS]
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
