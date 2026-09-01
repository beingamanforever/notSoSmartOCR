"""Exact cell match on the reviewed table failure panel.

Regenerate from the repository root:
    MPLCONFIGDIR=/private/tmp/notso-ocr-mpl python artifacts/table-cell-comparison.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from orx_figstyle import MUTED, PALETTE, WIDE, figure, save, use_style

DATA = Path(__file__).with_name("table-cell-comparison.csv")
OUTPUT = Path(__file__).with_name("table-cell-comparison")


def load_rows() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(
        rows,
        key=lambda row: int(row["correct_cells"]) / int(row["total_cells"]),
    )


def main() -> None:
    use_style()
    rows = load_rows()
    names = [row["method"] for row in rows]
    correct = [int(row["correct_cells"]) for row in rows]
    totals = [int(row["total_cells"]) for row in rows]
    scores = [100 * value / total for value, total in zip(correct, totals)]
    colors = [
        PALETTE["cyan"]
        if row["method"] == "Tesseract fusion"
        else PALETTE["blue"]
        if row["family"] == "fusion"
        else MUTED
        for row in rows
    ]

    fig, axes = figure(width=WIDE, ratio=0.43)
    bars = axes.barh(
        names,
        scores,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        height=0.68,
    )
    axes.set_xlim(0, 115)
    axes.set_xticks(range(0, 101, 20))
    axes.set_xlabel("Exact cell match (%)")
    axes.grid(axis="x")
    axes.grid(axis="y", visible=False)
    for bar, count, total, score in zip(bars, correct, totals, scores):
        axes.annotate(
            f"{count}/{total} ({score:.1f}%)",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
