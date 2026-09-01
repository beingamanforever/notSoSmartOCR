"""Route outcomes on the twice-reviewed 41-page hard panel.

Regenerate from the repository root:
    MPLCONFIGDIR=/private/tmp/notso-ocr-mpl python artifacts/hard-case-routing.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from orx_figstyle import MUTED, PALETTE, WIDE, figure, save, use_style

DATA = Path(__file__).with_name("hard-case-routing.csv")
OUTPUT = Path(__file__).with_name("hard-case-routing")


def load_rows() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    use_style()
    rows = load_rows()
    names = [row["pipeline"] for row in rows]
    accepts = [int(row["accept_local"]) for row in rows]
    reviews = [int(row["review"]) for row in rows]

    fig, axes = figure(width=WIDE, ratio=0.34)
    axes.barh(
        names,
        accepts,
        color=PALETTE["orange"],
        edgecolor="black",
        linewidth=0.45,
        height=0.56,
        label="Accepted locally",
    )
    axes.barh(
        names,
        reviews,
        left=accepts,
        color=PALETTE["blue"],
        edgecolor="black",
        linewidth=0.45,
        height=0.56,
        label="Routed to review",
    )
    axes.set_xlim(0, 47)
    axes.set_xticks(range(0, 42, 10))
    axes.set_xlabel("Pages in fixed hard panel")
    axes.grid(axis="x")
    axes.grid(axis="y", visible=False)
    axes.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.5), ncols=2)
    for index, row in enumerate(rows):
        success = int(row["operational_success"])
        complete = int(row["manual_complete"])
        total = int(row["total"])
        axes.annotate(
            f"{success}/{total} operational; {complete}/{total} complete",
            xy=(total, index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
