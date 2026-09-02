"""Five-page OCR error comparison. Regenerate: python hard_panel_comparison.py"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from evidence import ROOT, decimal, table_rows
from orx_figstyle import WIDE, figure, save, use_style

SOURCE = ROOT / "README.md"
OUTPUT = Path(__file__).with_name("hard_panel_comparison")
METHODS = ("Frozen baseline", "Final safe route")
ROWS = ("Frozen old baseline", "Final safe wide-band v5")
METRICS = ("CER", "WER", "Hallucination", "Missed character")
COLORS = ("#B9C0C8", "#15324B")


def load() -> dict[str, list[float]]:
    rows = table_rows(SOURCE, "Five-page configuration")
    return {
        method: [decimal(value) for value in rows[row][:4]]
        for method, row in zip(METHODS, ROWS, strict=True)
    }


def main() -> None:
    use_style()
    values = load()
    fig, ax = figure(width=WIDE, ratio=0.42)
    x = np.arange(len(METRICS))
    width = 0.36

    for index, method in enumerate(METHODS):
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            values[method],
            width,
            label=method,
            color=COLORS[index],
            edgecolor="#15324B",
            linewidth=0.55,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=6, padding=2)

    highest = max(value for row in values.values() for value in row)
    upper = math.ceil((highest + 0.08) * 10) / 10
    ax.set_ylim(0, upper)
    ax.set_ylabel("Error rate")
    ax.set_xlabel("OCR metric")
    ax.set_xticks(x, METRICS)
    ax.tick_params(axis="x", length=0)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=1,
        edgecolor="#667085",
        fontsize=7,
    )
    ax.text(
        0.01,
        0.96,
        "One run per configuration; no confidence intervals",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#667085",
    )
    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
