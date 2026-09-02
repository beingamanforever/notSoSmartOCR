"""C14 crop comparison. Regenerate: python handwriting_adapter_comparison.py"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from evidence import ROOT, decimal, fraction, table_rows
from orx_figstyle import WIDE, figure, save, use_style

SOURCE = ROOT / "artifacts" / "research" / "phi4_finetuning_cost.md"
OUTPUT = Path(__file__).with_name("handwriting_adapter_comparison")
METHODS = ("Stock Phi-4", "v4 adapter")
METRICS = (
    ("Exact", fraction),
    ("CER", decimal),
    ("Hallucinated-character rate", decimal),
    ("Missed-character rate", decimal),
)
COLORS = ("#B9C0C8", "#D28A00")


def load() -> dict[str, list[float]]:
    rows = table_rows(SOURCE, "Metric", occurrence=2)
    columns: dict[str, list[float]] = {method: [] for method in METHODS}
    for metric, parser in METRICS:
        stock, adapter = rows[metric][:2]
        columns[METHODS[0]].append(parser(stock))
        columns[METHODS[1]].append(parser(adapter))
    return columns


def main() -> None:
    use_style()
    values = load()
    labels = (
        "Exact match\n(higher is better)",
        "CER",
        "Hallucinated\ncharacter",
        "Missed\ncharacter",
    )
    fig, ax = figure(width=WIDE, ratio=0.42)
    x = np.arange(len(labels))
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
    ax.set_ylabel("Rate")
    ax.set_xlabel("Fixed C14 crop metric")
    ax.set_xticks(x, labels)
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
        "88 paired inferences per arm; one run; no confidence intervals",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#667085",
    )
    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
