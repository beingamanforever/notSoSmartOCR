"""Specialist transfer limits. Regenerate: python specialist_limits.py"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evidence import ROOT, decimal, fraction, table_rows
from orx_figstyle import WIDE, save, use_style

CONTROL_SOURCE = ROOT / "experiments" / "CHECKBOX_SPECIALIST_EVALUATION.md"
BROAD_SOURCE = Path(__file__).with_name("specialist_limits.csv")
OUTPUT = Path(__file__).with_name("specialist_limits")
COLORS = ("#15324B", "#A8B0B9", "#D9DDE2")


def load() -> tuple[list[float], list[str], list[float], list[str]]:
    clear = table_rows(CONTROL_SOURCE, "Metric")
    with BROAD_SOURCE.open(encoding="utf-8", newline="") as source:
        broad = {row["metric"]: row for row in csv.DictReader(source)}

    clear_coverage = fraction(clear["Detection coverage"][1].replace(" controls", ""))

    control_values = [
        clear_coverage,
        float(broad["broad_control_safe_match"]["value"]),
    ]
    control_labels = [
        clear["Detection coverage"][1].replace(" controls", ""),
        broad["broad_control_safe_match"]["denominator"].replace(" annotations", ""),
    ]
    table_values = [
        float(broad["table_presence_f1"]["value"]),
        float(broad["row_count_accuracy"]["value"]),
        float(broad["column_count_accuracy"]["value"]),
    ]
    table_labels = [
        broad["table_presence_f1"]["denominator"],
        broad["row_count_accuracy"]["denominator"],
        broad["column_count_accuracy"]["denominator"],
    ]
    if decimal(clear["State macro-F1"][0]) != 1.0:
        raise ValueError("Clear-control evidence changed")
    return control_values, control_labels, table_values, table_labels


def add_labels(ax, bars, rates: list[float], denominators: list[str]) -> None:
    for bar, rate, denominator in zip(bars, rates, denominators, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 0.025, 0.07),
            f"{rate:.3f}\n{denominator}",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def main() -> None:
    use_style()
    control_values, control_denoms, table_values, table_denoms = load()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDE, WIDE * 0.43),
        gridspec_kw={"width_ratios": (0.9, 1.35)},
        layout="constrained",
    )

    control_x = np.arange(2)
    control_bars = axes[0].bar(
        control_x,
        control_values,
        color=COLORS[:2],
        edgecolor="#15324B",
        linewidth=0.55,
        width=0.58,
    )
    add_labels(axes[0], control_bars, control_values, control_denoms)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Coverage rate")
    axes[0].set_xlabel("Control protocol")
    axes[0].set_xticks(
        control_x,
        ("Clear detection\n2-page slice", "Broad safe match\n169-page panel"),
    )
    axes[0].tick_params(axis="x", length=0)
    axes[0].text(
        0.99,
        0.98,
        "(a) Controls",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#344054",
        weight="bold",
    )

    table_x = np.arange(3)
    table_bars = axes[1].bar(
        table_x,
        table_values,
        color=COLORS,
        edgecolor="#15324B",
        linewidth=0.55,
        width=0.62,
    )
    add_labels(axes[1], table_bars, table_values, table_denoms)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("F1 or accuracy")
    axes[1].set_xlabel("Broad 169-page table protocol")
    axes[1].set_xticks(
        table_x,
        ("Presence F1", "Row-count\naccuracy", "Column-count\naccuracy"),
    )
    axes[1].tick_params(axis="x", length=0)
    axes[1].text(
        0.99,
        0.98,
        "(b) Tables",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#344054",
        weight="bold",
    )
    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
