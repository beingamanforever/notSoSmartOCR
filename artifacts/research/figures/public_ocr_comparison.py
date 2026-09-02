"""ClinOCR reader comparison. Regenerate: python public_ocr_comparison.py"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evidence import ROOT, decimal, fraction, table_rows
from orx_figstyle import WIDE, save, use_style

SOURCE = ROOT / "artifacts" / "ocr-evidence-report.md"
OUTPUT = Path(__file__).with_name("public_ocr_comparison")
METHODS = ("Tesseract", "Selective Nemotron")
COLORS = ("#B9C0C8", "#15324B")


def load() -> tuple[dict[str, list[float]], list[tuple[float, float, float]]]:
    rows = table_rows(SOURCE, "Reader")
    report = SOURCE.read_text(encoding="utf-8")
    deltas = re.search(
        r"paired CER change is (-?[\d.]+) with cluster 95% interval\s*"
        r"\[(-?[\d.]+), (-?[\d.]+)\]\. The WER change is (-?[\d.]+) "
        r"with interval\s*\[(-?[\d.]+), (-?[\d.]+)\]",
        report,
    )
    if deltas is None:
        raise ValueError("ClinOCR evidence format changed")

    tess = rows["Tesseract"]
    nemo = rows["Selective Nemotron"]
    if not tess[0].endswith("/328") or not nemo[0].endswith("/328"):
        raise ValueError("Expected the frozen 328-page ClinOCR panel")
    rates = {
        METHODS[0]: [decimal(tess[1]), decimal(tess[2]), fraction(tess[0])],
        METHODS[1]: [decimal(nemo[1]), decimal(nemo[2]), fraction(nemo[0])],
    }
    intervals = [float(value) for value in deltas.groups()]
    return rates, [tuple(intervals[:3]), tuple(intervals[3:])]


def main() -> None:
    use_style()
    rates, intervals = load()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDE, WIDE * 0.43),
        gridspec_kw={"width_ratios": (1.35, 1)},
        layout="constrained",
    )

    metrics = ("CER", "WER", "Coverage")
    x = np.arange(len(metrics))
    width = 0.36
    labels = {
        METHODS[0]: ("0.488", "0.611", "299/328"),
        METHODS[1]: ("0.331", "0.418", "282/328"),
    }
    for index, method in enumerate(METHODS):
        bars = axes[0].bar(
            x + (index - 0.5) * width,
            rates[method],
            width,
            color=COLORS[index],
            edgecolor="#15324B",
            linewidth=0.55,
            label=method,
        )
        axes[0].bar_label(bars, labels=labels[method], fontsize=6, padding=2)

    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("Rate")
    axes[0].set_xlabel("Failure-inclusive metric, all 328 pages")
    axes[0].set_xticks(x, metrics)
    axes[0].tick_params(axis="x", length=0)
    axes[0].legend(
        loc="upper left",
        frameon=True,
        framealpha=1,
        edgecolor="#667085",
        fontsize=7,
    )
    axes[0].text(
        0.99,
        0.98,
        "(a) Absolute rates",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#344054",
        weight="bold",
    )

    y = np.arange(2)
    for index, (delta, low, high) in enumerate(intervals):
        axes[1].errorbar(
            delta,
            y[index],
            xerr=[[delta - low], [high - delta]],
            fmt="o",
            color="#15324B",
            ecolor="#15324B",
            elinewidth=1.1,
            capsize=3,
            markersize=5,
            zorder=3,
        )
        axes[1].text(
            high + 0.006,
            y[index],
            f"{delta:.3f}",
            ha="left",
            va="center",
            fontsize=6.5,
        )
    axes[1].axvline(0, color="#667085", linewidth=0.8)
    axes[1].set_xlim(-0.25, 0.02)
    axes[1].set_ylim(-0.6, 1.6)
    axes[1].set_yticks(y, ("CER", "WER"))
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Paired metric")
    axes[1].set_xlabel("Nemotron - Tesseract error-rate delta")
    axes[1].grid(axis="x")
    axes[1].text(
        0.99,
        0.98,
        "(b) Paired deltas",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#344054",
        weight="bold",
    )
    axes[1].text(
        0.02,
        0.03,
        "95% cluster bootstrap CI\n16 template clusters; 10,000 draws",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#667085",
    )

    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
