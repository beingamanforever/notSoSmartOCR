"""Small vendored subset of the OpenResearch publication figure style."""

from __future__ import annotations

import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

COLUMN = 3.25
TEXT = 5.5
WIDE = 6.75

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
}
MUTED = "#CFCFCF"
SANS = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "TeX Gyre Heros",
    "Nimbus Sans",
    "Liberation Sans",
    "DejaVu Sans",
]

_font: str | None = None


def use_style() -> None:
    """Apply the shared OpenResearch print style."""
    global _font
    installed = {item.name for item in mpl.font_manager.fontManager.ttflist}
    _font = next((name for name in SANS if name in installed), None)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SANS,
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "savefig.transparent": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0,
        }
    )


def figure(width: float = COLUMN, ratio: float = 0.68):
    """Create one axes at its final printed size."""
    return plt.subplots(figsize=(width, width * ratio), layout="constrained")


def _audit(fig: mpl.figure.Figure) -> list[str]:
    problems = []
    if mpl.rcParams["pdf.fonttype"] != 42:
        problems.append("fonts are not Type 42")
    if _font is None or _font.startswith("DejaVu"):
        problems.append("no publication font is available")
    width = fig.get_size_inches()[0]
    if not any(abs(width - known) < 0.02 for known in (COLUMN, TEXT, WIDE)):
        problems.append(f"unknown printed width: {width:.2f} in")
    for axes in fig.axes:
        if axes.get_title():
            problems.append("an axes has a title")
        if not axes.get_xlabel():
            problems.append("an axes has no x label")
    tiny = {
        round(item.get_fontsize(), 1)
        for item in fig.findobj(mpl.text.Text)
        if item.get_text().strip() and item.get_fontsize() < 5
    }
    if tiny:
        problems.append(f"text below 5 pt: {sorted(tiny)}")
    return problems


def save(
    fig: mpl.figure.Figure,
    stem: str,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> list[str]:
    """Write vector outputs and print the figure audit result."""
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    paths = []
    for extension in formats:
        path = f"{stem}.{extension}"
        fig.savefig(path, format=extension)
        paths.append(path)
    problems = _audit(fig)
    plt.close(fig)
    if problems:
        print(f"FIGURE AUDIT {stem}: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    else:
        print(f"figure audit {stem}: clean", file=sys.stderr)
    return paths
