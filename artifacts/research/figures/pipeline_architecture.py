"""Evidence-linked OCR architecture. Regenerate: python pipeline_architecture.py"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches

from orx_figstyle import WIDE, figure, save, use_style

OUTPUT = Path(__file__).with_name("pipeline_architecture")
NAVY = "#15324B"
AMBER = "#D28A00"
TEXT = "#1F2933"
MUTED = "#667085"
LINE = "#AEB7C2"
PAPER = "#F7F8FA"


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    detail: str,
    *,
    accent: bool = False,
) -> None:
    edge = AMBER if accent else NAVY
    face = "#FFF5DC" if accent else "#FFFFFF"
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15 if accent else 0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.018,
        y + height * 0.63,
        label,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + 0.018,
        y + height * 0.31,
        detail,
        ha="left",
        va="center",
        fontsize=6.1,
        color=TEXT,
        linespacing=1.2,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    dashed: bool = False,
    curve: float = 0,
) -> None:
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.9,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(arrow)


def main() -> None:
    use_style()
    fig, ax = figure(width=WIDE, ratio=0.48)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_xlabel("Document flow")
    ax.set_facecolor(PAPER)

    add_box(ax, 0.02, 0.67, 0.13, 0.18, "PDF or image", "Local input\npage sequence")
    add_box(
        ax,
        0.20,
        0.67,
        0.15,
        0.18,
        "Page preparation",
        "300 dpi PDF\nEXIF normalization",
    )
    add_box(
        ax,
        0.40,
        0.67,
        0.20,
        0.18,
        "Local reader",
        "Orientation recovery\ntiles + wide bands",
    )
    add_box(
        ax,
        0.66,
        0.67,
        0.27,
        0.18,
        "Ordered TextRegion IR",
        "Literal text + pixel box + order\nprovider + alternatives + structure",
    )

    add_box(
        ax,
        0.57,
        0.25,
        0.19,
        0.18,
        "Specialist stages",
        "TATR table geometry\ngeometric controls",
    )
    add_box(
        ax,
        0.32,
        0.25,
        0.19,
        0.18,
        "Evidence verification",
        "Literal risks + disagreement\nfailures remain visible",
    )
    add_box(
        ax,
        0.06,
        0.25,
        0.19,
        0.18,
        "Route and render",
        "Review or accept local\nJSON + Markdown + layout",
    )
    add_box(
        ax,
        0.82,
        0.25,
        0.14,
        0.18,
        "Handwriting",
        "User-selected crop\nPhi-4 adapter",
        accent=True,
    )

    add_arrow(ax, (0.15, 0.76), (0.20, 0.76))
    add_arrow(ax, (0.35, 0.76), (0.40, 0.76))
    add_arrow(ax, (0.60, 0.76), (0.66, 0.76))
    add_arrow(ax, (0.74, 0.67), (0.67, 0.43), curve=0.12)
    add_arrow(ax, (0.57, 0.34), (0.51, 0.34))
    add_arrow(ax, (0.32, 0.34), (0.25, 0.34))
    add_arrow(ax, (0.86, 0.67), (0.87, 0.43), color=AMBER, dashed=True, curve=0.18)
    add_arrow(ax, (0.93, 0.43), (0.92, 0.67), color=AMBER, dashed=True, curve=0.18)

    ax.text(0.825, 0.53, "selected crop", fontsize=5.7, color=AMBER, ha="right")
    ax.text(
        0.955,
        0.53,
        "candidate +\nprovenance",
        fontsize=5.7,
        color=AMBER,
        ha="left",
        va="center",
    )
    ax.text(
        0.5,
        0.08,
        "No silent overwrite  |  private pages stay local  |  evidence IDs survive rendering",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MUTED,
    )
    save(fig, str(OUTPUT))


if __name__ == "__main__":
    main()
