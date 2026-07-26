#!/usr/bin/env python3
"""Generate the Qwen-backbone × flow-position ablation figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = (
    REPO_ROOT
    / "output"
    / "backbone_flow_head_joint_ablation"
    / "evidence"
    / "summary_seed42.json"
)

# Okabe–Ito colorblind-safe palette.
COLORS = {
    "E2-Q1": "#0072B2",
    "E2-Q0": "#D55E00",
    "E2b-Q0": "#009E73",
}
SELECTED_COLOR = "#CC79A7"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    configure_style()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = {
        (row["backbone"], row["position"]): row
        for row in summary["rows"]
    }
    decision = summary["decision"]
    fid_limit = decision["best_fid"] + decision["fid_noninferiority_margin"]
    is_limit = (
        decision["best_inception_score_mean"]
        - decision["is_noninferiority_margin"]
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.65))
    xmin, xmax = 22.84, 23.82
    ymin, ymax = 62.90, 65.20

    region = Rectangle(
        (xmin, is_limit),
        fid_limit - xmin,
        ymax - is_limit,
        facecolor="#E8F2EE",
        edgecolor="none",
        alpha=0.75,
        zorder=0,
    )
    ax.add_patch(region)
    ax.text(
        23.335,
        65.09,
        "quality-noninferior set",
        color="#4E7568",
        fontsize=7.5,
        ha="center",
        va="bottom",
    )

    for backbone, color in COLORS.items():
        fh0 = rows[(backbone, "FH0")]
        fh4 = rows[(backbone, "FH4")]
        arrow = FancyArrowPatch(
            (fh0["fid"], fh0["inception_score_mean"]),
            (fh4["fid"], fh4["inception_score_mean"]),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.3,
            color=color,
            alpha=0.75,
            shrinkA=6,
            shrinkB=7,
            zorder=2,
        )
        ax.add_patch(arrow)
        ax.scatter(
            fh0["fid"],
            fh0["inception_score_mean"],
            s=45,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )
        ax.scatter(
            fh4["fid"],
            fh4["inception_score_mean"],
            s=48,
            marker="D",
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    selected = rows[("E2-Q0", "FH4")]
    ax.scatter(
        selected["fid"],
        selected["inception_score_mean"],
        s=160,
        marker="*",
        facecolor=SELECTED_COLOR,
        edgecolor="#222222",
        linewidth=0.65,
        zorder=6,
    )
    ax.annotate(
        "selected: E2-Q0 + FH4",
        (selected["fid"], selected["inception_score_mean"]),
        xytext=(20, -3),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": "#555555",
            "linewidth": 0.7,
        },
    )

    backbone_handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="-",
            color=color,
            markerfacecolor=color,
            markeredgewidth=0,
            markersize=5,
            linewidth=1.2,
            label=backbone,
        )
        for backbone, color in COLORS.items()
    ]
    position_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markersize=5,
            label="DF1-FH0",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=5,
            label="DF1-FH4",
        ),
    ]
    legend_backbone = ax.legend(
        handles=backbone_handles,
        loc="lower left",
        ncol=3,
        handlelength=1.4,
        columnspacing=1.2,
    )
    ax.add_artist(legend_backbone)
    ax.legend(handles=position_handles, loc="lower right", ncol=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("FID ↓")
    ax.set_ylabel("Inception Score ↑")
    ax.set_axisbelow(True)
    fig.tight_layout()

    fig.savefig(ASSET_DIR / "joint_ablation.pdf")
    fig.savefig(ASSET_DIR / "joint_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
