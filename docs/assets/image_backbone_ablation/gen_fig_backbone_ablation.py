#!/usr/bin/env python3
"""Generate the archived Selfless Flow image-backbone ablation figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
EVIDENCE = REPO_ROOT / "output" / "image_backbone_ablation" / "evidence"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#D0D0D0"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ASSET_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(ASSET_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def screening_figure(summary: dict) -> None:
    rows = {row["id"]: row for row in summary["aggregates"]}
    retained_parent = {"E2", "E2b"}
    s2d = {"E3", "E5", "E6a", "E6b", "E6", "E7a", "E7b", "E7"}
    stage = {"E1", "E4a", "E4b", "E4"}

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for variant_id, row in rows.items():
        if variant_id in s2d:
            color, marker, size, zorder = VERMILLION, "x", 45, 3
        elif variant_id in retained_parent:
            color, marker, size, zorder = BLUE, "o", 55, 5
        elif variant_id in stage:
            color, marker, size, zorder = ORANGE, "^", 42, 4
        else:
            color, marker, size, zorder = GRAY, "o", 36, 2
        ax.scatter(
            row["fid_mean"],
            row["is_mean"],
            c=color,
            marker=marker,
            s=size,
            linewidth=1.4,
            zorder=zorder,
        )
        offset = (4, 4)
        if variant_id in {"E6a", "E7a"}:
            offset = (4, -10)
        elif variant_id == "E2":
            offset = (4, 9)
        elif variant_id == "E2b":
            offset = (-27, -2)
        elif variant_id == "E4":
            offset = (4, 7)
        ax.annotate(
            variant_id,
            (row["fid_mean"], row["is_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_xlabel("FID ↓")
    ax.set_ylabel("Inception Score ↑")
    ax.set_title("Seed-42 screening: quality frontier and rejected branches")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.invert_xaxis()
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=BLUE, label="Retained parent"),
        plt.Line2D(
            [],
            [],
            marker="^",
            linestyle="",
            color=ORANGE,
            label="Historical stage branch",
        ),
        plt.Line2D(
            [],
            [],
            marker="x",
            linestyle="",
            color=VERMILLION,
            label="Rejected S2D branch",
        ),
        plt.Line2D([], [], marker="o", linestyle="", color=GRAY, label="Other screen"),
    ]
    ax.legend(handles=handles, ncol=2, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "backbone_screening")


def confirmation_figure(summary: dict) -> None:
    ordered_ids = ["E0", "E1", "E2b", "E2", "E4b", "E4"]
    rows = {row["id"]: row for row in summary["aggregates"]}
    colors = [
        GRAY,
        ORANGE,
        GREEN,
        BLUE,
        PURPLE,
        ORANGE,
    ]
    x = np.arange(len(ordered_ids))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    fid = [rows[key]["fid_mean"] for key in ordered_ids]
    fid_err = [rows[key]["fid_sample_std"] for key in ordered_ids]
    inception = [rows[key]["is_mean"] for key in ordered_ids]
    inception_err = [rows[key]["is_sample_std"] for key in ordered_ids]

    axes[0].bar(x, fid, yerr=fid_err, color=colors, capsize=3, edgecolor="white")
    axes[0].axhline(rows["E0"]["fid_mean"], color=GRAY, linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("FID ↓")
    axes[0].set_title("Three-seed confirmation")
    axes[0].set_ylim(23.8, 27.9)

    axes[1].bar(
        x,
        inception,
        yerr=inception_err,
        color=colors,
        capsize=3,
        edgecolor="white",
    )
    axes[1].axhline(rows["E0"]["is_mean"], color=GRAY, linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("Inception Score ↑")
    axes[1].set_title("Mean ± sample SD")
    axes[1].set_ylim(57.2, 63.0)

    for ax in axes:
        ax.set_xticks(x, ordered_ids)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, "backbone_confirmation")


def retained_figure(summary: dict) -> None:
    rows = {row["id"]: row for row in summary["aggregates"]}
    ordered_ids = ["E2-Q1", "E2-Q0", "E2b-Q0"]
    colors = [BLUE, GREEN, ORANGE]
    x = np.arange(len(ordered_ids))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.15))
    fid = [rows[key]["fid_mean"] for key in ordered_ids]
    fid_err = [rows[key]["fid_sample_std"] for key in ordered_ids]
    inception = [rows[key]["is_mean"] for key in ordered_ids]
    inception_err = [rows[key]["is_sample_std"] for key in ordered_ids]

    axes[0].bar(x, fid, yerr=fid_err, color=colors, capsize=3, edgecolor="white")
    axes[0].set_ylabel("FID ↓")
    axes[0].set_title("Retained backbone set")
    axes[0].set_ylim(24.2, 26.1)

    axes[1].bar(
        x,
        inception,
        yerr=inception_err,
        color=colors,
        capsize=3,
        edgecolor="white",
    )
    axes[1].set_ylabel("Inception Score ↑")
    axes[1].set_title("Three seeds per variant")
    axes[1].set_ylim(59.8, 63.0)

    for ax in axes:
        ax.set_xticks(x, ordered_ids, rotation=15)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, "backbone_retained_variants")


def main() -> None:
    style()
    screening = load_json(
        EVIDENCE / "screening_and_confirmation" / "expanded_seed42_summary.json"
    )
    confirmation = load_json(
        EVIDENCE / "screening_and_confirmation" / "confirmation_d1_summary.json"
    )
    retained = load_json(
        EVIDENCE / "mask_position_q_factor" / "legacy_bridge_summary.json"
    )
    screening_figure(screening)
    confirmation_figure(confirmation)
    retained_figure(retained)
    print(ASSET_DIR)


if __name__ == "__main__":
    main()
