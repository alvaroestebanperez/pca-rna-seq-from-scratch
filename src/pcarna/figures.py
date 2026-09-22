"""Figures for the note, in light and dark variants.

Colours are slots 1-3 of the validated categorical palette. Three is the cap
for scatter: with every pair of colours visually adjacent, no ordering of more
than three clears the colour-vision separation floors. Identity beyond three
categories is carried by something other than hue -- here, by the line joining
a patient's samples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["THEMES", "SITE_ORDER", "SITE_LABEL", "figure_stages", "figure_scree", "save_both"]

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#8a8985", grid="#e5e4e0", neutral="#7a7975",
                  series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#8a8985", grid="#333330", neutral="#9a9990",
                 series=("#3987e5", "#d95926", "#199e70")),
}

SITE_ORDER = ["ov", "om_met", "met"]
SITE_LABEL = {"ov": "Primary ovary", "om_met": "Omental met.", "met": "Non-omental met."}

STAGE_LABEL = {
    "raw": "1 · Raw counts",
    "filtered": "2 · Low-expression genes removed",
    "cpm": "3 · CPM normalised",
    "log2_cpm": "4 · log2(CPM + 1)",
    "variable": "5 · Top 500 variable genes",
}


def _style_axes(ax, theme) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["secondary"], labelsize=7.5, length=3)


def figure_stages(results, metadata, library_sizes, theme_name: str):
    """PC1-PC2 at each preprocessing stage: the same samples, five geometries.

    Each panel carries |r| between PC1 and library size, which is the point of
    the figure: filtering leaves the depth axis untouched and the log transform,
    not the normalisation, is what finally breaks it.
    """
    theme = THEMES[theme_name]
    colours = dict(zip(SITE_ORDER, theme["series"]))
    stages = list(results)

    fig, axes = plt.subplots(1, len(stages), figsize=(15.0, 3.5), facecolor=theme["surface"])
    for ax, stage in zip(axes, stages):
        result = results[stage]
        scores, ratio = result.scores, result.explained_variance_ratio
        _style_axes(ax, theme)
        for site in SITE_ORDER:
            mask = (metadata["site"] == site).to_numpy()
            ax.scatter(scores["PC1"][mask], scores["PC2"][mask], s=34,
                       c=colours[site], edgecolors=theme["surface"],
                       linewidths=1.0, zorder=3, label=SITE_LABEL[site])
        r = abs(scores["PC1"].corr(library_sizes))
        ax.set_title(STAGE_LABEL[stage], color=theme["primary"], fontsize=9,
                     loc="left", pad=8)
        ax.set_xlabel(
            f"PC1 · {ratio.iloc[0] * 100:.1f}%   |r| with depth {r:.2f}",
            color=theme["secondary"], fontsize=8,
        )
        ax.set_ylabel(f"PC2 · {ratio.iloc[1] * 100:.1f}%", color=theme["secondary"], fontsize=8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    legend = axes[0].legend(frameon=False, fontsize=7.5, loc="best",
                            handletextpad=0.4, borderpad=0.2)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])
    fig.tight_layout()
    return fig


def figure_scree(result, theme_name: str):
    """Variance per component, and the running total.

    Two panels rather than two y-axes: individual shares run to 20% and the
    cumulative to 100%, and overlaying them on separate scales would invite
    reading one curve against the other's axis.
    """
    theme = THEMES[theme_name]
    ratios = result.explained_variance_ratio * 100
    positions = np.arange(1, len(ratios) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), facecolor=theme["surface"])
    for ax in axes:
        _style_axes(ax, theme)
        ax.set_xlabel("Principal component", color=theme["secondary"], fontsize=9)
        ax.set_xlim(0.3, len(ratios) + 0.7)

    axes[0].bar(positions, ratios.to_numpy(), width=0.68, color=theme["series"][0], zorder=3)
    axes[0].set_ylabel("Share of variance (%)", color=theme["secondary"], fontsize=9)
    axes[0].set_title(
        f"{len(ratios)} components carry variance; 30 centred samples have rank {len(ratios)}",
        color=theme["primary"], fontsize=9.5, loc="left", pad=10,
    )

    cumulative = ratios.cumsum().to_numpy()
    axes[1].plot(positions, cumulative, color=theme["series"][0], linewidth=2,
                 marker="o", markersize=3.5, zorder=3,
                 markeredgecolor=theme["surface"], markeredgewidth=0.8)
    axes[1].set_ylabel("Cumulative variance (%)", color=theme["secondary"], fontsize=9)
    axes[1].set_ylim(0, 103)
    axes[1].set_title("Running total", color=theme["primary"], fontsize=9.5, loc="left", pad=10)
    for n in (2, 10):
        axes[1].annotate(
            f"PC1–PC{n}: {cumulative[n - 1]:.0f}%",
            xy=(n, cumulative[n - 1]), xytext=(n + 1.5, cumulative[n - 1] - 12),
            color=theme["secondary"], fontsize=8,
            arrowprops=dict(arrowstyle="-", color=theme["muted"], linewidth=0.8),
        )
    fig.tight_layout()
    return fig


def save_both(builder, out_dir: Path, stem: str, *args, formats=("svg",)) -> list[Path]:
    """Render `builder` once per theme. Dark mode is a selected palette stepped
    for the dark surface, not an automatic inversion of the light one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme_name in THEMES:
        fig = builder(*args, theme_name)
        for suffix in formats:
            path = out_dir / f"{stem}-{theme_name}.{suffix}"
            fig.savefig(path, format=suffix, facecolor=THEMES[theme_name]["surface"],
                        bbox_inches="tight", dpi=130)
            written.append(path)
        plt.close(fig)
    return written
