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

__all__ = [
    "THEMES",
    "SITE_ORDER",
    "SITE_LABEL",
    "figure_centring",
    "figure_stages",
    "figure_scree",
    "figure_loadings",
    "mes_scores",
    "save_both",
]

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


def figure_centring(log2_cpm, gene_a: str, gene_b: str, theme_name: str):
    """What centring does, on two genes.

    Selected for comparable mean *and* dispersion on the scale in use, so the
    cloud is not a near-vertical line: POSTN (mean 7.14, sd 2.03) and COL11A1
    (7.52, 1.87) are the two most variable genes of the 15 in the signature,
    with means 0.38 apart and dispersions 0.16 apart. They are also the pair
    used in the companion animation, so the article and the video show the
    reader the same cloud.

    Both panels share one set of axis limits, computed over the original
    values, the centred values and zero. Letting each panel autoscale -- even
    with zero included -- would rescale the cloud and hide the translation,
    which is the only thing the figure is about.
    """
    theme = THEMES[theme_name]
    x, y = log2_cpm[gene_a], log2_cpm[gene_b]
    xc, yc = x - x.mean(), y - y.mean()

    span = np.concatenate([x, y, xc, yc, [0.0]])
    pad = 0.08 * (span.max() - span.min())
    lo, hi = span.min() - pad, span.max() + pad

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), facecolor=theme["surface"])
    panels = [("Original values", x, y), ("After subtracting each gene's mean", xc, yc)]

    for ax, (title, px, py) in zip(axes, panels):
        _style_axes(ax, theme)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")          # distances must stay readable as distances
        ax.axhline(0, color=theme["muted"], linewidth=0.9, zorder=1)
        ax.axvline(0, color=theme["muted"], linewidth=0.9, zorder=1)

        ax.scatter(px, py, s=54, c=theme["series"][0], edgecolors=theme["surface"],
                   linewidths=1.2, zorder=4)
        ax.scatter([0], [0], marker="P", s=110, c=theme["neutral"],
                   edgecolors=theme["surface"], linewidths=1.2, zorder=5)
        ax.scatter([px.mean()], [py.mean()], marker="D", s=62,
                   facecolors="none", edgecolors=theme["series"][1],
                   linewidths=2.0, zorder=6)

        ax.set_title(title, color=theme["primary"], fontsize=10, loc="left", pad=10)
        ax.set_xlabel(f"{gene_a}  ·  log2(CPM + 1)", color=theme["secondary"], fontsize=9)
    axes[0].set_ylabel(f"{gene_b}  ·  log2(CPM + 1)", color=theme["secondary"], fontsize=9)

    # Direct labels rather than a legend box: two marks, named where they sit.
    axes[0].annotate("origin", xy=(0, 0), xytext=(lo + 0.5 * pad, 0.9),
                     color=theme["secondary"], fontsize=8.5)
    axes[0].annotate("centroid", xy=(x.mean(), y.mean()), xytext=(x.mean() + 0.6, y.mean() - 1.4),
                     color=theme["secondary"], fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=theme["muted"], linewidth=0.8))
    # Placed in the empty lower-right quadrant: the cloud runs lower-left to
    # upper-right, so anything nearer the origin collides with a point.
    axes[1].annotate("origin and centroid\nnow coincide", xy=(0.25, -0.25),
                     xytext=(0.45 * hi, 0.55 * lo), color=theme["secondary"],
                     fontsize=8.5, ha="left", va="top",
                     arrowprops=dict(arrowstyle="-", color=theme["muted"], linewidth=0.8))

    fig.tight_layout()
    return fig


def mes_scores(log2_cpm, signature, sites) -> pd.DataFrame:
    """The Mes signature score per sample, both ways it can be computed.

    `authors` follows Hu et al.: threshold each gene at its median across the
    primary tumours, score 1/0, sum the 15 and rescale to 0-1.
    `z` is the plainer continuous alternative: z-score each gene across
    samples, then average.

    This is an *expression signature*, not a measurement of stromal content.
    Hu et al. showed it tracks pathologist-counted fibroblasts in this dataset's
    metastases (r = 0.703 omental, 0.893 non-omental) and not in its primary
    tumours (r = 0.170, ns) -- the cell counting is theirs, not ours.
    """
    mes = log2_cpm[signature.index]
    primary_median = mes[(sites == "ov").to_numpy()].median()
    z = (mes - mes.mean()) / mes.std()
    return pd.DataFrame(
        {"authors": (mes > primary_median).sum(axis=1) / mes.shape[1], "z": z.mean(axis=1)}
    )


def figure_loadings(result, symbols, signature, scores_table, metadata, theme_name: str,
                    n_top: int = 8):
    """What the components are made of, and where the signature sits.

    Any association between the signature score and the components is
    descriptive rather than independent evidence: both are computed from the
    same matrix, so a correlation between them is partly a restatement.
    """
    theme = THEMES[theme_name]
    loadings = result.loadings["PC1"]
    named = loadings.rename(index=symbols)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), facecolor=theme["surface"])
    for ax in axes:
        _style_axes(ax, theme)

    # A -- the genes that build PC1.
    extremes = pd.concat([named.nsmallest(n_top), named.nlargest(n_top)]).sort_values()
    positions = np.arange(len(extremes))
    colours = [theme["series"][1] if v < 0 else theme["series"][0] for v in extremes]
    axes[0].barh(positions, extremes.to_numpy(), color=colours, height=0.68, zorder=3)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(extremes.index, fontsize=7.5, color=theme["secondary"])
    axes[0].axvline(0, color=theme["muted"], linewidth=0.9, zorder=4)
    axes[0].set_xlabel("PC1 loading", color=theme["secondary"], fontsize=9)
    axes[0].set_title(f"Genes building PC1 · {result.explained_variance_ratio.iloc[0] * 100:.1f}% of variance",
                      color=theme["primary"], fontsize=9.5, loc="left", pad=10)

    # B -- the signature against the whole loading distribution.
    present = [g for g in signature.index if g in loadings.index]
    magnitude = loadings.abs()
    axes[1].hist(magnitude.to_numpy(), bins=40, color=theme["grid"], zorder=2)
    for gene in present:
        axes[1].axvline(magnitude[gene], color=theme["series"][2], linewidth=1.6, zorder=3)

    # One annotation rather than four rotated labels: the signature genes sit
    # so close together at the low end that individual labels collide.
    percentile = magnitude.rank(pct=True)[present].max() * 100
    listed = ", ".join(sorted(symbols[g] for g in present))
    top = axes[1].get_ylim()[1]
    axes[1].annotate(
        f"{listed}\nall below the {percentile:.0f}th percentile",
        xy=(magnitude[present].max(), top * 0.55),
        xytext=(magnitude.max() * 0.42, top * 0.78),
        color=theme["secondary"], fontsize=8, va="top",
        arrowprops=dict(arrowstyle="-", color=theme["series"][2], linewidth=1.0),
    )
    axes[1].set_xlabel("|PC1 loading|", color=theme["secondary"], fontsize=9)
    axes[1].set_ylabel("Genes", color=theme["secondary"], fontsize=9)
    axes[1].set_title(
        f"Where the signature sits ({len(present)} of {len(signature)} genes analysed)",
        color=theme["primary"], fontsize=9.5, loc="left", pad=10,
    )

    # C -- the signature score by site.
    order = SITE_ORDER
    rng = np.random.default_rng(0)
    for i, site in enumerate(order):
        values = scores_table.loc[(metadata["site"] == site).to_numpy(), "z"]
        jitter = rng.uniform(-0.13, 0.13, len(values))
        axes[2].scatter(np.full(len(values), i) + jitter, values, s=48,
                        c=THEMES[theme_name]["series"][i], edgecolors=theme["surface"],
                        linewidths=1.1, zorder=3)
        axes[2].plot([i - 0.28, i + 0.28], [values.mean()] * 2, color=theme["primary"],
                     linewidth=1.8, zorder=4, solid_capstyle="round")
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels([SITE_LABEL[s] for s in order], fontsize=8, color=theme["secondary"])
    axes[2].set_ylabel("Mes signature score (mean z)", color=theme["secondary"], fontsize=9)
    axes[2].set_xlim(-0.55, len(order) - 0.45)
    axes[2].set_title("Signature score by site", color=theme["primary"],
                      fontsize=9.5, loc="left", pad=10)

    fig.tight_layout()
    return fig
