"""Printed reports: every number the note quotes, computed here rather than by hand.

Nothing in this module returns a figure or a file. It takes the already-built
stages and PCA results and prints them, so the numbers in the article and the
numbers in the repository cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcarna import distance_matrix, group_distances, permanova
from pcarna.figures import SITE_ORDER, mes_scores

__all__ = [
    "report_filtering",
    "report_variance",
    "report_within_patient_sites",
    "report_ranking_scale_effect",
    "report_group_distances",
    "report_permanova",
    "report_signature",
]


def report_filtering(stages, counts: pd.DataFrame) -> None:
    kept = stages.matrices["filtered"]
    retained = kept.to_numpy().sum() / counts.to_numpy().sum() * 100
    per_sample = kept.sum(axis=1) / stages.library_sizes * 100
    print(
        f"{len(stages.kept_genes)} / {counts.shape[1]} genes retained "
        f"({retained:.1f}% of reads; {per_sample.min():.1f}-{per_sample.max():.1f}% per sample)"
    )


def report_variance(results, library_sizes: pd.Series) -> None:
    """Explained variance per stage, and how far PC1/PC2 track sequencing depth."""
    print(
        f"\n{'stage':<10}{'PC1':>7}{'PC2':>7}{'PC3':>7}{'PC4':>7}{'PC5':>7}"
        f"{'|r| PC1':>10}{'|r| PC2':>9}"
    )
    for stage, result in results.items():
        ratios = "".join(
            f"{v * 100:6.1f}%" for v in result.explained_variance_ratio.iloc[:5])
        r1 = abs(result.scores["PC1"].corr(library_sizes))
        r2 = abs(result.scores["PC2"].corr(library_sizes))
        print(f"{stage:<10}{ratios}{r1:>10.3f}{r2:>9.3f}")


def report_within_patient_sites(pair_distances: pd.DataFrame) -> None:
    print("\nWithin-patient distance between each pair of sites (variable stage):")
    print(pair_distances.round(1).to_string())

    # Ranked inside each patient rather than averaged across them: patients
    # differ in overall spread, and averaging would mix that variation with the
    # site effect it is meant to measure.
    largest = pair_distances.idxmax(axis=1).value_counts()
    smallest = pair_distances.idxmin(axis=1).value_counts()
    expected = len(pair_distances) / pair_distances.shape[1]
    print(
        f"\nWhich pair is largest / smallest within a patient "
        f"(n={len(pair_distances)}, {expected:.1f} expected each if site has no effect):"
    )
    for pair in pair_distances.columns:
        print(
            f"  {pair:<12} largest {largest.get(pair, 0):>2}   smallest {smallest.get(pair, 0):>2}")


def report_ranking_scale_effect(stages, n_top: int = 500) -> None:
    """Why variable genes are ranked on the log scale, shown rather than argued.

    RNA-seq variance is a function of expression level, so ranking on CPM
    selects the most abundant genes rather than the most variable ones. The
    two rankings turn out to be almost disjoint.
    """
    cpm = stages.matrices["cpm"]
    log2 = stages.matrices["log2_cpm"]
    by_cpm = cpm.var(axis=0).nlargest(n_top).index
    by_log2 = stages.gene_variance.nlargest(n_top).index
    mean_cpm = cpm.mean(axis=0)

    print(
        f"\nTop {n_top} most variable genes, ranked on CPM vs on log2-CPM:"
        f"\n  shared by both rankings: {len(set(by_cpm) & set(by_log2))} / {n_top}"
        f"\n  median expression (CPM) -- ranked on CPM: {mean_cpm[by_cpm].median():.1f}"
        f"  |  ranked on log2: {mean_cpm[by_log2].median():.1f}"
        f"  |  all kept genes: {mean_cpm.median():.1f}"
    )


def report_group_distances(scores: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Mean distance within a group against between groups, per factor.

    Self-distances are excluded: the diagonal is zero and would deflate the
    within-group mean by n zeros among n^2 entries -- a bias that grows as
    groups shrink, and so would favour whichever factor has smaller groups.
    """
    distances = distance_matrix(scores)
    print(f"\n{'factor':<12}{'groups':>7}{'within':>9}{'between':>9}{'ratio':>8}{'pairs w/b':>13}")
    for factor in ("patient_id", "site"):
        result = group_distances(distances, metadata[factor])
        print(
            f"{factor:<12}{result.n_groups:>7}{result.within:>9.2f}{result.between:>9.2f}"
            f"{result.ratio:>8.3f}{result.n_within_pairs:>7}/{result.n_between_pairs:<5}"
        )


def report_permanova(scores: pd.DataFrame, metadata: pd.DataFrame, *,
                     n_permutations: int = 9999, random_state: int = 20260922) -> None:
    """The contrast the note is named after.

    Site is tested twice on purpose. The statistic is identical either way;
    only the null changes. Free permutation breaks the patient blocks and so
    admits the between-patient variation into the null, which buries a real
    site effect. Restricting the shuffle to within a patient preserves the
    blocking and asks the only question a paired design can answer.

    The same restriction cannot test patient: its label does not vary inside
    its own block, so every shuffle would reproduce the observed value.
    """
    distances = distance_matrix(scores)
    contrasts = [
        ("patient (free)", dict(groups=metadata["patient_id"])),
        ("site (free)", dict(groups=metadata["site"])),
        ("site (restricted)", dict(groups=metadata["site"], strata=metadata["patient_id"])),
    ]
    print(f"\nPERMANOVA, {n_permutations} permutations, seed {random_state}")
    print(f"{'contrast':<20}{'df':>4}{'pseudo-F':>10}{'R2':>8}{'p':>9}{'perms >= obs':>14}  scheme")
    for label, kwargs in contrasts:
        result = permanova(
            distances, n_permutations=n_permutations, random_state=random_state, **kwargs
        )
        reached = int((result.null >= result.statistic).sum())
        print(
            f"{label:<20}{result.df_between:>4}{result.statistic:>10.3f}"
            f"{result.r_squared:>8.3f}{result.p_value:>9.4f}"
            f"{reached:>8} / {n_permutations:<5}  {result.scheme}"
        )


def report_signature(
    stages, annotations: pd.DataFrame, results, signature: pd.Series,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """The Mes signature: its two scorings, and where it sits in PC1.

    Any association between the score and the components is descriptive, not
    independent evidence -- both come from the same matrix.
    """
    scores = mes_scores(stages.matrices["log2_cpm"], signature, metadata["site"])
    loadings = results["variable"].loadings["PC1"]
    present = [g for g in signature.index if g in loadings.index]
    percentile = loadings.abs().rank(pct=True)[present].max() * 100

    print(
        f"\nMes signature ({len(signature)} genes, all mapped and all past the filter)"
        f"\n  the two scorings agree: r = {scores['authors'].corr(scores['z']):.3f} "
        f"(Spearman {scores['authors'].corr(scores['z'], method='spearman'):.3f})"
    )
    by_site = scores.assign(site=metadata["site"]).groupby("site")["z"].mean()
    print("  mean score by site: " + "  ".join(f"{s} {by_site[s]:+.3f}" for s in SITE_ORDER))
    print(
        f"  in the variable-gene set: {len(present)} of {len(signature)}, "
        f"all below the {percentile:.0f}th percentile of |PC1 loading|"
    )
    for pc in ("PC1", "PC2"):
        print(f"  r(score, {pc}) = {results['variable'].scores[pc].corr(scores['z']):+.3f}")
    return scores
