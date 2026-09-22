"""PERMANOVA: analysis of variance computed from a distance matrix.

The partition of variance is the same as in a classic ANOVA, but every sum of
squares is obtained from pairwise distances rather than from coordinates:

    SS_total  = sum(D**2) / (2 * n)
    SS_within = sum over groups of sum(D_g**2) / (2 * n_g)
    SS_between = SS_total - SS_within

That identity (Koenig-Huygens) is what makes the method work on any distance,
Euclidean or not. With one-dimensional data and Euclidean distance the
resulting pseudo-F is exactly the F of a one-way ANOVA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "sum_of_squares_total",
    "sum_of_squares_within",
    "pseudo_f",
    "PermanovaResult",
    "permanova",
]


def sum_of_squares_total(dist: pd.DataFrame) -> float:
    """Total sum of squares about the centroid, from distances alone."""
    return (dist.to_numpy() ** 2).sum() / (2 * dist.shape[0])


def sum_of_squares_within(dist: pd.DataFrame, groups: pd.Series) -> float:
    """Sum of squares within groups: each group normalised by its own size."""
    groups = groups.reindex(dist.index)
    values = dist.to_numpy()
    ss_within = 0.0
    for group in groups.unique():
        mask = (groups == group).to_numpy()
        block = values[np.ix_(mask, mask)]
        ss_within += (block ** 2).sum() / (2 * mask.sum())
    return ss_within


def pseudo_f(dist: pd.DataFrame, groups: pd.Series) -> float:
    """Pseudo-F statistic for `groups` over the distance matrix `dist`."""
    groups = groups.reindex(dist.index)
    ss_total = sum_of_squares_total(dist)
    ss_within = sum_of_squares_within(dist, groups)
    if ss_within == 0:
        raise ValueError(
            "Sum of squares within groups is zero, cannot compute pseudo-F.")
    ss_between = ss_total - ss_within
    df_between = groups.nunique() - 1
    df_within = dist.shape[0] - groups.nunique()
    return (ss_between / df_between) / (ss_within / df_within)


@dataclass(frozen=True)
class PermanovaResult:
    """Outcome of a PERMANOVA.

    statistic: the observed pseudo-F
    p_value: fraction of permutations reaching it, with the observed value
        counted as one of them -- so it can never be 0
    r_squared: SS_between / SS_total, the share of variation the factor
        accounts for. This is the effect size; `p_value` is only evidence
        against the null and says nothing about magnitude.
    n_permutations: how many shuffles the null was built from
    scheme: "free" or "restricted within <strata name>"
    """

    statistic: float
    p_value: float
    r_squared: float
    ss_between: float
    ss_within: float
    ss_total: float
    df_between: int
    df_within: int
    n_permutations: int
    scheme: str
    null: np.ndarray


def _shuffle(labels: np.ndarray, rng: np.random.Generator,
             strata: np.ndarray | None) -> np.ndarray:
    """Shuffle `labels`, either freely or independently inside each stratum."""
    if strata is None:
        return rng.permutation(labels)

    shuffled = labels.copy()
    for value in np.unique(strata):
        positions = np.flatnonzero(strata == value)
        shuffled[positions] = rng.permutation(labels[positions])
    return shuffled


def permanova(
    dist: pd.DataFrame,
    groups: pd.Series,
    *,
    strata: pd.Series | None = None,
    n_permutations: int = 999,
    random_state: int | None = None,
) -> PermanovaResult:
    """Test whether `groups` explains the structure in `dist`.

    The null is built by shuffling the group labels and recomputing pseudo-F.

    `strata` restricts that shuffling: labels are permuted only *within* each
    stratum, which preserves the blocking structure and removes the stratum's
    own effect from the null. For a paired design -- 10 patients x 3 sites --
    testing `site` with `strata=patient` asks whether the sites separate
    consistently inside a patient, which is the only interpretable question
    when the patient effect is large.

    The same restriction cannot test the stratifying factor itself: a patient
    label does not vary within its own block, so every shuffle would reproduce
    the observed value. Test `patient` with free permutation instead.
    """
    groups = groups.reindex(dist.index)
    if groups.isna().any():
        raise ValueError("every sample in `dist` needs a group label")

    observed = pseudo_f(dist, groups)
    ss_total = sum_of_squares_total(dist)
    ss_within = sum_of_squares_within(dist, groups)
    n_groups = int(groups.nunique())

    labels = groups.to_numpy()
    strata_values = None
    scheme = "free"
    if strata is not None:
        strata = strata.reindex(dist.index)
        if strata.isna().any():
            raise ValueError("every sample in `dist` needs a stratum label")
        if strata.nunique() == len(dist):
            raise ValueError("each stratum holds one sample: nothing can be permuted")
        strata_values = strata.to_numpy()
        scheme = f"restricted within {strata.name or 'strata'}"

    rng = np.random.default_rng(random_state)
    index = dist.index
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = pd.Series(_shuffle(labels, rng, strata_values), index=index)
        null[i] = pseudo_f(dist, permuted)

    # The observed value counts as one of the possible arrangements: under the
    # null it is no less likely than any shuffle. Without it p could read 0,
    # which claims more than 999 shuffles can support.
    p_value = (int((null >= observed).sum()) + 1) / (n_permutations + 1)

    return PermanovaResult(
        statistic=observed,
        p_value=p_value,
        r_squared=(ss_total - ss_within) / ss_total,
        ss_between=ss_total - ss_within,
        ss_within=ss_within,
        ss_total=ss_total,
        df_between=n_groups - 1,
        df_within=len(dist) - n_groups,
        n_permutations=n_permutations,
        scheme=scheme,
        null=null,
    )
