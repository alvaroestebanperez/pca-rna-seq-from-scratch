"""Pairwise distances between samples, and how they split by group.

Nothing here is specific to PCA: the input is any samples x features table.
Applied to PCA scores over *all* components it gives exactly the distances of
the original feature space, since a PCA is a rotation and rotations preserve
distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["distance_matrix", "GroupDistances",
           "group_distances", "group_distance_table"]


def distance_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance between every pair of rows of ``X``.

    Returns an ``(n_samples, n_samples)`` frame labelled with ``X.index`` on
    both axes: symmetric, zero on the diagonal, never negative.

    Expects a low-dimensional table such as PCA scores. The intermediate array
    is ``(n, n, n_features)``, so passing a full expression matrix (30 x 19712)
    would allocate ~140 MB where the scores allocate ~200 KB.
    """
    values = X.to_numpy(dtype=np.float64)
    distances = np.sqrt(
        ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(distances, index=X.index, columns=X.index)


@dataclass(frozen=True)
class GroupDistances:
    """How distances split by a grouping factor.

    within: mean distance between two *different* samples of the same group
    between: mean distance between samples of different groups
    ratio: between / within -- above 1 means the groups are separated
    n_groups, n_within_pairs, n_between_pairs: what each mean was computed over
    """

    within: float
    between: float
    ratio: float
    n_groups: int
    n_within_pairs: int
    n_between_pairs: int


def _pair_values(dist: pd.DataFrame, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Split the upper triangle into same-group and different-group distances.

    Only the upper triangle is used, excluding the diagonal. The diagonal holds
    each sample's distance to itself, which is zero and is not a distance
    "within" anything -- including it would deflate the within-group mean by a
    factor that depends on group size (n zeros among n^2 entries). With patients
    in groups of 3 and sites in groups of 10 that bias is 33% against 10%, which
    would make the smaller-grouped factor look tighter for purely arithmetic
    reasons.
    """
    if not dist.index.equals(dist.columns):
        raise ValueError(
            "dist must be square and labelled identically on both axes")
    groups = groups.reindex(dist.index)
    if groups.isna().any():
        missing = groups.index[groups.isna()].tolist()
        raise ValueError(
            f"no group label for {len(missing)} sample(s): {missing[:5]}")

    rows, cols = np.triu_indices(len(dist), k=1)
    values = dist.to_numpy()[rows, cols]
    same_group = groups.to_numpy()[rows] == groups.to_numpy()[cols]
    return values[same_group], values[~same_group]


def group_distances(dist: pd.DataFrame, groups: pd.Series) -> GroupDistances:
    """Summarise ``dist`` as within-group vs between-group mean distance."""
    within, between = _pair_values(dist, groups)
    if within.size == 0:
        raise ValueError(
            "no within-group pairs: every group has a single sample")
    if between.size == 0:
        raise ValueError("no between-group pairs: all samples share one group")

    within_mean = float(within.mean())
    between_mean = float(between.mean())
    return GroupDistances(
        within=within_mean,
        between=between_mean,
        ratio=between_mean / within_mean,
        n_groups=int(groups.reindex(dist.index).nunique()),
        n_within_pairs=int(within.size),
        n_between_pairs=int(between.size),
    )


def group_distance_table(dist: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Mean distance between every pair of groups, as a group x group frame.

    The diagonal is the within-group mean, computed without self-distances; the
    off-diagonal entries are between-group means, where every entry is a real
    pair of distinct samples and nothing needs excluding.
    """
    groups = groups.reindex(dist.index)
    labels = pd.Index(pd.unique(groups.dropna()), name=groups.name)
    table = pd.DataFrame(index=labels, columns=labels, dtype=float)

    values = dist.to_numpy()
    for i, a in enumerate(labels):
        mask_a = (groups == a).to_numpy()
        for b in labels[i:]:
            mask_b = (groups == b).to_numpy()
            block = values[np.ix_(mask_a, mask_b)]
            if a == b:
                mean = block[np.triu_indices_from(block, k=1)].mean()
            else:
                mean = block.mean()
            table.loc[a, b] = table.loc[b, a] = mean
    return table
