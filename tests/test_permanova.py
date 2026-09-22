"""Checks for the PERMANOVA statistic and its permutation schemes.

The anchor is that with one-dimensional data and Euclidean distance the
pseudo-F is *exactly* the F of a classic one-way ANOVA, so scipy's `f_oneway`
is an independent definition to check against rather than another route through
the same code.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pcarna import distance_matrix
from pcarna.permanova import (
    permanova,
    pseudo_f,
    sum_of_squares_total,
    sum_of_squares_within,
)


@pytest.fixture
def one_dimensional():
    """Three groups of different sizes, separated means, a single variable."""
    rng = np.random.default_rng(20260922)
    sizes = [5, 6, 7]
    labels = np.repeat(["a", "b", "c"], sizes)
    values = rng.normal(size=sum(sizes)) + np.repeat([0.0, 1.5, 3.0], sizes)
    index = pd.Index([f"s{i:02d}" for i in range(sum(sizes))], name="sample")
    return (
        pd.DataFrame({"v": values}, index=index),
        pd.Series(labels, index=index, name="group"),
    )


@pytest.fixture
def paired():
    """10 blocks x 3 conditions, with a large block effect and a small one
    inside each block -- the structure of the real dataset."""
    rng = np.random.default_rng(7)
    n_blocks, conditions = 10, ["x", "y", "z"]
    rows, block, cond = [], [], []
    block_offset = rng.normal(scale=10.0, size=(n_blocks, 4))
    cond_offset = np.array([[0.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    for b in range(n_blocks):
        for c, name in enumerate(conditions):
            rows.append(block_offset[b] + cond_offset[c] + rng.normal(scale=0.4, size=4))
            block.append(f"b{b}")
            cond.append(name)
    index = pd.Index([f"{b}_{c}" for b, c in zip(block, cond)], name="sample")
    return (
        pd.DataFrame(rows, index=index),
        pd.Series(cond, index=index, name="condition"),
        pd.Series(block, index=index, name="block"),
    )


def test_pseudo_f_equals_classic_anova(one_dimensional):
    """With one variable and Euclidean distance the pseudo-F *is* the ANOVA F.
    Catches any error in how the sums of squares are formed from distances."""
    X, groups = one_dimensional
    observed = pseudo_f(distance_matrix(X), groups)
    by_group = [X["v"][groups == g].to_numpy() for g in groups.unique()]
    assert observed == pytest.approx(stats.f_oneway(*by_group).statistic)


def test_total_sum_of_squares_matches_coordinates(one_dimensional):
    """SS_total from distances must equal the classic sum of squared deviations
    about the centroid -- the identity the whole method rests on."""
    X, _ = one_dimensional
    classic = ((X.to_numpy() - X.to_numpy().mean(axis=0)) ** 2).sum()
    assert sum_of_squares_total(distance_matrix(X)) == pytest.approx(classic)


def test_variance_partition_adds_up(one_dimensional):
    """Catches a within-group term normalised by n instead of each group's own
    size, which would silently break SS_total = SS_within + SS_between."""
    X, groups = one_dimensional
    dist = distance_matrix(X)
    total = sum_of_squares_total(dist)
    within = sum_of_squares_within(dist, groups)
    result = permanova(dist, groups, n_permutations=99, random_state=0)
    assert within + result.ss_between == pytest.approx(total)
    assert result.r_squared == pytest.approx((total - within) / total)


def test_group_order_does_not_change_the_statistic(one_dimensional):
    """Labels are aligned by name, not position: a reordered Series must give
    the same answer. Without the reindex the masks select the wrong samples."""
    X, groups = one_dimensional
    dist = distance_matrix(X)
    shuffled = groups.sample(frac=1, random_state=3)
    assert pseudo_f(dist, shuffled) == pytest.approx(pseudo_f(dist, groups))


def test_restricted_permutation_finds_what_free_permutation_buries(paired):
    """The point of the whole exercise: with a large block effect, a real but
    small within-block effect is invisible under free permutation and clear
    when the shuffling is restricted to the blocks. Same statistic, two nulls."""
    X, condition, block = paired
    dist = distance_matrix(X)

    free = permanova(dist, condition, n_permutations=999, random_state=0)
    restricted = permanova(dist, condition, strata=block, n_permutations=999, random_state=0)

    assert free.statistic == pytest.approx(restricted.statistic)
    assert free.p_value > 0.05
    assert restricted.p_value < 0.01
    assert restricted.null.mean() < free.null.mean()


def test_p_value_can_never_be_zero(one_dimensional):
    """The observed arrangement counts as one of the permutations, so p has a
    floor of 1/(n+1). A p of 0 would claim more than the shuffles support."""
    X, groups = one_dimensional
    n = 99
    result = permanova(distance_matrix(X), groups, n_permutations=n, random_state=0)

    reached = int((result.null >= result.statistic).sum())
    assert result.p_value == pytest.approx((reached + 1) / (n + 1))
    assert result.p_value >= 1 / (n + 1)


def test_strata_with_one_sample_each_is_rejected(one_dimensional):
    """Nothing can be permuted inside a stratum of size one, so the null would
    equal the observed value and p would always be 1/(n+1). Refuse instead."""
    X, groups = one_dimensional
    singletons = pd.Series(range(len(X)), index=X.index, name="singleton")
    with pytest.raises(ValueError, match="one sample"):
        permanova(distance_matrix(X), groups, strata=singletons, n_permutations=9)
