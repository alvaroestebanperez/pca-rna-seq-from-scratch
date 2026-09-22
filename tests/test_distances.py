"""Checks for pairwise distances and how they split by group."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from pcarna import distance_matrix, group_distance_table, group_distances, pca


@pytest.fixture
def points():
    rng = np.random.default_rng(20260922)
    return pd.DataFrame(
        rng.normal(size=(12, 5)),
        index=pd.Index([f"s{i:02d}" for i in range(12)], name="sample"),
        columns=[f"g{i}" for i in range(5)],
    )


def test_matches_scipy(points):
    """An independent implementation of the same definition."""
    np.testing.assert_allclose(
        distance_matrix(points).to_numpy(), squareform(pdist(points.to_numpy())), atol=1e-12
    )


def test_is_a_metric_and_keeps_labels(points):
    """Symmetry, a zero diagonal and non-negativity define a distance matrix;
    the labels on both axes are what lets callers group by name later."""
    dist = distance_matrix(points)
    values = dist.to_numpy()
    np.testing.assert_allclose(values, values.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(values), 0, atol=1e-12)
    assert (values >= 0).all()
    assert dist.index.equals(points.index) and dist.columns.equals(points.index)


def test_pca_rotation_preserves_distances(points):
    """A PCA over all components is a rotation, so distances in component space
    equal distances in the original feature space. This is what justifies doing
    the whole analysis on scores rather than on genes."""
    scores = pca(points).scores
    np.testing.assert_allclose(
        distance_matrix(scores).to_numpy(), distance_matrix(points).to_numpy(), atol=1e-10
    )


def test_within_group_mean_excludes_self_distances():
    """The diagonal is a sample's distance to itself: zero, and not a distance
    "within" anything. Including it deflates the within-group mean by n zeros
    among n^2 entries -- a bias that grows as groups get smaller, and so would
    favour whichever factor has the smaller groups."""
    # Two groups of 2, every pair exactly 10 apart inside a group.
    index = pd.Index(["a1", "a2", "b1", "b2"], name="sample")
    values = np.array(
        [[0, 10, 30, 30], [10, 0, 30, 30], [30, 30, 0, 10], [30, 30, 10, 0]], dtype=float
    )
    dist = pd.DataFrame(values, index=index, columns=index)
    groups = pd.Series(["a", "a", "b", "b"], index=index, name="grp")

    result = group_distances(dist, groups)
    assert result.within == pytest.approx(10.0)  # 5.0 if the diagonal leaked in
    assert result.between == pytest.approx(30.0)
    assert result.ratio == pytest.approx(3.0)
    assert result.n_within_pairs == 2 and result.n_between_pairs == 4

    table = group_distance_table(dist, groups)
    assert table.loc["a", "a"] == pytest.approx(10.0)
    assert table.loc["a", "b"] == pytest.approx(30.0)


def test_group_labels_align_by_name(points):
    """Reordering the label Series must not change the answer."""
    dist = distance_matrix(points)
    groups = pd.Series(["x", "y"] * 6, index=points.index, name="grp")
    a = group_distances(dist, groups)
    b = group_distances(dist, groups.sample(frac=1, random_state=1))
    assert a.within == pytest.approx(b.within) and a.between == pytest.approx(b.between)


def test_missing_label_is_rejected(points):
    dist = distance_matrix(points)
    groups = pd.Series(["x"] * 11, index=points.index[:11], name="grp")
    with pytest.raises(ValueError, match="no group label"):
        group_distances(dist, groups)
