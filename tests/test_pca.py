# Tests for the PCA implementation in src/pcarna/pca.py
import pandas as pd
import numpy as np
import pytest

from pcarna import pca, PCAResult


@pytest.fixture
def wide_data():
    return pd.DataFrame(
        np.random.default_rng(24945).random((10, 30)),
        columns=[f"gene{i+1}" for i in range(30)],
        index=[f"sample{i+1}" for i in range(10)]
    )


@pytest.fixture
def tall_data():
    return pd.DataFrame(
        np.random.default_rng(24250).random((20, 5)),
        columns=[f"gene{i+1}" for i in range(5)],
        index=[f"sample{i+1}" for i in range(20)]
    )

def align_signs(a, b):
    """Flip each column of `a` so its sign matches the matching column of `b`.

    A PCA is only defined up to a sign per component, so comparing two
    implementations without this is meaningless. Comparing absolute values
    would also work but is weaker: it cannot tell a genuine sign disagreement
    inside a component from the arbitrary global one.
    """
    return a * np.sign((a * b).sum(axis=0))


def test_reconstruction_round_trip(wide_data):
    # Perform PCA
    result = pca(wide_data)

    # Reconstruct the original data from the PCA scores and loadings
    reconstructed = result.scores.dot(result.loadings.T) + result.mean.values

    # Check that the reconstructed data is close to the original data
    np.testing.assert_allclose(reconstructed, wide_data.values, rtol=1e-6)


def test_limit_components(wide_data, tall_data):
    # Attempt to perform PCA with more components than features
    with pytest.raises(ValueError):
        pca(wide_data, n_components=50)
    # Attempt to perform PCA with more components than samples
    with pytest.raises(ValueError):
        pca(tall_data, n_components=25)

# Compare with the mathematical definition of PCA


def test_pca_definition(wide_data):
    # Center the data
    X = wide_data.values
    X_centered = X - X.mean(axis=0)

    # Perform PCA
    result = pca(wide_data)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Only the first k eigenvectors are determined: the covariance matrix is
    # 30x30 but the centred data has rank 9, so the remaining 21 span a null
    # space where any orthonormal basis is equally valid.
    k = result.loadings.shape[1]
    assert (eigvals[:k] > 1e-10).all(), "test data lost rank"
    assert (eigvals[k:] < 1e-10).all(), "more non-zero eigenvalues than components"

    np.testing.assert_allclose(result.explained_variance.values, eigvals[:k], rtol=1e-6)
    np.testing.assert_allclose(
        align_signs(result.loadings.values, eigvecs[:, :k]), eigvecs[:, :k], atol=1e-8)

# Compare with scikit-learn's PCA implementation scikit has 10 components by default. We need to specify n=9


def test_sklearn_comparison(wide_data):
    from sklearn.decomposition import PCA as SKPCA

    # Perform PCA using our implementation
    result = pca(wide_data)

    # n_components is a constructor argument, not a fit_transform one. Left at
    # its default, sklearn returns min(n_samples, n_features) = 10 components,
    # one more than the centred data can support -- its variance is ~1e-32.
    k = result.loadings.shape[1]
    sk_pca = SKPCA(n_components=k)
    sk_scores = sk_pca.fit_transform(wide_data.values)
    sk_loadings = sk_pca.components_.T

    np.testing.assert_allclose(
        result.explained_variance_ratio.values, sk_pca.explained_variance_ratio_, rtol=1e-6)
    np.testing.assert_allclose(
        align_signs(result.loadings.values, sk_loadings), sk_loadings, atol=1e-8)
    np.testing.assert_allclose(
        align_signs(result.scores.values, sk_scores), sk_scores, atol=1e-8)

# Deterministic behavior test


def test_deterministic(wide_data):
    result1 = pca(wide_data)
    result2 = pca(wide_data)
    np.testing.assert_allclose(
        result1.scores.values, result2.scores.values, rtol=1e-6)
    np.testing.assert_allclose(
        result1.loadings.values, result2.loadings.values, rtol=1e-6)

# Check that the labels of the scores and loadings match the original data


def test_labels(wide_data):
    result = pca(wide_data)
    assert all(result.scores.index == wide_data.index)
    assert all(result.loadings.index == wide_data.columns)

# Eckart-Young theorem test


def test_eckart_young(wide_data):
    result = pca(wide_data)
    reconstructed = result.scores.dot(result.loadings.T) + result.mean.values
    np.testing.assert_allclose(reconstructed, wide_data.values, rtol=1e-6)
