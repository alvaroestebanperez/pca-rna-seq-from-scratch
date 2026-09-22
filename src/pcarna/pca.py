import pandas as pd
import numpy as np
from dataclasses import dataclass

# 1. Define a dataclass to store PCA results.


@dataclass(frozen=True)
class PCAResult:
    '''
    Dataclass to store the results of a Principal Component Analysis (PCA).

    Attributes:
        scores: (n_samples, n_components) - samples in PC space.
        loadings: (n_features, n_components) - principal component loadings (eigenvectors).
        explained_variance: (n_components) - variance explained by each principal component.
        explained_variance_ratio: (n_components) - proportion of variance explained by each principal component.
        mean: (n_features) - mean of each feature in the original data.
    '''
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.Series
    explained_variance_ratio: pd.Series
    mean: pd.Series

# 2. Function to calculate max_components (maximum number of principal components)


def max_components(n_samples: int, n_features: int) -> int:
    """
    Calculate the maximum number of principal components for PCA. n_samples - 1 because when we center the data we consume 1 degree of freedom.

    Args:
        n_samples: Number of samples in the dataset.
        n_features: Number of features in the dataset.

    Returns:
        Maximum number of principal components.
    """
    return max(0, min(n_samples - 1, n_features))

# 3. Perform PCA using SVD and return results as a PCAResult dataclass.


def pca(X: pd.DataFrame, n_components: int | None = None) -> PCAResult:
    """
    Perform Principal Component Analysis (PCA) on the given dataset.

    Args:
        X: Input data as a pandas DataFrame.
        n_components: Number of principal components to compute. If None, use the maximum possible.

    Returns:
        PCAResult containing scores, loadings, explained variance, explained variance ratio, and mean.
    """
    if X is None or X.empty:
        raise ValueError("X must be a non-empty DataFrame")

    # Checked before isnull/isinf: those assume numeric data, and a text column
    # makes np.isinf fail with a TypeError that never names the culprit.
    non_numeric = X.columns[~X.dtypes.map(pd.api.types.is_numeric_dtype)]
    if len(non_numeric) > 0:
        raise ValueError(
            f"X must be entirely numeric; {len(non_numeric)} column(s) are not, "
            f"first: {list(non_numeric[:5])}"
        )

    n_missing = int(X.isna().to_numpy().sum())
    if n_missing:
        raise ValueError(f"X must not contain NaN; found {n_missing}")

    n_infinite = int((~np.isfinite(X.to_numpy())).sum())
    if n_infinite:
        raise ValueError(
            f"X must not contain infinite values; found {n_infinite}")

    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError(
            f"need at least 2 samples to decompose, got {n_samples}")

    limit = max_components(n_samples, n_features)
    if n_components is None:
        n_components = limit
    elif not 1 <= n_components <= limit:
        raise ValueError(
            f"n_components must be between 1 and {limit} (centred data of "
            f"{n_samples} samples and {n_features} features has rank at most "
            f"{limit}), got {n_components}"
        )

    # Center the data
    mean = X.mean(axis=0)  # Mean of each gene

    # Every gene expression value is centered by subtracting the mean of that gene
    X_centered = X - mean

    # SVD (singular value decomposition)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    variance_all = s**2 / (n_samples - 1)
    total_variance = variance_all.sum()
    # Select the top n_components
    U = U[:, :n_components]
    s = s[:n_components]
    Vt = Vt[:n_components, :]

    components_names = [
        f"PC{i+1}"
        for i in range(n_components)
    ]

    scores = pd.DataFrame(
        U * s,
        index=X.index,
        columns=components_names
    )

    loadings = pd.DataFrame(
        Vt.T,
        index=X.columns,
        columns=components_names
    )

    explained_variance = pd.Series(
        variance_all[:n_components],
        index=components_names
    )
    explained_variance_ratio = pd.Series(
        explained_variance / total_variance
    )

    for comp in loadings.columns:
        dominant = loadings[comp].abs().idxmax()
        if loadings.loc[dominant, comp] < 0:
            loadings[comp] *= -1
            scores[comp] *= -1

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean
    )
