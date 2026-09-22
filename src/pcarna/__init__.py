"""PCA from first principles, applied to matched HGSOC RNA-seq (GSE133296)."""

from pcarna.distances import (
    GroupDistances,
    distance_matrix,
    group_distance_table,
    group_distances,
)
from pcarna.permanova import PermanovaResult, permanova, pseudo_f
from pcarna import figures, reports  # noqa: F401
from pcarna.pca import PCAResult, max_components, pca

__all__ = [
    "pca",
    "PCAResult",
    "max_components",
    "distance_matrix",
    "group_distances",
    "group_distance_table",
    "GroupDistances",
    "permanova",
    "PermanovaResult",
    "pseudo_f",
]
