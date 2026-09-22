"""PCA from first principles, applied to matched HGSOC RNA-seq (GSE133296)."""

from pcarna.pca import PCAResult, max_components, pca

__all__ = ["pca", "PCAResult", "max_components"]
