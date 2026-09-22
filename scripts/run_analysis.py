"""Analysis of GSE133296: five preprocessing stages, a PCA on each, and the
question the note is named after -- does transcriptomic variation follow the
patient or the anatomical site?

Reads the tables cached by `fetch_data.py`; writes nothing yet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path

from pcarna import PCAResult, distance_matrix, pca

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE = ROOT / "reference"
RAW_COUNTS_FILE = DATA / "counts_raw.tsv.gz"
ANNOTATIONS_FILE = DATA / "annotations.tsv.gz"
METADATA_FILE = DATA / "metadata.tsv.gz"
MES_SIGNATURE_FILE = REFERENCE / "mes_signature.tsv"

N_SAMPLES = 30
N_GENES = 57773
MIN_CPM = 1
MIN_SAMPLES = 10          # the smallest group size: a gene expressed at one
                          # site only must survive, since that is the biology
N_VARIABLE_GENES = 500
SITE_PAIRS = [("ov", "om_met"), ("ov", "met"), ("om_met", "met")]


@dataclass(frozen=True)
class Stages:
    """The five preprocessing stages, plus what later steps need from the way
    they were built.

    matrices: stage name -> samples x genes frame. Every stage has 30 rows;
        `raw` has all genes, the rest only those passing the CPM filter.
    library_sizes: total raw reads per sample, from the unfiltered matrix --
        the depth PC1 is expected to track before normalisation.
    kept_genes: the genes surviving the filter.
    gene_variance: variance of each gene on the log2-CPM scale, which is what
        the top-N selection ranks on and what the sensitivity figure needs.
    """

    matrices: dict[str, pd.DataFrame]
    library_sizes: pd.Series
    kept_genes: pd.Index
    gene_variance: pd.Series


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Counts (samples x genes), gene annotation, and sample metadata.

    Counts are stored genes-in-rows, the convention for count matrices, and
    transposed here because every downstream step wants samples in rows.
    Metadata is reindexed to the count matrix so that anything positional --
    colours passed to matplotlib, for instance -- cannot go out of step.
    """
    counts = pd.read_csv(
        RAW_COUNTS_FILE, sep="\t", index_col="Gene_ID", dtype={"Gene_ID": str}
    ).T
    if counts.shape != (N_SAMPLES, N_GENES):
        raise ValueError(f"expected {(N_SAMPLES, N_GENES)} counts, got {counts.shape}")

    annotations = pd.read_csv(ANNOTATIONS_FILE, sep="\t", index_col="Gene_ID", dtype=str)

    metadata = pd.read_csv(
        METADATA_FILE, sep="\t", index_col="sample", dtype={"patient_id": str}
    ).reindex(counts.index)
    if metadata.isna().to_numpy().any():
        raise ValueError("metadata does not cover every sample in the count matrix")

    return counts, annotations, metadata


def build_stages(counts: pd.DataFrame) -> Stages:
    """Raw -> filtered -> CPM -> log2-CPM -> most variable genes.

    Library sizes come from the *unfiltered* matrix: how deeply a sample was
    sequenced is a property of the experiment, not of which genes we chose to
    keep. Recomputing them after filtering would make the denominator depend on
    the threshold, so two analyses with different filters would stop being
    comparable.
    """
    library_sizes = counts.sum(axis=1)
    cpm = counts.div(library_sizes, axis=0) * 1e6
    keep = (cpm >= MIN_CPM).sum(axis=0) >= MIN_SAMPLES
    kept_genes = counts.columns[keep]

    counts_filtered = counts[kept_genes]
    cpm_filtered = cpm.loc[:, keep]
    # +1 is not neutral: it compresses low values far more than high ones,
    # which is part of why the geometry moves between stages 3 and 4.
    log2_cpm = np.log2(cpm_filtered + 1)

    # Ranked on the log scale, never on CPM: raw variance is a function of
    # expression level, so ranking on CPM re-selects the most abundant genes.
    gene_variance = log2_cpm.var(axis=0)
    top_genes = gene_variance.nlargest(N_VARIABLE_GENES).index

    # Shape alone cannot tell counts from CPM -- both are (30, n_kept) -- so
    # the check has to be on the values.
    if not (cpm_filtered.sum(axis=1) <= 1e6).all():
        raise ValueError("cpm_filtered does not look like CPM: rows exceed 1e6")

    return Stages(
        matrices={
            "raw": counts,
            "filtered": counts_filtered,
            "cpm": cpm_filtered,
            "log2_cpm": log2_cpm,
            "variable": log2_cpm[top_genes],
        },
        library_sizes=library_sizes,
        kept_genes=kept_genes,
        gene_variance=gene_variance,
    )


def run_pca(stages: Stages) -> dict[str, PCAResult]:
    return {name: pca(matrix) for name, matrix in stages.matrices.items()}


def report_filtering(stages: Stages, counts: pd.DataFrame) -> None:
    kept = stages.matrices["filtered"]
    retained = kept.to_numpy().sum() / counts.to_numpy().sum() * 100
    per_sample = kept.sum(axis=1) / stages.library_sizes * 100
    print(
        f"{len(stages.kept_genes)} / {counts.shape[1]} genes retained "
        f"({retained:.1f}% of reads; {per_sample.min():.1f}-{per_sample.max():.1f}% per sample)"
    )


def report_variance(results: dict[str, PCAResult], library_sizes: pd.Series) -> None:
    """Explained variance per stage, and how far PC1/PC2 track sequencing depth."""
    print(
        f"\n{'stage':<10}{'PC1':>7}{'PC2':>7}{'PC3':>7}{'PC4':>7}{'PC5':>7}"
        f"{'|r| PC1':>10}{'|r| PC2':>9}"
    )
    for stage, result in results.items():
        ratios = "".join(f"{v * 100:6.1f}%" for v in result.explained_variance_ratio.iloc[:5])
        r1 = abs(result.scores["PC1"].corr(library_sizes))
        r2 = abs(result.scores["PC2"].corr(library_sizes))
        print(f"{stage:<10}{ratios}{r1:>10.3f}{r2:>9.3f}")


def within_patient_site_distances(
    scores: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """One row per patient, one column per pair of sites.

    The global site comparison is confounded: patients differ far more than
    sites do, so a large patient effect masks a real site one. With a paired
    design the interpretable question is whether the sites separate
    consistently *inside* each patient.
    """
    distances = distance_matrix(scores)
    patients = sorted(metadata["patient_id"].unique(), key=int)
    return pd.DataFrame(
        {
            f"{a}-{b}": [distances.loc[f"{p}_{a}", f"{p}_{b}"] for p in patients]
            for a, b in SITE_PAIRS
        },
        index=pd.Index(patients, name="patient"),
    )


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
        print(f"  {pair:<12} largest {largest.get(pair, 0):>2}   smallest {smallest.get(pair, 0):>2}")


def report_ranking_scale_effect(stages: Stages) -> None:
    """Why variable genes are ranked on the log scale, shown rather than argued.

    RNA-seq variance is a function of expression level, so ranking on CPM
    selects the most abundant genes rather than the most variable ones. The
    two rankings turn out to be almost disjoint.
    """
    cpm = stages.matrices["cpm"]
    log2 = stages.matrices["log2_cpm"]
    by_cpm = cpm.var(axis=0).nlargest(N_VARIABLE_GENES).index
    by_log2 = stages.gene_variance.nlargest(N_VARIABLE_GENES).index
    mean_cpm = cpm.mean(axis=0)

    print(
        f"\nTop {N_VARIABLE_GENES} most variable genes, ranked on CPM vs on log2-CPM:"
        f"\n  shared by both rankings: {len(set(by_cpm) & set(by_log2))} / {N_VARIABLE_GENES}"
        f"\n  median expression (CPM) -- ranked on CPM: {mean_cpm[by_cpm].median():.1f}"
        f"  |  ranked on log2: {mean_cpm[by_log2].median():.1f}"
        f"  |  all kept genes: {mean_cpm.median():.1f}"
    )


def main() -> None:
    counts, annotations, metadata = load_tables()
    stages = build_stages(counts)
    results = run_pca(stages)

    report_filtering(stages, counts)
    report_ranking_scale_effect(stages)
    report_variance(results, stages.library_sizes)
    report_within_patient_sites(
        within_patient_site_distances(results["variable"].scores, metadata)
    )


if __name__ == "__main__":
    main()
