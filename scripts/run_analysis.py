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
from pcarna.figures import write_all
from pcarna.reports import (
    report_filtering,
    report_group_distances,
    report_permanova,
    report_ranking_scale_effect,
    report_signature,
    report_variance,
    report_within_patient_sites,
)

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
CENTRING_GENES = ("POSTN", "COL11A1")   # also the pair used in the animation
N_PERMUTATIONS = 9999
RANDOM_STATE = 20260922
FIGURES = ROOT / "figures"


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
        raise ValueError(
            f"expected {(N_SAMPLES, N_GENES)} counts, got {counts.shape}")

    annotations = pd.read_csv(
        ANNOTATIONS_FILE, sep="\t", index_col="Gene_ID", dtype=str)

    metadata = pd.read_csv(
        METADATA_FILE, sep="\t", index_col="sample", dtype={"patient_id": str}
    ).reindex(counts.index)
    if metadata.isna().to_numpy().any():
        raise ValueError(
            "metadata does not cover every sample in the count matrix")

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
        raise ValueError(
            "cpm_filtered does not look like CPM: rows exceed 1e6")

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


def map_signature(annotations: pd.DataFrame, stages: Stages) -> pd.Series:
    """Map the Mes signature's gene symbols onto the matrix's Ensembl IDs.

    The translation uses the annotation that shipped inside the authors' own
    supplementary workbook, which is the GRCh37/hg19 build their pipeline
    aligned to. Fetching a current annotation instead would map symbols against
    a different genome -- symbols get renamed, reassigned and merged over time.

    Returns a Series mapping Ensembl ID -> symbol. The ID is the stable key and
    what the matrices are indexed by; the symbol is a presentation label, so
    the caller renames only when it is about to plot.
    """
    symbols = pd.read_csv(MES_SIGNATURE_FILE, sep="\t", comment="#")["gene_symbol"]
    mapped = annotations.loc[annotations["Gene_Name"].isin(symbols), "Gene_Name"]

    missing = sorted(set(symbols) - set(mapped))
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(symbols)} signature genes are absent from the "
            f"annotation: {missing}"
        )

    # A symbol resolving to several Ensembl IDs (paralogues, annotated
    # pseudogenes) would silently weight that gene more than the others.
    duplicated = mapped.value_counts()
    duplicated = duplicated[duplicated > 1]
    if not duplicated.empty:
        raise ValueError(
            f"{len(duplicated)} symbol(s) map to more than one Ensembl ID: "
            f"{duplicated.to_dict()}"
        )

    # The score is computed on the filtered matrices, so a signature gene that
    # failed the CPM filter would break selection with a KeyError that never
    # mentions the signature.
    dropped = mapped.index.difference(stages.kept_genes)
    if len(dropped) > 0:
        raise ValueError(
            f"{len(dropped)} signature gene(s) did not survive the "
            f"CPM>={MIN_CPM} in >={MIN_SAMPLES} filter: "
            f"{mapped.loc[dropped].tolist()}"
        )

    return mapped
def main() -> None:
    counts, annotations, metadata = load_tables()
    stages = build_stages(counts)
    results = run_pca(stages)

    signature = map_signature(annotations, stages)
    scores = results["variable"].scores

    report_filtering(stages, counts)
    report_ranking_scale_effect(stages)
    report_variance(results, stages.library_sizes)
    report_group_distances(scores, metadata)
    report_within_patient_sites(within_patient_site_distances(scores, metadata))
    report_permanova(scores, metadata,
                     n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE)
    signature_scores = report_signature(stages, annotations, results, signature, metadata)

    written = write_all(stages, results, metadata, annotations, signature,
                        signature_scores, FIGURES, CENTRING_GENES)
    print(f"\nwrote {len(written)} figure files to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
