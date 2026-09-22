import pandas as pd
from pathlib import Path
import numpy as np
from pcarna import distance_matrix, pca, PCAResult

DATA = Path(__file__).resolve().parents[1] / "data"
RAW_COUNTS_FILE = DATA / "counts_raw.tsv.gz"
ANNOTATIONS_FILE = DATA / "annotations.tsv.gz"
METADATA_FILE = DATA / "metadata.tsv.gz"

# Raw data: tcounts_all
# Transpose filtered: counts_filtered
# CPM filtered: cpm_filtered
# log2-CPM: log2_cpm
# Variable genes: log2_cpm_variable

counts_all = pd.read_csv(RAW_COUNTS_FILE, sep="\t",
                         index_col="Gene_ID", dtype={"Gene_ID": str})
annotations = pd.read_csv(ANNOTATIONS_FILE, sep="\t",
                          index_col="Gene_ID", dtype=str)

# Counts: Gene_ID x samples -> samples x Gene_ID after transpose
# Annotations: Gene_ID x attributes
tcounts_all = counts_all.T
assert tcounts_all.shape == (30, 57773)

# Filter out genes: we want to keep genes with at least 1 CPM in at least 10 samples
library_sizes = tcounts_all.sum(axis=1)
cpm = tcounts_all.div(library_sizes, axis=0) * 1e6
keep = (cpm >= 1).sum(axis=0) >= 10
filtered_genes = tcounts_all.columns[keep]

counts_filtered = tcounts_all[filtered_genes]
cpm_filtered = cpm.loc[:, keep]

n_kept = int(keep.sum())

# Structure
assert counts_filtered.shape == (30, n_kept)

retained = counts_filtered.sum().sum() / counts_all.sum().sum() * 100
per_sample = counts_filtered.sum(axis=1) / library_sizes * 100
print(f"{n_kept} / {tcounts_all.shape[1]} genes retained "
      f"({retained:.1f}% of reads; {per_sample.min():.1f}-{per_sample.max():.1f}% per sample)")

# log2-CPM (we need to add 1 to avoid log2(0))
log2_cpm = np.log2(cpm_filtered + 1)

# Variable genes: genes with high variance across samples
gene_variance_log2 = log2_cpm.var(axis=0)
top_genes = gene_variance_log2.nlargest(500).index
log2_cpm_variable_top500 = log2_cpm.loc[:, top_genes]

gene_variance_cpm = cpm_filtered.var(axis=0)
top_genes_cpm = gene_variance_cpm.nlargest(500).index
cpm_filtered_variable_top500 = cpm_filtered.loc[:, top_genes_cpm]

# Overlapping top 500 variable genes between log2-CPM and CPM
overlap_top500 = len(set(top_genes) &
                     set(top_genes_cpm))
print(f"Number of overlapping top 500 variable genes: {overlap_top500}")

# If we order by variance, the genes with higher cpms will weigh more in the selection even if they vary less in relative terms

stages = {
    "raw": tcounts_all,
    "filtered": counts_filtered,
    "cpm": cpm_filtered,
    "log2_cpm": log2_cpm,
    "variable": log2_cpm_variable_top500,
}

results = {stage: pca(data) for stage, data in stages.items()}

# Explained variance per stage, and how much PC1/PC2 track sequencing depth.
header = (f"{'stage':<10}{'PC1':>7}{'PC2':>7}{'PC3':>7}{'PC4':>7}{'PC5':>7}"
          f"{'|r| PC1':>10}{'|r| PC2':>9}")
print(header)
for stage, result in results.items():
    ratios = "".join(
        f"{v * 100:6.1f}%" for v in result.explained_variance_ratio.iloc[:5])
    r1 = abs(result.scores["PC1"].corr(library_sizes))
    r2 = abs(result.scores["PC2"].corr(library_sizes))
    print(f"{stage:<10}{ratios}{r1:>10.3f}{r2:>9.3f}")

# Load Metadata
metadata = pd.read_csv(METADATA_FILE, sep="\t",
                       index_col="sample", dtype={"patient_id": str})
metadata = metadata.reindex(counts_filtered.index)
assert metadata.index.equals(counts_filtered.index)


# --- Site, assessed within patient -------------------------------------------
# The global site comparison is confounded: patients differ far more than sites
# (ratio 1.585 vs 1.005), so a large patient effect can mask a real site effect.
# With a paired design the interpretable question is whether the three sites
# separate consistently *inside* each patient.
SITE_PAIRS = [("ov", "om_met"), ("ov", "met"), ("om_met", "met")]

sample_distances = distance_matrix(results["variable"].scores)
patients = sorted(metadata["patient_id"].unique(), key=int)

pair_distances = pd.DataFrame(
    {
        f"{a}-{b}": [sample_distances.loc[f"{p}_{a}", f"{p}_{b}"] for p in patients]
        for a, b in SITE_PAIRS
    },
    index=pd.Index(patients, name="patient"),
)

print("\nWithin-patient distance between each pair of sites (variable stage):")
print(pair_distances.round(1).to_string())

# Rank inside each patient rather than averaging across them: patients differ in
# overall spread, so raw column means would mix that variation with the site
# effect they are meant to measure.
largest = pair_distances.idxmax(axis=1).value_counts()
smallest = pair_distances.idxmin(axis=1).value_counts()
expected = len(patients) / len(SITE_PAIRS)

print(f"\nWhich pair is largest / smallest within a patient "
      f"(n={len(patients)}, {expected:.1f} expected each if site has no effect):")
for pair in (f"{a}-{b}" for a, b in SITE_PAIRS):
    print(f"  {pair:<12} largest {largest.get(pair, 0):>2}   smallest {smallest.get(pair, 0):>2}")
