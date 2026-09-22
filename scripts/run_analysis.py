import pandas as pd
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
RAW_COUNTS_FILE = DATA / "counts_raw.tsv.gz"
ANNOTATIONS_FILE = DATA / "annotations.tsv.gz"

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
