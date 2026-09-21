# PCA from first principles — HGSOC primary tumours and metastases

> **What drives transcriptomic variation in high-grade serous ovarian carcinoma:
> patient identity or metastatic site?**

Principal component analysis implemented from scratch with NumPy and validated against
scikit-learn, applied to matched primary/metastasis RNA-seq from 10 HGSOC patients.

Companion article: at [alvaroesteban.dev](https://www.alvaroesteban.dev).

## The data

[GSE133296](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133296) — 30 RNA-seq
samples, 10 patients × 3 anatomical sites, collected at primary debulking surgery before
chemotherapy:

| Site | Label | n |
|------|-------|---|
| Primary ovarian tumour | `ov` | 10 |
| Omental metastasis | `om_met` | 10 |
| Non-omental intraperitoneal metastasis | `met` | 10 |

Fully balanced, no missing samples. Material is **FFPE**, macrodissected by a pathologist
against an H&E template. 57,773 genes. Associated publication: PMID
[32766252](https://pubmed.ncbi.nlm.nih.gov/32766252/).

The ten non-omental metastases come from ten *different* anatomical sites — pelvic lymph
node, small intestine mesentery, diaphragm, bladder peritoneum, sigmoid serosa, cul de
sac, and others. So "site" is not a clean three-level factor: it is ovary (n=10), omentum
(n=10), and ten singletons. `data/samples.tsv` records the exact site per sample.

## A defect in the published supplementary file

The supplementary workbook mislabels one column: in the **raw** count block the header
`2_om_met` appears twice and `3_om_met` is absent. The normalized block is correct.

`scripts/fetch_data.py` detects this, proves which sample the column actually holds, and
repairs the label out loud:

```
DEFECT: raw block duplicates ['2_om_met'] and is missing ['3_om_met']
REPAIR: column 27 is labelled '2_om_met' but holds '3_om_met' --
        constant-rescaling CV 1.2e-15 against '3_om_met' vs 1.1e+01 against the printed label
```

The proof is internal consistency. The authors' normalization is a single scalar per
sample, so dividing a normalized column by its true raw column gives a constant — a
coefficient of variation at machine precision (~1e-15). The mislabelled column matches
`3_om_met`'s normalized column at CV 1.2e-15 and its printed label at CV 11. Two
different samples being exactly proportional across 57,773 genes does not happen.

No samples are lost and the 10 × 3 design survives intact. But an analysis that trusts
the raw header as published silently analyses patient 2 twice and patient 3 never.

### A second, independent defect in the GEO metadata

For patient 2, GEO's `organ site` contradicts the tissue type declared in the same
record: `2_met` is a non-omental metastasis annotated "omentum", and `2_ov` is a primary
ovarian tumour annotated "small intestine mesentery". Neither can be true.

Unlike the count columns, there is **no internal evidence that identifies the intended
values**, so these are flagged rather than repaired — `data/samples.tsv` carries an
`organ_site_flag` column. The `site` field (ovary / omental / non-omental) comes from the
sample title and is corroborated independently by `source_name` and the `tissue`
characteristic, so it stays trustworthy and is the variable to analyse.

If the file is ever corrected upstream, the sha256 check fails and the script stops
rather than analysing something else quietly.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Exact versions are pinned in `uv.lock`.

```bash
uv sync --extra dev
uv run python scripts/fetch_data.py   # ~25 MB download, ~1 min parse
uv run pytest                          # validates the implementation vs scikit-learn
```

`fetch_data.py` writes into `data/` (gitignored): raw and normalized count matrices,
gene annotation, and a sample table with patient, site, organ site and library size.

## Layout

```
src/pcarna/pca.py      PCA via SVD, from first principles
scripts/fetch_data.py  download, verify, repair, cache
tests/test_pca.py      validation against scikit-learn
```

## Two things the implementation is careful about

**Component signs are arbitrary.** If `(u, v)` is a singular pair, so is `(-u, -v)` — same
decomposition, mirrored plot. Two correct implementations can disagree on sign for no
scientific reason, so `pca()` forces the largest-magnitude loading of each component
positive, making output reproducible across machines and reruns.

**Centred data of n samples has rank ≤ n−1.** With 30 samples, component 30 carries
numerical noise, not signal. `pca()` refuses to return it rather than inviting someone to
interpret a direction that does not exist — a live trap in RNA-seq, where features
outnumber samples by three orders of magnitude.

## License

MIT — see [LICENSE](LICENSE).

Source data is from GEO and remains subject to its original terms.
