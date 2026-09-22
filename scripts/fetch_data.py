from pathlib import Path
from openpyxl import load_workbook
from hashlib import sha256
import pandas as pd
import gzip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILE = DATA / "GSE133296_counts.xlsx"
METADATA = DATA / "GSE133296_series_matrix.txt.gz"

# Original checksum of the excel file
CHECKSUM = "1ed17784e63406746694b5a4f249ca73a17a89d3dfe189a6724a8739acd68ed3"


def calculate_checksum(file_path):
    hasher = sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to repair the excel file header if there are duplicate column names. The original file contains a duplicate column that needs to be corrected.


def repair_excel_file(file_path):
    DUPLICATE = "2_om_met_Raw.Read.Count"
    CORRECTED = "3_om_met_Raw.Read.Count"

    wb = load_workbook(
        file_path,
        read_only=True,
        data_only=True,
    )

    try:
        header = [c.value for c in wb.active[1]]

    finally:
        wb.close()

    names = list(header)
    duplicate_positions = [
        index
        for index, name in enumerate(names)
        if name == DUPLICATE
    ]
    names[duplicate_positions[1]] = CORRECTED
    return names

# Function to extract and process the counts data from the excel file


def read_counts():
    # Load the excel file and create 3 dataframes: 1. counts raw, 2. counts normalized, 3. annotations
    names = repair_excel_file(FILE)
    assert len(names) == 63, "The number of columns in the header is not 63"
    df = pd.read_excel(FILE, header=0, names=names).set_index("Gene_ID")
    df_counts_raw = df.filter(regex=r"_Raw\.Read\.Count$")
    df_counts_normalized = df.filter(regex=r"_Normalized\.Read\.Count$")
    df_annotations = df.filter(regex=r"^(?!.*Read\.Count).*")

    # Take the suffix of the column names for counts dataframes
    df_counts_raw.columns = df_counts_raw.columns.str.removesuffix(
        "_Raw.Read.Count")
    df_counts_normalized.columns = df_counts_normalized.columns.str.removesuffix(
        "_Normalized.Read.Count")

    # Checks shapes, dtypes
    print(
        f"df_counts_raw shape: {df_counts_raw.shape}, dtypes: {df_counts_raw.dtypes}")
    print(
        f"df_counts_normalized shape: {df_counts_normalized.shape}, dtypes: {df_counts_normalized.dtypes}")
    print(
        f"df_annotations shape: {df_annotations.shape}, dtypes: {df_annotations.dtypes}")

    return df_counts_raw, df_counts_normalized, df_annotations


# Function to read and extract the sample metadata from the excel file
# What `organ site` must be, given the tissue the same record declares.
ORGAN_SITE_RULES = {
    "human primary ovarian cancer": lambda site: site == "ovary",
    "human omental metastasis": lambda site: site == "omentum",
    "human non-omental metastasis": lambda site: site not in ("omentum", "ovary"),
}


def read_sample_metadata():
    title = None
    organ_site = None
    tissue = None

    # rt: read text(str)
    with gzip.open(METADATA, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            values = [v.strip('"') for v in parts[1:]]
            if line.startswith("!Sample_title"):
                title = values
            elif line.startswith("!Sample_characteristics_ch1"):
                # '"organ site: left pelvic lymph node"'
                if "organ site:" in values[0]:
                    organ_site = [v.removeprefix(
                        "organ site:").strip() for v in values]
                elif "tissue:" in values[0]:
                    # GEO writes a double space in some of these values.
                    tissue = [" ".join(v.removeprefix("tissue:").split())
                              for v in values]

    # These would mean the file is not the one we think it is.
    assert title is not None, "!Sample_title not found"
    assert organ_site is not None, "no !Sample_characteristics_ch1 line with 'organ site:'"
    assert tissue is not None, "no !Sample_characteristics_ch1 line with 'tissue:'"
    assert len(title) == len(organ_site) == len(tissue) == 30, (
        f"expected 30 samples, got {len(title)}/{len(organ_site)}/{len(tissue)}"
    )

    # Title: <patient_id>_<site>, id [1-10], site [ov, om_met, met]
    sample_metadata = pd.DataFrame(
        {
            "patient_id": [int(t.split("_", 1)[0]) for t in title],
            "site": [t.split("_", 1)[1] for t in title],
            "organ_site": organ_site,
            "tissue": tissue,
        },
        index=pd.Index(title, name="sample"),
    )

    # Known defect: for patient 2, `organ site` contradicts the tissue declared in the record
    sample_metadata["organ_site_ok"] = [
        ORGAN_SITE_RULES[row.tissue](row.organ_site)
        for row in sample_metadata.itertuples()
    ]

    flagged = sample_metadata[~sample_metadata["organ_site_ok"]]
    if not flagged.empty:
        print(
            f"DEFECT: {len(flagged)} sample(s) whose organ site contradicts their tissue")
        for sample, row in flagged.iterrows():
            print(
                f"    {sample}: tissue '{row.tissue}' but organ site '{row.organ_site}'")
        print("    flagged, not repaired -- use the 'site' column for analysis.")

    return sample_metadata


def write_tables(df_counts_raw, df_counts_normalized, df_annotations, df_metadata):
    # Dictionary to store the processed dataframes
    dataframes = {
        "counts_raw": df_counts_raw,
        "counts_normalized": df_counts_normalized,
        "annotations": df_annotations,
        "metadata": df_metadata,
    }

    # Save the processed dataframes to separate TSV files
    for name, df in dataframes.items():
        df.to_csv(DATA / f"{name}.tsv.gz", sep="\t")
        # Check that the file was written accurately
        assert df.equals(pd.read_csv(DATA / f"{name}.tsv.gz", sep="\t",
                                     index_col="Gene_ID" if name != "metadata" else 0)), f"{name}.tsv.gz was not written accurately"


def main():
    if calculate_checksum(FILE) != CHECKSUM:
        raise ValueError("The excel file has been changed!")

    df_counts_raw, df_counts_normalized, df_annotations = read_counts()
    df_metadata = read_sample_metadata()
    write_tables(df_counts_raw, df_counts_normalized,
                 df_annotations, df_metadata)


if __name__ == "__main__":
    main()
