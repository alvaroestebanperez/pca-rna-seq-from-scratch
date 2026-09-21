from pathlib import Path
from openpyxl import load_workbook
from hashlib import sha256
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILE = DATA / "GSE133296_counts.xlsx"

# Check that the excel hasn't been changed
CHECKSUM = "1ed17784e63406746694b5a4f249ca73a17a89d3dfe189a6724a8739acd68ed3"


def calculate_checksum(file_path):
    hasher = sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


if calculate_checksum(FILE) != CHECKSUM:
    raise ValueError("The excel file has been changed!")

# We need to repair the excel file. 2_om_met_Raw.Read.Count is repeted twice. Exploring the header I realiced that the second occurrence should be renamed to 3_om_met_Raw.Read.Count.

wb = load_workbook(
    FILE,
    read_only=True,
    data_only=True,
)

try:
    header = [c.value for c in wb.active[1]]

finally:
    wb.close()

names = list(header)
dup = [name for name in names if names.count(name) > 1]
if dup:
    print(f"Duplicate column names found: {dup}")
    # Rename the second occurrence of the duplicate column
    for i, name in enumerate(names):
        if name == dup[0] and i != names.index(dup[0]):
            names[i] = "3_om_met_Raw.Read.Count"

print(f"Updated column names: {names}")

# Load the excel file and create 3 dataframes: 1. counts raw, 2. counts normalized, 3. annotations

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

# Dictionary to store the processed dataframes
dataframes = {
    "counts_raw": df_counts_raw,
    "counts_normalized": df_counts_normalized,
    "annotations": df_annotations,
}

# Save the processed dataframes to separate TSV files
for name, df in dataframes.items():
    df.to_csv(DATA / f"{name}.tsv.gz", sep="\t")
    # Check that the file was written accurately
    assert df.equals(pd.read_csv(DATA / f"{name}.tsv.gz", sep="\t",
                     index_col="Gene_ID")), f"{name}.tsv.gz was not written accurately"
