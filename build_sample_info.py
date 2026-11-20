# Build sample_info.csv for GSE78220 (replaces the Step 3 linking code).
# Key fix: link GEO to the expression columns by the GEO sample TITLE (Pt1, Pt16, Pt27A, Pt27B ...),
# not by the "patient id" characteristic, which does not distinguish Pt27A from Pt27B.

# %% 0. Settings
import gzip
import os

import pandas as pd

DATA_DIR = os.environ.get(
    "DATA_DIR", "/Users/dteen/Desktop/All Folders/School/Applied AI/Group Project/Data"
)
expr_path = os.path.join(DATA_DIR, "GSE78220_PatientFPKM.xlsx")
matrix_path = os.path.join(DATA_DIR, "GSE78220_series_matrix.txt.gz")
out_path = os.path.join(DATA_DIR, "sample_info.csv")

# Show everything: no row, column or cell-width limits when printing tables.
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# %% 1. Expression column names (this is the sample list the models will use)
expr_cols = list(pd.read_excel(expr_path, nrows=0).columns)
sample_cols = expr_cols[1:]                       # first column is 'Gene'
print("Expression samples:", len(sample_cols))

# %% 2. Read the GEO series matrix metadata lines
meta = {}
with gzip.open(matrix_path, "rt") as f:
    for line in f:
        if line.startswith("!Sample_"):
            parts = line.rstrip("\n").split("\t")
            key = parts[0].replace("!Sample_", "")
            meta.setdefault(key, []).append([v.strip('"') for v in parts[1:]])
n = len(meta["geo_accession"][0])
print("GEO samples:", n)

# %% 3. One row per GEO sample; characteristics are read by their 'key: value' text
rows = []
for i in range(n):
    d = {"geo_accession": meta["geo_accession"][0][i], "title": meta["title"][0][i]}
    for line in meta["characteristics_ch1"]:
        v = line[i]
        if ": " in v:
            key, val = v.split(": ", 1)
            key = key.strip()
            d[key] = f"{d[key]}; {val}" if key in d else val
    rows.append(d)
geo = pd.DataFrame(rows)
print("GEO columns:", geo.columns.tolist())

# %% 4. Link by title: expression column 'Pt27A.baseline' -> title 'Pt27A'
expr_ids = pd.DataFrame({"sample_id": sample_cols})
expr_ids["patient_id"] = expr_ids["sample_id"].str.split(".").str[0]          # Pt27A
expr_ids["timing_from_name"] = expr_ids["sample_id"].str.split(".").str[1]    # baseline / OnTx
expr_ids["patient_base"] = expr_ids["patient_id"].str.replace(r"(?<=\d)[A-Za-z]$", "", regex=True)  # Pt27

geo_small = geo.rename(columns={
    "title": "patient_id",
    "biopsy time": "timing_geo",
    "anti-pd-1 response": "response_original",
    "patient id": "geo_patient_id_field",      # kept only to document the Pt27 discrepancy
})[["geo_accession", "patient_id", "timing_geo", "response_original", "geo_patient_id_field"]]

merged = expr_ids.merge(geo_small, on="patient_id", how="outer", indicator=True, validate="one_to_one")
print("\nMerge result (must be all 'both'):")
print(merged["_merge"].value_counts())
if (merged["_merge"] != "both").any():
    print(merged[merged["_merge"] != "both"])
    raise SystemExit("Some samples did not link. Fix this before using sample_info.csv.")

# %% 5. Final table and checks
sample_info = merged[[
    "sample_id", "geo_accession", "patient_id", "patient_base",
    "timing_from_name", "timing_geo", "response_original", "geo_patient_id_field",
]].copy()

print(sample_info.to_string())
print("\nRows:", len(sample_info), "| unique sample_id:", sample_info["sample_id"].nunique(),
      "| unique patients:", sample_info["patient_base"].nunique())
print("\nMissing values per column:\n", sample_info.isna().sum())
print("\nOriginal response categories:\n", sample_info["response_original"].value_counts(dropna=False))
print("\nTiming (name vs GEO):\n", pd.crosstab(sample_info["timing_from_name"],
                                               sample_info["timing_geo"].fillna("MISSING")))
print("\nWhere GEO's 'patient id' field differs from the title:")
print(sample_info[sample_info["patient_id"] != sample_info["geo_patient_id_field"]]
      [["sample_id", "geo_accession", "patient_id", "geo_patient_id_field"]])

# %% 6. Save
sample_info.to_csv(out_path, index=False)
print("\nSaved:", out_path)
