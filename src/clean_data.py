"""Phase 2, Step 5: create a cleaned copy of the dataset.

The raw file is never modified. The cleaned copy is saved separately.
"""

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
raw_path = project_root / "data" / "student-por.csv"
clean_path = project_root / "data" / "student-por-clean.csv"

df = pd.read_csv(raw_path, sep=";")
rows_before, cols_before = df.shape

# 1. Remove accidental leading/trailing spaces in text columns.
#    Our checks found no problems, so this is a safe precaution.
text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    df[col] = df[col].str.strip()

# 2. Flag students whose final grade is 0 (1 = yes, 0 = no).
#    We keep these rows, but the flag lets us analyse the data with and
#    without them later.
df["g3_zero"] = (df["G3"] == 0).astype(int)

# 3. Save the cleaned copy. index=False stops pandas writing an extra
#    row-number column. The cleaned file uses commas, not semicolons.
df.to_csv(clean_path, index=False)

print("Raw shape:    ", (rows_before, cols_before))
print("Cleaned shape:", df.shape)
print("Rows removed: ", rows_before - df.shape[0])
print()
print("g3_zero counts (0 = normal, 1 = final grade of zero):")
print(df["g3_zero"].value_counts())
print()
print("Saved to:", clean_path)