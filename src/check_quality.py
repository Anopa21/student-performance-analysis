"""Phase 2, Step 3: check duplicates and look for invalid or inconsistent values."""

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por.csv"
df = pd.read_csv(data_path, sep=";")

# 1. Duplicate rows: rows identical in every column
print("=== DUPLICATE ROWS ===")
print("Number of fully duplicated rows:", df.duplicated().sum())
print()

# 2. Numeric columns: min and max reveal impossible values
#    (e.g. a negative age or a grade above 20)
print("=== NUMERIC COLUMNS: RANGES ===")
print(df.describe().loc[["min", "max", "mean"]].T)
print()

# 3. Text columns: list every distinct value, to spot typos or odd labels
print("=== TEXT COLUMNS: DISTINCT VALUES ===")
text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    print(f"{col}: {sorted(df[col].unique())}")
print()

# 4. Grades of zero: investigate the G1 = 0 row we noticed earlier
print("=== GRADES EQUAL TO ZERO ===")
for grade in ["G1", "G2", "G3"]:
    print(f"{grade} == 0: {(df[grade] == 0).sum()} students")
print()
print("Students with G3 == 0 (first 10 rows shown):")
print(df.loc[df["G3"] == 0, ["school", "age", "failures", "absences", "G1", "G2", "G3"]].head(10))