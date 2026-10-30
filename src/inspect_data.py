"""Phase 2, Step 2: inspect data types and check for missing values."""

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por.csv"
df = pd.read_csv(data_path, sep=";")

# 1. Data types: is each column stored as numbers or as text?
print("=== DATA TYPES ===")
print(df.dtypes)
print()

# 2. How many columns of each type?
print("=== COUNT OF COLUMNS PER TYPE ===")
print(df.dtypes.value_counts())
print()

# 3. Missing values: count of empty cells in each column
print("=== MISSING VALUES PER COLUMN ===")
print(df.isnull().sum())
print()
print("Total missing values in the whole dataset:", df.isnull().sum().sum())