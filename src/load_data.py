"""Phase 2, Step 1: load the raw dataset and take a first look."""

from pathlib import Path

import pandas as pd

# Show every column when printing, instead of hiding the middle ones with "..."
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Build the path to the data file relative to this script
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por.csv"

# This file separates values with semicolons (;) rather than commas,
# so we must tell pandas using sep=";"
df = pd.read_csv(data_path, sep=";")

print("Shape (rows, columns):", df.shape)
print()
print("Column names:")
print(list(df.columns))
print()
print("First 5 rows:")
print(df.head())