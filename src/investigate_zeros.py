"""Phase 2, Step 4: investigate students with grades of zero."""

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por.csv"
df = pd.read_csv(data_path, sep=";")

cols = ["school", "age", "failures", "absences", "G1", "G2", "G3"]

# 1. All students whose final grade is zero (not just the first 10)
zero_g3 = df[df["G3"] == 0]
print("=== ALL STUDENTS WITH G3 == 0 ===")
print(zero_g3[cols])
print()

# 2. Which school are they from, compared with the whole dataset?
print("=== SCHOOL: G3 == 0 STUDENTS vs EVERYONE ===")
print("G3 == 0 students:")
print(zero_g3["school"].value_counts())
print("All students:")
print(df["school"].value_counts())
print()

# 3. Is absences == 0 unusual among them?
rest = df[df["G3"] != 0]
print("=== SHARE OF STUDENTS WITH absences == 0 ===")
print(f"Among G3 == 0 students: {(zero_g3['absences'] == 0).mean():.1%}")
print(f"Among all other students: {(rest['absences'] == 0).mean():.1%}")
print()

# 4. The student with G1 == 0
print("=== STUDENT(S) WITH G1 == 0 ===")
print(df.loc[df["G1"] == 0, cols])
print()

# 5. Students with G2 == 0 but a non-zero final grade
print("=== G2 == 0 BUT G3 != 0 ===")
print(df.loc[(df["G2"] == 0) & (df["G3"] != 0), cols])