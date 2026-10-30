"""Phase 3, Step 1: distribution of student grades (G1, G2, G3)."""

from pathlib import Path

import matplotlib

# "Agg" saves charts to files without opening a pop-up window,
# which keeps the script from pausing in the terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)

# 1. Summary statistics for the three grades
print("=== SUMMARY STATISTICS ===")
summary = df[["G1", "G2", "G3"]].agg(["count", "mean", "median", "std", "min", "max", "skew"])
print(summary.round(2))
print()

# 2. Same statistics for G3, with and without the 15 zero-grade students
print("=== G3: ALL STUDENTS vs EXCLUDING G3 == 0 ===")
print("All students:        mean = {:.2f}, median = {:.1f}".format(df["G3"].mean(), df["G3"].median()))
without_zeros = df[df["g3_zero"] == 0]["G3"]
print("Excluding zero grades: mean = {:.2f}, median = {:.1f}".format(without_zeros.mean(), without_zeros.median()))
print()

# 3. Histogram of the final grade G3
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["G3"], discrete=True, color="steelblue", edgecolor="white", ax=ax)
ax.axvline(df["G3"].mean(), color="crimson", linestyle="--", label=f"Mean = {df['G3'].mean():.2f}")
ax.axvline(df["G3"].median(), color="darkgreen", linestyle=":", label=f"Median = {df['G3'].median():.1f}")
ax.set_title("Distribution of Final Grades (G3)")
ax.set_xlabel("Final grade (0-20 scale)")
ax.set_ylabel("Number of students")
ax.set_xticks(range(0, 21, 2))
ax.legend()
fig.tight_layout()

output_path = images_dir / "01_g3_distribution.png"
fig.savefig(output_path, dpi=150)
plt.close(fig)
print("Chart saved to:", output_path)