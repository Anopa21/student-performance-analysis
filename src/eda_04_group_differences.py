"""Phase 3, Step 4: differences in the final grade (G3) across student groups."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)
no_zeros = df[df["g3_zero"] == 0]

group_columns = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# 1. One row per group: size, and mean G3 with and without the zero-grade students
rows = []
for col in group_columns:
    for value in sorted(df[col].unique()):
        subset = df[df[col] == value]
        subset_nz = no_zeros[no_zeros[col] == value]
        rows.append({
            "variable": col,
            "group": value,
            "students": len(subset),
            "mean_G3_all": subset["G3"].mean(),
            "mean_G3_no_zeros": subset_nz["G3"].mean(),
        })
table = pd.DataFrame(rows)
print("=== MEAN FINAL GRADE BY GROUP ===")
print(table.round(2).to_string(index=False))
print()

# 2. For each variable: gap between its two groups (larger mean minus smaller mean),
#    sorted from largest to smallest gap, using the data WITHOUT the zero-grade students.
gaps = []
for col in group_columns:
    means = table[table["variable"] == col].set_index("group")["mean_G3_no_zeros"]
    gaps.append({
        "variable": col,
        "higher_group": means.idxmax(),
        "lower_group": means.idxmin(),
        "gap_in_mean_G3": means.max() - means.min(),
    })
gap_table = pd.DataFrame(gaps).sort_values("gap_in_mean_G3", ascending=False)
print("=== GAP IN MEAN G3 BETWEEN GROUPS (EXCLUDING G3 == 0), LARGEST FIRST ===")
print(gap_table.round(2).to_string(index=False))
print()

# 3. Chart: box plots for six variables, excluding the zero-grade students
sns.set_theme(style="whitegrid")
chart_columns = ["school", "sex", "address", "higher", "internet", "schoolsup"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
for ax, col in zip(axes.flat, chart_columns):
    sns.boxplot(data=no_zeros, x=col, y="G3", color="steelblue", ax=ax)
    ax.set_title(f"G3 by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Final grade G3 (0-20 scale)")
fig.suptitle("Final Grade by Student Group (Excluding G3 = 0)", fontsize=14)
fig.tight_layout()
path = images_dir / "05_group_differences.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)