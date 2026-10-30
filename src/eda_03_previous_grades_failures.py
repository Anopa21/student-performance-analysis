"""Phase 3, Step 3: previous grades (G1, G2) and past failures vs the final grade (G3)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)
no_zeros = df[df["g3_zero"] == 0]

# 1. Correlation of G1 and G2 with G3, with and without the zero-grade students.
#    Pearson measures straight-line association; Spearman measures whether
#    higher values tend to go with higher values (rank-based).
print("=== CORRELATION WITH G3 ===")
rows = []
for col in ["G1", "G2"]:
    for label, data in [("All students", df), ("Excluding G3 == 0", no_zeros)]:
        r, _ = pearsonr(data[col], data["G3"])
        rho, _ = spearmanr(data[col], data["G3"])
        rows.append({"variable": col, "group": label, "pearson_r": r, "spearman_rho": rho})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()

# 2. How far does G3 differ from G2? (excluding zeros, so that a drop to 0
#    does not dominate the numbers)
change = no_zeros["G3"] - no_zeros["G2"]
print("=== G3 minus G2 (EXCLUDING G3 == 0) ===")
print(change.describe().round(2))
print(f"Share of students with G3 within 1 point of G2: {(change.abs() <= 1).mean():.1%}")
print()

# 3. Past failures: group sizes and G3 statistics
print("=== STUDENTS PER FAILURES VALUE ===")
print(df["failures"].value_counts().sort_index())
print()
print("=== G3 BY PAST FAILURES (ALL STUDENTS) ===")
print(df.groupby("failures")["G3"].agg(["count", "mean", "median", "std"]).round(2))
print()
print("=== G3 BY PAST FAILURES (EXCLUDING G3 == 0) ===")
print(no_zeros.groupby("failures")["G3"].agg(["count", "mean", "median", "std"]).round(2))
print()

rho_all, _ = spearmanr(df["failures"], df["G3"])
rho_nz, _ = spearmanr(no_zeros["failures"], no_zeros["G3"])
print("=== SPEARMAN: failures vs G3 ===")
print(f"All students:    rho = {rho_all:.3f}")
print(f"Excluding zeros: rho = {rho_nz:.3f}")
print()

# 4. Chart A: scatter plots of G1 and G2 against G3
sns.set_theme(style="whitegrid")
df["Final grade"] = df["g3_zero"].map({0: "G3 above 0", 1: "G3 = 0"})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, col in zip(axes, ["G1", "G2"]):
    sns.scatterplot(
        data=df, x=col, y="G3", hue="Final grade",
        palette={"G3 above 0": "steelblue", "G3 = 0": "crimson"},
        alpha=0.6, ax=ax,
    )
    ax.plot([0, 20], [0, 20], color="grey", linestyle="--", label="G3 = " + col)
    ax.set_title(f"Final Grade (G3) vs {col}")
    ax.set_xlabel(f"{col} (0-20 scale)")
    ax.set_ylabel("G3 (0-20 scale)")
    ax.legend(title="")
fig.tight_layout()
path_a = images_dir / "03_previous_grades_vs_g3.png"
fig.savefig(path_a, dpi=150)
plt.close(fig)

# 5. Chart B: box plots of G3 by past failures
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
sns.boxplot(data=df, x="failures", y="G3", color="steelblue", ax=axes[0])
axes[0].set_title("Final Grade by Past Failures (All Students)")
sns.boxplot(data=no_zeros, x="failures", y="G3", color="seagreen", ax=axes[1])
axes[1].set_title("Final Grade by Past Failures (Excluding G3 = 0)")
for ax in axes:
    ax.set_xlabel("Number of past class failures")
    ax.set_ylabel("Final grade G3 (0-20 scale)")
fig.tight_layout()
path_b = images_dir / "04_failures_vs_g3.png"
fig.savefig(path_b, dpi=150)
plt.close(fig)

print("Charts saved to:")
print(" ", path_a)
print(" ", path_b)