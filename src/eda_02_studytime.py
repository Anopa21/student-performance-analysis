"""Phase 3, Step 2: relationship between weekly study time and the final grade (G3)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)

# studytime is a coded category. Map the codes to readable labels
# (verify these against student.txt from the dataset download).
labels = {1: "<2 h", 2: "2-5 h", 3: "5-10 h", 4: ">10 h"}
order = list(labels.values())
df["studytime_label"] = df["studytime"].map(labels)

# 1. How many students are in each study-time group?
print("=== STUDENTS PER STUDY-TIME GROUP ===")
print(df["studytime_label"].value_counts().reindex(order))
print()

# 2. G3 statistics per group, all students
print("=== G3 BY STUDY TIME (ALL STUDENTS) ===")
stats_all = df.groupby("studytime_label")["G3"].agg(["count", "mean", "median", "std"]).reindex(order)
print(stats_all.round(2))
print()

# 3. Same, excluding the 15 zero-grade students
no_zeros = df[df["g3_zero"] == 0]
print("=== G3 BY STUDY TIME (EXCLUDING G3 == 0) ===")
stats_nz = no_zeros.groupby("studytime_label")["G3"].agg(["count", "mean", "median", "std"]).reindex(order)
print(stats_nz.round(2))
print()

# 4. Spearman rank correlation. We use Spearman, not Pearson, because
#    studytime is an ordered category rather than a true measurement.
rho_all, p_all = spearmanr(df["studytime"], df["G3"])
rho_nz, p_nz = spearmanr(no_zeros["studytime"], no_zeros["G3"])
print("=== SPEARMAN CORRELATION: studytime vs G3 ===")
print(f"All students:        rho = {rho_all:.3f}, p = {p_all:.4f}")
print(f"Excluding zeros:     rho = {rho_nz:.3f}, p = {p_nz:.4f}")
print()

# 5. Box plots side by side: all students vs excluding zeros
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

sns.boxplot(data=df, x="studytime_label", y="G3", order=order, color="steelblue", ax=axes[0])
axes[0].set_title("Final Grade by Weekly Study Time (All Students)")

sns.boxplot(data=no_zeros, x="studytime_label", y="G3", order=order, color="seagreen", ax=axes[1])
axes[1].set_title("Final Grade by Weekly Study Time (Excluding G3 = 0)")

for ax in axes:
    ax.set_xlabel("Weekly study time")
    ax.set_ylabel("Final grade G3 (0-20 scale)")

fig.tight_layout()
output_path = images_dir / "02_studytime_vs_g3.png"
fig.savefig(output_path, dpi=150)
plt.close(fig)
print("Chart saved to:", output_path)