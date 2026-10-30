"""Phase 4, Step 1: check how close the final grade G3 is to a normal distribution."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)
g3_all = df["G3"]
g3_nz = df.loc[df["g3_zero"] == 0, "G3"]

# 1. Skewness and kurtosis, with and without the zero-grade students.
#    Skew: 0 = symmetric, negative = longer tail on the low side.
#    Excess kurtosis: 0 = tails like a normal distribution, positive = heavier tails.
print("=== SHAPE OF G3 ===")
shape = pd.DataFrame(
    {
        "students": [len(g3_all), len(g3_nz)],
        "mean": [g3_all.mean(), g3_nz.mean()],
        "median": [g3_all.median(), g3_nz.median()],
        "skew": [stats.skew(g3_all), stats.skew(g3_nz)],
        "excess_kurtosis": [stats.kurtosis(g3_all), stats.kurtosis(g3_nz)],
    },
    index=["All students", "Excluding G3 == 0"],
)
print(shape.round(3))
print()

# 2. Shapiro-Wilk normality test.
#    The null hypothesis is "the data come from a normal distribution".
#    A small p-value is evidence AGAINST normality. With several hundred
#    observations the test flags even small departures, so we read it
#    together with the Q-Q plot and not on its own.
print("=== SHAPIRO-WILK NORMALITY TEST ===")
for label, data in [("All students", g3_all), ("Excluding G3 == 0", g3_nz)]:
    stat, p = stats.shapiro(data)
    print(f"{label}: W = {stat:.4f}, p = {p:.2e}")
print()

# 3. Q-Q plots. Points close to the red line mean close to normal.
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (title, data) in zip(
    axes,
    [("All students", g3_all), ("Excluding G3 = 0", g3_nz)],
):
    stats.probplot(data, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("steelblue")
    ax.get_lines()[0].set_markeredgecolor("steelblue")
    ax.get_lines()[1].set_color("crimson")
    ax.set_title(f"Q-Q Plot of Final Grade (G3): {title}")
    ax.set_xlabel("Theoretical quantiles (normal distribution)")
    ax.set_ylabel("Observed G3 values")
fig.tight_layout()
path = images_dir / "08_g3_qqplot.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)