"""Phase 4, Step 3: bootstrap confidence intervals for Spearman correlations with G3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, spearmanr

pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"
images_dir = project_root / "images"

df = pd.read_csv(data_path)

N_BOOT = 2000
rng = np.random.default_rng(42)  # fixed seed so results are reproducible

# Variables to examine. Two-category variables are coded 0/1 and the
# label shows which group is 1.
variables = {
    "failures": df["failures"],
    "higher (yes=1)": (df["higher"] == "yes").astype(int),
    "Medu": df["Medu"],
    "studytime": df["studytime"],
    "school (MS=1)": (df["school"] == "MS").astype(int),
    "Fedu": df["Fedu"],
    "absences": df["absences"],
    "Dalc": df["Dalc"],
    "G1": df["G1"],
    "G2": df["G2"],
}


def bootstrap_ci(x, y, n_boot=N_BOOT):
    """95% bootstrap confidence interval for the Spearman correlation."""
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Spearman correlation = Pearson correlation of the ranks
        rhos[i] = np.corrcoef(rankdata(x[idx]), rankdata(y[idx]))[0, 1]
    return np.nanpercentile(rhos, [2.5, 97.5])


def build_table(mask, label):
    subset = df[mask]
    y = subset["G3"].to_numpy()
    rows = []
    for name, series in variables.items():
        x = series[mask].to_numpy()
        rho, p = spearmanr(x, y)
        low, high = bootstrap_ci(x, y)
        rows.append({
            "variable": name,
            "rho": rho,
            "ci_low": low,
            "ci_high": high,
            "ci_includes_0": bool(low <= 0 <= high),
        })
    table = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    print("=" * 70)
    print(f"{label}  (n = {mask.sum()}, {N_BOOT} bootstrap resamples)")
    print("=" * 70)
    print(table.round(3).to_string(index=False))
    print()
    return table


build_table(df["g3_zero"].notna(), "ALL STUDENTS")
table_nz = build_table(df["g3_zero"] == 0, "EXCLUDING G3 == 0")

# Chart: excluding zeros, without G1 and G2 so the others are readable
plot = table_nz[~table_nz["variable"].isin(["G1", "G2"])].sort_values("rho")
positions = np.arange(len(plot))

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(
    plot["rho"], positions,
    xerr=[plot["rho"] - plot["ci_low"], plot["ci_high"] - plot["rho"]],
    fmt="o", color="steelblue", ecolor="grey", capsize=4,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(positions)
ax.set_yticklabels(plot["variable"])
ax.set_title("Spearman Correlation with Final Grade (G3) and 95% Bootstrap Confidence Intervals\nExcluding students with G3 = 0")
ax.set_xlabel("Spearman correlation coefficient")
ax.set_ylabel("Variable")
fig.tight_layout()
path = images_dir / "09_correlation_confidence_intervals.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)