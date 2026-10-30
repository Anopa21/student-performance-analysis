"""Phase 4, Step 2: test whether differences in G3 between groups are larger than chance."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"

df = pd.read_csv(data_path)
no_zeros = df[df["g3_zero"] == 0]

binary_cols = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]


def holm_adjust(p_values):
    """Holm correction for running many tests at once."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def two_group_tests(data):
    rows = []
    for col in binary_cols:
        value_a, value_b = sorted(data[col].unique())
        a = data.loc[data[col] == value_a, "G3"]
        b = data.loc[data[col] == value_b, "G3"]
        u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial correlation: positive means group A tends to score higher
        effect = 2 * u_stat / (len(a) * len(b)) - 1
        rows.append({
            "variable": col,
            "group_A": value_a,
            "group_B": value_b,
            "n_A": len(a),
            "n_B": len(b),
            "median_A": a.median(),
            "median_B": b.median(),
            "effect_A_vs_B": effect,
            "p_value": p,
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_value"])
    table["significant_after_holm"] = table["p_holm"] < 0.05
    table = table.sort_values("p_value").reset_index(drop=True)
    table["p_value"] = table["p_value"].map("{:.2e}".format)
    table["p_holm"] = table["p_holm"].map("{:.2e}".format)
    table["effect_A_vs_B"] = table["effect_A_vs_B"].round(3)
    return table


def multi_group_tests(data):
    rows = []
    for col in ["studytime", "failures"]:
        groups = [g["G3"].values for _, g in data.groupby(col)]
        h_stat, p = stats.kruskal(*groups)
        n = len(data)
        epsilon_sq = h_stat / (n - 1)  # simple effect size, between 0 and 1
        rows.append({
            "variable": col,
            "groups": len(groups),
            "H": round(h_stat, 2),
            "p_value": f"{p:.2e}",
            "epsilon_squared": round(epsilon_sq, 3),
        })
    return pd.DataFrame(rows)


for label, data in [("ALL STUDENTS", df), ("EXCLUDING G3 == 0", no_zeros)]:
    print("=" * 70)
    print(f"{label}  (n = {len(data)})")
    print("=" * 70)
    print()
    print("--- Mann-Whitney U tests (two-category variables) ---")
    print("effect_A_vs_B > 0 means group A tends to have the higher grade.")
    print(two_group_tests(data).to_string(index=False))
    print()
    print("--- Kruskal-Wallis tests (multi-group variables) ---")
    print(multi_group_tests(data).to_string(index=False))
    print()