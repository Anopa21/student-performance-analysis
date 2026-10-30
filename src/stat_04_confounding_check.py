"""Phase 4, Step 4: do key relationships hold within subgroups? (a confounding check)"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

pd.set_option("display.width", 200)

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por-clean.csv"

df = pd.read_csv(data_path)
data = df[df["g3_zero"] == 0].copy()  # excluding the 15 zero-grade students

data["failed_before"] = (data["failures"] > 0).map({False: "0 failures", True: "1+ failures"})
data["higher_label"] = data["higher"].map({"yes": "wants higher ed.", "no": "does not"})


def corr_by_group(variable, group_col):
    """Spearman correlation of `variable` with G3 inside each subgroup."""
    rows = []
    for value, subset in data.groupby(group_col):
        if len(subset) < 20:
            continue
        rho, _ = spearmanr(subset[variable], subset["G3"])
        rows.append({
            "grouped_by": group_col,
            "subgroup": value,
            "students": len(subset),
            f"rho_{variable}_vs_G3": round(rho, 3),
        })
    return pd.DataFrame(rows)


# 1. Study time vs G3, within subgroups
print("=== STUDYTIME vs G3, WITHIN SUBGROUPS (excluding G3 == 0) ===")
rho_overall, _ = spearmanr(data["studytime"], data["G3"])
print(f"Overall: rho = {rho_overall:.3f}  (n = {len(data)})")
print()
for group_col in ["school", "failed_before", "higher_label", "sex"]:
    print(corr_by_group("studytime", group_col).to_string(index=False))
    print()

# 2. School gap within other subgroups: is MS vs GP still different?
print("=== MEDIAN AND MEAN G3 BY SCHOOL, WITHIN SUBGROUPS ===")
for group_col in ["failed_before", "higher_label"]:
    summary = (
        data.groupby([group_col, "school"])["G3"]
        .agg(students="count", mean="mean", median="median")
        .round(2)
    )
    print(summary)
    print()

# 3. Study time and school: do the two schools differ in how much students study?
print("=== STUDY TIME CODE BY SCHOOL (share of students) ===")
print((pd.crosstab(data["school"], data["studytime"], normalize="index") * 100).round(1))
print()

# 4. Failures by school
print("=== SHARE OF STUDENTS WITH 1+ PAST FAILURES, BY SCHOOL ===")
print((pd.crosstab(data["school"], data["failed_before"], normalize="index") * 100).round(1))