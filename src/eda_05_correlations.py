"""Phase 3, Step 5: which variables have the strongest relationships with G3?"""

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

# Two-category variables: (value coded 0, value coded 1).
# The label shows which value is coded 1, so the sign of the correlation is readable.
binary_map = {
    "school": ("GP", "MS"),
    "sex": ("F", "M"),
    "address": ("U", "R"),
    "famsize": ("LE3", "GT3"),
    "Pstatus": ("T", "A"),
    "schoolsup": ("no", "yes"),
    "famsup": ("no", "yes"),
    "paid": ("no", "yes"),
    "activities": ("no", "yes"),
    "nursery": ("no", "yes"),
    "higher": ("no", "yes"),
    "internet": ("no", "yes"),
    "romantic": ("no", "yes"),
}

# Numeric or ordered-code variables
numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]

# Build a fully numeric copy of the data
encoded = df[numeric_cols + ["G3", "g3_zero"]].copy()
labels = {col: col for col in numeric_cols}
for col, (zero_value, one_value) in binary_map.items():
    encoded[col] = (df[col] == one_value).astype(int)
    labels[col] = f"{col} ({one_value}=1)"

feature_cols = numeric_cols + list(binary_map)


def spearman_with_g3(data):
    """Spearman correlation of every feature with G3."""
    return data[feature_cols + ["G3"]].corr(method="spearman")["G3"].drop("G3")


rho_all = spearman_with_g3(encoded)
rho_nz = spearman_with_g3(encoded[encoded["g3_zero"] == 0])

table = pd.DataFrame({"rho_all": rho_all, "rho_no_zeros": rho_nz})
table["abs_no_zeros"] = table["rho_no_zeros"].abs()
table = table.sort_values("abs_no_zeros", ascending=False)
table.index = [labels[name] for name in table.index]

# 1. Ranked table, strongest first
print("=== SPEARMAN CORRELATION WITH G3 (STRONGEST FIRST, BY EXCLUDING-ZEROS VALUE) ===")
print(table[["rho_all", "rho_no_zeros"]].round(3))
print()
print("Note: G1 and G2 are earlier grades in the same subject, so their strong")
print("correlation with G3 is expected. They are excluded from the chart below.")
print()

# 2. Bar chart without G1 and G2, so the other variables are readable
plot_data = table.drop(index=["G1", "G2"]).sort_values("rho_no_zeros")
colors = ["indianred" if v < 0 else "steelblue" for v in plot_data["rho_no_zeros"]]

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 9))
ax.barh(plot_data.index, plot_data["rho_no_zeros"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Spearman Correlation of Each Variable with Final Grade (G3)\nExcluding students with G3 = 0")
ax.set_xlabel("Spearman correlation coefficient")
ax.set_ylabel("Variable")
fig.tight_layout()
path_bar = images_dir / "06_correlations_with_g3.png"
fig.savefig(path_bar, dpi=150)
plt.close(fig)

# 3. Heatmap: the 8 strongest non-grade variables plus G1, G2, G3
reverse_labels = {value: key for key, value in labels.items()}
top_names = [name for name in table.index if name not in ("G1", "G2")][:8]
heat_cols = [reverse_labels[name] for name in top_names] + ["G1", "G2", "G3"]

corr = encoded[encoded["g3_zero"] == 0][heat_cols].corr(method="spearman")
corr.index = corr.columns = [labels.get(c, c) for c in heat_cols]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Spearman Correlation Heatmap: Strongest Variables and Grades\nExcluding students with G3 = 0")
fig.tight_layout()
path_heat = images_dir / "07_correlation_heatmap.png"
fig.savefig(path_heat, dpi=150)
plt.close(fig)

print("Charts saved to:")
print(" ", path_bar)
print(" ", path_heat)