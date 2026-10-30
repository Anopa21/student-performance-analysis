"""Phase 5, Step 6: which features does the model rely on? (permutation importance)"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_common import (
    BACKGROUND_FEATURES,
    CATEGORICAL_COLS,
    IMAGES_DIR,
    RANDOM_STATE,
    TARGET,
    load_data,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", None)

FEATURES = [c for c in BACKGROUND_FEATURES if c != "absences"]
N_REPEATS = 10  # how many times each feature is shuffled within each fold


def build_model(model, feature_cols):
    """Pipeline: one-hot encode text columns, pass numeric columns through, then fit the model."""
    categorical = [c for c in feature_cols if c in CATEGORICAL_COLS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="passthrough",
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


model_makers = {
    "Linear regression": lambda: LinearRegression(),
    "Random forest": lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}

df = load_data()
X = df[FEATURES]
y = df[TARGET]
strata = (y == 0).astype(int)
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, strata))

records = []
for model_name, make_model in model_makers.items():
    for fold, (train_idx, val_idx) in enumerate(splits):
        pipe = build_model(make_model(), FEATURES)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Importance = how much the mean absolute error rises when a feature is shuffled
        result = permutation_importance(
            pipe, X.iloc[val_idx], y.iloc[val_idx],
            scoring="neg_mean_absolute_error",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for feature, importance in zip(FEATURES, result.importances_mean):
            records.append({"model": model_name, "feature": feature, "fold": fold, "mae_increase": importance})

results = pd.DataFrame(records)
summary = results.groupby(["model", "feature"])["mae_increase"].agg(["mean", "std"]).reset_index()

table = summary.pivot(index="feature", columns="model", values=["mean", "std"])
table.columns = [f"{'MAE_up' if stat == 'mean' else 'std'} | {model}" for stat, model in table.columns]
table = table.sort_values("MAE_up | Random forest", ascending=False)

print("=== PERMUTATION IMPORTANCE: increase in MAE (grade points) when a feature is shuffled ===")
print("Averaged over 5 held-out folds. std = spread between folds.")
print("Values near 0 (or negative) mean the model did not rely on that feature.")
print()
print(table.round(3).to_string())
print()

# Chart: top 12 features for each model
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, model_name in zip(axes, model_makers):
    top = (
        summary[summary["model"] == model_name]
        .sort_values("mean", ascending=False)
        .head(12)
        .sort_values("mean")
    )
    ax.barh(top["feature"], top["mean"], xerr=top["std"], color="steelblue", ecolor="grey", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{model_name}: Top 12 Features")
    ax.set_xlabel("Increase in MAE when feature is shuffled (grade points)")
    ax.set_ylabel("Feature")
fig.suptitle("Permutation Importance (background features only, held-out folds)", fontsize=13)
fig.tight_layout()
path = IMAGES_DIR / "11_permutation_importance.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)