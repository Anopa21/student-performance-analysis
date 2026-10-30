"""Phase 5, Step 5: error analysis using out-of-fold predictions (scenario A, without absences)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_common import (
    BACKGROUND_FEATURES,
    CATEGORICAL_COLS,
    IMAGES_DIR,
    RANDOM_STATE,
    TARGET,
    evaluate,
    load_data,
)

pd.set_option("display.width", 220)

FEATURES = [c for c in BACKGROUND_FEATURES if c != "absences"]


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


df = load_data()
X = df[FEATURES]
y = df[TARGET]
is_zero = y == 0

# Precomputed splits, stratified on "is the final grade zero". Each student is
# in the validation part of exactly one fold, so every prediction is out-of-fold.
strata = is_zero.astype(int)
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, strata))

models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}

predictions = {}
for name, model in models.items():
    predictions[name] = cross_val_predict(build_model(model, FEATURES), X, y, cv=splits)

# 1. Overall out-of-fold performance, and how the zero-grade students are handled
rows = []
for name, pred in predictions.items():
    pred = pd.Series(pred, index=df.index)
    overall = evaluate(y, pred)
    nonzero = evaluate(y[~is_zero], pred[~is_zero])
    nz_error = (pred[~is_zero] - y[~is_zero]).abs()
    rows.append({
        "model": name,
        "MAE_all": overall["MAE"],
        "RMSE_all": overall["RMSE"],
        "R2_all": overall["R2"],
        "MAE_nonzero": nonzero["MAE"],
        "within_1_pt_nonzero": (nz_error <= 1).mean(),
        "within_2_pt_nonzero": (nz_error <= 2).mean(),
        "MAE_zero_students": (pred[is_zero] - y[is_zero]).abs().mean(),
        "avg_prediction_zero_students": pred[is_zero].mean(),
    })
print("=== OUT-OF-FOLD PERFORMANCE (scenario A, without absences; n = 649) ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()

# 2. Error by actual grade band (students with G3 above 0): do models pull
#    predictions toward the middle?
bands = pd.cut(y, bins=[0, 9, 14, 20], labels=["low (1-9)", "middle (10-14)", "high (15-20)"])
print("=== ERROR BY ACTUAL GRADE BAND (students with G3 above 0) ===")
band_rows = []
for name in ["Linear regression", "Random forest"]:
    pred = pd.Series(predictions[name], index=df.index)
    signed = pred - y
    for band in bands.cat.categories:
        mask = bands == band
        band_rows.append({
            "model": name,
            "actual_G3_band": band,
            "students": int(mask.sum()),
            "mean_actual": y[mask].mean(),
            "mean_predicted": pred[mask].mean(),
            "mean_error (pred - actual)": signed[mask].mean(),
            "MAE": signed[mask].abs().mean(),
        })
print(pd.DataFrame(band_rows).round(2).to_string(index=False))
print()
print("A positive mean error means the model over-predicts for that group; negative means under-predicts.")
print()

# 3. Chart: actual vs predicted, and residuals, for linear regression
lin_pred = pd.Series(predictions["Linear regression"], index=df.index)
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.2, 0.2, size=len(df))  # small jitter so whole-number grades do not hide each other

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.scatter((y + jitter)[~is_zero], lin_pred[~is_zero], alpha=0.45, color="steelblue", label="G3 above 0")
ax.scatter((y + jitter)[is_zero], lin_pred[is_zero], alpha=0.8, color="crimson", label="G3 = 0")
ax.plot([0, 20], [0, 20], color="grey", linestyle="--", label="Perfect prediction")
ax.set_title("Linear Regression: Actual vs Predicted Final Grade\n(out-of-fold predictions, background features only)")
ax.set_xlabel("Actual final grade G3")
ax.set_ylabel("Predicted final grade")
ax.legend()

ax = axes[1]
residuals = (lin_pred - y)[~is_zero]
sns.histplot(residuals, bins=25, color="steelblue", edgecolor="white", ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Prediction Errors (Predicted minus Actual)\nStudents with G3 above 0")
ax.set_xlabel("Prediction error in grade points")
ax.set_ylabel("Number of students")

fig.tight_layout()
path = IMAGES_DIR / "10_ml_actual_vs_predicted.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print("Chart saved to:", path)