"""Phase 5, Step 3: repeated cross-validation, plus a naive "same as the previous grade" baseline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_common import CATEGORICAL_COLS, FEATURE_SETS, RANDOM_STATE, TARGET, evaluate, load_data

pd.set_option("display.width", 220)

N_SPLITS = 5
N_REPEATS = 3


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


def make_models():
    """Fresh, untrained models (a new set is needed for every fold)."""
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }


df = load_data()
y_all = df[TARGET]
strata = (y_all == 0).astype(int)  # keeps zero-grade students spread across folds

# Naive baselines: simply copy the most recent earlier grade
naive_columns = {"B: background + G1": "G1", "C: background + G1 + G2": "G2"}

records = []
for scenario, feature_cols in FEATURE_SETS.items():
    X_all = df[feature_cols]

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, strata)):
            X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
            y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

            for model_name, model in make_models().items():
                pipe = build_model(model, feature_cols)
                pipe.fit(X_tr, y_tr)
                train_m = evaluate(y_tr, pipe.predict(X_tr))
                val_m = evaluate(y_val, pipe.predict(X_val))
                records.append({
                    "scenario": scenario, "model": model_name,
                    "train_MAE": train_m["MAE"], "val_MAE": val_m["MAE"],
                    "val_RMSE": val_m["RMSE"], "val_R2": val_m["R2"],
                })

            naive_col = naive_columns.get(scenario)
            if naive_col:
                naive_m = evaluate(y_val, X_val[naive_col])
                records.append({
                    "scenario": scenario, "model": f"Naive: G3 = {naive_col}",
                    "train_MAE": np.nan, "val_MAE": naive_m["MAE"],
                    "val_RMSE": naive_m["RMSE"], "val_R2": naive_m["R2"],
                })

results = pd.DataFrame(records)
summary = results.groupby(["scenario", "model"], sort=False).agg(
    train_MAE=("train_MAE", "mean"),
    val_MAE=("val_MAE", "mean"),
    val_MAE_std=("val_MAE", "std"),
    val_RMSE=("val_RMSE", "mean"),
    val_R2=("val_R2", "mean"),
    val_R2_std=("val_R2", "std"),
)
summary["MAE_gap"] = summary["val_MAE"] - summary["train_MAE"]

print(f"=== CROSS-VALIDATION: {N_SPLITS} folds x {N_REPEATS} repeats = {N_SPLITS * N_REPEATS} scores per model ===")
print(summary.round(3).to_string())
print()
print("val_* = average over the held-out folds; _std = how much the score varies between folds.")
print("MAE_gap = validation error minus training error (a large positive gap suggests overfitting).")