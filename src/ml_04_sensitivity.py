"""Phase 5, Step 4: sensitivity checks on the background-only scenario (A)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_common import (
    BACKGROUND_FEATURES,
    CATEGORICAL_COLS,
    RANDOM_STATE,
    TARGET,
    evaluate,
    load_data,
)

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


# Each variant: (feature list, whether to remove zero-grade students from the TRAINING data)
variants = {
    "1. Full background features": (BACKGROUND_FEATURES, False),
    "2. Without absences": ([c for c in BACKGROUND_FEATURES if c != "absences"], False),
    "3. Full features, trained without zero-grade students": (BACKGROUND_FEATURES, True),
}

df = load_data()
strata = (df[TARGET] == 0).astype(int)

records = []
for repeat in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)

    for train_idx, val_idx in skf.split(df, strata):
        val_df = df.iloc[val_idx]
        val_nz = val_df[val_df[TARGET] > 0]

        for variant, (feature_cols, drop_zeros) in variants.items():
            train_df = df.iloc[train_idx]
            if drop_zeros:
                train_df = train_df[train_df[TARGET] > 0]

            for model_name, model in make_models().items():
                pipe = build_model(model, feature_cols)
                pipe.fit(train_df[feature_cols], train_df[TARGET])

                m_all = evaluate(val_df[TARGET], pipe.predict(val_df[feature_cols]))
                m_nz = evaluate(val_nz[TARGET], pipe.predict(val_nz[feature_cols]))

                records.append({
                    "variant": variant,
                    "model": model_name,
                    "MAE_all": m_all["MAE"],
                    "RMSE_all": m_all["RMSE"],
                    "R2_all": m_all["R2"],
                    "MAE_no_zeros": m_nz["MAE"],
                    "R2_no_zeros": m_nz["R2"],
                })

results = pd.DataFrame(records)
summary = results.groupby(["variant", "model"], sort=False).agg(
    MAE_all=("MAE_all", "mean"),
    MAE_all_std=("MAE_all", "std"),
    RMSE_all=("RMSE_all", "mean"),
    R2_all=("R2_all", "mean"),
    MAE_no_zeros=("MAE_no_zeros", "mean"),
    R2_no_zeros=("R2_no_zeros", "mean"),
)

print(f"=== SENSITIVITY CHECKS, SCENARIO A (background only), {N_SPLITS} folds x {N_REPEATS} repeats ===")
print(summary.round(3).to_string())
print()
print("_all      = validation students including those with G3 = 0")
print("_no_zeros = validation students with G3 above 0 only")
print("Compare each variant with variant 1 (the version we used so far).")