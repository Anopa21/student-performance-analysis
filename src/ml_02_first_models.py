"""Phase 5, Step 2: train linear regression and random forest on three feature scenarios."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_common import (
    CATEGORICAL_COLS,
    FEATURE_SETS,
    RANDOM_STATE,
    evaluate,
    load_data,
    split_data,
)

pd.set_option("display.width", 220)

df = load_data()


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


models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}

rows = []
for scenario, feature_cols in FEATURE_SETS.items():
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)
    test_nonzero = y_test > 0

    for model_name, model in models.items():
        pipe = build_model(model, feature_cols)
        pipe.fit(X_train, y_train)

        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)

        train_m = evaluate(y_train, train_pred)
        test_m = evaluate(y_test, test_pred)
        test_nz_m = evaluate(y_test[test_nonzero], test_pred[test_nonzero])

        rows.append({
            "scenario": scenario,
            "model": model_name,
            "train_MAE": train_m["MAE"],
            "test_MAE": test_m["MAE"],
            "test_RMSE": test_m["RMSE"],
            "test_R2": test_m["R2"],
            "test_MAE_no_zeros": test_nz_m["MAE"],
            "test_R2_no_zeros": test_nz_m["R2"],
        })

results = pd.DataFrame(rows)
print("=== RESULTS (single 80/20 split; test set = 130 students) ===")
print(results.round(3).to_string(index=False))
print()
print("train_MAE vs test_MAE: a much lower train error than test error suggests overfitting.")