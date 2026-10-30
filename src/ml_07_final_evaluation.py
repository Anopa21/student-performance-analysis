"""Phase 5, Step 7: final evaluation on the held-out test set. Run once; do not tune on these results."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_common import (
    BACKGROUND_FEATURES,
    CATEGORICAL_COLS,
    IMAGES_DIR,
    evaluate,
    load_data,
    split_data,
)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

BASE = [c for c in BACKGROUND_FEATURES if c != "absences"]
SCENARIOS = {
    "A: background only": (BASE, None),
    "B: + G1": (BASE + ["G1"], "G1"),
    "C: + G1 + G2": (BASE + ["G1", "G2"], "G2"),
}
N_BOOT = 2000
rng = np.random.default_rng(42)  # fixed seed so the intervals are reproducible


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


def bootstrap_mae_ci(abs_err):
    """95% bootstrap interval for the mean absolute error."""
    idx = rng.integers(0, len(abs_err), size=(N_BOOT, len(abs_err)))
    return np.percentile(abs_err[idx].mean(axis=1), [2.5, 97.5])


def bootstrap_diff_ci(err_a, err_b):
    """95% bootstrap interval for mean(err_a) - mean(err_b), resampling the same students for both."""
    idx = rng.integers(0, len(err_a), size=(N_BOOT, len(err_a)))
    return np.percentile(err_a[idx].mean(axis=1) - err_b[idx].mean(axis=1), [2.5, 97.5])


df = load_data()
rows, diff_rows = [], []

for scenario, (cols, naive_col) in SCENARIOS.items():
    X_train, X_test, y_train, y_test = split_data(df, cols)
    y_true = y_test.to_numpy()
    nonzero = y_true > 0

    preds = {
        "Baseline (mean)": build_model(DummyRegressor(strategy="mean"), cols).fit(X_train, y_train).predict(X_test),
        "Linear regression": build_model(LinearRegression(), cols).fit(X_train, y_train).predict(X_test),
    }
    if naive_col:
        preds[f"Naive: G3 = {naive_col}"] = X_test[naive_col].to_numpy()

    abs_errs = {name: np.abs(p - y_true) for name, p in preds.items()}

    for name, p in preds.items():
        m = evaluate(y_true, p)
        m_nz = evaluate(y_true[nonzero], p[nonzero])
        low, high = bootstrap_mae_ci(abs_errs[name])
        rows.append({
            "scenario": scenario,
            "model": name,
            "kind": "Naive (copy latest grade)" if name.startswith("Naive") else name,
            "MAE": m["MAE"], "MAE_CI_low": low, "MAE_CI_high": high,
            "RMSE": m["RMSE"], "R2": m["R2"],
            "MAE_no_zeros": m_nz["MAE"], "R2_no_zeros": m_nz["R2"],
        })

    pairs = [("Linear regression", "Baseline (mean)")]
    if naive_col:
        pairs.append(("Linear regression", f"Naive: G3 = {naive_col}"))
    for a, b in pairs:
        diff = abs_errs[a].mean() - abs_errs[b].mean()
        low, high = bootstrap_diff_ci(abs_errs[a], abs_errs[b])
        diff_rows.append({
            "scenario": scenario,
            "comparison": f"{a} minus {b}",
            "MAE_difference": diff, "CI_low": low, "CI_high": high,
            "CI_excludes_0": bool(low > 0 or high < 0),
        })

results = pd.DataFrame(rows)
print(f"=== TEST SET RESULTS: {len(y_test)} test students ({(~nonzero).sum()} with G3 = 0); models trained on {len(X_train)} ===")
print(results.drop(columns="kind").round(3).to_string(index=False))
print()
print("=== DIFFERENCE IN MAE (negative = linear regression has the LOWER error) ===")
print(pd.DataFrame(diff_rows).round(3).to_string(index=False))
print()
print("MAE_CI_* and CI_* are 95% bootstrap intervals over the test students.")
print("If an interval for a difference includes 0, the advantage could be chance.")

# Chart: test MAE by scenario, with bootstrap intervals
kinds = ["Baseline (mean)", "Linear regression", "Naive (copy latest grade)"]
colors = {"Baseline (mean)": "lightgrey", "Linear regression": "steelblue", "Naive (copy latest grade)": "darkorange"}
width = 0.25

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, kind in enumerate(kinds):
    xs, heights, err_low, err_high = [], [], [], []
    for j, scenario in enumerate(SCENARIOS):
        match = results[(results["scenario"] == scenario) & (results["kind"] == kind)]
        if match.empty:
            continue
        r = match.iloc[0]
        xs.append(j + (i - 1) * width)
        heights.append(r["MAE"])
        err_low.append(r["MAE"] - r["MAE_CI_low"])
        err_high.append(r["MAE_CI_high"] - r["MAE"])
    ax.bar(xs, heights, width, yerr=[err_low, err_high], capsize=4, color=colors[kind], label=kind)

ax.set_xticks(range(len(SCENARIOS)))
ax.set_xticklabels(list(SCENARIOS))
ax.set_xlabel("Prediction scenario (information available)")
ax.set_ylabel("Mean absolute error on test set (grade points; lower is better)")
ax.set_title("Test-Set Error of Final Grade Predictions\nError bars: 95% bootstrap intervals over 130 test students")
ax.set_ylim(0, 3.2)
ax.legend(loc="upper center", ncol=3)
fig.tight_layout()
path = IMAGES_DIR / "12_test_mae_by_scenario.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print()
print("Chart saved to:", path)