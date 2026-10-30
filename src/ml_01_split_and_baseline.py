"""Phase 5, Step 1: train/test split and baseline predictions."""

import pandas as pd
from sklearn.dummy import DummyRegressor

from ml_common import BACKGROUND_FEATURES, evaluate, load_data, split_data

pd.set_option("display.width", 200)

df = load_data()
X_train, X_test, y_train, y_test = split_data(df, BACKGROUND_FEATURES)

# 1. Sizes of the two sets, and how the zero-grade students were divided
print("=== SPLIT ===")
print(f"Total students:  {len(df)}")
print(f"Training set:    {len(X_train)}")
print(f"Test set:        {len(X_test)}")
print(f"Zero-grade students in training set: {(y_train == 0).sum()}")
print(f"Zero-grade students in test set:     {(y_test == 0).sum()}")
print()

# 2. Mean of G3 in each set: they should be similar
print("=== MEAN G3 ===")
print(f"Training set: {y_train.mean():.2f}")
print(f"Test set:     {y_test.mean():.2f}")
print()

# 3. Baselines: always predict one fixed number, learned from the training set only
rows = []
test_nonzero = y_test > 0
for strategy in ["mean", "median"]:
    baseline = DummyRegressor(strategy=strategy)
    baseline.fit(X_train, y_train)
    pred = baseline.predict(X_test)

    train_metrics = evaluate(y_train, baseline.predict(X_train))
    test_metrics = evaluate(y_test, pred)
    test_nz_metrics = evaluate(y_test[test_nonzero], pred[test_nonzero])

    rows.append({"baseline": strategy, "predicts": round(float(pred[0]), 2),
                 "set": "train", **train_metrics})
    rows.append({"baseline": strategy, "predicts": round(float(pred[0]), 2),
                 "set": "test (all)", **test_metrics})
    rows.append({"baseline": strategy, "predicts": round(float(pred[0]), 2),
                 "set": "test (excl. zeros)", **test_nz_metrics})

print("=== BASELINE RESULTS ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()
print("MAE  = average size of the error, in grade points")
print("RMSE = like MAE but penalises large errors more")
print("R2   = share of the variation explained (0 = no better than the average)")