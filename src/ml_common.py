"""Shared helpers for the machine learning steps: data loading, feature lists, splitting, metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student-por-clean.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

RANDOM_STATE = 42
TARGET = "G3"

# Columns holding text categories (need encoding before most models can use them)
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# Numeric columns that are NOT grades
NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]

# Feature sets for the three scenarios.
# g3_zero is deliberately absent from every set: it is derived from G3,
# so using it would be data leakage.
BACKGROUND_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS
FEATURE_SETS = {
    "A: background only": BACKGROUND_FEATURES,
    "B: background + G1": BACKGROUND_FEATURES + ["G1"],
    "C: background + G1 + G2": BACKGROUND_FEATURES + ["G1", "G2"],
}


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def split_data(df, feature_cols, test_size=0.2):
    """Split into train and test sets, keeping zero-grade students spread evenly."""
    X = df[feature_cols]
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=(df[TARGET] == 0),
    )


def evaluate(y_true, y_pred):
    """Return MAE, RMSE and R-squared for a set of predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }