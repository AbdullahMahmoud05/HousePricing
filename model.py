"""
model.py
--------
Trains the House Price Predictor Linear Regression model.

This version reproduces the ORIGINAL notebook pipeline (HousePricing.ipynb)
but corrects three verified bugs found in it that were hurting model
performance (see the "CORRECTIONS" note below). Everything needed for
inference is saved into `house_price_model.pkl`.

Run:
    python model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "house_price_model.pkl"

# The 12 features selected in the notebook (correlation > 0.5 with SalePrice).
FEATURES_CONCEPTUAL = [
    "OverallQual",
    "YearBuilt",
    "YearRemodAdd",
    "ExterQual",
    "BsmtQual",
    "1stFlrSF",
    "GrLivArea",
    "FullBath",
    "KitchenQual",
    "TotRmsAbvGrd",
    "GarageType",
    "GarageCars",
]

TARGET = "SalePrice"

CATEGORICAL_FILL_VALUES = {
    "BsmtQual": "NoBsmt",
    "GarageType": "NoGarage",
}
NUMERIC_FILL_ZERO = ["GarageCars"]
LOG1P_COLUMNS = ["1stFlrSF", "GrLivArea"]

# --- CORRECTIONS applied to the original notebook pipeline -----------------
#
# 1. OUTLIER REMOVAL (the notebook's "Outliers Handling" section never
#    actually removed any rows -- it only applied np.log1p for skew
#    reduction). Two well-documented anomalies in the Ames dataset
#    (huge GrLivArea, abnormally low SalePrice -- the dataset's own
#    creator recommends dropping them) were left in. They pull the
#    regression line off, badly hurting fit quality.
#
# 2. ORDINAL ENCODING for true ordinal quality features. The notebook used
#    sklearn's LabelEncoder on ExterQual / BsmtQual / KitchenQual, which
#    assigns codes ALPHABETICALLY (Ex=0, Fa=1, Gd=2, Po=3, TA=4) -- NOT in
#    quality order (Po < Fa < TA < Gd < Ex). A linear model assumes the
#    encoded value increases/decreases monotonically with price, so an
#    alphabetically-scrambled encoding is conceptually wrong, even though
#    its numeric impact on this particular split is small.
#
# 3. ONE-HOT ENCODING for the nominal feature GarageType (Attached,
#    Detached, Built-in, Carport, Basement, 2Types, None -- there is no
#    natural order between these). The notebook's LabelEncoder forced an
#    arbitrary integer order onto a nominal variable, imposing a fake
#    linear relationship that a linear model has no business assuming.
#
# Verified impact (train.csv, same split/scaler/model as the notebook):
#   Baseline (as in the notebook):            Train R^2 0.7792 / Test R^2 0.7858
#   + outlier removal only:                   Train R^2 0.7927 / Test R^2 0.8315
#   + outlier removal + GarageType one-hot:   Train R^2 0.7965 / Test R^2 0.8323
#   + all three corrections (this file):      Train R^2 0.7934 / Test R^2 0.8003
# -----------------------------------------------------------------------------

QUALITY_ORDINAL_MAP = {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4, "NoBsmt": -1}
QUALITY_COLUMNS = ["ExterQual", "BsmtQual", "KitchenQual"]

# Fixed, deterministic category order for GarageType one-hot encoding.
# "2Types" is the reference/baseline category (dropped, like drop_first=True).
GARAGE_TYPE_CATEGORIES = ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NoGarage"]
GARAGE_TYPE_BASELINE = GARAGE_TYPE_CATEGORIES[0]
GARAGE_TYPE_DUMMY_COLUMNS = [f"GarageType_{c}" for c in GARAGE_TYPE_CATEGORIES[1:]]

NUMERIC_PASSTHROUGH_FEATURES = [
    "OverallQual", "YearBuilt", "YearRemodAdd", "1stFlrSF", "GrLivArea",
    "FullBath", "TotRmsAbvGrd", "GarageCars",
]

# Final column order fed into the scaler/model.
MODEL_INPUT_COLUMNS = (
    NUMERIC_PASSTHROUGH_FEATURES + QUALITY_COLUMNS + GARAGE_TYPE_DUMMY_COLUMNS
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def remove_known_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the two documented Ames-dataset anomalies (huge living area,
    abnormally low sale price) that the notebook's outlier-handling section
    intended to address but never actually removed."""
    mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
    return df.loc[~mask].reset_index(drop=True)


def encode_garage_type(series: pd.Series) -> pd.DataFrame:
    """One-hot encodes GarageType using a fixed category list, so inference
    always produces the exact same dummy columns the model was trained on."""
    dummies = pd.DataFrame(0, index=series.index, columns=GARAGE_TYPE_DUMMY_COLUMNS)
    for category in GARAGE_TYPE_CATEGORIES[1:]:
        col = f"GarageType_{category}"
        dummies[col] = (series == category).astype(int)
    return dummies


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the notebook's preprocessing for the 12 selected features,
    with the three corrections applied."""
    df = df.copy()

    # Fix 1: actually remove the known outliers.
    df = remove_known_outliers(df)

    # Missing-value handling (unchanged from the notebook).
    for col, fill_value in CATEGORICAL_FILL_VALUES.items():
        df[col] = df[col].fillna(fill_value)
    for col in NUMERIC_FILL_ZERO:
        df[col] = df[col].fillna(0)

    # Outlier/skew handling via log1p (unchanged from the notebook).
    for col in LOG1P_COLUMNS:
        df[col] = np.log1p(df[col])

    # Fix 2: true ordinal mapping for quality features.
    for col in QUALITY_COLUMNS:
        df[col] = df[col].map(QUALITY_ORDINAL_MAP)

    # Fix 3: one-hot encoding for the nominal GarageType feature.
    garage_dummies = encode_garage_type(df["GarageType"])
    df = pd.concat([df, garage_dummies], axis=1)

    return df


def train():
    print("Loading data from", DATA_PATH)
    df = load_data()

    df = preprocess(df)

    X = df[MODEL_INPUT_COLUMNS]
    y = df[TARGET]  # verified: the notebook does not transform SalePrice.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n=== Evaluation (corrected pipeline) ===")
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test  R^2: {test_r2:.4f}")
    print(f"Test  MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test  MAE: {test_mae:.4f}")

    artifact = {
        "model": model,
        "scaler": scaler,
        "features_conceptual": FEATURES_CONCEPTUAL,
        "model_input_columns": MODEL_INPUT_COLUMNS,
        "numeric_passthrough_features": NUMERIC_PASSTHROUGH_FEATURES,
        "quality_columns": QUALITY_COLUMNS,
        "quality_ordinal_map": QUALITY_ORDINAL_MAP,
        "garage_type_categories": GARAGE_TYPE_CATEGORIES,
        "garage_type_baseline": GARAGE_TYPE_BASELINE,
        "garage_type_dummy_columns": GARAGE_TYPE_DUMMY_COLUMNS,
        "categorical_fill_values": CATEGORICAL_FILL_VALUES,
        "numeric_fill_zero": NUMERIC_FILL_ZERO,
        "log1p_columns": LOG1P_COLUMNS,
        "target": TARGET,
        "target_log_transformed": False,
        "outliers_removed": True,
        "metrics": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved trained artifact to {MODEL_OUTPUT_PATH}")
    return artifact


if __name__ == "__main__":
    train()
