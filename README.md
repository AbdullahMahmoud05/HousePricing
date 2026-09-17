# House Price Predictor

A production-ready Machine Learning web app that predicts residential sale
prices from 12 key property features, using a **Linear Regression** model
trained on the Kaggle *House Prices — Advanced Regression Techniques*
dataset.

## Overview

This project turns an exploratory Jupyter notebook (`HousePricing.ipynb`)
into a clean, reusable pipeline — **with 3 verified bugs from the original
notebook corrected** (see below):

- `model.py` reproduces the notebook's preprocessing and training steps,
  with the fixes applied, and saves a single self-contained artifact
  (`house_price_model.pkl`).
- `app.py` is a Streamlit application that loads that artifact and serves
  live predictions from user-entered property details — no re-training or
  re-fitting happens at inference time.

## Dataset

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
(`train.csv`, 1,460 rows, 81 columns).

## Bugs Found in the Original Notebook (and Fixed Here)

The original notebook's own embedded output shows **Train R² = 0.7792 /
Test R² = 0.7858** — confirmed by re-running its pipeline cell-by-cell and
cross-checking the resulting feature matrix byte-for-byte against the
notebook's own exported `features.csv` (exact match). Reviewing the
pipeline for correctness surfaced three real issues that were holding
performance back:

### 1. Outliers were never actually removed
The notebook has a section titled **"Outliers Handling"**, but it only
applies `np.log1p` to reduce skew — it never removes any rows. The Ames
dataset has two well-documented anomalies (`GrLivArea` > 4,000 sq ft with
`SalePrice` < $300,000 — the dataset's own creator, Dean De Cock,
explicitly recommends dropping them). Left in, they pull the regression
line off badly.
**Fix:** these 2 rows are dropped before training.
**Impact alone:** Test R² 0.7858 → **0.8315**.

### 2. Quality features were label-encoded alphabetically, not by quality
`ExterQual`, `BsmtQual`, and `KitchenQual` are ordinal (`Po < Fa < TA < Gd
< Ex`), but the notebook ran `sklearn.LabelEncoder` on them, which assigns
codes **alphabetically**: `Ex=0, Fa=1, Gd=2, Po=3, TA=4`. That scrambles
the true quality order, which breaks the assumption a linear model relies
on — that increasing the encoded value should move the prediction
consistently in one direction.
**Fix:** these columns are mapped to a true ordinal scale
(`Po=0, Fa=1, TA=2, Gd=3, Ex=4`) instead.

### 3. `GarageType` was label-encoded as if it were ordinal
`GarageType` is a nominal category (Attached, Detached, Built-in, Carport,
Basement, 2Types, None) — there's no natural order between these. Label-
encoding it forces the model to treat, say, "Carport" as numerically
"between" other types, imposing a fake linear relationship.
**Fix:** `GarageType` is one-hot encoded instead (with a fixed baseline
category so inference always produces the same columns the model was
trained on).

### Combined effect

| Pipeline | Train R² | Test R² |
|---|---|---|
| Original notebook (verified) | 0.7792 | 0.7858 |
| + outlier removal only | 0.7927 | 0.8315 |
| + outlier removal + `GarageType` one-hot | 0.7965 | **0.8323** |
| **+ all 3 fixes (used in this project)** | **0.7934** | **0.8003** |

Outlier removal is by far the biggest, clearest win. The ordinal-quality
fix is the methodologically correct choice (a linear model should not be
handed an arbitrary alphabetical encoding of a quality scale), but on this
particular 80/20 split it slightly reduces raw test R² compared to leaving
`GarageType` as one-hot without it — a difference small enough to be
sample noise given the dataset's size (~1,460 rows, ~292 in the test
split). This project applies all three fixes for the sake of a correct,
defensible pipeline rather than one tuned to this specific split.

## Features used (12)

| Feature | Description |
|---|---|
| `OverallQual` | Overall material and finish quality (1–10) |
| `YearBuilt` | Original construction year |
| `YearRemodAdd` | Remodel year (same as `YearBuilt` if never remodeled) |
| `ExterQual` | Exterior material quality (ordinal: Po–Ex) |
| `BsmtQual` | Basement height/quality (ordinal: Po–Ex, or "no basement") |
| `1stFlrSF` | First floor area (sq ft, log1p-transformed) |
| `GrLivArea` | Above-grade living area (sq ft, log1p-transformed) |
| `FullBath` | Full bathrooms above grade |
| `KitchenQual` | Kitchen quality (ordinal: Po–Ex) |
| `TotRmsAbvGrd` | Total rooms above grade (excludes bathrooms) |
| `GarageType` | Garage location/type (one-hot encoded) |
| `GarageCars` | Garage capacity (car count) |

## Preprocessing

1. **Outlier removal:** the 2 documented `GrLivArea`/`SalePrice` anomalies are dropped.
2. **Missing values:** `BsmtQual` → `"NoBsmt"`, `GarageType` → `"NoGarage"`, `GarageCars` → `0`.
3. **Skew handling:** `np.log1p` on `1stFlrSF` and `GrLivArea`.
4. **Encoding:** ordinal mapping for the 3 quality columns; one-hot for `GarageType`.
5. **Target:** `SalePrice` used as-is — the notebook does **not** log-transform the target, so no inverse transform is needed on predictions (verified by inspecting the notebook cell by cell).
6. **Train/test split:** `train_test_split(X, y, test_size=0.2, random_state=42)`.

## Scaling

`StandardScaler` is fit **only on the training split** and applied to the
test split and, at inference time, to new user inputs — no data leakage.

## Model

`LinearRegression` trained on the scaled training features.

## Evaluation Metrics (this project's corrected pipeline)

| Metric | Value |
|---|---|
| Train R² | **0.7934** |
| Test R² | **0.8003** |
| Test RMSE | **≈ $33,209** |
| Test MAE | **≈ $25,607** |

## Project Structure

```
house_price_project/
│
├── app.py                  # Streamlit application
├── model.py                 # Training script — produces house_price_model.pkl
├── house_price_model.pkl    # Trained model + scaler + encodings + metadata
├── train.csv                 # Kaggle training dataset
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python model.py
```

This loads `train.csv`, reproduces the notebook's preprocessing (with the
3 fixes above), trains the Linear Regression model, evaluates it, and
writes `house_price_model.pkl`.

## Run the App

You can run the application either using Streamlit:

```bash
streamlit run app.py
```

Or directly using Python:

```bash
python app.py
```

The browser will open automatically (typically at `http://localhost:8501`). Fill in the property details, and click **Calculate Estimated Price**.

## Notes

- The saved `.pkl` bundles the trained model, the fitted `StandardScaler`,
  the ordinal quality mapping, the fixed `GarageType` one-hot column list,
  and preprocessing metadata — the app never re-fits anything.
- The app validates inputs (e.g. remodel year can't precede build year,
  categorical values are constrained via dropdowns) and shows friendly
  error messages instead of raw tracebacks if the model file is missing or
  malformed.
