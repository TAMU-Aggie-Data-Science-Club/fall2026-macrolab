"""
Linear regression for Texas House Price Index (TXSTHPI), 3 months ahead.

Run from the repo root:  python House_Price_Index/model/linear_regression.py
Needs first:             python House_Price_Index/model/baseline_model.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HPI_DIR = Path(__file__).resolve().parent.parent
DATA, RESULTS, MODEL_DIR = HPI_DIR / "data", HPI_DIR / "results", HPI_DIR / "model"
RESULTS.mkdir(exist_ok=True)

PREDICTORS = ["hpi"]
TARGET = "actual_future"
TEST_SIZE = 0.20

# ----------------------------------------------------------------------
# 1. Load data (oldest -> newest)
# ----------------------------------------------------------------------
path = DATA / "modeling_data.csv"
if not path.exists():
    raise FileNotFoundError("Run baseline_model.py first to create modeling_data.csv")

df = (pd.read_csv(path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True))

missing_cols = [c for c in PREDICTORS + [TARGET] if c not in df.columns]
if missing_cols:
    raise KeyError(f"Columns not found in modeling_data.csv: {missing_cols}")

# ----------------------------------------------------------------------
# 2. Keep only predictors + target, check numeric, drop missing
# ----------------------------------------------------------------------
data = df[["date"] + PREDICTORS + [TARGET]].copy()   # date is kept only to label outputs

for col in PREDICTORS + [TARGET]:
    before = data[col].isna().sum()
    data[col] = pd.to_numeric(data[col], errors="coerce")   # text -> number
    newly_missing = data[col].isna().sum() - before
    if newly_missing:
        print(f"WARNING: {newly_missing} non-numeric value(s) in '{col}' became NaN")

print("Dtypes:\n", data.dtypes, "\n")
rows_before = len(data)
data = data.dropna()
print(f"Rows after dropping missing: {len(data)} (dropped {rows_before - len(data)})\n")

# ----------------------------------------------------------------------
# 3. Chronological 80/20 split
# ----------------------------------------------------------------------
split = int(len(data) * (1 - TEST_SIZE))
train, test = data.iloc[:split], data.iloc[split:]
print(f"Train: {len(train)} rows ({train.date.min().date()} -> {train.date.max().date()})")
print(f"Test : {len(test)} rows ({test.date.min().date()} -> {test.date.max().date()})\n")

X_train, y_train = train[PREDICTORS].values, train[TARGET].values
X_test, y_test = test[PREDICTORS].values, test[TARGET].values

# ----------------------------------------------------------------------
# 4. Train: least squares with a column of 1s so the intercept is solved too
# ----------------------------------------------------------------------
X_train_b = np.c_[np.ones(len(X_train)), X_train]
theta, *_ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
intercept, coefficients = theta[0], theta[1:]

# ----------------------------------------------------------------------
# 5. Predict
# ----------------------------------------------------------------------
def predict(X):
    return np.c_[np.ones(len(X)), X] @ theta

train_pred, test_pred = predict(X_train), predict(X_test)

# ----------------------------------------------------------------------
# 6. Evaluate: MAE, RMSE, R^2
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    errors = y_true - y_pred
    ss_res = (errors ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "model": label,
        "MAE": np.abs(errors).mean(),
        "RMSE": np.sqrt((errors ** 2).mean()),
        "R2": 1 - ss_res / ss_tot,
    }

metrics = pd.DataFrame([
    evaluate(y_train, train_pred, "linear_regression_train"),
    evaluate(y_test, test_pred, "linear_regression_test"),
]).round(4)
print(metrics.to_string(index=False), "\n")

# ----------------------------------------------------------------------
# 7. Print coefficients & intercept
# ----------------------------------------------------------------------
print(f"Intercept: {intercept:.4f}")
for name, coef in zip(PREDICTORS, coefficients):
    print(f"Coefficient for {name}: {coef:.4f}")

# ----------------------------------------------------------------------
# 8. Save outputs. Baseline = current hpi on the SAME test rows, so the
#    Monday comparison is apples-to-apples.
# ----------------------------------------------------------------------
preds = test[["date", TARGET]].copy()
preds["baseline_prediction"] = df.loc[test.index, "hpi"].values
preds["regression_prediction"] = test_pred
preds["regression_residual"] = preds[TARGET] - preds["regression_prediction"]
preds.to_csv(RESULTS / "regression_predictions.csv", index=False)

metrics.to_csv(RESULTS / "regression_metrics.csv", index=False)

coefs = pd.DataFrame({"feature": PREDICTORS + ["intercept"],
                      "coefficient": list(coefficients) + [intercept]})
coefs.to_csv(RESULTS / "coefficients.csv", index=False)

with open(MODEL_DIR / "linear_regression_model.json", "w") as f:
    json.dump({"predictors": PREDICTORS, "target": TARGET,
               "intercept": float(intercept),
               "coefficients": [float(c) for c in coefficients]}, f, indent=2)

print("\nSaved predictions, metrics, coefficients, and trained model.")