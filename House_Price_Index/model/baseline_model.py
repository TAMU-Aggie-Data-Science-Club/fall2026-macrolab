"""
Persistence baseline for Texas House Price Index (TXSTHPI).
Horizon: 3 months. HPI is QUARTERLY, so 3 months ahead = 1 row (shift(-1)).
Baseline rule: predicted future value = most recently observed value.

Run from anywhere:  python model/baseline_model.py
Expects:            data/TXSTHPI_clean.csv  (columns: date, hpi, ...)
Writes:             data/modeling_data.csv
                    results/baseline_predictions.csv
                    results/baseline_metrics.csv
                    figures/baseline_actual_vs_predicted.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[2]
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"
for d in (DATA, RESULTS, FIGS):
    d.mkdir(exist_ok=True)

HORIZON_ROWS = 1      # 3 months on quarterly data = 1 row
TEST_SIZE = 0.20

# ----------------------------------------------------------------------
# 1. Build modeling_data.csv: date, variable, future target (oldest -> newest)
# ----------------------------------------------------------------------
src = pd.read_csv(REPO / "data" / "processed" / "TXSTHPI_clean.csv", parse_dates=["date"])
df = (src[["date", "hpi"]]
      .sort_values("date")
      .reset_index(drop=True))

df["actual_future"] = df["hpi"].shift(-HORIZON_ROWS)   # value 3 months later

# ----------------------------------------------------------------------
# 2. Baseline predictions, drop rows with no future target
# ----------------------------------------------------------------------
df["baseline_prediction"] = df["hpi"]
df = df.dropna(subset=["actual_future"]).reset_index(drop=True)

df[["date", "hpi", "actual_future"]].to_csv(DATA / "modeling_data.csv", index=False)

# ----------------------------------------------------------------------
# 3. Chronological 80/20 split (no shuffling: time series)
# ----------------------------------------------------------------------
split = int(len(df) * (1 - TEST_SIZE))
train, test = df.iloc[:split], df.iloc[split:]
print(f"Train: {len(train)} rows ({train.date.min().date()} -> {train.date.max().date()})")
print(f"Test : {len(test)} rows ({test.date.min().date()} -> {test.date.max().date()})")

# ----------------------------------------------------------------------
# 4. Evaluate: MAE, RMSE, R^2
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    return {
        "model": label,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

metrics = pd.DataFrame([
    evaluate(train["actual_future"], train["baseline_prediction"], "baseline_train"),
    evaluate(test["actual_future"], test["baseline_prediction"], "baseline_test"),
]).round(4)
print("\n", metrics.to_string(index=False))

# ----------------------------------------------------------------------
# 5. Save predictions + metrics
# ----------------------------------------------------------------------
test[["date", "hpi", "actual_future", "baseline_prediction"]].to_csv(
    RESULTS / "baseline_predictions.csv", index=False)
metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)

# ----------------------------------------------------------------------
# 6. Actual vs predicted graph (test set)
# ----------------------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(test["date"], test["actual_future"], label="Actual", linewidth=2)
plt.plot(test["date"], test["baseline_prediction"], "--", label="Baseline (persistence)")
plt.title("HPI: Actual vs Baseline Prediction (Test Set, 3 Months Ahead)")
plt.xlabel("Date")
plt.ylabel("House Price Index")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "baseline_actual_vs_predicted.png", dpi=150)
print("\nSaved results and figure.")