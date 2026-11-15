"""
Compare the persistence baseline and the linear regression for HPI (3 months ahead).

Run from the repo root, AFTER the two model scripts:
    python House_Price_Index/model/baseline_model.py
    python House_Price_Index/model/linear_regression.py
    python House_Price_Index/model/compare_models.py

Reads:   results/baseline_predictions.csv
         results/regression_predictions.csv
         results/coefficients.csv
Writes:  results/model_comparison.csv
         figures/actual_vs_predicted.png
         figures/residual_plot.png
         model/baseline_model.json   (the persistence rule, our "trained" baseline)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HPI_DIR = Path(__file__).resolve().parent.parent
RESULTS, FIGS, MODEL_DIR = HPI_DIR / "results", HPI_DIR / "figures", HPI_DIR / "model"
FIGS.mkdir(exist_ok=True)

TARGET = "actual_future"

# ----------------------------------------------------------------------
# 1. Load predictions and make sure both models are scored on the SAME rows
# ----------------------------------------------------------------------
base = pd.read_csv(RESULTS / "baseline_predictions.csv", parse_dates=["date"])
reg = pd.read_csv(RESULTS / "regression_predictions.csv", parse_dates=["date"])

if not base["date"].reset_index(drop=True).equals(reg["date"].reset_index(drop=True)):
    raise ValueError(
        "Baseline and regression test sets cover different dates. "
        "Re-run baseline_model.py and linear_regression.py so both use the same data/split."
    )

comp = pd.DataFrame({
    "date": reg["date"],
    "actual": reg[TARGET],
    "baseline": base["baseline_prediction"],
    "regression": reg["regression_prediction"],
})
comp["baseline_residual"] = comp["actual"] - comp["baseline"]
comp["regression_residual"] = comp["actual"] - comp["regression"]
print(f"Comparing on {len(comp)} test rows "
      f"({comp.date.min().date()} -> {comp.date.max().date()})\n")

# ----------------------------------------------------------------------
# 2. Model comparison table (MAE, RMSE, R^2 on the shared test set)
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    err = y_true - y_pred
    ss_res = (err ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "model": label,
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "R2": 1 - ss_res / ss_tot,
    }

table = pd.DataFrame([
    evaluate(comp["actual"], comp["baseline"], "Persistence baseline"),
    evaluate(comp["actual"], comp["regression"], "Linear regression"),
])

# Improvement of regression over baseline (positive = regression is better)
b, r = table.iloc[0], table.iloc[1]
improvement = pd.DataFrame([{
    "model": "Regression vs baseline (% change)",
    "MAE": (b["MAE"] - r["MAE"]) / b["MAE"] * 100,
    "RMSE": (b["RMSE"] - r["RMSE"]) / b["RMSE"] * 100,
    "R2": (r["R2"] - b["R2"]),     # difference in R^2 points, not %
}])
table = pd.concat([table, improvement], ignore_index=True).round(4)
table.to_csv(RESULTS / "model_comparison.csv", index=False)
print(table.to_string(index=False))
print("\n(Improvement row: MAE/RMSE are % reduction, positive = regression better; "
      "R2 is the difference in R2.)")

winner = "Linear regression" if r["RMSE"] < b["RMSE"] else "Persistence baseline"
print(f"Lower test RMSE: {winner}\n")

# ----------------------------------------------------------------------
# 3. Actual vs predicted graph (actual, baseline, regression)
# ----------------------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(comp["date"], comp["actual"], label="Actual", linewidth=2.5, color="black")
plt.plot(comp["date"], comp["baseline"], "--", label="Baseline (persistence)")
plt.plot(comp["date"], comp["regression"], ":", linewidth=2, label="Linear regression")
plt.title("HPI 3 Months Ahead: Actual vs Predicted (Test Set)")
plt.xlabel("Date")
plt.ylabel("House Price Index")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "actual_vs_predicted.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 4. Residual plot (actual - predicted)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].axhline(0, color="gray", linewidth=1)
ax[0].plot(comp["date"], comp["baseline_residual"], "o--", label="Baseline", alpha=0.8)
ax[0].plot(comp["date"], comp["regression_residual"], "s:", label="Linear regression", alpha=0.8)
ax[0].set_title("Residuals over time")
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Actual - Predicted")
ax[0].legend()

ax[1].axhline(0, color="gray", linewidth=1)
ax[1].scatter(comp["regression"], comp["regression_residual"])
ax[1].set_title("Regression residuals vs predicted value")
ax[1].set_xlabel("Predicted HPI")
ax[1].set_ylabel("Actual - Predicted")

plt.tight_layout()
plt.savefig(FIGS / "residual_plot.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5. Save the baseline "model" and check every deliverable exists
# ----------------------------------------------------------------------
with open(MODEL_DIR / "baseline_model.json", "w") as f:
    json.dump({"type": "persistence",
               "rule": "predicted future HPI = most recently observed HPI",
               "horizon": "3 months (1 quarterly row)"}, f, indent=2)

expected = [
    RESULTS / "baseline_predictions.csv",
    RESULTS / "regression_predictions.csv",
    RESULTS / "model_comparison.csv",
    RESULTS / "coefficients.csv",
    FIGS / "actual_vs_predicted.png",
    FIGS / "residual_plot.png",
    MODEL_DIR / "baseline_model.json",
    MODEL_DIR / "linear_regression_model.json",
]
print("Deliverable check:")
for p in expected:
    print(f"  [{'ok' if p.exists() else 'MISSING'}] {p.relative_to(HPI_DIR)}")