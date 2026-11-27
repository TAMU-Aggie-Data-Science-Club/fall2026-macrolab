"""
XGBoost for Texas House Prince Index (TXSTHPI), 3 months ahead.
HPI is quarterly, so 3 months ahead = 1 row (the 'actual_future' column).

Two models will be trained on the SAME chronological 80/20 split used by the baseline and the linear regression:

    Model A "xgboost"   (as instructed) predictor: hpi -> target: actual_future
    Model B "xgboost_change" predictors: hpi growth features -> target: the CHANGE in hPI, then predicted HPI = current hpi + predicted change
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

HPI_DIR = Path(__file__).resolve().parent.parent
DATA, RESULTS, FIGS, MODEL_DIR = (HPI_DIR / "data", HPI_DIR / "results",
                                  HPI_DIR / "figures", HPI_DIR / "model")
for d in (RESULTS, FIGS):
    d.mkdir(exist_ok=True)

TARGET = "actual_future"
TEST_SIZE = 0.20
PREDICTORS_A = ["hpi"]                                
FEATURES_B = ["hpi_qoq_growth", "hpi_yoy_growth", "hpi_qoq_growth_lag1"]

PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3,
              subsample=0.8, random_state=42)

# ----------------------------------------------------------------------
# 1. Load data, check numeric, build Model B features (all backward-looking)
# ----------------------------------------------------------------------
path = DATA / "modeling_data.csv"
if not path.exists():
    raise FileNotFoundError("Run baseline_model.py first to create modeling_data.csv")

df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
for col in ["hpi", TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["hpi", TARGET]).reset_index(drop=True)

df["hpi_qoq_growth"] = df["hpi"].pct_change(1) * 100
df["hpi_yoy_growth"] = df["hpi"].pct_change(4) * 100
df["hpi_qoq_growth_lag1"] = df["hpi_qoq_growth"].shift(1)
df["change"] = df[TARGET] - df["hpi"]

# ----------------------------------------------------------------------
# 2. Chronological 80/20 split (same rows as the baseline & regression)
# ----------------------------------------------------------------------
split = int(len(df) * (1 - TEST_SIZE))
train, test = df.iloc[:split], df.iloc[split:].copy()
print(f"Train: {len(train)} rows ({train.date.min().date()} -> {train.date.max().date()})")
print(f"Test : {len(test)} rows ({test.date.min().date()} -> {test.date.max().date()})\n")

above = (test["hpi"] > train["hpi"].max()).mean() * 100
print(f"Training HPI range: {train.hpi.min():.1f} - {train.hpi.max():.1f}")
print(f"Test HPI range    : {test.hpi.min():.1f} - {test.hpi.max():.1f}")
print(f"{above:.0f}% of test rows have HPI above anything seen in training "
      "(trees cannot extrapolate there)\n")

# ----------------------------------------------------------------------
# 3. Model A: hpi -> actual_future (as instructed)
# ----------------------------------------------------------------------
model_a = xgb.XGBRegressor(**PARAMS)
model_a.fit(train[PREDICTORS_A], train[TARGET])
train_pred_a = model_a.predict(train[PREDICTORS_A])
test["xgboost_prediction"] = model_a.predict(test[PREDICTORS_A])

# ----------------------------------------------------------------------
# 4. Model B: growth features -> change in HPI, then add back current hpi
# ----------------------------------------------------------------------
train_b = train.dropna(subset=FEATURES_B)        # first few rows lack lag/growth values
model_b = xgb.XGBRegressor(**PARAMS)
model_b.fit(train_b[FEATURES_B], train_b["change"])
train_pred_b = train_b["hpi"] + model_b.predict(train_b[FEATURES_B])
test["xgboost_change_prediction"] = test["hpi"] + model_b.predict(test[FEATURES_B])

print("Model B feature importance:")
for name, imp in zip(FEATURES_B, model_b.feature_importances_):
    print(f"  {name:22s} {imp:.3f}")
print()

# ----------------------------------------------------------------------
# 5. Evaluate: MAE, RMSE, R^2
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    err = np.asarray(y_true) - np.asarray(y_pred)
    ss_res = (err ** 2).sum()
    ss_tot = ((np.asarray(y_true) - np.mean(y_true)) ** 2).sum()
    return {"model": label, "MAE": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()), "R2": 1 - ss_res / ss_tot}

metrics = pd.DataFrame([
    evaluate(train[TARGET], train_pred_a, "xgboost_train"),
    evaluate(test[TARGET], test["xgboost_prediction"], "xgboost_test"),
    evaluate(train_b[TARGET], train_pred_b, "xgboost_change_train"),
    evaluate(test[TARGET], test["xgboost_change_prediction"], "xgboost_change_test"),
]).round(4)
print(metrics.to_string(index=False), "\n")
metrics.to_csv(RESULTS / "xgboost_metrics.csv", index=False)

# ----------------------------------------------------------------------
# 6. Compare against baseline & linear regression (same test rows)
# ----------------------------------------------------------------------
comp = test[["date", TARGET, "hpi", "xgboost_prediction", "xgboost_change_prediction"]].copy()
comp = comp.rename(columns={TARGET: "actual"})
comp["baseline"] = comp["hpi"]

reg_path = RESULTS / "regression_predictions.csv"
if reg_path.exists():
    reg = pd.read_csv(reg_path, parse_dates=["date"])
    if reg["date"].reset_index(drop=True).equals(comp["date"].reset_index(drop=True)):
        comp["regression"] = reg["regression_prediction"].values
    else:
        print("NOTE: regression test dates differ; skipping regression in the comparison.\n")
else:
    print("NOTE: regression_predictions.csv not found; run linear_regression.py first.\n")

rows = [("Persistence baseline", "baseline")]
if "regression" in comp:
    rows.append(("Linear regression", "regression"))
rows += [("XGBoost (hpi -> future hpi)", "xgboost_prediction"),
         ("XGBoost (growth features -> change)", "xgboost_change_prediction")]
comparison = pd.DataFrame([evaluate(comp["actual"], comp[c], name) for name, c in rows]).round(4)
print("Test-set comparison:")
print(comparison.to_string(index=False), "\n")
comparison.to_csv(RESULTS / "xgboost_comparison.csv", index=False)

# ----------------------------------------------------------------------
# 7. Save predictions and trained models
# ----------------------------------------------------------------------
out = comp.copy()
out["xgboost_residual"] = out["actual"] - out["xgboost_prediction"]
out["xgboost_change_residual"] = out["actual"] - out["xgboost_change_prediction"]
out.drop(columns="hpi").to_csv(RESULTS / "xgboost_predictions.csv", index=False)
model_a.save_model(MODEL_DIR / "xgboost_model.json")
model_b.save_model(MODEL_DIR / "xgboost_change_model.json")

# ----------------------------------------------------------------------
# 8. Actual vs predicted graph
# ----------------------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(comp["date"], comp["actual"], label="Actual", linewidth=2.5, color="black")
plt.plot(comp["date"], comp["baseline"], "--", label="Baseline (persistence)", alpha=0.8)
if "regression" in comp:
    plt.plot(comp["date"], comp["regression"], ":", linewidth=2, label="Linear regression")
plt.plot(comp["date"], comp["xgboost_prediction"], label="XGBoost (level)")
plt.plot(comp["date"], comp["xgboost_change_prediction"], label="XGBoost (change)")
plt.title("HPI 3 Months Ahead: Actual vs Predicted (Test Set)")
plt.xlabel("Date")
plt.ylabel("House Price Index")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "xgboost_actual_vs_predicted.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 9. Residual plot (actual - predicted)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].axhline(0, color="gray", linewidth=1)
ax[0].plot(out["date"], out["xgboost_residual"], "o-", label="XGBoost (level)")
ax[0].plot(out["date"], out["xgboost_change_residual"], "s-", label="XGBoost (change)")
ax[0].set_title("Residuals over time")
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Actual - Predicted")
ax[0].legend()

ax[1].axhline(0, color="gray", linewidth=1)
ax[1].scatter(out["xgboost_prediction"], out["xgboost_residual"], label="XGBoost (level)")
ax[1].scatter(out["xgboost_change_prediction"], out["xgboost_change_residual"],
              marker="s", label="XGBoost (change)")
ax[1].set_title("Residuals vs predicted value")
ax[1].set_xlabel("Predicted HPI")
ax[1].set_ylabel("Actual - Predicted")
ax[1].legend()
plt.tight_layout()
plt.savefig(FIGS / "xgboost_residual_plot.png", dpi=150)
plt.close()

print("Saved predictions, metrics, comparison, models, and both figures.")
