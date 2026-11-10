import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared modeling data
df = pd.read_csv("wti-crude-oil/data/modeling_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Use the same chronological 80/20 split as linear regression
split = int(len(df) * 0.8)
test = df.iloc[split:].copy()

# Persistence baseline: next month's price equals today's price
actual = test["actual_future"]
predicted = test["oil_current"]

print("Baseline MAE:", round(mean_absolute_error(actual, predicted), 4))
print("Baseline RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 4))
print("Baseline R-squared:", round(r2_score(actual, predicted), 4))

# Save predictions
results = test[["Date", "actual_future"]].copy()
results["Target_Date"] = results["Date"] + pd.DateOffset(months=1)
results["baseline_prediction"] = predicted.to_numpy()

results.to_csv(
    "wti-crude-oil/results/baseline_predictions.csv",
    index=False
)

print("Baseline predictions saved!")
