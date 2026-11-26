import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared modeling data
df = pd.read_csv("wti-crude-oil/data/modeling_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Use the same predictors as linear regression
predictors = [
    "oil_current",
    "oil_lag_1",
    "oil_lag_2",
    "manufacturing_index"
]

# Split data into 80% training and 20% testing
split = int(len(df) * 0.8)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# Train XGBoost model
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

model.fit(train[predictors], train["actual_future"])

# Generate predictions
actual = test["actual_future"]
predicted = model.predict(test[predictors])

# Calculate evaluation metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("Training observations:", len(train))
print("Testing observations:", len(test))

print("XGBoost MAE:", round(mae, 4))
print("XGBoost RMSE:", round(rmse, 4))
print("XGBoost R-squared:", round(r2, 4))


# Save test-period predictions
results = test[["Date", "actual_future"]].copy()
results["Target_Date"] = results["Date"] + pd.DateOffset(months=1)
results["xgboost_prediction"] = predicted

results.to_csv(
    "wti-crude-oil/results/xgboost_predictions.csv",
    index=False
)

# Save evaluation metrics
metrics = pd.DataFrame({
    "Model": ["XGBoost"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

metrics.to_csv(
    "wti-crude-oil/results/xgboost_metrics.csv",
    index=False
)

# Save trained model
joblib.dump(
    model,
    "wti-crude-oil/results/xgboost_model.joblib"
)


import matplotlib.pyplot as plt

# Actual vs. predicted graph
plt.figure(figsize=(12, 6))

plt.plot(results["Target_Date"], actual,
         label="Actual", marker="o")

plt.plot(results["Target_Date"], predicted,
         label="XGBoost", marker="o")

plt.title("WTI Crude Oil: Actual vs. XGBoost Predictions")
plt.xlabel("Date")
plt.ylabel("Oil Price (USD per Barrel)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "wti-crude-oil/figures/xgboost_actual_vs_predicted.png",
    dpi=300
)

plt.close()

# Residual plot
residuals = actual - predicted

plt.figure(figsize=(10, 6))
plt.scatter(predicted, residuals)
plt.axhline(y=0, color="red", linestyle="--")

plt.title("XGBoost Residual Plot")
plt.xlabel("Predicted Oil Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "wti-crude-oil/figures/xgboost_residual_plot.png",
    dpi=300
)

plt.close()
