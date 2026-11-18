import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the prepared dataset
df = pd.read_csv("wti-crude-oil/data/modeling_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

predictors = [
    "oil_current",
    "oil_lag_1",
    "oil_lag_2",
    "manufacturing_index"
]

# Chronological 80/20 split
split = int(len(df) * 0.8)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# Persistence baseline: next month = current month
baseline = test["oil_current"].to_numpy()

# Train linear regression using training observations only
model = LinearRegression()
model.fit(train[predictors], train["actual_future"])
regression = model.predict(test[predictors])

actual = test["actual_future"].to_numpy()

# Evaluate both on exactly the same test observations
def evaluate(name, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

comparison = pd.DataFrame([
    evaluate("Persistence Baseline", baseline),
    evaluate("Linear Regression", regression)
])

print("Training observations:", len(train))
print("Testing observations:", len(test))
print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))

# Save evaluation metrics
comparison.to_csv(
    "wti-crude-oil/results/model_comparison.csv",
    index=False
)

# Save both sets of predictions
results = test[["Date", "actual_future"]].copy()
results["Target_Date"] = results["Date"] + pd.DateOffset(months=1)
results["baseline_prediction"] = baseline
results["regression_prediction"] = regression

results[
    ["Date", "Target_Date", "actual_future", "baseline_prediction"]
].to_csv(
    "wti-crude-oil/results/baseline_predictions.csv",
    index=False
)

results[
    ["Date", "Target_Date", "actual_future", "regression_prediction"]
].to_csv(
    "wti-crude-oil/results/regression_predictions.csv",
    index=False
)

# Save coefficients and intercept
coefficients = pd.DataFrame({
    "Predictor": predictors + ["Intercept"],
    "Coefficient": list(model.coef_) + [model.intercept_]
})
coefficients.to_csv(
    "wti-crude-oil/results/coefficients.csv",
    index=False
)

print("\nCoefficients:")
print(coefficients.round(4).to_string(index=False))

# Save trained regression model
joblib.dump(model, "wti-crude-oil/results/linear_regression.joblib")

# Graph 1: Actual vs both predictions
plt.figure(figsize=(11, 5))
plt.plot(results["Target_Date"], actual, label="Actual price")
plt.plot(results["Target_Date"], baseline, label="Baseline prediction")
plt.plot(results["Target_Date"], regression, label="Linear regression")
plt.xlabel("Target month")
plt.ylabel("WTI oil price (USD per barrel)")
plt.title("WTI Oil Price: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("wti-crude-oil/figures/actual_vs_predicted.png")
plt.close()

# Graph 2: Residuals (actual minus predicted)
plt.figure(figsize=(11, 5))
plt.plot(
    results["Target_Date"],
    actual - baseline,
    marker="o",
    label="Baseline residual"
)
plt.plot(
    results["Target_Date"],
    actual - regression,
    marker="o",
    label="Linear regression residual"
)
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Target month")
plt.ylabel("Residual (USD per barrel)")
plt.title("WTI Oil Price: Model Residuals")
plt.legend()
plt.tight_layout()
plt.savefig("wti-crude-oil/figures/residual_plot.png")
plt.close()

print("\nAll comparison files saved successfully!")
