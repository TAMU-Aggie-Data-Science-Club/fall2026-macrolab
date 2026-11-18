import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared modeling data
df = pd.read_csv("wti-crude-oil/data/modeling_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

predictors = [
    "oil_current",
    "oil_lag_1",
    "oil_lag_2",
    "manufacturing_index"
]

# Split data chronologically into 80% training and 20% testing
split = int(len(df) * 0.8)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# Train linear regression model
model = LinearRegression()
model.fit(train[predictors], train["actual_future"])

# Generate predictions
actual = test["actual_future"]
predicted = model.predict(test[predictors])

# Calculate evaluation metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("Linear Regression MAE:", round(mae, 4))
print("Linear Regression RMSE:", round(rmse, 4))
print("Linear Regression R-squared:", round(r2, 4))

# Print coefficients and intercept
print("\nCoefficients:")
for name, coefficient in zip(predictors, model.coef_):
    print(name, ":", round(coefficient, 4))

print("Intercept:", round(model.intercept_, 4))

# Save predictions
results = test[["Date", "actual_future"]].copy()
results["Target_Date"] = results["Date"] + pd.DateOffset(months=1)
results["regression_prediction"] = predicted

results.to_csv(
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

# Save trained model
joblib.dump(model, "wti-crude-oil/results/linear_regression.joblib")

print("\nLinear regression results saved!")

