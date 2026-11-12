
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned WTI oil prices
oil = pd.read_csv("data/cleaned/wti-crude-oil/MCOILWTICO_cleaned.csv")
oil["Date"] = pd.to_datetime(oil["Date"])
oil["WTI_Price_USD_per_Barrel"] = pd.to_numeric(
    oil["WTI_Price_USD_per_Barrel"], errors="coerce"
)

# Load manufacturing index
manufacturing = pd.read_csv("Manufacturing_Index_data.csv")
manufacturing["Date"] = pd.to_datetime(
    manufacturing["Date"], format="%b-%y", errors="coerce"
)
manufacturing["production"] = pd.to_numeric(
    manufacturing["production"], errors="coerce"
)
manufacturing = manufacturing.dropna(subset=["Date", "production"])

# Match both datasets by month
df = pd.merge(
    oil[["Date", "WTI_Price_USD_per_Barrel"]],
    manufacturing[["Date", "production"]],
    on="Date",
    how="inner"
)
df = df.sort_values("Date").reset_index(drop=True)

# Create predictors from current and recent oil prices
df["oil_current"] = df["WTI_Price_USD_per_Barrel"]
df["oil_lag_1"] = df["oil_current"].shift(1)
df["oil_lag_2"] = df["oil_current"].shift(2)
df["manufacturing_index"] = df["production"]

# Target: next month's oil price
df["actual_future"] = df["oil_current"].shift(-1)

predictors = [
    "oil_current",
    "oil_lag_1",
    "oil_lag_2",
    "manufacturing_index"
]

# Keep required columns and remove missing values
df = df[["Date"] + predictors + ["actual_future"]]
df = df.dropna().reset_index(drop=True)

# Split chronologically: 80% training, 20% testing
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[predictors]
y_train = train["actual_future"]
X_test = test[predictors]
y_test = test["actual_future"]

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future oil prices
predictions = model.predict(X_test)

# Evaluates model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Total observations:", len(df))
print("Training observations:", len(train))
print("Testing observations:", len(test))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R-squared:", round(r2, 4))

# Print coefficients and intercept
print("\nCoefficients:")
for name, coefficient in zip(predictors, model.coef_):
    print(name, ":", round(coefficient, 4))

print("Intercept:", round(model.intercept_, 4))

# Save predictions
results = test[["Date", "actual_future"]].copy()
results["predicted_future"] = predictions
results["Target_Date"] = results["Date"] + pd.DateOffset(months=1)
results.to_csv(
    "models/wti-crude-oil/linear_regression_predictions.csv",
    index=False
)

# Actual vs predicted graph
plt.figure(figsize=(11, 5))
plt.plot(
    results["Target_Date"], results["actual_future"],
    label="Actual future price"
)
plt.plot(
    results["Target_Date"], results["predicted_future"],
    label="Linear regression prediction"
)
plt.xlabel("Target month")
plt.ylabel("WTI oil price (USD per barrel)")
plt.title("WTI Oil Price: Actual vs Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("models/wti-crude-oil/linear_regression_graph.png")
plt.show()
