import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/cleaned/wti-crude-oil/MCOILWTICO_cleaned.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Predict the oil price one month ahead
df["actual_future"] = df["WTI_Price_USD_per_Barrel"].shift(-1)
df["baseline_prediction"] = df["WTI_Price_USD_per_Barrel"]

# Remove rows with missing actual or predicted values
df = df.dropna(subset=["actual_future", "baseline_prediction"]).copy()

# Split chronologically into 80% training and 20% testing
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

# Calculate test-set evaluation metrics
actual = test["actual_future"]
predicted = test["baseline_prediction"]

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)

print("Total observations:", len(df))
print("Training observations:", len(train))
print("Testing observations:", len(test))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R-squared:", round(r2, 4))

# Save modeling spreadsheet
df.to_csv("models/wti-crude-oil/baseline_predictions.csv", index=False)

# Actual vs predicted graph for test data
plt.figure(figsize=(11, 5))
future_dates = test["Date"] + pd.DateOffset(months=1)
plt.plot(future_dates, actual, label="Actual future price")
plt.plot(future_dates, predicted, label="Baseline prediction")
plt.xlabel("Prediction date")
plt.ylabel("WTI oil price (USD per barrel)")
plt.title("WTI Oil Price: Actual vs Baseline Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("models/wti-crude-oil/actual_vs_predicted.png")
plt.show()
