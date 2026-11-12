import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure chronological order and prepare the core dataframe
df_baseline = df[['observation_date', 'TXNA']].copy()
df_baseline['observation_date'] = pd.to_datetime(df_baseline['observation_date'])
df_baseline = df_baseline.sort_values('observation_date').reset_index(drop=True)

# Create future target (TXNA 3 months in the future, shifting up by 3 periods for monthly data)
df_baseline['future_target'] = df_baseline['TXNA'].shift(-3)

# Create baseline prediction (using current TXNA as prediction for 3 months ahead)
df_baseline['baseline_prediction'] = df_baseline['TXNA']

# Remove rows that do not have a future target (the last 3 months)
df_baseline_clean = df_baseline.dropna(subset=['future_target']).copy()

# Display the prepared dataframe
print("--- Prepared Baseline Dataframe (First 5 Rows) ---")
display(df_baseline_clean[['observation_date', 'TXNA', 'future_target', 'baseline_prediction']].head())

# Split into training (80%) and testing (20%) sets chronologically
split_idx = int(len(df_baseline_clean) * 0.80)
train_df = df_baseline_clean.iloc[:split_idx]
test_df = df_baseline_clean.iloc[split_idx:]

# Evaluate the baseline predictions on the test set
y_true = test_df['future_target']
y_pred = test_df['baseline_prediction']

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\n--- Baseline Model Evaluation (Test Set) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Coefficient of Determination (R2 Score): {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(train_df['observation_date'], train_df['future_target'], label='Train Actual (Future Target)', color='gray', alpha=0.7)
plt.plot(test_df['observation_date'], y_true, label='Test Actual (Future Target)', color='#1f77b4', linewidth=2)
plt.plot(test_df['observation_date'], y_pred, label='Baseline Prediction (Today\'s TXNA)', color='#ff7f0e', linestyle='--', linewidth=2)

plt.title('Baseline Model: Actual vs Predicted (3-Month Forecast)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('TXNA Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
