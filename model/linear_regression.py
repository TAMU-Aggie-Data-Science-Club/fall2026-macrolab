from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Re-load the baseline dataframe from drive or workspace
file_path = '/content/drive/MyDrive/TXNA-2.csv'
df = pd.read_csv(file_path)
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Load the other three CSV files
mcoil_df = pd.read_csv('/content/MCOILWTICO.csv')
manuf_df = pd.read_csv('/content/Manufacturing_Index_data.csv')
txur_df = pd.read_csv('/content/TXUR.csv')

# Standardize dates using robust format checking to handle 'Aug-16' properly
def standardize_date_col(temp_df):
    for col in ['DATE', 'Date', 'observation_date']:
        if col in temp_df.columns:
            # Try parsing with format '%b-%y' if strings look like 'Aug-16'
            first_val = str(temp_df[col].iloc[0]).strip()
            if '-' in first_val and not first_val.split('-')[0].isdigit() and len(first_val.split('-')[-1]) == 2:
                temp_df['observation_date'] = pd.to_datetime(temp_df[col], format='%b-%y')
            else:
                temp_df['observation_date'] = pd.to_datetime(temp_df[col], errors='coerce')
            
            if col != 'observation_date':
                temp_df.drop(columns=[col], inplace=True)
            return temp_df
    raise KeyError("No date column found in dataframe.")

mcoil_df = standardize_date_col(mcoil_df)
manuf_df = standardize_date_col(manuf_df)
txur_df = standardize_date_col(txur_df)

# 2. Sequentially merge all dataframes together on 'observation_date'
df_merged = df[['observation_date', 'TXNA']].copy()
df_merged = pd.merge(df_merged, mcoil_df, on='observation_date', how='inner')
df_merged = pd.merge(df_merged, manuf_df, on='observation_date', how='inner')
df_merged = pd.merge(df_merged, txur_df, on='observation_date', how='inner')

# Assign back to df
df = df_merged
print("Combined Workspace Dataframe Columns:", df.columns.tolist())

# 3. Check and clean each column to ensure they are purely numeric
for col in df.columns:
    if col != 'observation_date':
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Cleaning non-numeric column: {col}")
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace('$', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill any NaNs created during conversion with forward fill
df = df.ffill().bfill().fillna(0)

print("\n--- Verified Data Types of Combined Dataframe ---")
print(df.dtypes)
display(df.head())

# Create the future target: TXNA 3 months in the future (shift of -3)
df['future_target'] = df['TXNA'].shift(-3)

# Drop the last 3 rows since they won't have a future target
df_model = df.dropna(subset=['future_target']).copy()

# Define the predictors and target variable
predictors = ['TXNA', 'TXUR', 'production', 'MCOILWTICO']
target = 'future_target'

# Chronological split (80% training, 20% testing)
split_idx = int(len(df_model) * 0.80)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train = train_df[predictors]
y_train = train_df[target]
X_test = test_df[predictors]
y_test = test_df[target]

# Initialize and fit the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Generate predictions
train_preds = lr_model.predict(X_train)
test_preds = lr_model.predict(X_test)

# Evaluate the model metrics
mae = mean_absolute_error(y_test, test_preds)
mse = mean_squared_error(y_test, test_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_preds)

print("--- Multivariate Linear Regression Results ---")
print(f"Training set size: {len(X_train)} months")
print(f"Testing set size: {len(X_test)} months\n")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Coefficient of Determination (R2 Score): {r2:.4f}\n")

print("--- Model Parameters ---")
print(f"Intercept: {lr_model.intercept_:.4f}")
for feature, coef in zip(predictors, lr_model.coef_):
    print(f"{feature} Coefficient: {coef:.4f}")
