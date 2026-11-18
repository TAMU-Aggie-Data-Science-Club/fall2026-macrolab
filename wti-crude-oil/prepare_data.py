import pandas as pd

# Load WTI oil prices
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

# Combine both datasets by month
df = pd.merge(
    oil[["Date", "WTI_Price_USD_per_Barrel"]],
    manufacturing[["Date", "production"]],
    on="Date",
    how="inner"
)

df = df.sort_values("Date").reset_index(drop=True)

# Create predictors
df["oil_current"] = df["WTI_Price_USD_per_Barrel"]
df["oil_lag_1"] = df["oil_current"].shift(1)
df["oil_lag_2"] = df["oil_current"].shift(2)
df["manufacturing_index"] = df["production"]

# Predict the following month's price
df["actual_future"] = df["oil_current"].shift(-1)

# Keep only required columns
df = df[
    [
        "Date",
        "oil_current",
        "oil_lag_1",
        "oil_lag_2",
        "manufacturing_index",
        "actual_future",
    ]
]

# Remove rows with missing predictors or targets
df = df.dropna().reset_index(drop=True)

# Save modeling data
df.to_csv("wti-crude-oil/data/modeling_data.csv", index=False)

print("Modeling dataset saved!")
print("Total observations:", len(df))
print(df.head())
