"""
Data Quality & Validation for TXSTHPI (Texas All-Transactions House Price Index, quarterly)
Input : TXSTHPI.csv  (columns: observation_date, TXSTHPI)
Output: TXSTHPI_clean.csv + plots + printed audit report
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "TXSTHPI.csv"

# ----------------------------------------------------------------------
# 0. LOAD + STANDARDIZE NAMES / FORMATS
# ----------------------------------------------------------------------
raw = pd.read_csv(FILE)
print("Raw shape:", raw.shape)
print("Raw columns:", raw.columns.tolist())
print("Raw dtypes:\n", raw.dtypes, "\n")

df = raw.copy()

# Column naming: strip whitespace, lowercase, snake_case
df.columns = (df.columns.str.strip()
                        .str.lower()
                        .str.replace(r"[^0-9a-z]+", "_", regex=True))
df = df.rename(columns={"observation_date": "date", "txsthpi": "hpi"})

# Formatting: parse dates, force numeric (bad strings like "." become NaN)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["hpi"] = pd.to_numeric(df["hpi"], errors="coerce")

# ----------------------------------------------------------------------
# 1. DUPLICATES (rows, dates, columns)
# ----------------------------------------------------------------------
print("=" * 60, "\n1. DUPLICATES")
print("Fully duplicated rows      :", df.duplicated().sum())
print("Duplicated dates           :", df["date"].duplicated().sum())
print("Duplicated column names    :", df.columns.duplicated().sum())
# Duplicate columns by content (same values under different names)
dup_cols = df.T.duplicated().sum()
print("Duplicated column contents :", dup_cols)

df = df.drop_duplicates()                              # drop identical rows
df = df.drop_duplicates(subset="date", keep="last")    # one row per date
df = df.loc[:, ~df.columns.duplicated()]               # drop duplicate col names

# ----------------------------------------------------------------------
# 2. DATE RANGE + CHRONOLOGICAL ORDER + FREQUENCY
# ----------------------------------------------------------------------
print("=" * 60, "\n2. DATES & ORDER")
print("Was sorted ascending?      :", df["date"].is_monotonic_increasing)
df = df.sort_values("date").reset_index(drop=True)

print("Date range                 :", df["date"].min().date(), "->", df["date"].max().date())
print("Null/unparseable dates     :", df["date"].isna().sum())

# All dates should be quarter starts (Jan/Apr/Jul/Oct, day 1)
bad_dates = df[~((df["date"].dt.day == 1) & (df["date"].dt.month.isin([1, 4, 7, 10])))]
print("Dates not on quarter start :", len(bad_dates))

# Build the full expected quarterly calendar and find gaps
full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="QS")  # QS = quarter start (Jan/Apr/Jul/Oct)
missing_quarters = full_idx.difference(df["date"])
print("Expected quarters          :", len(full_idx))
print("Actual rows                :", len(df))
print("Missing quarters           :", list(missing_quarters.date))

# ----------------------------------------------------------------------
# 3. MISSING VALUES
# ----------------------------------------------------------------------
print("=" * 60, "\n3. MISSING VALUES")
# Reindex to the full calendar so missing quarters show up as NaN rows
df = df.set_index("date").reindex(full_idx).rename_axis("date")

miss_n = df["hpi"].isna().sum()
miss_pct = miss_n / len(df) * 100
print(f"Missing hpi values: {miss_n} ({miss_pct:.2f}%)")

# Fill strategy: this is a PRICE INDEX, so filling with 0 would be wrong
# (0 = prices collapsed to nothing, and it would wreck growth rates/outliers).
# Use time-based linear interpolation for interior gaps instead.
# Use zeros ONLY for count/flow style columns (e.g., number of sales), not indexes.
df["hpi_was_missing"] = df["hpi"].isna()
df["hpi"] = df["hpi"].interpolate(method="time", limit_area="inside")
print("Remaining NaNs after fill  :", df["hpi"].isna().sum())

# ----------------------------------------------------------------------
# 4. UNITS / VALUE SANITY CHECKS
# ----------------------------------------------------------------------
print("=" * 60, "\n4. UNITS & SANITY")
print("dtype                      :", df["hpi"].dtype)
print("Negative or zero values    :", (df["hpi"] <= 0).sum())
print("Max decimals observed      :", raw.iloc[:, 1].astype(str).str.split(".").str[1].str.len().max())
# Index is rebased so that 1980-01-01 = 100 (FRED convention for this series)
print("Value at 1980-01-01 (base) :", df.loc["1980-01-01", "hpi"])
df["hpi"] = df["hpi"].round(2)

# ----------------------------------------------------------------------
# 5. OUTLIERS
# ----------------------------------------------------------------------
print("=" * 60, "\n5. OUTLIERS")
# A trending index will always look "outlying" in LEVELS, so test the
# quarter-over-quarter % change instead (stationary-ish).
df["qoq_pct"] = df["hpi"].pct_change() * 100

# Method A: IQR on QoQ change
q1, q3 = df["qoq_pct"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["outlier_iqr"] = (df["qoq_pct"] < lo) | (df["qoq_pct"] > hi)

# Method B: z-score on QoQ change
z = (df["qoq_pct"] - df["qoq_pct"].mean()) / df["qoq_pct"].std()
df["outlier_z"] = z.abs() > 3

print(f"IQR bounds for QoQ %       : [{lo:.2f}, {hi:.2f}]")
print("Flagged by IQR             :", df["outlier_iqr"].sum())
print("Flagged by |z|>3           :", df["outlier_z"].sum())
print("\nFlagged quarters:")
print(df.loc[df["outlier_iqr"] | df["outlier_z"], ["hpi", "qoq_pct"]].round(2))

# DECISION: RETAIN. The flagged points cluster in 1970s-80s inflation/oil boom
# and 2021-22 (pandemic-era housing surge). These are real, documented
# market events, not data-entry errors, so removing them would bias the series.
# We keep them and carry the flag columns for modeling/robustness checks.
# Only remove if a value is impossible (negative, typo-like jump that reverses).
df["outlier_any"] = df["outlier_iqr"] | df["outlier_z"]

# ----------------------------------------------------------------------
# 6. VALUE COUNTS & DISTRIBUTION
# ----------------------------------------------------------------------
print("=" * 60, "\n6. DISTRIBUTION & VALUE COUNTS")
print(df["hpi"].describe().round(2), "\n")
print("Skew    :", round(df["hpi"].skew(), 3))
print("Kurtosis:", round(df["hpi"].kurt(), 3), "\n")

# Observations per quarter (should be perfectly balanced)
print("Rows per quarter-of-year:")
print(df.index.quarter.value_counts().sort_index(), "\n")

# Observations per decade
print("Rows per decade:")
print(((df.index.year // 10) * 10).value_counts().sort_index(), "\n")

# Frequency of HPI values in bins (levels)
bins = pd.cut(df["hpi"], bins=[0, 100, 150, 200, 300, 400, 600])
print("HPI value-count by bin:")
print(bins.value_counts().sort_index(), "\n")

# Frequency of QoQ changes in bins
chg_bins = pd.cut(df["qoq_pct"], bins=[-10, -2, -1, 0, 1, 2, 4, 10])
print("QoQ % change frequency:")
print(chg_bins.value_counts().sort_index(), "\n")

# Any repeated exact values? (could signal stale/copied data)
print("Repeated HPI values:", df["hpi"].duplicated().sum())

# ----------------------------------------------------------------------
# 7. PLOTS
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8))

ax[0, 0].plot(df.index, df["hpi"])
ax[0, 0].set_title("TXSTHPI over time")

ax[0, 1].hist(df["hpi"], bins=20)
ax[0, 1].set_title("Distribution of HPI levels")

ax[1, 0].plot(df.index, df["qoq_pct"])
ax[1, 0].scatter(df.index[df["outlier_any"]], df.loc[df["outlier_any"], "qoq_pct"], color="red", zorder=3)
ax[1, 0].axhline(hi, ls="--", c="gray"); ax[1, 0].axhline(lo, ls="--", c="gray")
ax[1, 0].set_title("QoQ % change (red = flagged)")

ax[1, 1].boxplot(df["qoq_pct"].dropna(), vert=False)
ax[1, 1].set_title("Boxplot of QoQ % change")

plt.tight_layout()
plt.savefig("txsthpi_quality_plots.png", dpi=150)

# ----------------------------------------------------------------------
# 8. SAVE CLEANED DATA
# ----------------------------------------------------------------------
df.reset_index().to_csv("TXSTHPI_clean.csv", index=False)
print("\nSaved TXSTHPI_clean.csv and txsthpi_quality_plots.png")