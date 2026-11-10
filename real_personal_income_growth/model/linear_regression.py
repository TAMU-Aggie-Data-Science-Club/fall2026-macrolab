# Linear Regression Model
import pandas as pd
import numpy as np

df = pd.read_csv('../data/modeling_data.csv')
import pandas as pd
import numpy as np

# Loads the cleaned modeling data and sets the predictor
df = pd.read_csv("real_personal_income_growth/data/modeling_data.csv")
predictors = ["income_growth"]

# This just splits the data 80% training, and then 20% testing
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[predictors].values
y_train = train_df["actual_future"].values
X_test = test_df[predictors].values
y_test = test_df["actual_future"].values

# This trains the linear regression model and adds a column of 1s to X_train so it can calculate the intercept
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
theta, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
intercept = theta[0]
coefficients = theta[1:]

# Generates the predictions on the testing data
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
predictions = X_test_b.dot(theta)

# Evaluates the model using MAE, RMSE, and R^2
errors = y_test - predictions
mae = np.abs(errors).mean()
rmse = np.sqrt((errors ** 2).mean())

# Finds the coefficient of determination which would be the R^2 value
ss_res = (errors ** 2).sum()
ss_tot = ((y_test - y_test.mean()) ** 2).sum()
r2 = 1 - (ss_res / ss_tot)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2: {r2:.2f}\n")

print(f"Intercept: {intercept:.4f}")
for name, coef in zip(predictors, coefficients):
    print(f"Coefficient for {name}: {coef:.4f}")