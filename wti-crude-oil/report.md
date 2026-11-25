#### WTI Crude Oil Model Comparison



###### Objective



The goal of this project was to predict WTI crude oil prices one month ahead using a persistence baseline model, a linear regression model, and an XGBoost model.



###### Data and Results



I used current oil prices, oil prices from the previous two months, and the manufacturing index as predictors for linear regression and XGBoost. The data were divided into training and testing sets to evaluate all three models. The results showed that linear regression had lower prediction errors and performed better overall compared to the baseline and XGBoost models.



###### Conclusion



The linear regression model was the most accurate at predicting future oil prices. Although XGBoost is a more complex machine learning model, it did not improve prediction accuracy. This may be due to the limited dataset size and model settings. Future improvements could include adjusting these settings and exploring additional economic indicators.

