# Deliverables & Timeline

> **How to read this file.** This is the PMs' best current estimate of what the project needs to ship and roughly when. It is a **living plan, not a contract** — deliverables will be added, dropped, split, or resequenced as the team learns more. The authoritative, up-to-the-minute picture always lives in the repo's **GitHub Issues and Project board**; this file is the high-level narrative that keeps everyone oriented.
>
> PMs: replace the placeholder rows below with your real deliverables. Keep each deliverable small enough to become one or a few issues.

## Milestones at a glance

| # | Deliverable | Description | Owner (role) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | Project scoping | Define the problem, success criteria, and out-of-scope items. | PM | Week 1 |
| 2 | Data sourcing & access | Identify and secure the data sources (see [`DATA.md`](DATA.md)). | PM + members | Weeks 1–2 |
| 3 | Data exploration (EDA) | Load, profile, and document the data; surface quality issues. | Members | Weeks 2–3 |
| 4 | Data cleaning & prep | Reproducible pipeline from raw → analysis-ready. | Members | Weeks 3–4 |
| 5 | Baseline model / analysis | First end-to-end result to beat. | Members | Weeks 4–5 |
| 6 | Iteration & evaluation | Improve on the baseline; agree on evaluation metrics. | Members + PM | Weeks 5–7 |
| 7 | Findings & deliverable | Report / dashboard / model artifact for the audience. | Members + PM | Weeks 7–8 |
| 8 | Handoff & retro | Documentation, reproducibility check, lessons learned. | PM | Week 8 |

## Timeline (rough)

```
Week:   1     2     3     4     5     6     7     8
        |-----|-----|-----|-----|-----|-----|-----|
Scope   ██
Data          ████
EDA                 ████
Prep                      ████
Baseline                        ██
Iterate                              ████████
Deliver                                       ████
Retro                                              ██
```

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Dates are estimates.** When reality diverges, update the issue and, if the shift is material, this file. Don't let this file quietly go stale.
- **"Done" is defined per issue** via acceptance criteria — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.

- Week of 10/05
Objective: Data Collection

Economic Health Indicators (variables we'll be using to our economic health index) 
Unemployment Rate - Keerthi
Non-farm employment - Jaira
Real Personal Income Growth - Alexis
Texas House Price Index - Brandon
WTI crude oil price - Katie
Dallas Fed Texas Manufacturing index - Andrew

Reason for shortlisting these 6 variables: Covers main economic drivers of labor, output, housing and energy

Suggested Data Sources:
FRED
Dallas Fed
BLS
Census
Kaggle

Time Period: 10 years (5 years minimum) with monthly/yearly data points

Subteams: Each team member will be in charge of one Texas Economic Health Indicator variable 

To-do:
Is everyone able to access the github repo for macrolab?
Assign a variable to each member
1 dataset for each variable due eod

Github set up
install git
clone repo (repo url can be accessed when u click on the green code button on macrolab github repo)
  git clone REPOSITORY_URL
  cd REPOSITORY_NAME
create own branch 
  git switch -c data/VARIABLE_NAME
Github cheat sheet: https://docs.google.com/document/d/1ZGp9oosNhfgPfZLwb4571PIa3mPZ-ZR9ISJtEC2dKgI/edit?tab=t.0
add datasets to repo
   git add FILE_NAME.csv
   git commit -m "Add Texas VARIABLE data"
   git push -u origin data/VARIABLE_NAME

Annoucements/Reminders
deliverales.md will be updated for a detailed semester plan soon
deliverables (data cleaning) for week of 10/11 (due 10/17) will be posted eod along with related resources
complete your deliverables eod (if u won't be able to work on it later then finish it during team meeting) & send update in status update channel
check discord
pls reach out if u need help

Week of 10/11
Objective:  Data Cleaning and Exploratory Data Analysis

To-do
Data Quality and Validation: 
Account for missing values (calculate missing %/values if they can be using the data available or add zeros whenever necessary)
Check and correct duplicate rows & columns, date ranges, chronological order of data, units, formatting, columns/rows naming etc.
Check for outliers and decide whether to retain or remove based on context
Validate value counts and distribution frequency
Calculate count, mean, median, standard deviation, variance, minimum, maximum, Q1, Q3, IQR and outliers.
Create bar graphs with meaningful categories (ex: categorize as "rising, falling or unchanged"  or "positive, negative or zero" as needed)
If your data can be represented well, then create histograms, scatter plots, box plots, etc. as necessary
Create time-series line chart

Deliverables:
Cleaned datasets should be saved in data/cleaned/[VARIABLE_NAME]
Post notes summary on missing values, any visualization/cleaning decisions or observations (can be bullet points) in status updates (be as detailed as possible, I would suggets taking notes as you got through each step so there is a record of explanation for everything)

I know this is a lot, so
1 on the to-do is due eod Saturday (10/17) as a hard deadline
2-5 will be due eod next wednesday (10/21), but it would be great to have them done by this weekend.

Next meeting: Saturday (10/17) 3pm 

For those of you who can't make it to the meeting, I'll be posting meeting notes and we can also set up a time to quickly hop on call if needed.

These are some resources that might be helpful: 
https://www.geeksforgeeks.org/data-analysis/data-cleaning-introduction/
https://www.geeksforgeeks.org/python/graph-plotting-in-python-set-1/

Please let me know if you need any help and communicate if you're having any issues. 

Due eod Saturday 10/17
Data Quality and Validation: 
Account for missing values (calculate missing %/values if they can be using the data available or add zeros whenever necessary)
Check for outliers and decide whether to retain or remove based on context
Check and correct duplicate rows & columns, date ranges, chronological order of data, units, formatting, columns/rows naming etc.
Validate value counts and distribution frequency

Due eod Wednesday 10/21
Calculate count, mean, median, standard deviation, variance, minimum, maximum, Q1, Q3, IQR and outliers.
Create bar graphs with meaningful categories (ex: categorize as "rising, falling or unchanged"  or "positive, negative or zero" as needed)
If your data can be represented well, then create histograms, scatter plots, box plots, etc. as necessary
Create time-series line chart

Deliverables:
Cleaned datasets should be saved in data/cleaned/[VARIABLE_NAME]
Post notes summary on missing values, any visualization/cleaning decisions or observations (can be bullet points) in status updates (be as detailed as possible, I would suggest taking notes as you got through each step so there is a record of explanation for everything)

Objective: Modeling

Due eod Wednesday (Baseline Model)
We'll be using persistence baseline so the future value should be the same as the most recently observed value.
If needed convert datasets into monthly average (if ur dataset has weekly or daily values)
create a spreadsheet with just date (need to be oldest to newest order), variable (ex: unemployment rate, income growth rate, etc.) and future target columns
for future target column, we need today's value to match what you want to predict in the future so move future observation upward by the time period length so that each value contains the value 3/1 month later. (Ex: df["actual_future"] = df["unemployment_rate"].shift(-3))
create baseline predictions (ex: df["baseline_prediction"] = df["unemployment_rate"]) & remove rows without a future target
split data into training and testing sets (80% training, 20% testing)
evaluate model using Mean Absolute Error, Root Mean Squared Error & Coefficient of Determination
create an actual vs predicted graph


These are the predictions time periods for each variable:
Unemployment    3 months
Nonfarm employment        3 months
Real personal income growth        1 quarter
House Price Index    3 months
WTI oil price    1 month
Manufacturing index    1 month

Due eod Saturday (Predictive Model - Linear regression)
Use predictors sent in the next message
keep only predictors & targets in dataset
check if columns are numeric (if text then convert to numeric)
split data 80% training,20% testing
train model & generate predictions
evaluate model using Mean Absolute Error, Root Mean Squared Error & Coefficient of Determination
print coefficients & intercepts

for predictors, 
Unemployment: Current unemployment, employment growth, manufacturing index, oil price
Nonfarm employment: Current employment, unemployment, manufacturing index, oil price
Personal income growth: Current income growth, employment growth, unemployment, housing
House Price Index: Current HPI, employment, unemployment, income growth
WTI oil price: Current oil price, recent oil prices, manufacturing index
Manufacturing index: Current index, employment growth, oil price, unemployment

Due eod next Monday (Compare baseline & lin reg models)
create model comparison table
create actual vs predicted graph (actual, baseline predictions & linear regression predictions)
Create residual plot
save predictions, evaluation metrics, coefficients & trained models

The folder structure should look like this on github:

VARIABLE NAME/
│
├── data/
│   └── modeling_data.csv
│
├── model/
│   ├── baseline_model.py
│   └── linear_regression.py
│
├── results/
│   ├── baseline_predictions.csv
│   ├── regression_predictions.csv
│   ├── model_comparison.csv
│   └── coefficients.csv
│
├── figures/
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
└── report.md

Deliverables
update github
send short summary about your model in the status updates once you're done

Next Meeting: 3pm Saturday Nov 7th on discord

Wednesday is just a checkpoint but Saturday is a hard deadline. Please communicate if you have any questions or issues or won't be able to complete all the deliverables on time and we'll work out a different timeline. 

Respurces:
https://skforecast.org/latest/user_guides/forecasting-baseline.html
https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
https://www.geeksforgeeks.org/machine-learning/simple-linear-regression-in-python/ 


