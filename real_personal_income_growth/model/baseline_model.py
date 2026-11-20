# Persistence Baseline Model
import pandas as pd
import numpy as np

df = pd.read_csv('../data/modeling_data.csv')
df['baseline_prediction'] = df['income_growth']
