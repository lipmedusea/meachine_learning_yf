import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("load_data/FY20_DD_modeling_partiton_sample.csv", nrows = 10000)

l = ["CAFE20_age",
"CAFE20_gender",
"CAFE20_region",
"CAFE20_levels"]

z = pd.get_dummies(df[l])


