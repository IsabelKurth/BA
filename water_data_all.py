import pandas as pd
import numpy as np

# water data for all countries 
water = pd.read_csv('dhs_final_labels.csv')
print(water.shape)

# extract countries and years with missing water data (NaN)
# get names and years of dropped ones 

water_na_free = water.dropna(subset=['water_index'])
print(water_na_free.shape)

dropped_rows = water[~water.index.isin(water_na_free.index)]
print(dropped_rows.head())

print(dropped_rows.groupby('cname').sum())
# print(dropped_rows.groupby('cname'))