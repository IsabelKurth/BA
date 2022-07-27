import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sys import platform 

# load image data 
satellite_data_all = pd.read_pickle('satellite_all.pkl')
satellite_data_night = pd.read_pickle('satellite_all_night.pkl')
street_data_all = pd.read_pickle('street_all.pkl')

# load water index data
water_data = pd.read_csv('dhs_final_labels.csv')

# combine data
combined_satellite = pd.merge(satellite_data_all, water_data, how="left", on=['DHSID_EA'])
combined_satellite_night = pd.merge(satellite_data_night, water_data, how="left", on=['DHSID_EA'])
combined_street = pd.merge(street_data_all, water_data, how="left", on=['DHSID_EA'])

### PREPROCESSING ###
# scale data #
## satellite ##
scaler_s_r = StandardScaler()
red_scaled = scaler_s_r.fit_transform(combined_satellite['red'].to_numpy().reshape(-1,1))
combined_satellite['red_scaled'] = red_scaled

scaler_s_g = StandardScaler()
green_scaled = scaler_s_g.fit_transform(combined_satellite['green'].to_numpy().reshape(-1,1))
combined_satellite['green_scaled'] = green_scaled

scaler_s_b = StandardScaler()
blue_scaled = scaler_s_b.fit_transform(combined_satellite['blue'].to_numpy().reshape(-1,1))
combined_satellite['blue_scaled'] = blue_scaled

## satellite night ##
scaler_n = StandardScaler()
mean_nl_scaled = scaler_n.fit_transform(combined_satellite_night['mean_nl'].to_numpy().reshape(-1,1))
combined_satellite_night['mean_scaled'] = mean_nl_scaled


## street ##
scaler_st_r = StandardScaler()
red_scaled_st = scaler_st_r.fit_transform(combined_street['red'].to_numpy().reshape(-1,1))
combined_street['red_scaled'] = red_scaled_st

scaler_st_g = StandardScaler()
green_scaled_st = scaler_st_g.fit_transform(combined_street['green'].to_numpy().reshape(-1,1))
combined_street['green_scaled'] = green_scaled_st

scaler_st_b = StandardScaler()
blue_scaled_st = scaler_st_b.fit_transform(combined_street['blue'].to_numpy().reshape(-1,1))
combined_street['blue_scaled'] = blue_scaled_st

# get relevant columns only 
just_need_satellite = combined_satellite[['DHSID_EA', 'red', 'green', 'blue', 'red_scaled', 'green_scaled', 'blue_scaled', 'water_index', 'year', 'path', 'country']]
just_need_satellite_night = combined_satellite_night[['DHSID_EA', 'mean_nl', 'mean_scaled', 'water_index', 'year', 'path', 'country']]
just_need_street = combined_street[['DHSID_EA', 'red', 'green', 'blue', 'red_scaled', 'green_scaled', 'blue_scaled', 'year', 'water_index', 'country']]

# drop rows with no water index data available 
just_need_satellite = just_need_satellite.dropna(subset=['water_index'])
just_need_satellite_night = just_need_satellite_night.dropna(subset=['water_index'])
just_need_street = just_need_street.dropna(subset=['water_index'])

# add rounded water index # 
def round_half(number, base=0.5):
    return base* round(number/base)

just_need_satellite['water_index_rnd'] = round_half(just_need_satellite['water_index'])
just_need_satellite_night['water_index_rnd'] = round_half(just_need_satellite_night['water_index'])
just_need_street['water_index_rnd'] = round_half(just_need_street['water_index'])

if platform == "linux" or platform == "linux2":
    just_need_satellite.to_pickle("../BA/finish_satellite.pkl")
    just_need_satellite_night.to_pickle("../BA/finish_satellite_night.pkl")
    just_need_street.to_pickle("../BA/finish_street.pkl")
elif platform == "win32" or platform == "win64":
    just_need_satellite.to_pickle("..\\BA\\finish_satellite.pkl")
    just_need_satellite_night.to_pickle("..\\BA\\finish_satellite_night.pkl")
    just_need_street.to_pickle("..\\BA\\finish_street.pkl")

# country split: 
just_need_satellite_train = just_need_satellite.loc[just_need_satellite['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
just_need_satellite_test = just_need_satellite.loc[just_need_satellite['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])]   
if platform == "linux" or platform == "linux2":
    just_need_satellite_train.to_pickle("../BA/finish_satellite_train.pkl")
    just_need_satellite_test.to_pickle("../BA/finish_satellite_test.pkl")

elif platform == "win32" or platform == "win64":
    just_need_satellite_train.to_pickle("..\\BA\\finish_satellite_train.pkl")
    just_need_satellite_test.to_pickle("..\\BA\\finish_satellite_test.pkl")

just_need_street_train = just_need_street.loc[just_need_street['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
just_need_street_test = just_need_street.loc[just_need_street['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])]   
if platform == "linux" or platform == "linux2":
    just_need_street_train.to_pickle("../BA/finish_street_train.pkl")
    just_need_street_test.to_pickle("../BA/finish_street_test.pkl")

elif platform == "win32" or platform == "win64":
    just_need_street_train.to_pickle("..\\BA\\finish_street_train.pkl")
    just_need_street_test.to_pickle("..\\BA\\finish_street_test.pkl")

   

"""
train = dataset.loc[dataset['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
test = dataset.oc[dataset['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])]    
"""

# split in DMSP and VIIRS
#satellite_viirs, satellite_dmsp = [x for _, x in just_need_satellite.groupby(just_need_satellite['year'] <= 2011)]
satellite_n_viirs, satellite_n_dmsp = [x for _, x in just_need_satellite_night.groupby(just_need_satellite_night['year'] <= 2011)]
#street_viirs, street_dmsp = [x for _, x in just_need_street.groupby(just_need_street['year'] <= 2011)]

sat_night_dmsp_train = satellite_n_dmsp.loc[satellite_n_dmsp['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
sat_night_dmsp_test = satellite_n_dmsp.loc[satellite_n_dmsp['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])]   
if platform == "linux" or platform == "linux2":
    satellite_n_dmsp_train.to_pickle("../BA/satellite_n_dmsp_train.pkl")
    satellite_n_dmsp_test.to_pickle("../BA/satellite_n_dmsp_test.pkl")

elif platform == "win32" or platform == "win64":
    satellite_n_dmsp_train.to_pickle("..\\BA\\satellite_n_dmsp_train.pkl")
    satellite_n_dmsp_test.to_pickle("..\\BA\\satellite_n_dmsp_test.pkl")

sat_night_viirs_train = satellite_n_viirs.loc[satellite_n_viirs['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
sat_night_viirs_test = satellite_n_viirs.loc[satellite_n_viirs['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])]   
if platform == "linux" or platform == "linux2":
    satellite_n_viirs_train.to_pickle("../BA/satellite_n_viirs_train.pkl")
    satellite_n_viirs_test.to_pickle("../BA/satellite_n_viirs_test.pkl")

elif platform == "win32" or platform == "win64":
    satellite_n_viirs_train.to_pickle("..\\BA\\satellite_n_viirs_train.pkl")
    satellite_n_viirs_test.to_pickle("..\\BA\\satellite_n_viirs_test.pkl")    
    
    

if platform == "linux" or platform == "linux2":
    #satellite_viirs.to_pickle("../BA/satellite_viirs.pkl")
    #satellite_dmsp.to_pickle("../BA/satellite_dmsp.pkl")
    satellite_n_viirs.to_pickle("../BA/satellite_n_viirs.pkl")
    satellite_n_dmsp.to_pickle("../BA/satellite_n_dmsp.pkl")
    #street_viirs.to_pickle("../BA/street_viirs.pkl")
    #street_dmsp.to_pickle("../BA/street_dmsp.pkl")

elif platform == "win32" or platform == "win64":    
    #satellite_viirs.to_pickle("..\\BA\\satellite_viirs.pkl")
    #satellite_dmsp.to_pickle("..\\BA\\satellite_dmsp.pkl")
    satellite_n_viirs.to_pickle("..\\BA\\satellite_n_viirs.pkl")
    satellite_n_dmsp.to_pickle("..\\BA\\satellite_n_dmsp.pkl")
    #street_viirs.to_pickle("..\\BA\\street_viirs.pkl")
    #street_dmsp.to_pickle("..\\BA\\street_dmsp.pkl")


# combine satellite and street #
combined_s_s_6 = pd.merge(just_need_satellite, just_need_street, how="inner", on=['DHSID_EA'])
just_need_s_s_6 = combined_s_s_6[['DHSID_EA', 'red_scaled_x', 'green_scaled_x', 'blue_scaled_x', 'red_scaled_y', 'green_scaled_y', 'blue_scaled_y', 'year_x', 'water_index_y', 'country_y']]
just_need_s_s_6 = just_need_s_s_6.rename(columns={'red_scaled_x': 'red_sat', 'green_scaled_x': 'green_sat', 'blue_scaled_x': 'blue_sat', 
'red_scaled_y': 'red_str', 'green_scaled_y': 'green_str', 'blue_scaled_y': 'blue_str', 'year_x': 'year', 'water_index_y': 'water_index', 'country_y': 'country'})
just_need_s_s_6['water_index_rnd'] = round_half(just_need_s_s_6['water_index'])
# s_s_6_viirs, s_s_6_dmsp = [x for _, x in just_need_s_s_6.groupby(just_need_s_s_6['year'] <= 2011)]

just_need_s_s_6_train = just_need_s_s_6.loc[just_need_s_s_6['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
just_need_s_s_6_test = just_need_s_s_6.loc[just_need_s_s_6['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])] 


if platform == "linux" or platform == "linux2":
    #s_s_6_viirs.to_pickle("../BA/s_s_6_viirs.pkl")
    #s_s_6_dmsp.to_pickle("../BA/s_s_6_dmsp.pkl")
    just_need_s_s_6.to_pickle("../BA/finish_s_s_6.pkl")
    just_need_s_s_6_train.to_pickle("../BA/s_s_6_train.pkl")
    just_need_s_s_6_test.to_pickle("../BA/s_s_6_test.pkl")
elif platform == "win32" or platform == "win64": 
    #s_s_6_viirs.to_pickle("..\\BA\\s_s_6_viirs.pkl")
    #s_s_6_dmsp.to_pickle("..\\BA\\s_s_6_dmsp.pkl")
    just_need_s_s_6.to_pickle("..\\BA\\finish_s_s_6.pkl")
    just_need_s_s_6_train.to_pickle("..\\BA\\s_s_6_train.pkl")
    just_need_s_s_6_test.to_pickle("..\\BA\\s_s_6_test.pkl")

    


# combine satellite and street and night#
combined_s_s_7 = pd.merge(just_need_satellite, just_need_street, how="inner", on=['DHSID_EA'])
combined_s_s_7 = pd.merge(combined_s_s_7, just_need_satellite_night, how="inner", on=['DHSID_EA'])
just_need_s_s_7 = combined_s_s_7[['DHSID_EA', 'red_scaled_x', 'green_scaled_x', 'blue_scaled_x', 'red_scaled_y', 'green_scaled_y', 'blue_scaled_y', 'mean_scaled', 'year_x', 'water_index_rnd', 'country_y']]
just_need_s_s_7 = just_need_s_s_7.rename(columns={'red_scaled_x': 'red_sat', 'green_scaled_x': 'green_sat', 'blue_scaled_x': 'blue_sat', 
'red_scaled_y': 'red_str', 'green_scaled_y': 'green_str', 'blue_scaled_y': 'blue_str', 'year_x': 'year', 'water_index_rnd': 'water_index', 'country_y': 'country'})
just_need_s_s_7['water_index_rnd'] = round_half(just_need_s_s_7['water_index'])
s_s_7_viirs, s_s_7_dmsp = [x for _, x in just_need_s_s_7.groupby(just_need_s_s_7['year'] <= 2011)]

s_s_7_dmsp_train = s_s_7_dmsp.loc[s_s_7_dmsp['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
s_s_7_dmsp_test = s_s_7_dmsp.loc[s_s_7_dmsp['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])] 

s_s_7_viirs_train = s_s_7_viirs.loc[s_s_7_viirs['country'].isin(['AL', 'BD', 'BF', 'BJ', 'BO', 'CD', 'CO', 'CM', 'DR', 'GA', 
    'GH', 'GN', 'GU','GY', 'HN', 'HT', 'ID', 'JO', 'KE', 'KM', 'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 
    'NG', 'NI', 'NM', 'PE', 'PH', 'SL', 'SN', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM', 'ZW'])]
s_s_7_viirs_test = s_s_7_viirs.loc[s_s_7_dmsp['country'].isin(['AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'HY', 'ML', 'NP', 'PK', 'RW', 'SZ'])] 

if platform == "linux" or platform == "linux2":
    s_s_7_viirs.to_pickle("../BA/s_s_7_viirs.pkl")
    s_s_7_dmsp.to_pickle("../BA/s_s_7_dmsp.pkl")
    just_need_s_s_7.to_pickle("../BA/finish_s_s_7.pkl")
    s_s_7_dmsp_train.to_pickle("../BA/s_s_7_dmsp_train.pkl")
    s_s_7_dmsp_test.to_pickle("../BA/s_s_7_dmsp_test.pkl")
    s_s_7_viirs_train.to_pickle("../BA/s_s_7_viirs_train.pkl")
    s_s_7_viirs_test.to_pickle("../BA/s_s_7_viirs_test.pkl")
elif platform == "win32" or platform == "win64": 
    s_s_7_viirs.to_pickle("..\\BA\\s_s_7_viirs.pkl")
    s_s_7_dmsp.to_pickle("..\\BA\\s_s_7_dmsp.pkl")
    just_need_s_s_7.to_pickle("..\\BA\\finish_s_s_7.pkl")
    s_s_7_dmsp_train.to_pickle("..\\BA\\s_s_7_dmsp_train.pkl")
    s_s_7_dmsp_test.to_pickle("..\\BA\\s_s_7_dmsp_test.pkl")
    s_s_7_viirs_train.to_pickle("..\\BA\\s_s_7_viirs_train.pkl")
    s_s_7_viirs_test.to_pickle("..\\BA\\s_s_7_viirs_test.pkl")


"""
# get how many rows are dropped because water index is NaN
water_na_free = water_data.dropna(subset=['water_index'])
dropped_rows = water_data[~water_data.index.isin(water_na_free.index)]
print(dropped_rows.head())
print(dropped_rows.groupby('cname').sum())
print(water_data.shape)
print(water_na_free.shape)

plt.hist(water_data['water_index'])
plt.show()

plt.hist(just_need_satellite['water_index_rnd'])
plt.show()

print(just_need_s_s_6.head(), flush=True)
"""