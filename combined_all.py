import pandas as pd

# load satellite data
satellite_data_all = pd.read_pickle('satellite_all.pkl')
satellite_data_night = pd.read_pickle('satellite_all_night.pkl')
# load street data
street_data_all = pd.read_pickle('street_all.pkl')

# load water index data
water_data = pd.read_csv('dhs_final_labels.csv')

# combine data
combined_satellite = pd.merge(satellite_data_all, water_data, how="left", on=['DHSID_EA'])
combined_satellite_night = pd.merge(satellite_data_night, water_data, how="left", on=['DHSID_EA'])
combined_street = pd.merge(street_data_all, water_data, how="left", on=['DHSID_EA'])

# get id, red, green, blue, water index 
just_need_satellite = combined_satellite[['DHSID_EA', 'red', 'green', 'blue', 'water_index', 'year', 'path']]
just_need_satellite_night = combined_satellite_night[['DHSID_EA', 'mean_nl', 'water_index', 'year', 'path']]
just_need_street = combined_street[['DHSID_EA', 'red', 'green', 'blue', 'water_index']]
just_need_satellite = just_need_satellite.dropna(subset=['water_index'])
just_need_satellite_night = just_need_satellite_night.dropna(subset=['water_index'])
just_need_street = just_need_street.dropna(subset=['water_index'])

just_need_satellite.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\finish_satellite.pkl")
just_need_satellite_night.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\finish_satellite_night.pkl")
just_need_street.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\finish_street.pkl")

water_na_free = water_data.dropna(subset=['water_index'])
dropped_rows = water_data[~water_data.index.isin(water_na_free.index)]
print(dropped_rows.head())
print(dropped_rows.groupby('cname').sum())
print(water_data.shape)
print(water_na_free.shape)

