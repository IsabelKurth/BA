import pandas as pd

# load satellite data
satellite_data_all = pd.read_pickle('satellite_all.pkl')
# load street data
street_data_all = pd.read_pickle('street_all.pkl')

# load water index data
water_data = pd.read_pickle('water.pkl')

# combine data
combined_satellite = pd.merge(satellite_data_all, water_data, how="left", on=['DHSID_EA'])
combined_street = pd.merge(street_data_all, water_data, how="left", on=['DHSID_EA'])

# get id, red, green, blue, water index 
just_need_satellite = combined_satellite[['DHSID_EA', 'red', 'green', 'blue', 'water_index']]
just_need_street = combined_street[['DHSID_EA', 'red', 'green', 'blue', 'water_index']]
just_need_satellite = just_need_satellite.dropna(subset=['water_index'])
just_need_street = just_need_street.dropna(subset=['water_index'])

just_need_satellite.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\finish_satellite.pkl")
just_need_street.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\finish_street.pkl")


