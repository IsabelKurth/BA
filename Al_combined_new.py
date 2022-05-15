import pandas as pd 

# load satellite data 
al_satellite_data = pd.read_pickle('AL_satellite_2008.pkl')
# load street data 
al_street_data = pd.read_pickle('AL_street_2008.pkl')

# load water index data 
al_data = pd.read_pickle('AL_water.pkl')

# combine all data for albania
AL_combined_data_satellite = pd.merge(al_satellite_data, al_data, how="left", on=['DHSID_EA'])
AL_combined_data_street = pd.merge(al_street_data, al_data, how="left", on=['DHSID_EA'])

# get id, red, green, blue, water index 
just_need_AL_satellite = AL_combined_data_satellite[['DHSID_EA', 'red', 'green', 'blue', 'water_index']]
just_need_AL_street = AL_combined_data_street[['DHSID_EA', 'red', 'green', 'blue', 'water_index']]

just_need_AL_satellite.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\AL_2008_finish_satellite.pkl")
just_need_AL_street.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\AL_2008_finish_street.pkl")