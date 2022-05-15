import pandas as pd

# general data for albania
al_data = pd.read_csv('AL.csv')

# water index for albania
al_water = pd.read_csv('dhs_final_labels.csv')
# print(al_water.iloc[1300])

al_water.to_pickle("C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL_water.pkl")


"""
# combined data for algeria 
al_combined = pd.merge(al_data, al_water, on='DHSID_EA')
AL_data = pd.DataFrame(al_combined)
AL_data.to_pickle("C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL_data.pkl")
"""