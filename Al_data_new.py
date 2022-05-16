import pandas as pd

# general data for albania
al_data = pd.read_csv('AL.csv')

# water index for albania
al_water = pd.read_csv('dhs_final_labels.csv')
# print(al_water.iloc[1300])

al_water.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\AL_water.pkl")
