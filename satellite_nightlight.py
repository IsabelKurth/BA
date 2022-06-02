import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 
import pickle


# all satellite images 
folder = "C:\\Users\\isabe\\Documents\\BA\\BA\\DHS_Data"

df_satellite_night = pd.DataFrame(columns=['DHSID_EA', 'mean_nl', 'imagename', 'path', 'year'])

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r  

for item in list_files(folder):
    list_data = np.load(item, allow_pickle = True)
    list_data_int = list_data['x']
    image_data = list_data[list_data.files[0]]
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    df_satellite_night = df_satellite_night.append({
            'DHSID_EA': id, 
            'mean_nl': list_data_int[-1].mean(dtype=np.float64),
            'imagename': firstsplit,
            'path': item,
            'year': firstsplit.str[2:6]
        }, ignore_index = True)

print(df_satellite_night.head())
print(df_satellite_night.keys())
df_satellite_night.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\satellite_all_night.pkl")  