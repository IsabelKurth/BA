import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 
import pickle
from sklearn.preprocessing import StandardScaler


# all satellite images 
folder = "C:\\Users\\isabe\\Documents\\BA\\BA\\DHS_Data"

df_satellite_night = pd.DataFrame(columns=['DHSID_EA', 'mean_nl', 'scaled_mean', 'imagename', 'path'])

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
    image_data = np.load(item)['x']
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    country = id[:2]
    df_satellite_night = df_satellite_night.append({
            'DHSID_EA': id, 
            'mean_nl': image_data[-1].mean(dtype=np.float64),
            'imagename': firstsplit,
            'path': item, 
            'country': country
        }, ignore_index = True)


print(df_satellite_night.head())
print(df_satellite_night.keys())
print(df_satellite_night['mean_nl'].max())
print(df_satellite_night['mean_nl'].min())
print(df_satellite_night.shape)
df_satellite_night.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\satellite_all_night.pkl")  