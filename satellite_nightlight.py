import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 
import pickle
from sklearn.preprocessing import StandardScaler

if platform == "linux" or platform == "linux2":
    folder = "../BA/DHS_Data"
elif platform == "win32" or platform == "win64":
    folder = "..\\BA\\DHS_Data"  


df_satellite_night = pd.DataFrame(columns=['DHSID_EA', 'mean_nl', 'scaled_mean', 'imagename', 'path', 'country'])

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
    image_data = np.load(item, allow_pickle=True)['x']
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    country = id[:2]
    mydict = {
            'DHSID_EA': id, 
            'mean_nl': image_data[-1].mean(dtype=np.float64),
            'imagename': firstsplit,
            'path': item, 
            'country': country
        }
    df_satellite_night = pd.concat([df_satellite_night, pd.DataFrame([mydict])], ignore_index = True)


print(df_satellite_night.head())
print(df_satellite_night.keys())
print(df_satellite_night['mean_nl'].max())
print(df_satellite_night['mean_nl'].min())
print(df_satellite_night.shape)

if platform == "linux" or platform == "linux2":
    df_satellite.to_pickle("../BA/satellite_all_night.pkl")   
elif platform == "win32" or platform == "win64":
    df_satellite.to_pickle("..\\BA\\satellite_all_night.pkl")  