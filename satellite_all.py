import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 
import pickle


# all satellite images 
folder = "C:\\Users\\isabe\\Documents\\BA\\BA\\DHS_Data"

df_satellite = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename', 'path'])

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r  


# print(list_files(folder))    

for item in list_files(folder):
    #list_data = np.load(item, allow_pickle = True)
    #image_data = list_data[list_data.files[0]]
    image_data = np.load(item)['x']
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    country = id[:2]
    df_satellite = df_satellite.append({
            'DHSID_EA': id, 
            'red':np.mean(image_data[2,:,:]), 
            'green': np.mean(image_data[1,:,:]), 
            'blue':np.mean(image_data[0,:,:]),
            'imagename': firstsplit,
            'path': item,
            'country': country
        }, ignore_index = True)

print(df_satellite.head())
print(df_satellite.keys())
df_satellite.to_pickle("..\\BA\\satellite_all.pkl")        