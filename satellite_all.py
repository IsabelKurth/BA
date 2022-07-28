import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 
import pickle
from sys import platform


if platform == "linux" or platform == "linux2":
    folder = "../BA/DHS_Data"
elif platform == "win32" or platform == "win64":
    folder = "..\\BA\\DHS_Data"   


df_satellite = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename', 'path', 'country'])

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r  


print(list_files(folder))    


for item in list_files(folder):
    #list_data = np.load(item, allow_pickle = True)
    #image_data = list_data[list_data.files[0]]
    image_data = np.load(item, allow_pickle=True)['x']
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    country = id[:2]
    mydict = {
            'DHSID_EA': id, 
            'red':np.mean(image_data[2,:,:]), 
            'green': np.mean(image_data[1,:,:]), 
            'blue':np.mean(image_data[0,:,:]),
            'imagename': firstsplit,
            'path': item,
            'country': country
        }
    df_satellite = pd.concat([df_satellite, pd.DataFrame([mydict])], ignore_index = True)

print(df_satellite.head())
print(df_satellite.keys())

if platform == "linux" or platform == "linux2":
    df_satellite.to_pickle("../BA/satellite_all.pkl")   
elif platform == "win32" or platform == "win64":
    df_satellite.to_pickle("..\\BA\\satellite_all.pkl")        

