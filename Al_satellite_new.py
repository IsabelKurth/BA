import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 

# all satellite images for AL 2005 
folder_AL_2008 = "C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\DHS_Data\\AL-2008-5#"
folder_AL_2017 = "C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\DHS_Data\\AL-2017-7#"
"""
# understand the data 
dic_data =  np.load('DHS_Data/AL-2008-5#/AL-2008-5#-00000026.npz')
image_data = dic_data[dic_data.files[0]]
print(dic_data.files)
print(image_data)
print(image_data.shape)
# 255x255x8px satellite image 
# bands: blue, green, red, infrared, infrared 2, thermal, near infrared, nightlights 

# blue 
print(image_data[0,:,:])
# green
print(image_data[1,:,:])
# red 
print(image_data[2,:,:])

print(image_data[0,:,:].shape)
"""

# dataframe to save RGB values and DHSID_EA id 
df_satellite_AL_2008 = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename'])
df_satellite_AL_2017 = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename'])

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r  

# AL 2008 data 
for item in list_files(folder_AL_2008):
    list_data = np.load(item)
    image_data = list_data[list_data.files[0]]
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    df_satellite_AL_2008 = df_satellite_AL_2008.append({
            'DHSID_EA': id, 
            'red':np.mean(image_data[2,:,:]), 
            'green': np.mean(image_data[1,:,:]), 
            'blue':np.mean(image_data[0,:,:]),
            'imagename': firstsplit
        }, ignore_index = True)


"""
 # AL 2017 data 
for item in list_files(folder_AL_2017):
    list_data = np.load(item)
    image_data = list_data[list_data.files[0]]
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    df_satellite_AL_2017 = df_satellite_AL_2017.append({
            'DHSID_EA': id, 
            'red':np.mean(image_data[2,:,:]), 
            'green':np.mean(image_data[1,:,:]), 
            'blue':np.mean(image_data[0,:,:]),
            'imagename': firstsplit
        }, ignore_index=True)


df_satellite_AL = pd.concat([df_satellite_AL_2008, df_satellite_AL_2017])
"""

df_satellite_AL_2008.to_pickle("C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL_satellite_2008.pkl")
#df_satellite_AL_2017.to_pickle("C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL_satellite_2017.pkl")
#df_satellite_AL.to_pickle("C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL_satellite.pkl")
