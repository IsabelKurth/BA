import numpy as np 
from numpy import asarray
import scipy.sparse
import pandas as pd 
import os 


# all satellite images for AL 2005 
folder_AL_2008 = "..\\DHS_Data\\AL-2008-5#"

# dataframe to save RGB values and DHSID_EA id 
df_satellite_AL_2008 = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename'])

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r  

"""
#example one image: 

dic_data =  np.load('DHS_Data/AL-2008-5#/AL-2008-5#-00000026.npz')
image_data = dic_data[dic_data.files[0]]
# print(image_data.shape)
# 255x255x8px satellite image 
# bands: blue, green, red, infrared, infrared 2, thermal, near infrared, nightlights 

# blue 
print(np.mean(image_data[0,:,:]))
# green
print(np.mean(np.ndarray.flatten(image_data[1,:,:])))
# red 
# print(image_data[2,:,:])

# print(image_data[0,:,:].shape)
"""

# AL 2008 data 
for item in list_files(folder_AL_2008):
    list_data = np.load(item)
    image_data = list_data[list_data.files[0]]
    firstsplit = (os.path.basename(item))
    id = os.path.splitext(firstsplit)[0]
    df_satellite_AL_2008 = df_satellite_AL_2008.append({
            'DHSID_EA': id, 
            'red':np.mean(image_data[2,:,:]), 
            'green': np.mean(np.ndarray.flatten(image_data[1,:,:])) * 255, 
            'blue':np.mean(np.ndarray.flatten(image_data[0,:,:])) * 255,
            'imagename': firstsplit
        }, ignore_index = True)


print(df_satellite_AL_2008)
df_satellite_AL_2008.to_pickle("C:\\Users\\isabe\\Documents\\BA\\BA\\AL_satellite_2008.pkl")

