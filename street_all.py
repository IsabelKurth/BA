from PIL import Image
import numpy as np
from numpy import asarray 
import pandas as pd 
import os 
from os import listdir
import io
import pathlib
import time
from sklearn.preprocessing import StandardScaler 
from sys import platform

start_time = time.time()

if platform == "linux" or platform == "linux2":
    folder_dir = "../BA/Street_images"
elif platform == "win32" or platform == "win64":
    folder_dir = "..\\BA\\Street_images"    

# data frame to save id, red value pixels, green value pixels, blue value pixels
df_street = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename'],
index = ['DHSID_EA'])


# folder with all downloaded countries 
# folder_dir = "..\\BA\\Street_images"

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(os.path.join(subdir, file))                                                                         
    return r  


for image in list_files(folder_dir):
    firstsplit = os.path.split(image)[0]
    id = os.path.split(firstsplit)[1]
    country = id[:2]
    data = asarray(Image.open(image))
    mydict = {
            'DHSID_EA': id, 
            'red': np.mean(np.ndarray.flatten(data[:,:,0])), 
            'green': np.mean(np.ndarray.flatten(data[:,:,1])), 
            'blue':np.mean(np.ndarray.flatten(data[:,:,2])),
            'imagename': os.path.basename(image), 
            'country': country}
    df_street = pd.concat([df_street,pd.DataFrame([mydict])], ignore_index = True)
    
  

df_street.to_pickle("..\\BA\\street_all.pkl")
print(df_street.head())

print("Process finished --- %s seconds ---" % (time.time() - start_time))
