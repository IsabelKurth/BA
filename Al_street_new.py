from PIL import Image
import numpy as np
from numpy import asarray 
import pandas as pd 
import os 
from os import listdir
import io
import pathlib

"""
# test mit einem einzelnen Bild
# Ziel RGB als Vector extrahieren 
image = Image.open('AL/AL-2008-5#-00000105/138149814963930.jpeg')
# print(os.path.split('AL/AL-2008-5#-00000105/138149814963930.jpeg')[1])
data = asarray(image)
print(data.shape)
print(data[:,:,2].shape) #blue
print(data[:,:,0]) #red
print(data[:,:,1]) #green
print(data[:,:,2]) #blue
"""

# data frame to save id, red value pixels, green value pixels, blue value pixels
df_street = pd.DataFrame(columns=['DHSID_EA', 'red', 'green', 'blue', 'imagename'],
index = ['DHSID_EA'])


folder_dir = "C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL"

  
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
    data = asarray(Image.open(image))
    df_street = df_street.append({
            'DHSID_EA': id, 
            'red': np.mean(np.ndarray.flatten(data[:,:,0])), 
            'green': np.mean(np.ndarray.flatten(data[:,:,1])), 
            'blue':np.mean(np.ndarray.flatten(data[:,:,2])),
            'imagename': os.path.basename(image)
        }, ignore_index=True)
    """
    temp = pd.DataFrame(
        {
            'DHSID_EA': id, 
            'red': asarray(data[:,:,0]), 
            'green': asarray(data[:,:,1]), 
            'blue':asarray(data[:,:,2]),
            'imagename': os.path.basename(image)
        }, index = ['DHSID_EA']
       
    )
    
    image_dict = {
            'DHSID_EA': id, 
            'red': #asarray(data[:,:,0]), asarray(data).iloc[:,:,0]
            'green': asarray(data[:,:,1]), 
            'blue':asarray(data[:,:,2]),
            'imagename': os.path.basename(image)
        }
    temp = pd.DataFrame(image_dict, index=pd.Index(range(1,100,1)))
    """

    #df_street = pd.concat([df_street, temp])

#print(df_street)    

df_street.to_pickle("C:\\Users\\isabe\\Documents\\BA_sustainbench\\BA_sustainbench\\AL_street_2008.pkl")
