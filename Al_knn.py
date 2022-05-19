import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix, accuracy_score

# load data
data_satellite = pd.read_pickle('AL_2008_finish_satellite.pkl')
data_street = pd.read_pickle('AL_2008_finish_street.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]
data_street_RGB = data_street.iloc[:,1:4]
data_satellite_RGB = data_satellite.iloc[:,1:4]

# features and labels
X_street = data_street_RGB.values
X_satellite = data_satellite_RGB.values

Y_street = data_street['water_index'].values
Y_satellite = data_satellite['water_index'].values

# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street, test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite, test_size = 0.2, train_size=0.8)

"""
# feature scaling
scaler = StandardScaler()
scaler.fit(x_test_street)

x_train_street = scaler.transform(x_train_street)
x_test_street = scaler.transform(x_train_street)
"""

print(x_train_street)
print(y_train_street)

y_train_street = np.digitize(y_train_street, [1, 2, 3, 4, 5])
y_test_street = np.digitize(y_test_street, [1, 2, 3, 4, 5]) 
print(y_train_street)

# training and predictions 
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_street, y_train_street)

y_pred = classifier.predict(x_test_street)
cm = confusion_matrix(y_test_street, y_pred)
ac = accuracy_score(y_test_street, y_pred)
print(cm)
print(ac)
