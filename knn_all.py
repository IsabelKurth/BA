import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# load data
data_satellite = pd.read_pickle('finish_satellite.pkl')
data_street = pd.read_pickle('finish_street.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]
data_street_RGB = data_street.iloc[:,1:4]
data_satellite_RGB = data_satellite.iloc[:,1:4]

# features and labels
X_street = data_street_RGB.values
X_satellite = data_satellite_RGB.values

Y_street = data_street['water_index'].values
Y_satellite = data_satellite['water_index'].values

# auf halbe Zahlen gerundet 
for x in range(len(Y_street)):
    if (Y_street[x] < 5 and Y_street[x] >= 4.5):
        Y_street[x] = 5
    elif (Y_street[x] < 4.5 and Y_street[x] > 4): 
        Y_street[x] = 4.5
    elif (Y_street[x] < 4 and Y_street[x] > 3.5): 
        Y_street[x] = 4
    elif (Y_street[x] < 3.5 and Y_street[x] > 3): 
        Y_street[x] = 3.5
    elif (Y_street[x] < 3 and Y_street[x] > 2.5): 
        Y_street[x] = 3
    elif (Y_street[x] < 2.5 and Y_street[x] > 2): 
        Y_street[x] = 2.5
    elif (Y_street[x] < 2 and Y_street[x] > 1.5): 
        Y_street[x] = 2
    elif (Y_street[x] < 1.5 and Y_street[x] > 1): 
        Y_street[x] = 1.5
    elif (Y_street[x] < 1 and Y_street[x] > 0.5): 
        Y_street[x] = 1
    elif (Y_street[x] < 0.5 and Y_street[x] > 0): 
        Y_street[x] = 0.5
    else: 
        Y_street[x] = 0                               

print(Y_street)

# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street, test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite, test_size = 0.2, train_size=0.8)


# auf ganze Zahlen gerundet 
y_train_street_grob = np.digitize(y_train_street, [1, 2, 3, 4, 5])
y_test_street_grob = np.digitize(y_test_street, [1, 2, 3, 4, 5]) 
print(y_train_street_grob)
y_train_satellite_grob = np.digitize(y_train_satellite, [1, 2, 3, 4, 5])
y_test_satellite_grob = np.digitize(y_test_satellite, [1, 2, 3, 4, 5]) 
print(y_train_satellite_grob)


# training and predictions 
### street ###
classifier_street = KNeighborsClassifier(n_neighbors=5)
classifier_street.fit(x_train_street, y_train_street_grob)

y_pred_street = classifier_street.predict(x_test_street)
cm_street = confusion_matrix(y_test_street_grob, y_pred_street)
ac_street = accuracy_score(y_test_street_grob, y_pred_street)
cl_matrix_street = classification_report(y_test_street_grob, y_pred_street)
print(cm_street)
print(ac_street)
print(cl_matrix_street)


error_street = []
for i in range(1,5):
    knn_street = KNeighborsClassifier(n_neighbors=i)
    knn_street.fit(x_train_street, y_train_street_grob)
    pred_i_street = knn_street.predict(x_test_street)
    error_street.append(np.mean(pred_i_street != y_test_street_grob))

plt.figure(figsize=(12,6))
plt.plot(range(1,5), error_street, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.show()


### satellite ###
classifier_satellite = KNeighborsClassifier(n_neighbors=5)
classifier_satellite.fit(x_train_satellite, y_train_satellite_grob)

y_pred_satellite = classifier_satellite.predict(x_test_satellite)
cm_satellite = confusion_matrix(y_test_satellite_grob, y_pred_satellite)
ac_satellite = accuracy_score(y_test_satellite_grob, y_pred_satellite)
cl_matrix_satellite = classification_report(y_test_satellite_grob, y_pred_satellite)
print(cm_satellite)
print(ac_satellite)
print(cl_matrix_satellite)


error_satellite = []
for i in range(1,5):
    knn_satellite = KNeighborsClassifier(n_neighbors=i)
    knn_satellite.fit(x_train_satellite, y_train_satellite_grob)
    pred_i_satellite = knn_satellite.predict(x_test_satellite)
    error_satellite.append(np.mean(pred_i_satellite != y_test_satellite_grob))

plt.figure(figsize=(12,6))
plt.plot(range(1,5), error_satellite, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.show()
