import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, r2_score

# load data
data_satellite = pd.read_pickle('finish_satellite.pkl')
data_street = pd.read_pickle('finish_street.pkl')
data_satellite_night = pd.read_pickle('finish_satellite_night.pkl')
data_satellite_night_dmsp= pd.read_pickle('satellite_n_dmsp.pkl')
data_satellite_night_viirs = pd.read_pickle('satellite_n_viirs.pkl')
data_6 = pd.read_pickle('finish_s_s_6.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]


# features and labels
X_street = data_street.iloc[:,4:7]
X_satellite = data_satellite.iloc[:,4:7]
X_satellite_night = data_satellite_night['mean_scaled']
X_satellite_night_dmsp = data_satellite_night_dmsp['mean_scaled']
X_satellite_night_viirs = data_satellite_night_viirs['mean_scaled']
X_6 = data_6.iloc[:,1:7]
print(X_6.shape)
Y_street = data_street['water_index_rnd']
Y_satellite = data_satellite['water_index_rnd']
Y_satellite_night = data_satellite_night['water_index_rnd']
Y_satellite_night_dmsp = data_satellite_night_dmsp['water_index_rnd']
Y_satellite_night_viirs = data_satellite_night_viirs['water_index_rnd']
Y_6 = data_6['water_index_rnd']
print(Y_6.shape)

# satellite: red
plt.scatter(data_satellite.iloc[:,1], data_satellite['water_index'], color='red')
# satellite: green
plt.scatter(data_satellite.iloc[:,2], data_satellite['water_index'], color='green')
# satellite: blue
plt.scatter(data_satellite.iloc[:,3], data_satellite['water_index'], color='blue')
#plt.show()


# street: red
plt.scatter(data_street.iloc[:,1], data_street['water_index'], color='red')
# street: green
plt.scatter(data_street.iloc[:,2], data_street['water_index'], color='green')
# street: blue
plt.scatter(data_street.iloc[:,3], data_street['water_index'], color='blue')
#plt.show()

# blue line on the left 
# criteria = data_street[data_street.iloc[:,3] < 4]


# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street.astype(str), test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite.astype(str), test_size = 0.2, train_size=0.8)
x_train_satellite_night, x_test_satellite_night, y_train_satellite_night, y_test_satellite_night = train_test_split(X_satellite_night, Y_satellite_night.astype(str), test_size = 0.2, train_size=0.8)
x_train_satellite_night_dmsp, x_test_satellite_night_dmsp, y_train_satellite_night_dmsp, y_test_satellite_night_dmsp = train_test_split(X_satellite_night_dmsp, Y_satellite_night_dmsp.astype(str), test_size = 0.2, train_size=0.8)
x_train_satellite_night_viirs, x_test_satellite_night_viirs, y_train_satellite_night_viirs, y_test_satellite_night_viirs = train_test_split(X_satellite_night_viirs, Y_satellite_night_viirs.astype(str), test_size = 0.2, train_size=0.8)
x_train_6, x_test_6, y_train_6, y_test_6 = train_test_split(X_6, Y_6.astype(str), test_size = 0.2, train_size=0.8)
print(x_train_6.shape)
print(y_train_6.shape)

# training and predictions 
### street ###
classifier_street = KNeighborsClassifier(n_neighbors=5)
classifier_street.fit(x_train_street, y_train_street)

y_pred_street = classifier_street.predict(x_test_street)
cm_street = confusion_matrix(y_test_street, y_pred_street)
ac_street = accuracy_score(y_test_street, y_pred_street)
cl_matrix_street = classification_report(y_test_street, y_pred_street)
r2score_street = r2_score(y_test_street, y_pred_street)
print(cm_street)
print(ac_street) # 0.86
print(cl_matrix_street)
print(r2score_street) # -0.19


error_street = []
for i in range(1,21):
    knn_street = KNeighborsClassifier(n_neighbors=i)
    knn_street.fit(x_train_street, y_train_street)
    pred_i_street = knn_street.predict(x_test_street)
    error_street.append(np.mean(pred_i_street != y_test_street))

plt.figure(figsize=(12,6))
plt.plot(range(1,21), error_street, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("street")
plt.show()


### satellite ###
classifier_satellite = KNeighborsClassifier(n_neighbors=5)
classifier_satellite.fit(x_train_satellite, y_train_satellite)

y_pred_satellite = classifier_satellite.predict(x_test_satellite)
cm_satellite = confusion_matrix(y_test_satellite, y_pred_satellite)
ac_satellite = accuracy_score(y_test_satellite, y_pred_satellite)
cl_matrix_satellite = classification_report(y_test_satellite, y_pred_satellite)
r2score_satellite = r2_score(y_test_satellite, y_pred_satellite)
print('cm satellite', cm_satellite)
print('accuracy satellite:', ac_satellite) # 0.55
print(cl_matrix_satellite)
print('r2 satellite', r2score_satellite) # 0.06


error_satellite = []
for i in range(1,21):
    knn_satellite = KNeighborsClassifier(n_neighbors=i)
    knn_satellite.fit(x_train_satellite, y_train_satellite)
    pred_i_satellite = knn_satellite.predict(x_test_satellite)
    error_satellite.append(np.mean(pred_i_satellite != y_test_satellite))

plt.figure(figsize=(12,6))
plt.plot(range(1,21), error_satellite, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("satellite")
plt.show()


### satellite night ###
# pandas series to array 
x_train_satellite_night = x_train_satellite_night.to_numpy().reshape(-1,1)
x_test_satellite_night = x_test_satellite_night.to_numpy().reshape(-1,1)

classifier_satellite_night = KNeighborsClassifier(n_neighbors=5)
classifier_satellite_night.fit(x_train_satellite_night, y_train_satellite_night)

y_pred_satellite_night = classifier_satellite_night.predict(x_test_satellite_night)
cm_satellite_night = confusion_matrix(y_test_satellite_night, y_pred_satellite_night)
ac_satellite_night = accuracy_score(y_test_satellite_night, y_pred_satellite_night)
cl_matrix_satellite_night = classification_report(y_test_satellite_night, y_pred_satellite_night)
r2score_satellite_night = r2_score(y_test_satellite_night, y_pred_satellite_night)
print(cm_satellite_night)
print('ac_night', ac_satellite_night) # 0.55
print(cl_matrix_satellite_night)
print ('r2 night', r2score_satellite_night) # -0.51

# histogram labels und Häufigkeit 

error_satellite_night = []
for i in range(1,21):
    knn_satellite_night = KNeighborsClassifier(n_neighbors=i)
    knn_satellite_night.fit(x_train_satellite_night, y_train_satellite_night)
    pred_i_satellite_night = knn_satellite_night.predict(x_test_satellite_night)
    error_satellite_night.append(np.mean(pred_i_satellite_night != y_test_satellite_night))

plt.figure(figsize=(12,6))
plt.plot(range(1,21), error_satellite_night, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("satellite night")
plt.show()


### satellite night: year split ###

## 1. DMSP
# pandas series to array 
x_train_satellite_night_dmsp = x_train_satellite_night_dmsp.to_numpy().reshape(-1,1)
x_test_satellite_night_dmsp = x_test_satellite_night_dmsp.to_numpy().reshape(-1,1)

classifier_satellite_night_dmsp = KNeighborsClassifier(n_neighbors=5)
classifier_satellite_night_dmsp.fit(x_train_satellite_night_dmsp, y_train_satellite_night_dmsp)

y_pred_satellite_night_dmsp = classifier_satellite_night_dmsp.predict(x_test_satellite_night_dmsp)
cm_satellite_night_dmsp = confusion_matrix(y_test_satellite_night_dmsp, y_pred_satellite_night_dmsp)
ac_satellite_night_dmsp = accuracy_score(y_test_satellite_night_dmsp, y_pred_satellite_night_dmsp)
cl_matrix_satellite_night_dmsp = classification_report(y_test_satellite_night_dmsp, y_pred_satellite_night_dmsp)
r2score_satellite_night_dmsp = r2_score(y_test_satellite_night_dmsp, y_pred_satellite_night_dmsp)
print(cm_satellite_night_dmsp)
print('ac_night_dmsp', ac_satellite_night_dmsp) # 0.55
print(cl_matrix_satellite_night_dmsp)
print ('r2 night_dmsp', r2score_satellite_night_dmsp) # -0.51

# histogram labels und Häufigkeit 

error_satellite_night_dmsp = []
for i in range(1,21):
    knn_satellite_night_dmsp = KNeighborsClassifier(n_neighbors=i)
    knn_satellite_night_dmsp.fit(x_train_satellite_night_dmsp, y_train_satellite_night_dmsp)
    pred_i_satellite_night_dmsp = knn_satellite_night_dmsp.predict(x_test_satellite_night_dmsp)
    error_satellite_night_dmsp.append(np.mean(pred_i_satellite_night_dmsp != y_test_satellite_night_dmsp))

plt.figure(figsize=(12,6))
plt.plot(range(1,21), error_satellite_night_dmsp, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("satellite night dmsp")
plt.show()

## 1. DMSP
# pandas series to array 
x_train_satellite_night_viirs = x_train_satellite_night_viirs.to_numpy().reshape(-1,1)
x_test_satellite_night_viirs = x_test_satellite_night_viirs.to_numpy().reshape(-1,1)

classifier_satellite_night_viirs = KNeighborsClassifier(n_neighbors=5)
classifier_satellite_night_viirs.fit(x_train_satellite_night_viirs, y_train_satellite_night_viirs)

y_pred_satellite_night_viirs = classifier_satellite_night_viirs.predict(x_test_satellite_night_viirs)
cm_satellite_night_viirs = confusion_matrix(y_test_satellite_night_viirs, y_pred_satellite_night_viirs)
ac_satellite_night_viirs = accuracy_score(y_test_satellite_night_viirs, y_pred_satellite_night_viirs)
cl_matrix_satellite_night_viirs = classification_report(y_test_satellite_night_viirs, y_pred_satellite_night_viirs)
r2score_satellite_night_viirs = r2_score(y_test_satellite_night_viirs, y_pred_satellite_night_viirs)
print(cm_satellite_night_viirs)
print('ac_night_viirs', ac_satellite_night_viirs) # 0.55
print(cl_matrix_satellite_night_viirs)
print ('r2 night_viirs', r2score_satellite_night_viirs) # -0.51

# histogram labels und Häufigkeit 

error_satellite_night_viirs = []
for i in range(1,21):
    knn_satellite_night_viirs = KNeighborsClassifier(n_neighbors=i)
    knn_satellite_night_viirs.fit(x_train_satellite_night_viirs, y_train_satellite_night_viirs)
    pred_i_satellite_night_viirs = knn_satellite_night_viirs.predict(x_test_satellite_night_viirs)
    error_satellite_night_viirs.append(np.mean(pred_i_satellite_night_viirs != y_test_satellite_night_viirs))

plt.figure(figsize=(12,6))
plt.plot(range(1,21), error_satellite_night_viirs, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("satellite night dmsp")
plt.show()



### street and satellite combined (6 features) ###
classifier_6 = KNeighborsClassifier(n_neighbors=5)
classifier_6.fit(x_train_6, y_train_6)

y_pred_6 = classifier_6.predict(x_test_6)
cm_6 = confusion_matrix(y_test_6, y_pred_6)
ac_6 = accuracy_score(y_test_6, y_pred_6)
cl_matrix_6 = classification_report(y_test_6, y_pred_6)
r2score_6 = r2_score(y_test_6, y_pred_6)
print(cm_6)
print('ac_6', ac_6) # 0.55
print(cl_matrix_6)
print ('r2 6', r2score_6) # -0.51

# histogram labels und Häufigkeit 

error_6 = []
for i in range(1,21):
    knn_6 = KNeighborsClassifier(n_neighbors=i)
    knn_6.fit(x_train_6, y_train_6)
    pred_i_6 = knn_6.predict(x_test_6)
    error_6.append(np.mean(pred_i_6 != y_test_6))

plt.figure(figsize=(12,6))
plt.plot(range(1,21), error_6, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title("satellite 6")
plt.show()

