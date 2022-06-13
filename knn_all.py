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
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]
data_street_RGB = data_street.iloc[:,1:4]
data_satellite_RGB = data_satellite.iloc[:,1:4]

# features and labels
X_street = data_street_RGB.values
X_satellite = data_satellite_RGB.values
X_satellite_night = data_satellite_night['mean_nl']

Y_street = data_street['water_index'].values
Y_satellite = data_satellite['water_index'].values
Y_satellite_night = data_satellite_night['water_index'].values


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

# auf halbe Zahlen gerundet 
for x in range(len(Y_street)):
    if (Y_street[x] < 5 and Y_street[x] >= 4.5):
        Y_street[x] = str(5)
    elif (Y_street[x] < 4.5 and Y_street[x] > 4): 
        Y_street[x] = str(4.5)
    elif (Y_street[x] < 4 and Y_street[x] > 3.5): 
        Y_street[x] = str(4)
    elif (Y_street[x] < 3.5 and Y_street[x] > 3): 
        Y_street[x] = str(3.5)
    elif (Y_street[x] < 3 and Y_street[x] > 2.5): 
        Y_street[x] = str(3)
    elif (Y_street[x] < 2.5 and Y_street[x] > 2): 
        Y_street[x] = str(2.5)
    elif (Y_street[x] < 2 and Y_street[x] > 1.5): 
        Y_street[x] = str(2)
    elif (Y_street[x] < 1.5 and Y_street[x] > 1): 
        Y_street[x] = str(1.5)
    elif (Y_street[x] < 1 and Y_street[x] > 0.5): 
        Y_street[x] = str(1)
    elif (Y_street[x] < 0.5 and Y_street[x] > 0): 
        Y_street[x] = str(0.5)
    else: 
        Y_street[x] = str(0)                               


Y_street_str = Y_street.astype(str)


for x in range(len(Y_satellite)):
    if (Y_satellite[x] < 5 and Y_satellite[x] >= 4.5):
        Y_satellite[x] = str(5)
    elif (Y_satellite[x] < 4.5 and Y_satellite[x] > 4): 
        Y_satellite[x] = str(4.5)
    elif (Y_satellite[x] < 4 and Y_satellite[x] > 3.5): 
        Y_satellite[x] = str(4)
    elif (Y_satellite[x] < 3.5 and Y_satellite[x] > 3): 
        Y_satellite[x] = str(3.5)
    elif (Y_satellite[x] < 3 and Y_satellite[x] > 2.5): 
        Y_satellite[x] = str(3)
    elif (Y_satellite[x] < 2.5 and Y_satellite[x] > 2): 
        Y_satellite[x] = str(2.5)
    elif (Y_satellite[x] < 2 and Y_satellite[x] > 1.5): 
        Y_satellite[x] = str(2)
    elif (Y_satellite[x] < 1.5 and Y_satellite[x] > 1): 
        Y_satellite[x] = str(1.5)
    elif (Y_satellite[x] < 1 and Y_satellite[x] > 0.5): 
        Y_satellite[x] = str(1)
    elif (Y_satellite[x] < 0.5 and Y_satellite[x] > 0): 
        Y_satellite[x] = str(0.5)
    else: 
        Y_satellite[x] = str(0)                               


Y_satellite_str = Y_satellite.astype(str)


for x in range(len(Y_satellite_night)):
    if (Y_satellite_night[x] < 5 and Y_satellite_night[x] >= 4.5):
        Y_satellite_night[x] = str(5)
    elif (Y_satellite_night[x] < 4.5 and Y_satellite_night[x] > 4): 
        Y_satellite_night[x] = str(4.5)
    elif (Y_satellite_night[x] < 4 and Y_satellite_night[x] > 3.5): 
        Y_satellite_night[x] = str(4)
    elif (Y_satellite_night[x] < 3.5 and Y_satellite_night[x] > 3): 
        Y_satellite_night[x] = str(3.5)
    elif (Y_satellite_night[x] < 3 and Y_satellite_night[x] > 2.5): 
        Y_satellite_night[x] = str(3)
    elif (Y_satellite_night[x] < 2.5 and Y_satellite_night[x] > 2): 
        Y_satellite_night[x] = str(2.5)
    elif (Y_satellite_night[x] < 2 and Y_satellite_night[x] > 1.5): 
        Y_satellite_night[x] = str(2)
    elif (Y_satellite_night[x] < 1.5 and Y_satellite_night[x] > 1): 
        Y_satellite_night[x] = str(1.5)
    elif (Y_satellite_night[x] < 1 and Y_satellite_night[x] > 0.5): 
        Y_satellite_night[x] = str(1)
    elif (Y_satellite_night[x] < 0.5 and Y_satellite_night[x] > 0): 
        Y_satellite_night[x] = str(0.5)
    else: 
        Y_satellite_night[x] = str(0)                               


Y_satellite_night_str = Y_satellite_night.astype(str)


# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street_str, test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite_str, test_size = 0.2, train_size=0.8)
x_train_satellite_night, x_test_satellite_night, y_train_satellite_night, y_test_satellite_night = train_test_split(X_satellite_night, Y_satellite_night_str, test_size = 0.2, train_size=0.8)



# auf ganze Zahlen gerundet (switxh case)
y_train_street_grob = np.digitize(y_train_street, [1, 2, 3, 4, 5])
y_test_street_grob = np.digitize(y_test_street, [1, 2, 3, 4, 5]) 
print(y_train_street_grob)
y_train_satellite_grob = np.digitize(y_train_satellite, [1, 2, 3, 4, 5])
y_test_satellite_grob = np.digitize(y_test_satellite, [1, 2, 3, 4, 5]) 
print(y_train_satellite_grob)


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
x_train_satellite_night = x_train_satellite_night.to_numpy()
x_train_satellite_night = x_train_satellite_night.reshape(-1,1)
x_test_satellite_night = x_test_satellite_night.to_numpy()
x_test_satellite_night = x_test_satellite_night.reshape(-1,1)

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

