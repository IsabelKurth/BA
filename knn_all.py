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


# training and predictions 
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_street, y_train_street_grob)

y_pred = classifier.predict(x_test_street)
cm = confusion_matrix(y_test_street_grob, y_pred)
ac = accuracy_score(y_test_street_grob, y_pred)
cl_matrix = classification_report(y_test_street_grob, y_pred)
print(cm)
print(ac)
print(cl_matrix)


error = []
for i in range(1,5):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_street, y_train_street_grob)
    pred_i = knn.predict(x_test_street)
    error.append(np.mean(pred_i != y_test_street_grob))

plt.figure(figsize=(12,6))
plt.plot(range(1,5), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.show()
