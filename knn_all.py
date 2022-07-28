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

"""
# features and labels
X_street = data_street.iloc[:,4:7]
X_satellite = data_satellite.iloc[:,4:7]
X_satellite_night = data_satellite_night['mean_scaled'].to_numpy().reshape(-1,1)
X_satellite_night_dmsp = data_satellite_night_dmsp['mean_scaled'].to_numpy().reshape(-1,1)
X_satellite_night_viirs = data_satellite_night_viirs['mean_scaled'].to_numpy().reshape(-1,1)
X_6 = data_6.iloc[:,1:7]
"""


### idea: function knn for all datasets 

# metrics for knn with optimal k
# input: dataset, features X (either rgb or mean_nl), optimal k
# output: evaluation metrics
def knn_optimal(dataset, X, k):
    Y = dataset['water_index_rnd']
    x_train, x_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size=0.2, train_size=0.8)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    cl_matrix = classification_report(y_test, y_pred)
    r2score = r2_score(y_test, y_pred)
    print(cm)
    print(ac)
    print(cl_matrix)
    print(r2score)


# find optimal l
# input: dataset and features 
# output: r^2 for all ks and optimal k
def knn_k(dataset, X):
    best_r2 = 0
    best_k = None
    error = []
    for i in range (1,21):
        Y = dataset['water_index_rnd']
        x_train, x_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size=0.2, train_size=0.8)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        r2score = r2_score(y_test, y_pred)
        print(f'k={i:2d} r^2 = {r2score:.3f}')
        error.append(np.mean(y_pred != y_test))
        if r2score > best_r2:
            best_r2 = r2score
            best_k = i
    return best_k
 
    print(best_k)
    plt.figure(figsize=(12,6))
    plt.plot(range(1, 21), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)  
    plt.show()  


    #knn_optimal(dataset, X, best_k)  

best_k_found = knn_k(data_satellite_night, data_satellite_night['mean_scaled'].to_numpy().reshape(-1,1))
knn_optimal(data_satellite_night, data_satellite_night['mean_scaled'].to_numpy().reshape(-1,1), best_k_found)

knn_k(data_street, data_street.iloc[:,4:7])
knn_optimal(data_street, data_street.iloc[:,4:7], 2)