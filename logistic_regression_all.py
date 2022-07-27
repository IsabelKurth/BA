import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# load data
# satellite
data_satellite = pd.read_pickle('finish_satellite.pkl')
data_satellite_viirs = pd.read_pickle('satellite_viirs.pkl')
data_satellite_dmsp = pd.read_pickle('satellite_dmsp.pkl')
data_satellite_train = pd.read_pickle('finish_satellite_train.pkl')
data_satellite_test = pd.read_pickle('finish_satellite_test.pkl')

# street
data_street = pd.read_pickle('finish_street.pkl')

# night
data_satellite_night = pd.read_pickle('finish_satellite_night.pkl')
data_satellite_night_dmsp= pd.read_pickle('satellite_n_dmsp.pkl')
data_satellite_night_viirs = pd.read_pickle('satellite_n_viirs.pkl')

# combined 
data_6 = pd.read_pickle('finish_s_s_6.pkl')
data_7 = pd.read_pickle('finish_s_s_7.pkl')
data_6_dmsp = pd.read_pickle('s_s_6_dmsp.pkl')
data_6_viirs = pd.read_pickle('s_s_6_viirs.pkl')
data_7_dmsp = pd.read_pickle('s_s_7_dmsp.pkl')
data_7_viirs = pd.read_pickle('s_s_7_viirs.pkl')

data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]


# features and labels
X_street = data_street.iloc[:,4:7] #scaled RGB
X_satellite = data_satellite.iloc[:,4:7] #scaled RGB
X_satellite_night = data_satellite_night['mean_scaled'].to_numpy().reshape(-1,1)
X_satellite_night_dmsp = data_satellite_night_dmsp['mean_scaled'].to_numpy().reshape(-1,1)
X_satellite_night_viirs = data_satellite_night_viirs['mean_scaled'].to_numpy().reshape(-1,1)
X_6 = data_6.iloc[:,1:7]
X_7 = data_7.iloc[:,1:8]

def logreg(dataset, X):
    Y = dataset['water_index_rnd']
    x_train, x_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.2, train_size=0.8)
    reg = LogisticRegression(multi_class='multinomial')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    score = reg.score(x_test, y_test)
    print('score:', score)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm)
    #plt.show()
    plt.hist(dataset['water_index'])
    #plt.show()
    plt.hist(y_pred)
    #plt.show()


#logreg(data_6, data_6.iloc[:,1:7])
#logreg(data_satellite, data_satellite.iloc[:, 4:7])
#logreg(data_satellite_viirs, data_satellite_viirs.iloc[:, 4:7])
#logreg(data_satellite_dmsp, data_satellite_dmsp.iloc[:, 4:7])


def logreg_country(dataset_train, dataset_test, X_train, X_test):
    Y_train = dataset_train['water_index_rnd'].astype(str)
    Y_test = dataset_test['water_index_rnd'].astype(str)
    reg = LogisticRegression(multi_class='multinomial')
    reg.fit(X_train, Y_train)
    y_pred = reg.predict(X_test)
    score = reg.score(X_test, Y_test)
    print('score:', score)
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm)
    #plt.show()
    plt.hist(y_pred)
    #plt.show()


logreg_country(data_satellite_train, data_satellite_test, data_satellite_train.iloc[:, 4:7], data_satellite_test.iloc[:, 4:7])


