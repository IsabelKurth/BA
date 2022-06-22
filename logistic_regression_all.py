import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


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
X_satellite_night = data_satellite_night['mean_scaled'].to_numpy().reshape(-1,1)
X_satellite_night_dmsp = data_satellite_night_dmsp['mean_scaled'].to_numpy().reshape(-1,1)
X_satellite_night_viirs = data_satellite_night_viirs['mean_scaled'].to_numpy().reshape(-1,1)
X_6 = data_6.iloc[:,1:7]

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
    plt.show()
    plt.hist(dataset['water_index'])
    plt.show()
    plt.hist(y_pred)
    plt.show()


print(logreg(data_6, data_6.iloc[:,1:7]))


