import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# load data 
data_satellite = pd.read_pickle('finish_satellite.pkl')
data_street = pd.read_pickle('finish_street.pkl')
data_satellite_night = pd.read_pickle('finish_satellite_night.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]

X_street = data_street.iloc[:,1:4]
X_satellite = data_satellite.iloc[:,1:4].to_numpy().reshape(-1,1)
X_satellite_night = data_satellite_night['mean_nl'].to_numpy().reshape(-1,1)


def linreg(dataset, X):
    Y = dataset['water_index'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, train_size=0.8)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #plt.scatter(y_test, y_pred)
    #plt.show()
    #sns.regplot(x=y_test, y=y_pred)
    #plt.show()

    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)
    

    print("Street MAE is", MAE)
    print("Street MSE is", MSE)
    print("Street RMSE is", np.sqrt(MSE))
    print("Street r2 score", score)

#linreg(data_street, data_street.iloc[:,1:4])
linreg(data_satellite, data_satellite.iloc[:,1:4].to_numpy().reshape(-1,1))
#linreg(data_satellite_night, data_satellite_night['mean_nl'].to_numpy().reshape(-1,1))