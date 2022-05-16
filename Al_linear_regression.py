#multiple linear regression
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load data 
data_satellite = pd.read_pickle('AL_2008_finish_satellite.pkl')
data_street = pd.read_pickle('AL_2008_finish_street.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]
print(data_street.head())
print(data_satellite.head())

data_joined = pd.merge(data_street, data_satellite, how="left", on=['DHSID_EA'])
data_joined.drop('water_index_x', axis=1, inplace=True)
data_joined.columns = ['DHSID_EA', 'red_street', 'green_street', 'blue_street', 'red_satellite', 'green_satellite', 'blue_satellite', 'water_index']
print(data_joined.head())

data_street_RGB = data_street.iloc[:,1:4]
data_satellite_RGB = data_satellite.iloc[:,1:4]


# X: red/green/blue values (column 1, 2, 3 of the dataset)
# shape 75, 3
X_street = data_street_RGB.values
X_satellite = data_satellite_RGB.values


# Y: water index
# shape 75, 
Y_street = data_street['water_index'].values
Y_satellite = data_satellite['water_index'].values


# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street, test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite, test_size = 0.2, train_size=0.8)


# train model 
model_street = LinearRegression()
model_street.fit(x_train_street, y_train_street)
print("Street coef is:", model_street.coef_)
print("Street intercept is:", model_street.intercept_)

model_satellite = LinearRegression()
model_satellite.fit(x_train_satellite, y_train_satellite)
print("Satellite coef is:", model_satellite.coef_)
print("Satellite intercept is:", model_satellite.intercept_)

# predict values and evaluate model 
predictions_street = model_street.predict(x_test_street)

predictions_satellite = model_satellite.predict(x_test_satellite)

plot_street = plt.scatter(y_test_street, predictions_street)
plot_satellite = plt.scatter(y_test_satellite, predictions_satellite)
print(plot_street)
print(plot_satellite)


# evaluation metrics 
# 1) mean absolute error (MAE): absolute difference between actual and predicted values
# sum all errors and divided by total number of observations 
# goal: minimum MAE

# 2) mean squared error (MSE): squared difference between actual and predicted value
# avoid cancallation of negative terms 
# output unit not same as input unit 
# version: RMSE: rooted 

# 3) R Squared: independant of the context, how much regression line is
# better than mean line 
# if regression line is perfect: 1
# if 0.8: model is capable to explain 80 percent of the variance of the data 
MAE_street = mean_absolute_error(y_test_street, predictions_street)
MSE_street = mean_squared_error(y_test_street, predictions_street)
score_street = r2_score(y_test_street, predictions_street)
print("Street MAE is", MAE_street)
print("Street MSE is", MSE_street)
print("Street RMSE is", np.sqrt(MSE_street))
print("Street r2 score", score_street)

MAE_satellite = mean_absolute_error(y_test_satellite, predictions_satellite)
MSE_satellite = mean_squared_error(y_test_satellite, predictions_satellite)
score_satellite = r2_score(y_test_satellite, predictions_satellite)
print("satellite MAE is", MAE_satellite)
print("satellite MSE is", MSE_satellite)
print("Satellite RMSE is", np.sqrt(MSE_satellite))
print("Satellite r2 score", score_satellite)
