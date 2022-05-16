#multiple linear regression
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# load data 
data_satellite = pd.read_pickle('AL_2008_finish_satellite.pkl')
data_street = pd.read_pickle('AL_2008_finish_street.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]

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
score_street = r2_score(y_test_street, predictions_street)
print("Street r2 score", score_street)

predictions_satellite = model_satellite.predict(x_test_satellite)
score_satellite = r2_score(y_test_satellite, predictions_satellite)
print("Satellite r2 score", score_satellite)

plt.scatter(y_test_street, predictions_street)
plt.scatter(y_test_satellite, predictions_satellite)