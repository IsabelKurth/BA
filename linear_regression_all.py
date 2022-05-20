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
import seaborn as sns

# load data 
data_satellite_all = pd.read_pickle('finish_satellite.pkl')
data_street_all = pd.read_pickle('finish_street.pkl')
data_street_all = data_street_all.iloc[1:,:]
data_satellite_all = data_satellite_all.iloc[1:,:]
print(data_street_all.shape)
print(data_satellite_all.shape)
print(data_street_all.head())
print(data_satellite_all.head())

data_satellite_all_RGB = data_satellite_all.iloc[:,1:4]
data_street_all_RGB = data_street_all.iloc[:,1:4]

X_satellite_all = data_satellite_all_RGB.values
Y_satellite_all = data_satellite_all['water_index'].values
X_street_all = data_street_all_RGB.values
Y_street_all = data_street_all['water_index'].values

x_train_satellite_all, x_test_satellite_all, y_train_satellite_all, y_test_satellite_all = train_test_split(X_satellite_all, Y_satellite_all, test_size = 0.2, train_size=0.8)
x_train_street_all, x_test_street_all, y_train_street_all, y_test_street_all = train_test_split(X_street_all, Y_street_all, test_size = 0.2, train_size=0.8)

model_satellite = LinearRegression()
model_satellite.fit(x_train_satellite_all, y_train_satellite_all)

pred_satellite = model_satellite.predict(x_test_satellite_all)
plt.scatter(y_test_satellite_all, pred_satellite)
plt.show()
sns.regplot(x=y_test_satellite_all, y=pred_satellite)
plt.show()


model_street = LinearRegression()
model_street.fit(x_train_street_all, y_train_street_all)

pred_street = model_street.predict(x_test_street_all)
plt.scatter(y_test_street_all, pred_street)
plt.show()
sns.regplot(x=y_test_street_all, y=pred_street)
plt.show()
