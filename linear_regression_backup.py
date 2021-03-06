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
data_satellite_night = pd.read_pickle('finish_satellite_night.pkl')
data_street_all = data_street_all.iloc[1:,:]
data_satellite_all = data_satellite_all.iloc[1:,:]
"""
print(data_street_all.shape)
print(data_satellite_all.shape)
print(data_satellite_night.shape)
print(data_street_all.head())
print(data_satellite_all.head())
print(data_satellite_night.head())
"""

data_satellite_all_RGB = data_satellite_all.iloc[:,1:4]
data_street_all_RGB = data_street_all.iloc[:,1:4]

X_satellite_all = data_satellite_all_RGB.values
Y_satellite_all = data_satellite_all['water_index'].values
X_satellite_night = data_satellite_night['mean_nl']
Y_satellite_night = data_satellite_night['water_index'].values
X_street_all = data_street_all_RGB.values
Y_street_all = data_street_all['water_index'].values

x_train_satellite_all, x_test_satellite_all, y_train_satellite_all, y_test_satellite_all = train_test_split(X_satellite_all, Y_satellite_all, test_size = 0.2, train_size=0.8)
x_train_street_all, x_test_street_all, y_train_street_all, y_test_street_all = train_test_split(X_street_all, Y_street_all, test_size = 0.2, train_size=0.8)
x_train_satellite_night, x_test_satellite_night, y_train_satellite_night, y_test_satellite_night = train_test_split(X_satellite_night, Y_satellite_night, test_size = 0.2, train_size=0.8)


model_satellite = LinearRegression()
model_satellite.fit(x_train_satellite_all, y_train_satellite_all)

pred_satellite = model_satellite.predict(x_test_satellite_all)
plt.scatter(y_test_satellite_all, pred_satellite)
plt.title("satellite")
plt.show()
sns.regplot(x=y_test_satellite_all, y=pred_satellite)
plt.show()

print(x_train_satellite_all.shape)
print(x_train_satellite_night.shape)
print(y_train_satellite_all.shape)
print(y_train_satellite_night.shape)

# pandas series to array 
x_train_satellite_night = x_train_satellite_night.to_numpy()
x_train_satellite_night = x_train_satellite_night.reshape(-1,1)
x_test_satellite_night = x_test_satellite_night.to_numpy()
x_test_satellite_night = x_test_satellite_night.reshape(-1,1)

model_satellite_night = LinearRegression()
model_satellite_night.fit(x_train_satellite_night, y_train_satellite_night)
pred_satellite_night = model_satellite_night.predict(x_test_satellite_night)
plt.scatter(y_test_satellite_night, pred_satellite_night)
plt.title("satellite night")
plt.show()
sns.regplot(x=y_test_satellite_night, y= pred_satellite_night)
plt.show()


model_street = LinearRegression()
model_street.fit(x_train_street_all, y_train_street_all)

pred_street = model_street.predict(x_test_street_all)
plt.scatter(y_test_street_all, pred_street)
plt.title("street")
plt.show()
sns.regplot(x=y_test_street_all, y=pred_street)
plt.show()


MAE_street = mean_absolute_error(y_test_street_all, pred_street)
MSE_street = mean_squared_error(y_test_street_all, pred_street)
score_street = r2_score(y_test_street_all, pred_street)
print("Street MAE is", MAE_street)
print("Street MSE is", MSE_street)
print("Street RMSE is", np.sqrt(MSE_street))
print("Street r2 score", score_street)

MAE_satellite = mean_absolute_error(y_test_satellite_all, pred_satellite)
MSE_satellite = mean_squared_error(y_test_satellite_all, pred_satellite)
score_satellite = r2_score(y_test_satellite_all, pred_satellite)
print("satellite MAE is", MAE_satellite)
print("satellite MSE is", MSE_satellite)
print("Satellite RMSE is", np.sqrt(MSE_satellite))
print("Satellite r2 score", score_satellite)

MAE_satellite_night = mean_absolute_error(y_test_satellite_night, pred_satellite_night)
MSE_satellite_night = mean_squared_error(y_test_satellite_night, pred_satellite_night)
score_satellite_night = r2_score(y_test_satellite_night, pred_satellite_night)
print("satellite_n MAE is", MAE_satellite_night)
print("satellite_n MSE is", MSE_satellite_night)
print("Satellite_n RMSE is", np.sqrt(MSE_satellite_night))
print("Satellite_n r2 score", score_satellite_night)
