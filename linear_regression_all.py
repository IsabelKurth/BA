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
data_satellite_all = pd.read_pickle('satellite_all.pkl')
#data_street = pd.read_pickle('AL_2008_finish_street.pkl')
#data_street = data_street.iloc[1:,:]
data_satellite_all = data_satellite_all.iloc[1:,:]
#print(data_street.head())
print(data_satellite_all.head())
print(data_satellite_all.shape)

data_satellite_all_RGB = data_satellite_all.iloc[:,1:4]

X_satellite_all = data_satellite_all_RGB.values
Y_satellite_all = data_satellite_all['water_index'].values

x_train_satellite_all, x_test_satellite_all, y_train_satellite_all, y_test_satellite_all = train_test_split(X_satellite_all, Y_satellite_all, test_size = 0.2, train_size=0.8)

model = LinearRegression()
model.fit(x_train_satellite_all, y_train_satellite_all)

pred = model.predict(x_test_satellite_all)
plt.scatter(y_test_satellite_all, pred)
plt.show()
sns.regplot(x=y_test_satellite_all, y=pred)
plt.show()
