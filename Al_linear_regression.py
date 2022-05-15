#multiple linear regression
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import r2_score

# load data 
data_satellite = pd.read_pickle('AL_2008_finish_satellite.pkl')
data_street = pd.read_pickle('AL_2008_finish_street.pkl')
data_street = data_street.iloc[1:,:]
print(data_street)

data_street_RGB = data_street.iloc[:,1:4]
# X: red/green/blue values (column 1, 2, 3 of the dataset)
# shape 75, 3
X = data_street_RGB.values


# Y: water index
# shape 75, 
Y = data_street['water_index'].values


# split data 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, train_size=0.8)
print(x_train.shape)
print(y_train.shape)


model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_)
print(model.intercept_)

predictions = model.predict(x_test)
score = r2_score(y_test, predictions)
print(score)


