import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


### plot sigmoid: logistic regression ###
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.arange(-5, 5, 0.1)

phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.xlabel('z')
plt.ylabel('$logistic(z)$')
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()


### plot regression line: linear regression ###
np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5
res = 0.5 * np.random.randn(100)
y = 2 + 0.3 * X + res

X = X.reshape((100, 1))
reg = LinearRegression()

# Fitting training data
reg = reg.fit(X, y)
ypred = reg.predict(X)
plt.clf()

# Calculating RMSE and R2 Score
mse = mean_squared_error(y, ypred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, y)

plt.plot(X, ypred, color='red', label='regression line')
plt.scatter(X, y, c='green', label='data')
plt.plot(mse)
#plt.show()



water_data = pd.read_csv('dhs_final_labels.csv')
water_data = water_data.dropna(subset=['water_index'])
print(water_data['year'].min())
print(water_data['year'].max())
print(water_data['year'].unique())


