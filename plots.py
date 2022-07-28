import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import npzviewer
from PIL import Image
"""
def normalize8(I):
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8)

# dfile = "C:\\Users\\isabe\\Documents\\BA\\BA\\DHS_Data\\CM-2011-6#\\CM-2011-6#-00000159.npz"
dfile = "C:\\Users\\isabe\\Documents\\BA\\BA\\DHS_Data\\AM-2016-7#\\AM-2016-7#-00000045.npz"
#dfile = "C:\\Users\\isabe\\Documents\\BA\\BA\\DHS_Data\\DR-2013-6#\\DR-2013-6#-00002296.npz"
data = np.load(dfile)
print(data.files)
image = data['x']
#image = normalize8(image)
print(image.shape, image.dtype)
#img = Image.fromarray(image)
#plt.imshow(image[:3].transpose(1,2,0))
plt.imshow(image[7], cmap='Greys')
plt.show()
plt.savefig('satellite day.png')
#npzviewer [dfile]


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

"""

water_data = pd.read_csv('dhs_final_labels.csv')
water_data = water_data.dropna(subset=['water_index'])
print(water_data['year'].min())
print(water_data['year'].max())
print(water_data['year'].unique())

water_new, water_old = [x for _, x in water_data.groupby(water_data['year'] < 2016)]
print(water_old.shape)
print(water_new.shape)
print(water_data.shape)

old = water_data.loc[water_data['year'] < 2016]
print(old.shape)

