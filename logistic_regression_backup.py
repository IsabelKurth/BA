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
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]

# features and labels
X_street = data_street.iloc[:,4:7].values
X_satellite = data_satellite.iloc[:,4:7].values
X_satellite_night = data_satellite_night['mean_scaled']

Y_street = data_street['water_index_rnd']
Y_satellite = data_satellite['water_index_rnd']
Y_satellite_night = data_satellite_night['water_index_rnd']


# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street.astype(str), test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite.astype(str), test_size = 0.2, train_size=0.8)
x_train_satellite_night, x_test_satellite_night, y_train_satellite_night, y_test_satellite_night = train_test_split(X_satellite_night, Y_satellite_night.astype(str), test_size = 0.2, train_size=0.8)


### street ###
logreg = LogisticRegression(multi_class = 'multinomial', max_iter = 1500)
logreg.fit(x_train_street, y_train_street)
pred_street = logreg.predict(x_test_street)
score = logreg.score(x_test_street, y_test_street)
print('street score', score) #0.85
#print(logreg.classes_)
#print(logreg.intercept_)
#print(logreg.coef_)

cm = confusion_matrix(y_test_street, pred_street)
print(cm)

plt.figure(figsize=(5,5))
sns.heatmap(cm)
plt.show()

plt.hist(data_street['water_index'])
plt.show()
plt.hist(pred_street)
plt.show()

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()


### satellite ###
logreg_sat = LogisticRegression(multi_class = 'multinomial', max_iter = 1500)
logreg_sat.fit(x_train_satellite, y_train_satellite)
pred_satellite = logreg_sat.predict(x_test_satellite)
score_sat = logreg_sat.score(x_test_satellite, y_test_satellite)
print('satellite score', score_sat) # 0.44

plt.hist(data_satellite['water_index'])
plt.show()
plt.hist(pred_satellite)
plt.show()

cm_sat = confusion_matrix(y_test_satellite, pred_satellite)
print(cm_sat)
# Ci,j: number of observations known to be in group i and predicted to be in group j 

plt.figure(figsize=(5,5))
sns.heatmap(cm_sat)
plt.show()

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()

### satellite night ###
x_train_satellite_night = x_train_satellite_night.to_numpy().reshape(-1,1)
x_test_satellite_night = x_test_satellite_night.to_numpy().reshape(-1,1)
logreg_night = LogisticRegression(multi_class = 'multinomial', max_iter = 1500)
logreg_night.fit(x_train_satellite_night, y_train_satellite_night)
pred_night = logreg_night.predict(x_test_satellite_night)
score_night = logreg_night.score(x_test_satellite_night, y_test_satellite_night)
print('night score', score_night) # 0.44
plt.hist(data_satellite_night['water_index'])
plt.show()
plt.hist(pred_night)
plt.show()

cm_night = confusion_matrix(y_test_satellite_night, pred_night)
print(cm_night)
# Ci,j: number of observations known to be in group i and predicted to be in group j 

plt.figure(figsize=(5,5))
sns.heatmap(cm_night)
plt.show()

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()


