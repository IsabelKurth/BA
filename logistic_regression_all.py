import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# load data
data_satellite = pd.read_pickle('finish_satellite.pkl')
data_street = pd.read_pickle('finish_street.pkl')
data_satellite_night = pd.read_pickle('finish_satellite_night.pkl')
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]
data_street_RGB = data_street.iloc[:,1:4]
data_satellite_RGB = data_satellite.iloc[:,1:4]

# features and labels
X_street = data_street_RGB.values
X_satellite = data_satellite_RGB.values
X_satellite_night = data_satellite_night['mean_nl']

Y_street = data_street['water_index'].values
Y_satellite = data_satellite['water_index'].values
Y_satellite_night = data_satellite_night['water_index'].values




# auf halbe Zahlen gerundet 
for x in range(len(Y_street)):
    if (Y_street[x] < 5 and Y_street[x] >= 4.5):
        Y_street[x] = str(5)
    elif (Y_street[x] < 4.5 and Y_street[x] > 4): 
        Y_street[x] = str(4.5)
    elif (Y_street[x] < 4 and Y_street[x] > 3.5): 
        Y_street[x] = str(4)
    elif (Y_street[x] < 3.5 and Y_street[x] > 3): 
        Y_street[x] = str(3.5)
    elif (Y_street[x] < 3 and Y_street[x] > 2.5): 
        Y_street[x] = str(3)
    elif (Y_street[x] < 2.5 and Y_street[x] > 2): 
        Y_street[x] = str(2.5)
    elif (Y_street[x] < 2 and Y_street[x] > 1.5): 
        Y_street[x] = str(2)
    elif (Y_street[x] < 1.5 and Y_street[x] > 1): 
        Y_street[x] = str(1.5)
    elif (Y_street[x] < 1 and Y_street[x] > 0.5): 
        Y_street[x] = str(1)
    elif (Y_street[x] < 0.5 and Y_street[x] > 0): 
        Y_street[x] = str(0.5)
    else: 
        Y_street[x] = str(0)                               

print(Y_street.shape)
Y_street_str = Y_street.astype(str)
print(Y_street_str)

for x in range(len(Y_satellite)):
    if (Y_satellite[x] < 5 and Y_satellite[x] >= 4.5):
        Y_satellite[x] = str(5)
    elif (Y_satellite[x] < 4.5 and Y_satellite[x] > 4): 
        Y_satellite[x] = str(4.5)
    elif (Y_satellite[x] < 4 and Y_satellite[x] > 3.5): 
        Y_satellite[x] = str(4)
    elif (Y_satellite[x] < 3.5 and Y_satellite[x] > 3): 
        Y_satellite[x] = str(3.5)
    elif (Y_satellite[x] < 3 and Y_satellite[x] > 2.5): 
        Y_satellite[x] = str(3)
    elif (Y_satellite[x] < 2.5 and Y_satellite[x] > 2): 
        Y_satellite[x] = str(2.5)
    elif (Y_satellite[x] < 2 and Y_satellite[x] > 1.5): 
        Y_satellite[x] = str(2)
    elif (Y_satellite[x] < 1.5 and Y_satellite[x] > 1): 
        Y_satellite[x] = str(1.5)
    elif (Y_satellite[x] < 1 and Y_satellite[x] > 0.5): 
        Y_satellite[x] = str(1)
    elif (Y_satellite[x] < 0.5 and Y_satellite[x] > 0): 
        Y_satellite[x] = str(0.5)
    else: 
        Y_satellite[x] = str(0)                               


Y_satellite_str = Y_satellite.astype(str)


for x in range(len(Y_satellite_night)):
    if (Y_satellite_night[x] < 5 and Y_satellite_night[x] >= 4.5):
        Y_satellite_night[x] = str(5)
    elif (Y_satellite_night[x] < 4.5 and Y_satellite_night[x] > 4): 
        Y_satellite_night[x] = str(4.5)
    elif (Y_satellite_night[x] < 4 and Y_satellite_night[x] > 3.5): 
        Y_satellite_night[x] = str(4)
    elif (Y_satellite_night[x] < 3.5 and Y_satellite_night[x] > 3): 
        Y_satellite_night[x] = str(3.5)
    elif (Y_satellite_night[x] < 3 and Y_satellite_night[x] > 2.5): 
        Y_satellite_night[x] = str(3)
    elif (Y_satellite_night[x] < 2.5 and Y_satellite_night[x] > 2): 
        Y_satellite_night[x] = str(2.5)
    elif (Y_satellite_night[x] < 2 and Y_satellite_night[x] > 1.5): 
        Y_satellite_night[x] = str(2)
    elif (Y_satellite_night[x] < 1.5 and Y_satellite_night[x] > 1): 
        Y_satellite_night[x] = str(1.5)
    elif (Y_satellite_night[x] < 1 and Y_satellite_night[x] > 0.5): 
        Y_satellite_night[x] = str(1)
    elif (Y_satellite_night[x] < 0.5 and Y_satellite_night[x] > 0): 
        Y_satellite_night[x] = str(0.5)
    else: 
        Y_satellite_night[x] = str(0)                               


Y_satellite_night_str = Y_satellite_night.astype(str)


# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street_str, test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite_str, test_size = 0.2, train_size=0.8)
x_train_satellite_night, x_test_satellite_night, y_train_satellite_night, y_test_satellite_night = train_test_split(X_satellite_night, Y_satellite_night_str, test_size = 0.2, train_size=0.8)


### street ###
# auf ganze Zahlen gerundet 
y_train_street_grob = np.digitize(y_train_street, [1, 2, 3, 4, 5])
y_test_street_grob = np.digitize(y_test_street, [1, 2, 3, 4, 5]) 

logreg = LogisticRegression(multi_class = 'multinomial', max_iter = 1e3)
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

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()


### satellite ###
# auf ganze Zahlen gerundet 
y_train_satellite_grob = np.digitize(y_train_satellite, [1, 2, 3, 4, 5])
y_test_satellite_grob = np.digitize(y_test_satellite, [1, 2, 3, 4, 5]) 

logreg_sat = LogisticRegression(multi_class = 'multinomial', max_iter = 1e3)
logreg_sat.fit(x_train_satellite, y_train_satellite)
pred_satellite = logreg_sat.predict(x_test_satellite)
score_sat = logreg_sat.score(x_test_satellite, y_test_satellite)
print('satellite score', score_sat) # 0.44

cm_sat = confusion_matrix(y_test_satellite, pred_satellite)
print(cm_sat)
# Ci,j: number of observations known to be in group i and predicted to be in group j 

plt.figure(figsize=(5,5))
sns.heatmap(cm_sat)
plt.show()

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()

### satellite night ###
x_train_satellite_night = x_train_satellite_night.to_numpy()
x_train_satellite_night = x_train_satellite_night.reshape(-1,1)
x_test_satellite_night = x_test_satellite_night.to_numpy()
x_test_satellite_night = x_test_satellite_night.reshape(-1,1)

logreg_night = LogisticRegression(multi_class = 'multinomial', max_iter = 1e3)
logreg_night.fit(x_train_satellite_night, y_train_satellite_night)
pred_night = logreg_night.predict(x_test_satellite_night)
score_night = logreg_night.score(x_test_satellite_night, y_test_satellite_night)
print('night score', score_night) # 0.44

cm_night = confusion_matrix(y_test_satellite_night, pred_night)
print(cm_night)
# Ci,j: number of observations known to be in group i and predicted to be in group j 

plt.figure(figsize=(5,5))
sns.heatmap(cm_night)
plt.show()

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()

