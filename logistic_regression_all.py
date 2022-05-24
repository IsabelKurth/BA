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
data_street = data_street.iloc[1:,:]
data_satellite = data_satellite.iloc[1:,:]
data_street_RGB = data_street.iloc[:,1:4]
data_satellite_RGB = data_satellite.iloc[:,1:4]

# features and labels
X_street = data_street_RGB.values
X_satellite = data_satellite_RGB.values

Y_street = data_street['water_index'].values
Y_satellite = data_satellite['water_index'].values

# hyperparameters
# batch_size = 
# learning_rate = 

# constants
num_classes = 5

# train test split
# split data 
x_train_street, x_test_street, y_train_street, y_test_street = train_test_split(X_street, Y_street, test_size = 0.2, train_size=0.8)
x_train_satellite, x_test_satellite, y_train_satellite, y_test_satellite = train_test_split(X_satellite, Y_satellite, test_size = 0.2, train_size=0.8)

### street ###
# auf ganze Zahlen gerundet 
y_train_street_grob = np.digitize(y_train_street, [1, 2, 3, 4, 5])
y_test_street_grob = np.digitize(y_test_street, [1, 2, 3, 4, 5]) 

logreg = LogisticRegression(multi_class = 'multinomial', max_iter = 1e3)
logreg.fit(x_train_street, y_train_street_grob)
pred_street = logreg.predict(x_test_street)
score = logreg.score(x_test_street, y_test_street_grob)
print(score) #0.85

cm = confusion_matrix(y_test_street_grob, pred_street)
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
logreg_sat.fit(x_train_satellite, y_train_satellite_grob)
pred_satellite = logreg_sat.predict(x_test_satellite)
score_sat = logreg_sat.score(x_test_satellite, y_test_satellite_grob)
print(score_sat) # 0.44

cm_sat = confusion_matrix(y_test_satellite_grob, pred_satellite)
print(cm_sat)
# Ci,j: number of observations known to be in group i and predicted to be in group j 

plt.figure(figsize=(5,5))
sns.heatmap(cm_sat)
plt.show()

#sns.regplot(x_train_street, y_train_street_grob, logistic=True, ci=None)
#plt.show()
