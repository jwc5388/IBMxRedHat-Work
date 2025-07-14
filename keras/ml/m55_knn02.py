from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
import random
import numpy as np
import xgboost as xgb
import pandas as pd
#1data

seed = 123

random.seed(seed)
np.random.seed(seed)

datasets = fetch_california_housing()
x = datasets.data
y= datasets.target
print(x.shape, y.shape)        

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNeighborsRegressor(n_neighbors=5)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('accuracy score:', acc)

print('r2 score:', r2_score(y_test, y_pred))


# r2 score: 0.6910488020112968
