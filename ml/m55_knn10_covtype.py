from sklearn.datasets import load_breast_cancer, load_wine, fetch_covtype
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
# 1. Load Data
dataset = fetch_covtype()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy score:', acc)

# print('f1 score:', f1_score(y_test, y_pred))


# accuracy score: 0.9287195683416091
