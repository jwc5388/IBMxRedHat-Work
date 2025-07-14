import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine


# 1. Load Data
dataset = load_wine()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5

kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

#2 model
model = MLPClassifier()


#3 훈련 
score = cross_val_score(model, x_train, y_train, cv = kfold)

print('acc:', score, '\n평균 acc:', round(np.mean(score), 4) )



y_pred = cross_val_predict(model, x_test, y_test, cv = kfold)

acc_score = accuracy_score(y_test, y_pred)

print('cross_val_predict ACC', acc_score)


# acc: [0.89655172 1.         1.         1.         0.96428571] 
# 평균 acc: 0.9722
# cross_val_predict ACC 0.9444444444444444