import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine


# 1. Load Data
dataset = load_wine()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2


n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)


#회귀는 stratfied 안된다
kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)


# model
model = RandomForestClassifier()
# model = HistGradientBoostingClassifier()


# train
score = cross_val_score(model, x,y, cv=kfold)   #fit 포함됨 그 안에 자동으로 훈련(fit)과 평가(predict, score) 과정이 포함되어 있습니다.

print('accuracy:', score, '\n평균 acc:',round( np.mean(score), 4) )


#       accuracy: [0.97222222 1.         1.         0.94285714 1.        ] 
# 평균 acc: 0.983


#       accuracy: [0.95196521 0.93279686 0.93595752 0.83392856 0.90365757] 
# 평균 acc: 0.9117