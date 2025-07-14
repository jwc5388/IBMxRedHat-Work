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
from sklearn.datasets import fetch_covtype


from sklearn.datasets import load_digits


# 1. Load digits dataset
digits = load_digits()
x = digits.data  # shape: (1797, 64)
y = digits.target  # shape: (1797,)

n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)


#회귀는 stratfied 안된다
kfold = StratifiedKFold(n_splits=n_split, shuffle= True, random_state=42)


# model
# model = RandomForestClassifier()
model = HistGradientBoostingClassifier()


# train
score = cross_val_score(model, x,y, cv=kfold)   #fit 포함됨

print('accuracy:', score, '\n평균 acc:',round( np.mean(score), 4) )


#       accuracy: [0.88099017 0.88597546 0.88653573 0.8621205  0.8544694 ] 
# 평균 acc: 0.874