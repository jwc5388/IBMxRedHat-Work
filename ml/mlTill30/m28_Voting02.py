from tabnanny import verbose
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
from bayes_opt import BayesianOptimization
import random
import time
from keras.callbacks import EarlyStopping
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data

#1 data
x,y = fetch_california_housing(return_X_y=True)


# ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
                        #  voting='soft',
                        #  voting = 'hard'
                         )

#3 train
model.fit(x_train, y_train)


#4 train, predict
result = model.score(x_test, y_test)
print('final score:', result)

#Bagging
# final score: 0.9666015507344905



# final score: 0.8780324088018382



# VotingClassifier
# final score: 0.8563376051498612