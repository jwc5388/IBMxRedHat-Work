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
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor, VotingClassifier
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

# === Load Data ===
path = basepath + '_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']


# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
                         voting='soft',
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
# final score: 0.8807939421753098