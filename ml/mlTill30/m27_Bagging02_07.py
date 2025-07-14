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
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor


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

#1 data
path = basepath +  '_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

print(train_csv.info()) # [652 rows x 9 columns]
print(test_csv.info()) #  [116 rows x 8 columns]
print(sample_submission_csv.info()) # [116 rows x 1 columns]



x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# model = DecisionTreeRegressor()
model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, 
                         n_jobs=-2, 
                         random_state=42, 
                        #  bootstrap=True,)           #false 성능 개 구림
)

# model = RandomForestRegressor(random_state=42)
#3 train
model.fit(x_train, y_train)


#4 train, predict
result = model.score(x_test, y_test)
print('final score:', result)


# final score: 0.7633587786259542