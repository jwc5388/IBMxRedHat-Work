
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)
#1 data
dataset = fetch_covtype()
x = dataset.data
y = dataset.target


# ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

model = XGBClassifier(
    n_estimators = 500,
    random_state = seed,
    gamma = 0,
    min_child_weight = 0,
    reg_alpha = 0,
    reg_lambda = 1,
    # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
    #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
    callbacks = [es]
    
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=1)




print('acc:', model.score(x_test, y_test))   # acc2: 0.9333333333333333
print(model.feature_importances_)# [0.01230742 0.02487084 0.5794107  0.38341108]


thresholds = np.sort(model.feature_importances_) #오름차순
print(thresholds) #[0.01230742 0.02487084 0.38341108 0.5794107 ]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit = False)
    # threshold 가 i값 이상인것을 모두 훈련시킨다. 
    # prefit = False : 모델이 아직 학습 되지 않았을때, model.fit 호출해서 훈련한다. (기본)
    # prefit = True : 이미 학습 된 모델을 전달 할 때, model.fit
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape) 

    select_model = XGBClassifier(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
        #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        callbacks = [es]
         
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, acc: %.4f%%' %(i, select_x_train.shape[1], score*100))



# Trech=0.001, n=54, acc: 83.1407%
# Trech=0.002, n=53, acc: 74.7339%
# Trech=0.003, n=52, acc: 74.7339%
# Trech=0.003, n=51, acc: 74.7339%
# Trech=0.004, n=50, acc: 74.7468%
# Trech=0.004, n=49, acc: 74.7468%
# Trech=0.004, n=48, acc: 74.6091%
# Trech=0.005, n=47, acc: 74.6091%
# Trech=0.005, n=46, acc: 74.6082%
# Trech=0.005, n=45, acc: 74.6082%
# Trech=0.005, n=44, acc: 74.5093%
# Trech=0.005, n=43, acc: 74.5093%
# Trech=0.006, n=42, acc: 74.5230%
# Trech=0.006, n=41, acc: 74.5721%
# Trech=0.007, n=40, acc: 74.4017%
# Trech=0.008, n=39, acc: 74.4051%
# Trech=0.008, n=38, acc: 74.0179%
# Trech=0.008, n=37, acc: 74.0179%
# Trech=0.009, n=36, acc: 74.0179%
# Trech=0.009, n=35, acc: 74.0179%
# Trech=0.010, n=34, acc: 73.9792%
# Trech=0.010, n=33, acc: 73.2313%
# Trech=0.011, n=32, acc: 73.2313%
# Trech=0.012, n=31, acc: 72.3441%
# Trech=0.012, n=30, acc: 72.4172%
# Trech=0.012, n=29, acc: 71.3450%
# Trech=0.013, n=28, acc: 71.3002%
# Trech=0.014, n=27, acc: 71.2520%
# Trech=0.015, n=26, acc: 71.1453%
# Trech=0.015, n=25, acc: 71.1453%
# Trech=0.015, n=24, acc: 71.1453%
# Trech=0.018, n=23, acc: 71.0472%
# Trech=0.019, n=22, acc: 71.0059%
# Trech=0.019, n=21, acc: 70.9887%
# Trech=0.020, n=20, acc: 70.9870%
# Trech=0.021, n=19, acc: 70.9767%
# Trech=0.021, n=18, acc: 70.8742%
# Trech=0.022, n=17, acc: 70.8656%
# Trech=0.024, n=16, acc: 70.8656%
# Trech=0.024, n=15, acc: 70.8656%
# Trech=0.024, n=14, acc: 70.8656%
# Trech=0.027, n=13, acc: 70.4052%
# Trech=0.029, n=12, acc: 70.4827%
# Trech=0.031, n=11, acc: 70.4646%
# Trech=0.034, n=10, acc: 70.4483%
# Trech=0.034, n=9, acc: 70.4121%
# Trech=0.037, n=8, acc: 69.6712%
# Trech=0.040, n=7, acc: 69.2770%
# Trech=0.041, n=6, acc: 69.0335%
# Trech=0.041, n=5, acc: 68.6738%
# Trech=0.049, n=4, acc: 68.1764%
# Trech=0.056, n=3, acc: 67.6213%
# Trech=0.058, n=2, acc: 48.7604%
# Trech=0.064, n=1, acc: 48.7604%