from sklearn.datasets import load_iris, load_diabetes
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
from sklearn.preprocessing import StandardScaler
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

path = basepath + '_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path+ 'sampleSubmission.csv', index_col=0)


# Prepare features and target
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'logloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

model = XGBRegressor(
    n_estimators = 500,
    random_state = seed,
    gamma = 0,
    min_child_weight = 0,
    reg_alpha = 0,
    reg_lambda = 1,
    # eval_metric = 'logloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
    #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
    callbacks = [es]
    
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=0)




print('r2:', model.score(x_test, y_test))   # acc2: 0.9333333333333333
print(model.feature_importances_)# [0.01230742 0.02487084 0.5794107  0.38341108]


thresholds = np.sort(model.feature_importances_) #오름차순
print(thresholds) #[0.01230742 0.02487084 0.38341108 0.5794107 ]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit = False)
    #threshold 가 i값 이상인것을 모두 훈련시킨다. 
    #prefit = False: 모델이 아직 학습되지 않았을떄, fit호출해서 훈련한다(기본값)
    #prefit = True: 이미 학습된 모델을 전달할때, fit 호출하지 않음.
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape) 

    select_model = XGBRegressor(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        callbacks = [es]
        
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Thresh=%.3f, n=%d, R2: %.4f%%' % (i, select_x_train.shape[1], score * 100))
#     # print(score)
#     # print('acc2:', model.score(select_x_test, y_test))  
     
     
# Thresh=0.047, n=8, R2: 16.1873%
# Thresh=0.056, n=7, R2: 16.2445%
# Thresh=0.062, n=6, R2: 16.2042%
# Thresh=0.100, n=5, R2: 16.1422%
# Thresh=0.104, n=4, R2: 15.7087%
# Thresh=0.135, n=3, R2: 15.5759%
# Thresh=0.146, n=2, R2: 14.1091%
# Thresh=0.350, n=1, R2: 9.2104%