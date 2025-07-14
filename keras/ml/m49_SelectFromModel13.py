
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, fetch_covtype, load_digits
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

# 1. Load Data
path = basepath + '_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']



# # ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


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


# Trech=0.002, n=93, acc: 71.5498%
# Trech=0.002, n=92, acc: 71.5498%
# Trech=0.002, n=91, acc: 71.5579%
# Trech=0.002, n=90, acc: 71.5740%
# Trech=0.003, n=89, acc: 71.5740%
# Trech=0.003, n=88, acc: 71.5740%
# Trech=0.003, n=87, acc: 71.5740%
# Trech=0.003, n=86, acc: 71.5740%
# Trech=0.003, n=85, acc: 71.5740%
# Trech=0.003, n=84, acc: 71.5740%
# Trech=0.003, n=83, acc: 71.5740%
# Trech=0.003, n=82, acc: 71.5740%
# Trech=0.003, n=81, acc: 71.5175%
# Trech=0.003, n=80, acc: 71.5175%
# Trech=0.003, n=79, acc: 71.5175%
# Trech=0.003, n=78, acc: 71.5175%
# Trech=0.003, n=77, acc: 71.5175%
# Trech=0.004, n=76, acc: 71.5094%
# Trech=0.004, n=75, acc: 71.5094%
# Trech=0.004, n=74, acc: 71.5094%
# Trech=0.004, n=73, acc: 71.5902%
# Trech=0.004, n=72, acc: 71.5902%
# Trech=0.004, n=71, acc: 71.5821%
# Trech=0.004, n=70, acc: 71.6063%
# Trech=0.004, n=69, acc: 71.6063%
# Trech=0.004, n=68, acc: 71.6063%
# Trech=0.004, n=67, acc: 71.6144%
# Trech=0.004, n=66, acc: 71.6063%
# Trech=0.004, n=65, acc: 71.6063%
# Trech=0.004, n=64, acc: 71.6306%
# Trech=0.004, n=63, acc: 71.6387%
# Trech=0.004, n=62, acc: 71.6467%
# Trech=0.004, n=61, acc: 71.6306%
# Trech=0.005, n=60, acc: 71.6306%
# Trech=0.005, n=59, acc: 71.6791%
# Trech=0.005, n=58, acc: 71.6306%
# Trech=0.005, n=57, acc: 71.6306%
# Trech=0.005, n=56, acc: 71.5336%
# Trech=0.005, n=55, acc: 71.5417%
# Trech=0.005, n=54, acc: 71.7033%
# Trech=0.005, n=53, acc: 71.6710%
# Trech=0.006, n=52, acc: 71.6710%
# Trech=0.006, n=51, acc: 71.6710%
# Trech=0.006, n=50, acc: 71.6629%
# Trech=0.006, n=49, acc: 71.4447%
# Trech=0.006, n=48, acc: 71.4205%
# Trech=0.006, n=47, acc: 71.3963%
# Trech=0.006, n=46, acc: 71.3720%
# Trech=0.006, n=45, acc: 71.3882%
# Trech=0.006, n=44, acc: 71.5659%
# Trech=0.006, n=43, acc: 71.3316%
# Trech=0.007, n=42, acc: 71.3882%
# Trech=0.007, n=41, acc: 71.3882%
# Trech=0.007, n=40, acc: 71.2589%
# Trech=0.008, n=39, acc: 71.2427%
# Trech=0.008, n=38, acc: 71.2427%
# Trech=0.008, n=37, acc: 71.0973%
# Trech=0.008, n=36, acc: 70.9761%
# Trech=0.008, n=35, acc: 70.9761%
# Trech=0.009, n=34, acc: 70.7822%
# Trech=0.009, n=33, acc: 70.7822%
# Trech=0.009, n=32, acc: 70.7094%
# Trech=0.009, n=31, acc: 70.9438%
# Trech=0.009, n=30, acc: 70.8226%
# Trech=0.010, n=29, acc: 70.7014%
# Trech=0.010, n=28, acc: 70.3620%
# Trech=0.010, n=27, acc: 70.3297%
# Trech=0.010, n=26, acc: 70.0953%
# Trech=0.011, n=25, acc: 69.9741%
# Trech=0.011, n=24, acc: 69.7317%
# Trech=0.012, n=23, acc: 69.4328%
# Trech=0.012, n=22, acc: 68.5844%
# Trech=0.013, n=21, acc: 68.4308%
# Trech=0.015, n=20, acc: 67.8248%
# Trech=0.015, n=19, acc: 67.9703%
# Trech=0.015, n=18, acc: 67.5259%
# Trech=0.017, n=17, acc: 67.1299%
# Trech=0.017, n=16, acc: 66.6532%
# Trech=0.018, n=15, acc: 65.5866%
# Trech=0.018, n=14, acc: 65.5220%
# Trech=0.020, n=13, acc: 65.2796%
# Trech=0.021, n=12, acc: 64.9240%
# Trech=0.022, n=11, acc: 63.1545%
# Trech=0.022, n=10, acc: 61.8132%
# Trech=0.023, n=9, acc: 60.4396%
# Trech=0.023, n=8, acc: 58.1125%
# Trech=0.025, n=7, acc: 55.4299%
# Trech=0.026, n=6, acc: 55.0016%
# Trech=0.028, n=5, acc: 54.4360%
# Trech=0.034, n=4, acc: 54.1451%
# Trech=0.056, n=3, acc: 50.2262%
# Trech=0.072, n=2, acc: 43.3824%
# Trech=0.098, n=1, acc: 38.2999%