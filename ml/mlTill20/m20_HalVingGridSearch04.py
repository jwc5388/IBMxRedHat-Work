
import numpy as np
from sklearn import base
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits, load_iris, fetch_covtype, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/TensorJae/Study25/')
    
basepath = os.path.join(BASE_PATH)

warnings.filterwarnings('ignore')


#1 data


path = basepath +  '_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
# kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

parameters = [
    {'n_estimators': [100,500], 'max_depth': [6,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #18
    
    {'max_depth' : [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    # {'min_child_weight': [2,3,5,10]}
]       #42



#2 model

xgb = XGBRegressor()
    # tree_method='gpu_hist',         # ✅ GPU 학습 방식 지정
    # predictor='gpu_predictor',      # ✅ 예측도 GPU로 (optional)
    # gpu_id=0,                       # ✅ 첫 번째 GPU 사용
    # n_jobs=-5,
    # verbosity=1)

model = HalvingGridSearchCV(xgb, parameters, cv = kfold, 
                            verbose = 1,
                            n_jobs=-5,
                            refit = True,
                            random_state=42,
                            factor = 3,  # HalvingGridSearchCV에서 factor는 각 단계에서 후보군의 수를 얼마나 줄일지를 결정합니다.
                            min_resources=10, # 1 iter 때의 최소 훈련 행의 갯수
                            max_resources = 120 #데이터 행의 갯수
                            
                            )
#refit = True 이면, 최적의 파라미터로 다시 학습을 시킨다.
#3 훈련

start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수:', model.best_estimator_)

print('최적의 파라미터:', model.best_params_)

#4 평가, 예측
print('best score:', model.best_score_ )

print('model.score:', model.score(x_test, y_test))

y_pred = model.predict(x_test)

# print('accuracy score:', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
# print('best accuracy score:', accuracy_score(y_test, y_pred_best))


print('time:', round(end-start, 2), 'seconds')

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   #ranktestscore 기준으로 오름차순 정렬

print(pd.DataFrame(model.cv_results_).columns)


path = '/workspace/TensorJae/Study25/_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm20_04_csv_results.csv')  #gs = gridsearch




joblib.dump(model.best_estimator_, path + 'm20_04_best_model.joblib')


# best score: 0.7597301005524858
# model.score: 0.7841466659425407
# time: 41.18 seconds


# best score: 0.5014062015018844
# model.score: 0.7869391730795359
# time: 157.09 seconds