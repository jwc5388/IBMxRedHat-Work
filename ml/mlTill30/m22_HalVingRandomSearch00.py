
import random
from tabnanny import verbose
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
import joblib
from sklearn.model_selection import HalvingRandomSearchCV

warnings.filterwarnings('ignore')
import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/TensorJae/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data
x,y = load_iris(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8, stratify=y)

n_split = 5
kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

parameters = [
    {'n_estimators': [100,500], 'max_depth': [6,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #18
    
    {'max_depth' : [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    # {'min_child_weight': [2,3,5,10]}
]       #42



#2 model

xgb = XGBClassifier()
# model = RandomizedSearchCV(xgb, parameters, cv = kfold,         # 54*5 = 270
#                      verbose = 1, 

#                      n_jobs=-5,            
#                      refit = True,
#                      random_state=42,
#                      n_iter=11)        


# model = HalvingGridSearchCV(xgb, parameters, cv = kfold, 
#                             verbose = 1,
#                             n_jobs=-5,
#                             refit = True,
#                             random_state=42,
#                             factor = 3,  # HalvingGridSearchCV에서 factor는 각 단계에서 후보군의 수를 얼마나 줄일지를 결정합니다.
#                             min_resources=10, # 1 iter 때의 최소 훈련 행의 갯수
#                             max_resources = 120 #데이터 행의 갯수
                            
#                             )

model = HalvingRandomSearchCV(xgb, parameters, cv = kfold, 
                            verbose = 1,
                            n_jobs=-5,
                            refit = True,
                            random_state=42,
                            factor = 3,  # HalvingGridSearchCV에서 factor는 각 단계에서 후보군의 수를 얼마나 줄일지를 결정합니다.
                            min_resources=10, # 1 iter 때의 최소 훈련 행의 갯수
                            max_resources = 120 #데이터 행의 갯수
                            
                            )

### 배율 min_resources# factor 
# ncandidates/factor
#refit = True 이면, 최적의 파라미터로 다시 학습을 시킨다.
#3 훈련

start = time.time()
model.fit(x_train, y_train)
end = time.time()

# Fitting 5 folds for each of 10 candidates, totalling 50 fits

print('최적의 매개변수:', model.best_estimator_)

print('최적의 파라미터:', model.best_params_)

#4 평가, 예측
print('best score:', model.best_score_ )

print('model.score:', model.score(x_test, y_test))

y_pred = model.predict(x_test)

print('accuracy score:', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('best accuracy score:', accuracy_score(y_test, y_pred_best))


print('time:', round(end-start, 2), 'seconds')

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   #ranktestscore 기준으로 오름차순 정렬

print(pd.DataFrame(model.cv_results_).columns)

path = basepath + '_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm22_00_csv_results.csv')  #gs = gridsearch

joblib.dump(model.best_estimator_, 'm22_00_best_model.joblib')


#halving grid search
# best score: 0.975
# model.score: 0.9666666666666667
# accuracy score: 0.9666666666666667
# best accuracy score: 0.9666666666666667
# time: 38.04 seconds


#. halving random search
# best score: 0.9666666666666668
# model.score: 0.9666666666666667
# accuracy score: 0.9666666666666667
# best accuracy score: 0.9666666666666667
# time: 38.29 seconds