import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


#1 data

#1 data
from sklearn.datasets import load_wine


# 1. Load Data
dataset = load_wine()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_split = 5
kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

parameters = [
    {'n_estimators': [100,500], 'max_depth': [6,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #18
    
    {'max_depth' : [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    # {'min_child_weight': [2,3,5,10]}
]       #42



#2 model

xgb = XGBClassifier(
    tree_method='gpu_hist',         # ✅ GPU 학습 방식 지정
    predictor='gpu_predictor',      # ✅ 예측도 GPU로 (optional)
    gpu_id=0,                       # ✅ 첫 번째 GPU 사용
    n_jobs=25,
    verbosity=1)

model = GridSearchCV(xgb, parameters, cv = kfold,         # 54*5 = 270
                     verbose = 1, 
                     n_jobs=25,             
                     refit = True)        # 271번.  false 로 하면 에러뜸

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

print('accuracy score:', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('best accuracy score:', accuracy_score(y_test, y_pred_best))


print('time:', round(end-start, 2), 'seconds')

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   #ranktestscore 기준으로 오름차순 정렬

print(pd.DataFrame(model.cv_results_).columns)

path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm15_gs_wine_csv_results.csv')  #gs = gridsearch