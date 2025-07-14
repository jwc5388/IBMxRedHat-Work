import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_california_housing
import pandas as pd
import joblib


warnings.filterwarnings('ignore')


#1 data


path = './Study25/_data/dacon/ddarung/'

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

xgb = XGBRegressor(
    tree_method='gpu_hist',         # ✅ GPU 학습 방식 지정
    predictor='gpu_predictor',      # ✅ 예측도 GPU로 (optional)
    gpu_id=0,                       # ✅ 첫 번째 GPU 사용
    n_jobs=25,
    verbosity=1)
# xgb = XGBClassifier()
model = GridSearchCV(xgb, parameters, cv = kfold,         # 54*5 = 270
                     verbose = 1, 
                     n_jobs=-1,             #1 번
                     refit = True)          # 271번


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

print('time:', round(end-start, 2), 'seconds')


# 최적의 파라미터: {'learning_rate': 0.1, 'min_child_weight': 2}
# best score: 0.7635897049523054
# model.score: 0.7775273418071964
# time: 4.83 seconds


path = '/workspace/TensorJae/Study25/_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm15_gs_cali_csv_results.csv')  #gs = gridsearch


path = '/workspace/TensorJae/Study25/_save/m15_cv_results/'
joblib.dump(model.best_estimator_, 'm16_05_best_model.joblib')
