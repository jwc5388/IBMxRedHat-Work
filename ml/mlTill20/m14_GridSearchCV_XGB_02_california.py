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

from sklearn.datasets import fetch_california_housing


warnings.filterwarnings('ignore')


#1 data
x,y = fetch_california_housing(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8)

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


# best score: 0.840487351583144
# model.score: 0.8446189810970788
# time: 48.81 seconds