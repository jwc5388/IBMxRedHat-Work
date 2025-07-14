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


warnings.filterwarnings('ignore')


#1 data

#1 data
path = './Study25/_data/diabetes/'

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

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

parameters = [
    {'n_estimators': [100,500], 'max_depth': [6,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #18
    
    {'max_depth' : [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}, #12
    
    # {'min_child_weight': [2,3,5,10]}
]       #42



#2 model

# xgb = XGBRegressor()
xgb = XGBClassifier()
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

print('accuracy score:', accuracy_score(y_test, y_pred))

print('time:', round(end-start, 2), 'seconds')


# best score: 0.758040293040293
# model.score: 0.7557251908396947
# accuracy score: 0.7557251908396947
# time: 2.88 seconds