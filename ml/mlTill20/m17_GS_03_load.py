import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits, load_iris, fetch_covtype, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
import joblib

warnings.filterwarnings('ignore')


#1 data

#1 data
x,y = fetch_california_housing(return_X_y=True)



# # ✅ LabelEncoder 적용
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)


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


path = '/workspace/TensorJae/Study25/_save/m15_cv_results/'

model = joblib.load(path + 'm15_03_best_model.joblib')


print('model.score:', model.score(x_test, y_test))

y_pred = model.predict(x_test)

print('accuracy score:', accuracy_score(y_test, y_pred))