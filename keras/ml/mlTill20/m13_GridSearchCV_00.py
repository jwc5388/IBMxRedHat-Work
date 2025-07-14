import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time



#1 data
x,y = load_digits(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8, stratify=y)

n_split = 5
kfold = StratifiedKFold(n_splits= n_split, shuffle=True, random_state=42)

parameters = [
    {"C":[1,10,100,1000], "kernel":['linear', 'sigmoid'], 'degree': [3,4,5]},  #24
    
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma': [0.001, 0.0001]},  #6
    
    {'C':[1,10,100,1000], 'kernel': ['sigmoid'],'gamma': [0.01,0.001,0.0001], 'degree': [3,4]} #24
]



#2 model


model = GridSearchCV(SVC(), parameters, cv = kfold,         # 54*5 = 270
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

# best score: 0.9902560007742934
# model.score: 0.9888888888888889
# accuracy score: 0.9888888888888889
# time: 2.3 seconds

