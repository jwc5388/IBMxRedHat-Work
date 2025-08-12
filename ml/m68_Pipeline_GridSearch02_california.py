# clifornia diabetes 2 3. xgboost

# cancer 6  lightgbm

# digits 11


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import r2_score

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

#1 data
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=777)

parameters = [
    {'xgb__n_estimators' : [100,200], 'xgb__max_depth': [5,6,10], 'xgb__min_samples_leaf': [3,10]},
    {'xgb__max_depth': [6,8,10,12], 'xgb__min_samples_leaf': [3,5,7,10]},
    {'xgb__min_samples_leaf':[3,5,7,9], 'xgb__min_samples_split': [2,3,5,10]},
    {'xgb__min_samples_split': [2,3,5,6]},
]



# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. model

# model = RandomForestClassifier()
#순서는 지켜야함 scaler 먼저
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe = Pipeline([('std', StandardScaler()), ('xgb', XGBRegressor())])

model = GridSearchCV(pipe, parameters, cv = 5, verbose=1, n_jobs=-1)


#3 train
model.fit(x_train, y_train)

#4 evaluate, predict
result = model.score(x_test, y_test)
print('model score:', result)



y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
# acc = accuracy_score(y_test, y_predict)
# print('accuracy score:', acc)

print('r2 score:', r2)


# model score: 0.8303959534857227
# r2 score: 0.8303959534857227