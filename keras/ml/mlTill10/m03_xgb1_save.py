import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#1 data
x , y = load_breast_cancer(return_X_y=True)
#이진분류이기때문에 stratiofy y 하면
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8, stratify=y)

print(x.shape, y.shape) #(569, 30) (569,)

parameters = {'n_estimators': 1000,
              'learning_rate': 0.3,
              'max_depth': 3,
              'gamma':1,
              'min_child_weight': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state': 3377,
            #   'verbose': 0,
              }

##################
# 변수 앞에 * 하나 **둘 차이 뭔지.
##################


#2 model
model = XGBClassifier(
    # **parameters,
    # n_estimators = 1000,
)

#3 train
model.set_params(**parameters,
                 early_stopping_rounds = 10)


model.fit(x_train, y_train,
          eval_set = [(x_test,y_test)],
          verbose=10)


result = model.score(x_test, y_test)
print("최종점수:", result)


y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('accuracy score:', acc)


# 최종점수: 0.9736842105263158
# accuracy score: 0.9736842105263158

path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_save/m01_job/'
# import joblib 
# joblib.dump(model, path + 'm01_joblib_save.joblib')

model.save_model(path + 'm03_xgb_save.dat')
