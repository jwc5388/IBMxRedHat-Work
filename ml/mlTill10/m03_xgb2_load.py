import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib 

#1 data
x , y = load_breast_cancer(return_X_y=True)
#이진분류이기때문에 stratiofy y 하면
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8, stratify=y)

print(x.shape, y.shape) #(569, 30) (569,)

##################
# 변수 앞에 * 하나 **둘 차이 뭔지.
##################
path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_save/m01_job/'


#2 model,#3 train ---불러오기
# model = joblib.load(path + 'm01_joblib_save.joblib')
model = XGBClassifier()
model.load_model(path + 'm03_xgb_save.dat')




#4

result = model.score(x_test, y_test)
print("최종점수:", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('accuracy score:', acc)

# accuracy score: 0.956140350877193


# 최종점수: 0.9736842105263158
# accuracy score: 0.9736842105263158

# joblib.dump(model, path + 'm01_joblib_save.joblib')
