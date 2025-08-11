import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

#1 data
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=777, stratify=y)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. model

# model = RandomForestClassifier()
#순서는 지켜야함 scaler 먼저
model = make_pipeline(StandardScaler(), RandomForestClassifier())

#3 train
model.fit(x_train, y_train)

#4 evaluate, predict
result = model.score(x_test, y_test)
print('model score:', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy score:', acc)