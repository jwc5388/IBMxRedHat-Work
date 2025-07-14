import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#1 data
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,0,0,1]

#2 model
# model = Perceptron()
# model =LogisticRegression()
model = LinearSVC()

#3 훈련
model.fit(x_data, y_data)

#4 평가 예측
y_predict = model.predict(x_data)
result = model.score(x_data,y_data)
print('model.score:', result)
acc = accuracy_score(y_predict, y_data)
print('accuracy score:', acc)


# model.score: 1.0
# accuracy score: 1.0


# model.score: 0.75
# accuracy score: 0.75



# model.score: 1.0
# accuracy score: 1.0