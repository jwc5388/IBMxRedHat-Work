import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

#1 data
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2 model
# model = Perceptron()      #구려
# model =LogisticRegression()   #
# model = LinearSVC()   #구려
# model = SVC()       #된다
model = DecisionTreeClassifier()


#3 훈련

model.fit(x_data, y_data)

#4 평가 예측
y_predict = model.predict(x_data)
result = model.score(x_data,y_data)
print('model.score:', result)    



acc = accuracy_score(np.round(y_predict), y_data)
print('accuracy score:', acc)
