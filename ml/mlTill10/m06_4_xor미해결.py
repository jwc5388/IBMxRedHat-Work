import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense



#1 data
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2 model
# model = Perceptron()
# model =LogisticRegression()
# model = LinearSVC()


model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))

#3 훈련

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['acc'])
model.fit(x_data, y_data, epochs=100)

#4 평가 예측
y_predict = model.predict(x_data)
result = model.evaluate(x_data,y_data)
print('model.evaluate:', result)    



acc = accuracy_score(np.round(y_predict), y_data)
print('accuracy score:', acc)


# model.evaluate: [0.7607640027999878, 0.5]
# accuracy score: 0.5