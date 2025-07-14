# 08-1 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#70% train, %30 evaluation
#1 data
# x = np.array([1,2,3,4,5,6,7,8,9,10]).T
# y = np.array([1,2,3,4,5,6,7,8,9,10]).T

# print(x.shape)
# print(y.shape)
# exit()

#training data
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

#훈련할떄 validation 함 
x_val = np.array([7,8])
y_val = np.array([7,8])

#evaluation/testing data
x_test = np.array([9,10])
y_test = np.array([9,10])

#2 model
model = Sequential()
model.add(Dense(1, input_dim= 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# compile, train

#evaluation 1 epochs 마다 happen
#val_loss is important -기준
#왜 validation 사용하는가?
# 말 그대로 검증용 데이터이다. 학습 중 overfitting 방지, 일반화 성능 모니터링
#성능을 올려주진 않지만 좋은 모델을 만들도록 도움을 줌(피드백 도구)

#fit에서 가중치를 보여줌. 가중치를 올려주진 않음!!!

model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=200, batch_size = 1, 
          validation_data = (x_val, y_val))


loss = model.evaluate(x_test,y_test)
result = model.predict(np.array([11]))

print("loss:", loss)
print("prediction of [11]:", result)