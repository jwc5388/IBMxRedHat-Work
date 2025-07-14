from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#에포는 100으로 고정
#loss 기준 0.32 미만으로 만들 것
#2 모델 구성
model = Sequential()

model.add(Dense(6,input_dim=1))
model.add(Dense(36))
model.add(Dense(72))
model.add(Dense(144))
model.add(Dense(72))
model.add(Dense(36))
model.add(Dense(1))


#3 컴파일, 훈련
epochs = 100

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs = epochs)



#4 평가 예측

loss = model.evaluate(x,y)
print("+++++++++++++++++++++++++++++")

print("epochs: ", epochs)

print('loss: ', loss)

result = model.predict([6])
print('6 prediction: ' , result)

# epochs:  100
# loss:  0.32427695393562317

# epochs:  200
# loss:  0.32384994626045227

# epochs:  100
# loss:  0.3243955671787262
# 1/1 [==============================] - 0s 74ms/step
# 6 prediction:  [[5.8901796]]