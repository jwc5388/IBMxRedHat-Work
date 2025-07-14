#배치를 적용한거
#쪼개서 넣어주면 정확도가 상승할 수 있다. 쪼개는걸 batch라고 한다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2 모델 구성
model = Sequential()

model.add(Dense(6,input_dim=1))
model.add(Dense(36))
model.add(Dense(72))
model.add(Dense(36))
model.add(Dense(1))


#3 컴파일, 훈련
epochs = 100

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs = epochs, batch_size = 2
          )

#여기서 예를 들어 배치 사이즈를 4로 하면, 데이터 크기는 6이면, 4 2 이런식으로 나뉘어서 훈련된다


#4 평가 예측

loss = model.evaluate(x,y)
print("+++++++++++++++++++++++++++++")

print("epochs: ", epochs)

print('loss: ', loss)

result = model.predict([6])
print('6 prediction: ' , result)


#Batch is a smallgroup of data samples that your model looks at at one time during training
# each batch is like one bite of the data
#여기 6/6 은 epochs 를 나눈거다 6으로 
#Batch는 낮을수록 정확도가 좋다 gpu메모리에 좋은건 아니다
#그렇다고 또 batch가 너무 작아도 문제가 생길 수도 있다
#데이터 크기에 따라 배치 사이즈도 적절히 맞춰서


# Epoch 100/100
# 6/6 [==============================] - 0s 792us/step - loss: 0.4721
# 1/1 [==============================] - 0s 106ms/step - loss: 0.3915


# batch 5
# epochs:  100
# loss:  0.32388168573379517

# batch 2
# epochs:  100
# loss:  0.32383859157562256