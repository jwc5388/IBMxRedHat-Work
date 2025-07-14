import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1 data
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], [9,8,7,6,5,4,3,2,1,0]]) #(2,5) again here, gave (10,3) data has to be (3,10)
y = np.array([1,2,3,4,5,6,7,8,9,10])

# transpose switches row and column
x = np.transpose(x)
print(x)

# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])  #(5,2) 행 열

print(x.shape) 
print(y.shape) #(5,)

#2 model
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
####행무시
####열우선 열우선 행무시 열우선 행무시 열우선 행무시 
#행은 몇개이든 당연히 다 훈련시킴. 
################column = 열 = 차원

#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 100, batch_size = 1)


#4 evaluate and predict
loss = model.evaluate(x,y)
results = model.predict([[11,2.0,-1]])

print("loss:, loss")
print('[11,2.0,-1]s predicted value:', results)


# the more the colum, the accuracy improves(열)