from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Conv2D, Flatten


#원본은 (N,5,5,1) 이미지                                                      #세로, 가로, 색깔(color) 
model = Sequential()                                                       #height width, channels
model.add(Conv2D(filters = 10, kernel_size=(2,2), input_shape=(5, 5, 1))) # (N,4,4,10)
model.add(Conv2D(filters = 5, kernel_size=(2,2)))                         # (3,3,5)
model.add(Flatten())
#dense 2dimension input and output
#dense 다차원 입력 출ㄹ력 된다
model.add(Dense(units = 10))    # input = (batch, input_dim)
model.add(Dense(3))
model.summary()

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))

model.summary()