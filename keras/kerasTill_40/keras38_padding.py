from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Input, MaxPooling2D


#2 model
model = Sequential()
model.add(Conv2D(filters = 10, kernel_size=(2,2), input_shape=(10,10,1), 
                 strides=1,
                 #keeps the shape as before if you do padding = 'same'
                 padding='same',
                #  padding = 'valid'
                 ))

model.add(Conv2D(filters=9, kernel_size=(3,3), 
                 strides=1,
                 #this is default
                 padding='valid'
                 ))

model.add(Conv2D(filters=8,kernel_size=4))

model.summary()