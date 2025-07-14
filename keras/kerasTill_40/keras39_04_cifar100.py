import tensorflow as tf
print("Using GPU:", tf.config.list_physical_devices('GPU'))


from keras.datasets import cifar100
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time


#1 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))   

#(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    #    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500]))
    
    
    
x_train = x_train/255.
x_test = x_test/255.


y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
# )
# datagen.fit(x_train)

#2 model
model = Sequential()
model.add(Conv2D(filters=64, 
                 kernel_size=(3,3),
                 input_shape = (32,32,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=256, 
                 kernel_size=(3,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, 
                 kernel_size=(3,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

exit()


model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True
)

start = time.time()

hist = model.fit(datagen.flow(x_train, y_train, batch_size=64),
          validation_data=(x_test, y_test),
          epochs=100,
          callbacks=[es],
          verbose=1)


end = time.time()

loss, acc = model.evaluate(x_test, y_test)

print('loss:', loss)
print('acc:', acc)


print('time:', end- start)

result = model.predict(x_test)

y_pred = np.argmax(result, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test,y_pred)

print('accuracy score:', acc_score)


# loss: 2.544921398162842
# acc: 0.34450000524520874
# time: 1230.7274162769318

# loss: 1.7060247659683228
# acc: 0.5497999787330627
# time: 3034.2118003368378
