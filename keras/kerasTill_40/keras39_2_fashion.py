# #36-2 copy
# import numpy as np
# import pandas as pd
# from keras.datasets import mnist, fashion_mnist
# import pandas as pd
# from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import time

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train)

# # print(x_train[0])
# # print(y_train[0]) 

# # exit()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)


# #이진인지 다중인지 확인!!!! 
# print(np.unique(y_train, return_counts=True))    #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))

# print(pd.value_counts(y_test))
# # exit()
# # aaa = 7
# # print(y_train[aaa])

# # import matplotlib.pyplot as plt

# # plt.imshow(x_train[aaa], 'twilight_shifted')
# # plt.show()


# x_train = x_train/255.
# x_test = x_test/255.


# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)


# #2 model

# model = Sequential()

# model.add(Conv2D(filters = 64, kernel_size=(3,3), 
#                 strides=1, 
#                 input_shape = (28,28,1), 
#                 activation='relu'))

# model.add(MaxPooling2D())
# model.add(Dropout(0.2))

# model.add(Conv2D(filters = 128, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))

# model.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))

# model.add(Flatten())

# model.add(Dense(units=16, activation='relu'))


# model.add(Dense(10, activation='softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])


# es = EarlyStopping(
#     monitor='val_loss',
#     mode = 'auto',
#     patience=20,
#     restore_best_weights=True
# )


# start = time.time()

# model.fit(x_train, y_train, epochs=500, batch_size=32, verbose= 1, validation_split=0.2, callbacks=[es])

# end = time.time()

# loss, acc = model.evaluate(x_test, y_test, verbose=1)
# print('loss:', loss)
# print('acc:', acc)

# print('time:', end-start)

# y_pred = model.predict(x_test)
# print(y_pred)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)
# acc_score = accuracy_score(y_test, y_pred)


# # loss: 0.49149590730667114
# # acc: 0.8234000205993652
# # time: 167.34843397140503


import numpy as np
import pandas as pd
import time
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# 1. Load Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28), (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28), (10000,)

# 2. Preprocessing
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# 3. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# 5. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# 7. Train
start = time.time()
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),  # no augmentation on validation set
    epochs=500,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 8. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print('✅ loss:', loss)
print('✅ acc:', acc)
print('⏱️ time:', end - start)

# 9. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test.values, axis=1)  # Use .values for pandas df

acc_score = accuracy_score(y_test_labels, y_pred)
print("✅ Final Accuracy Score (sklearn):", acc_score)
