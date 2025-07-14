# from keras.datasets import cifar10
# import numpy as np
# import pandas as pd
# from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
# from keras.models import Sequential, Model
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import time

# #1 data
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)

# print(x_test.shape, y_test.shape)#(10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]))

# x_train = x_train/255.
# x_test = x_test/255.


# y_train = pd.get_dummies(y_train.reshape(-1))
# y_test = pd.get_dummies(y_test.reshape(-1))

# #2 model
# model = Sequential()
# model.add(Conv2D(filters=32, 
#                  kernel_size=(3,3),
#                  input_shape = (32,32,3),
#                  activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=64, 
#                  kernel_size=(3,3),
#                  activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=32, 
#                  kernel_size=(3,3),
#                  activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))

# model.add(Flatten())

# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=15,
#     restore_best_weights=True
# )

# start = time.time()

# hist = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])

# end = time.time()

# loss, acc = model.evaluate(x_test, y_test)

# print('loss:', loss)
# print('acc:', acc)


# print('time:', end- start)

# result = model.predict(x_test)

# y_pred = np.argmax(result, axis=1)
# y_test = np.argmax(y_test, axis=1)

# acc_score = accuracy_score(y_test,y_pred)

# print('accuracy score:', acc_score)
# # aaa = 7
# # print(y_train[aaa])

# # import matplotlib.pyplot as plt

# # plt.imshow(x_train[aaa])
# # plt.show()


from keras.datasets import cifar10
import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))

# 2. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# 3. Build model
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

# 4. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# 6. Train
start = time.time()
model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=100,
    callbacks=[es, lr],
    verbose=1
)
end = time.time()

# 7. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("✅ loss:", loss)
print("✅ acc:", acc)
print("⏱️ time:", end - start)

# 8. Predict
result = model.predict(x_test)
y_pred = np.argmax(result, axis=1)
y_true = np.argmax(y_test.values, axis=1)

acc_score = accuracy_score(y_true, y_pred)
print("✅ Final Accuracy Score:", acc_score)


# ✅ loss: 0.5149935483932495
# ✅ acc: 0.8289999961853027
# ⏱️ time: 2788.2778754234314