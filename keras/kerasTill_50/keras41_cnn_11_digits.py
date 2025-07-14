import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time

# 1. Load digits dataset
digits = load_digits()
x = digits.data  # shape: (1797, 64)
y = digits.target  # shape: (1797,)

# print(y.shape)

# aaa = 7
# print(y[aaa])

# import matplotlib.pyplot as plt
# plt.imshow(x[aaa].reshape(8,8), 'twilight_shifted')
# plt.show()

# exit()

# x = digits.data
# aaa = x[0].reshape(8,8)
# print(aaa)

# print(y[0])

# # exit()
# print()

# 2. Normalize & reshape
x = x / 16.0  # since pixel range is 0–16
x = x.reshape((-1, 8, 8, 1))  # add channel dim for Conv2D

# 3. One-hot encode labels
y = to_categorical(y, num_classes=10)  # shape becomes (1797, 10)

# 4. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 5. Build Model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(8, 8, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D())
model.add(Dropout(0.3))

# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))  # ✅ classification output

model.summary()

# 6. Compile model
model.compile(
    loss='categorical_crossentropy',  # ✅ correct loss for one-hot classification
    optimizer='adam',
    metrics=['accuracy']
)

# 7. EarlyStopping
es = EarlyStopping(
    monitor='val_loss',       # 기준: 검증 손실
    patience=10,              # 10 epoch 개선 없으면 멈춤
    mode='min',               # 손실이므로 'min'
    verbose=1,
    restore_best_weights=True
)

# 8. Train
start = time.time()

hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]  # 👈 여기에 추가!
)

end = time.time()

# 9. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', acc)
print('걸린시간:', end - start)
