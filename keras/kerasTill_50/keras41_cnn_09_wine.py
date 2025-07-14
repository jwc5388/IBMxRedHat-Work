from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import time

# 1. Load and preprocess data
dataset = load_wine()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2

print(np.unique(y, return_counts=True))  # 3 classes

# 2. One-hot encode the labels
y = to_categorical(y, num_classes=3)

# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# 4. Normalize
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Reshape for Conv2D input
x_train = x_train.reshape(-1, 13, 1, 1).astype('float32')
x_test = x_test.reshape(-1, 13, 1, 1).astype('float32')

# 6. Build model
model = Sequential()
model.add(Input(shape=(13, 1, 1)))
model.add(Conv2D(64, (3, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))  # 3 classes → softmax

# 7. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 8. EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 9. Train
start = time.time()
model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 10. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('✅ loss:', loss)
print('✅ accuracy:', acc)
print('✅ 걸린시간:', end - start)


# ✅ loss: 0.17220523953437805
# ✅ accuracy: 0.9444444179534912
# ✅ 걸린시간: 10.619065999984741