from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# 1. Load data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target  # labels: 1 ~ 7

print(x.shape, y.shape)  # (581012, 54), (581012,)
print(np.unique(y, return_counts=True))

# 2. Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 3. One-hot encode y (shift labels to 0~6)
y = pd.get_dummies(y).values  # shape: (581012, 7)

# 4. Split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Reshape for CNN
x_train = x_train.reshape(-1, 9, 6, 1).astype('float32')  # 9x6 = 54
x_test = x_test.reshape(-1, 9, 6, 1).astype('float32')

# ✅ y는 그대로 두기 (one-hot)
# y_train = y_train.reshape(-1, 1) ❌ 삭제
# y_test = y_test.reshape(-1, 1) ❌ 삭제

# 6. Build model
model = Sequential()
model.add(Input(shape=(9, 6, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))  # ✅ 7 classes

model.summary()

# 7. Compile
model.compile(
    loss='categorical_crossentropy',  # ✅ 다중분류 손실 함수
    optimizer='adam',
    metrics=['accuracy']              # ✅ 정확도
)

# 8. Callback
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

# 9. Train
start = time.time()
model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]
)
end = time.time()

# 10. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('✅ loss:', loss)
print('✅ accuracy:', acc)
print('✅ 걸린시간:', end - start)
