from keras.datasets import cifar100
import numpy as np
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

# 2. Reshape & Normalize
x_train = x_train.reshape(-1, 32*32*3) / 255.0
x_test = x_test.reshape(-1, 32*32*3) / 255.0

# 3. One-hot encode labels
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)
print(y_train.shape, y_test.shape)  # (50000, 100), (10000, 100)

# 4. Reshape for LSTM input
x_train = x_train.reshape(-1, 3072, 1)
x_test = x_test.reshape(-1, 3072, 1)

# 5. Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(3072, 1)),
    LSTM(64, return_sequences=False, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(100, activation='softmax')  # 100 classes
])

model.summary()

# 6. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']  # 'accuracy' 명시 권장
)

# 7. Callback
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 8. Train
start = time.time()
model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
    callbacks=[es]
)
end = time.time()

# 9. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('✅ loss:', loss)
print('✅ accuracy:', acc)
print('⏱️ 걸린시간:', end - start)

# 10. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print("✅ Final Accuracy Score (sklearn):", acc_score)
