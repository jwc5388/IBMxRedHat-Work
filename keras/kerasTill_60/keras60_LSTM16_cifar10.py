import numpy as np
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 다중 GPU 설정
strategy = tf.distribute.MirroredStrategy()
print("✅ Num of GPUs:", strategy.num_replicas_in_sync)

# 1. Load data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("🧹 Data loaded:", x_train.shape, y_train.shape)

# 2. Preprocessing
x_train = x_train.reshape(-1, 32 * 32 * 3) / 255.0
x_test = x_test.reshape(-1, 32 * 32 * 3) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape for LSTM input
x_train = x_train.reshape(-1, 3072, 1)
x_test = x_test.reshape(-1, 3072, 1)

# 수동 validation 분할
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# 3. Build model inside strategy scope
with strategy.scope():
    model = Sequential([
        LSTM(64, return_sequences=True, dropout=0.2, input_shape=(3072, 1)),
        LSTM(64, return_sequences=False, dropout=0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

model.summary()

# 4. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 5. Train
start = time.time()
model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[es],
    verbose=1
)
end = time.time()

# 6. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('✅ loss:', loss)
print('✅ accuracy:', acc)
print('⏱️ time:', round(end - start, 2), 'seconds')

# 7. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print("✅ Accuracy Score (sklearn):", acc_score)
