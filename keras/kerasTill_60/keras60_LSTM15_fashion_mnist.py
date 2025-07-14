import numpy as np
import pandas as pd
import time
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1. Load Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28)
print(x_test.shape, y_test.shape)    # (10000, 28, 28)

# 2. Reshape & Normalize
x_train = x_train.reshape(-1, 784, 1) / 255.0
x_test = x_test.reshape(-1, 784, 1) / 255.0

# 3. One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 4. Build Model
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(784,1)),
    LSTM(64, return_sequences=False, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10-class classification
])
model.summary()

# 5. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# 7. Train
start = time.time()
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[es, lr],
    verbose=1
)
end = time.time()

# 8. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('✅ loss:', loss)
print('✅ accuracy:', acc)
print('⏱️ time:', end - start)

# 9. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print("✅ Accuracy Score (sklearn):", accuracy_score(y_test_labels, y_pred))
