import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# 1. Load dataset
dataset = load_wine()
x = dataset.data         # (178, 13)
y = dataset.target       # (178,), class: 0,1,2

print("✅ 클래스 분포:", np.unique(y, return_counts=True))  # sanity check

# 2. One-hot encode target
y = to_categorical(y, num_classes=3)  # (178, 3)

# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# 4. Normalize
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Reshape for LSTM
x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)

# 6. Build model
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(13, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 다중 클래스 → softmax
])
model.summary()

# 7. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 8. EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

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
loss, acc = model.evaluate(x_test, y_test, verbose=0)

print("\n📊 Evaluation Metrics:")
print(f"✅ Loss: {loss:.4f}")
print(f"✅ Accuracy: {acc:.4f}")
print(f"⏱️ 걸린시간: {end - start:.2f}초")
