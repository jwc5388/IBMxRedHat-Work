import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 1. Load dataset
digits = load_digits()
x = digits.data            # (1797, 64)
y = digits.target          # (1797,)
print("✅ 데이터 shape:", x.shape, y.shape)

# 2. Normalize and reshape
x = x / 16.0               # 픽셀 범위가 0~16이므로 정규화
x = x.reshape((-1, 64, 1)) # LSTM 입력을 위해 (64, 1)로 reshape

# 3. One-hot encode labels
y = to_categorical(y, num_classes=10)  # (1797, 10)

# 4. Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 5. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(64, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10개의 숫자 클래스
])
model.summary()

# 6. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 7. EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 8. Train
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

# 9. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("\n📊 Evaluation Metrics:")
print(f"✅ Loss      : {loss:.4f}")
print(f"✅ Accuracy  : {acc:.4f}")
print(f"⏱️ 걸린시간   : {end - start:.2f}초")
