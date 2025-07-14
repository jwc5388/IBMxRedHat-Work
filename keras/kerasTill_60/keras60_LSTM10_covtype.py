import numpy as np
import pandas as pd
import time

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# 1. Load data
datasets = fetch_covtype()
x = datasets.data                  # (581012, 54)
y = datasets.target                # (581012,)
print("✅ Original shape:", x.shape, y.shape)
print("✅ 클래스 분포:", np.unique(y, return_counts=True))  # 클래스 1~7

# 2. One-hot encode labels (1~7 → 0~6)
y = pd.get_dummies(y).values       # (581012, 7)

# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# 4. Normalize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Reshape for LSTM
x_train = x_train.reshape(-1, 54, 1)  # (N, 54, 1)
x_test = x_test.reshape(-1, 54, 1)

# 6. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(54, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')  # 7개의 클래스
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
    mode='min',
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
print(f"✅ Loss      : {loss:.4f}")
print(f"✅ Accuracy  : {acc:.4f}")
print(f"⏱️ 걸린시간   : {end - start:.2f}초")
