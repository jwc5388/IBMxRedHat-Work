import numpy as np
import pandas as pd
import time
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# 1. Load Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2. Preprocessing (28, 28) → (28, 28)
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# One-hot encoding
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# 3. Build Model (Conv1D)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28)),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')  # 10 classes
])

# 4. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 5. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.9, verbose=1)

# 6. Train
start = time.time()
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es, lr],
    verbose=1
)
end = time.time()

# 7. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n📊 Final Evaluation Metrics:")
print(f"✅ Loss         : {loss:.4f}")
print(f"✅ Accuracy     : {acc:.4f}")
print(f"⏱️  Time         : {end - start:.2f}초")

# 8. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_labels, y_pred)
print(f"✅ Sklearn Accuracy Score : {acc_score:.4f}")

# 📊 Final Evaluation Metrics:
# ✅ Loss         : 0.3982
# ✅ Accuracy     : 0.8579
# ⏱️  Time         : 21.34초
