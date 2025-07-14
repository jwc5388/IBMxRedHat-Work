from keras.datasets import cifar100
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print("📦 Raw x_train:", x_train.shape)  # (50000, 32, 32, 3)
print("📦 Raw x_test :", x_test.shape)   # (10000, 32, 32, 3)

# 2. Reshape for Conv1D: (32, 32, 3) → (32, 96)
x_train = x_train.reshape(-1, 32, 96).astype('float32') / 255.
x_test = x_test.reshape(-1, 32, 96).astype('float32') / 255.
print("🔁 Reshaped x_train:", x_train.shape)  # (50000, 32, 96)
print("🔁 Reshaped x_test :", x_test.shape)   # (10000, 32, 96)

# 3. One-hot encode labels
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
print("🏷️ One-hot y_train:", y_train.shape)  # (50000, 100)
print("🏷️ One-hot y_test :", y_test.shape)   # (10000, 100)


# 📦 Raw x_train: (50000, 32, 32, 3)
# 📦 Raw x_test : (10000, 32, 32, 3)
# 🔁 Reshaped x_train: (50000, 32, 96)
# 🔁 Reshaped x_test : (10000, 32, 96)
# 🏷️ One-hot y_train: (50000, 100)
# 🏷️ One-hot y_test : (10000, 100)

# 4. Build Conv1D Model
model = Sequential([
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=(32, 96)),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(100, activation='softmax')
])

# 5. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# 7. Train
start = time.time()
history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[es, lr],
                    verbose=1)
end = time.time()

# 8. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("\n📊 Final Evaluation:")
print(f"✅ Loss     : {loss:.4f}")
print(f"✅ Accuracy : {acc:.4f}")
print(f"⏱️  Time     : {end - start:.2f}초")

# 9. Predict & Accuracy
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_true, y_pred)
print(f"✅ Sklearn Accuracy Score: {acc_score:.4f}")


# 📊 Final Evaluation:
# ✅ Loss     : 4.2464
# ✅ Accuracy : 0.0612
# ⏱️  Time     : 12.48초