from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np
import time

# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. Preprocessing
# Reshape: (32, 32, 3) → (32, 96) by flattening each row (32 rows × 3 channels = 96 features)
x_train = x_train.reshape(-1, 32, 96).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32, 96).astype('float32') / 255.0

# One-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 3. Build Conv1D Model
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
    Dense(10, activation='softmax')
])

# 4. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# 6. Train
start = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=64,
          validation_split=0.2,
          callbacks=[es, lr],
          verbose=1)
end = time.time()

# 7. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("\n📊 Final Evaluation Metrics:")
print(f"✅ Loss       : {loss:.4f}")
print(f"✅ Accuracy   : {acc:.4f}")
print(f"⏱️  Time       : {end - start:.2f}초")

# 8. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test_labels, y_pred)
print(f"✅ Sklearn Accuracy Score : {acc_score:.4f}")


# 📊 Final Evaluation Metrics:
# ✅ Loss       : 1.1023
# ✅ Accuracy   : 0.6216
# ⏱️  Time       : 239.05초
# 313/313 [==============================] - 2s 5ms/step
# ✅ Sklearn Accuracy Score : 0.6216