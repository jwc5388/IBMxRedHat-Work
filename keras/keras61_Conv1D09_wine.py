import numpy as np
import pandas as pd
import time
import tensorflow as tf

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 1. Load and preprocess data
dataset = load_wine()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2

print(np.unique(y, return_counts=True))  # class 분포 확인

# 2. One-hot encode the labels
y = to_categorical(y, num_classes=3)

# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# 4. Normalize
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. Reshape for Conv1D
x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)

# 6. Build Conv1D model
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', padding='same', input_shape=(13, 1)),
    BatchNormalization(),
    Conv1D(128, kernel_size=2, activation='relu'),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

# 7. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 8. EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

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
y_pred_proba = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# 11. Calculate metrics
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# 12. Output
print(f"\n📊 Final Evaluation:")
print(f"✅ Loss (categorical_crossentropy): {loss:.4f}")
print(f"✅ Accuracy                      : {acc:.4f}")
print(f"✅ F1 Score (weighted)          : {f1:.4f}")
print(f"⏱️  Training Time                : {end - start:.2f} sec")

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred_classes))


# 📊 Final Evaluation:
# ✅ Loss (categorical_crossentropy): 0.0024
# ✅ Accuracy                      : 1.0000
# ✅ F1 Score (weighted)          : 1.0000
# ⏱️  Training Time                : 24.43 sec

# 📄 Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        12
#            1       1.00      1.00      1.00        14
#            2       1.00      1.00      1.00        10

#     accuracy                           1.00        36
#    macro avg       1.00      1.00      1.00        36
# weighted avg       1.00      1.00      1.00        36