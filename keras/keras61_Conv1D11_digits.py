import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

# 1. Load data
digits = load_digits()
x = digits.data  # (1797, 64)
y = digits.target  # (1797,)

# 2. Normalize & reshape
x = x / 16.0  # pixel range is 0–16
x = x.reshape((-1, 64, 1))  # for Conv1D: (samples, timesteps, features)

# 3. One-hot encode labels
y = to_categorical(y, num_classes=10)  # (1797, 10)

# 4. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 5. Build Conv1D model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(64, 1)),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')  # 10 classes
])

model.summary()

# 6. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 7. Callback
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
    verbose=1,
    callbacks=[es]
)
end = time.time()

# 9. Evaluate + 성능지표
loss, acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"\n📊 Final Evaluation:")
print(f"✅ Loss      : {loss:.4f}")
print(f"✅ Accuracy  : {acc:.4f}")
print(f"✅ F1 Score  : {f1:.4f}")
print(f"⏱️  Time     : {end - start:.2f} sec")

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred))


#   Final Evaluation:
# ✅ Loss      : 0.0679
# ✅ Accuracy  : 0.9778
# ✅ F1 Score  : 0.9778
# ⏱️  Time     : 20.76 sec

# 📄 Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        33
#            1       0.96      0.93      0.95        28
#            2       1.00      1.00      1.00        33
#            3       1.00      0.97      0.99        34
#            4       1.00      1.00      1.00        46
#            5       0.96      0.98      0.97        47
#            6       0.97      0.97      0.97        35
#            7       1.00      0.97      0.99        34
#            8       0.94      0.97      0.95        30
#            9       0.95      0.97      0.96        40

#     accuracy                           0.98       360
#    macro avg       0.98      0.98      0.98       360
# weighted avg       0.98      0.98      0.98       360