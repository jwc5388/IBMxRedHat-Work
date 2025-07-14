from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

# 1. Load data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target  # labels: 1~7

# 2. Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 3. One-hot encode y
y = pd.get_dummies(y).values  # shape: (581012, 7)

# 4. Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# 5. Reshape for Conv1D (N, 54 features → 54 timesteps, 1 feature)
x_train = x_train.reshape(-1, 54, 1)
x_test = x_test.reshape(-1, 54, 1)

# 6. Build Conv1D model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(54, 1)),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 classes
])

# 7. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 8. Callback
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
    verbose=1,
    callbacks=[es]
)
end = time.time()

# 10. Evaluate + Performance Metrics
loss, acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)
f1 = f1_score(y_true, y_pred, average='weighted')

# 11. Output
print(f"\n📊 Final Evaluation:")
print(f"✅ Loss (categorical_crossentropy): {loss:.4f}")
print(f"✅ Accuracy                      : {acc:.4f}")
print(f"✅ F1 Score (weighted)           : {f1:.4f}")
print(f"⏱️  Training Time                : {end - start:.2f} sec")

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred))


# 📊 Final Evaluation:
# ✅ Loss (categorical_crossentropy): 0.4938
# ✅ Accuracy                      : 0.7881
# ✅ F1 Score (weighted)           : 0.7751
# ⏱️  Training Time                : 330.96 sec

# 📄 Classification Report:
#               precision    recall  f1-score   support

#            0       0.83      0.73      0.78     42368
#            1       0.78      0.89      0.83     56661
#            2       0.67      0.89      0.76      7151
#            3       0.77      0.09      0.16       549
#            4       0.86      0.17      0.29      1899
#            5       0.63      0.13      0.21      3473
#            6       0.83      0.73      0.78      4102

#     accuracy                           0.79    116203
#    macro avg       0.77      0.52      0.54    116203
# weighted avg       0.79      0.79      0.78    116203