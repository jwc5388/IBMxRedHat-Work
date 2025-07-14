import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

# 1. Load Data
path = '/workspace/TensorJae/Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Label Encoding
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unnecessary columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Split into x and y
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# 5. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=33
)

# 6. Scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 7. Reshape for LSTM
x_train = x_train.reshape(-1, 10, 1)  # (N, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

# 8. Build LSTM Model for binary classification
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(10, 1)),
    LSTM(64, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # ✅ binary classification
])
model.summary()

# 9. Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 10. EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# 11. Train
start = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 12. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

# 13. Metrics
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# 14. Print
print("\n📊 Evaluation Metrics:")
print(f"✅ Loss (Binary Crossentropy): {loss:.4f}")
print(f"✅ Accuracy                  : {acc:.4f}")
print(f"✅ F1 Score                  : {f1:.4f}")
print(f"✅ ROC AUC                   : {roc_auc:.4f}")
print(f"⏱️ Training Time            : {end - start:.2f} sec")

print("\n📌 Confusion Matrix:")
print(cm)
