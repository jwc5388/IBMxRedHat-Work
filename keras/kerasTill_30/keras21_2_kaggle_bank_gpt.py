import numpy as np
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

# 1. Load Data
path = 'Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col = 0)


# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# 5. Apply MinMaxScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=333, stratify=y
)

# 7. Handle class imbalance (important!)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 8. Build improved model
model = Sequential([
    Dense(128, input_dim=10, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 9. Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 10. Callbacks
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)

# 11. Train
hist = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es],
    class_weight=class_weights,
    verbose=1
)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) # 9x6 size 
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.plot(hist.history['accuracy'], c= 'green', label = 'accuracy')
plt.title("Kaggle Bank")
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.legend(loc='upper right') #우측 상단에 label 표시
plt.grid() #격자 표시

# 12. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Optional: classification report
y_pred = model.predict(x_test)
y_pred_binary = np.round(y_pred).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

# 13. Predict for submission
y_submit = model.predict(test_scaled)
# y_submit = np.round(y_submit).astype(int)

# 14. Save submission
submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_0527_improved.csv')
print("✅ Submission file saved: submission_0527_improved.csv")
