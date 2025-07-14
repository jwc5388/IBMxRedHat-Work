
# === Import Libraries ===
import pandas as pd
import numpy as np
import os
from datetime import datetime
# from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import AUC, Precision, Recall

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

# === Set Random Seed for Reproducibility ===
np.random.seed(42)

# === Set Time-based Save Path ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_path = f'/workspace/TensorJae/Study25/_data/dacon/cancer/{timestamp}'
os.makedirs(base_path, exist_ok=True)

# === Load Data ===
data_path = '/workspace/TensorJae/Study25/_data/dacon/cancer/'
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# === Scale Data ===
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# === Feature Selection with XGBoost ===
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_scaled, y)
importances = xgb.feature_importances_
threshold = np.percentile(importances, 25)  # remove bottom 25% features
selected_indices = np.where(importances > threshold)[0]
x_selected = x_scaled[:, selected_indices]
test_selected = test_scaled[:, selected_indices]

# === Stratified Train/Validation Split ===
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(x_selected, y):
    x_train, x_val = x_selected[train_idx], x_selected[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# === Apply SMOTE only to training set ===
# x_train, y_train = SMOTE(random_state=333).fit_resample(x_train, y_train)

print(x.shape, y.shape)

# === Reshape for LSTM ===
x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_val = x_val.reshape(-1, x_val.shape[1], 1)
test_selected = test_selected.reshape(-1, test_selected.shape[1], 1)


# === Build the Model ===
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.2, input_shape=(x_train.shape[1], 1)),
    LSTM(64, return_sequences=False, dropout=0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# === Compile Model ===
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
)

# === Callbacks ===
model_path = os.path.join(base_path, f'model_mcp_{timestamp}.h5')
mcp = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# === Train Model ===
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=32,
    callbacks=[es, lr, mcp],
    verbose=1
)

# === Evaluate Model ===
loss, accuracy, auc, precision, recall = model.evaluate(x_val, y_val, verbose=0)
y_val_pred = model.predict(x_val).ravel()

# === Find Best Threshold for F1 Score ===
thresholds = np.arange(0.3, 0.7, 0.01)
f1_scores = [f1_score(y_val, (y_val_pred > t).astype(int)) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]



# === Print Metrics ===
print(f'✅ loss: {loss:.4f}')
print(f'✅ acc : {accuracy:.4f}')
print(f'✅ AUC : {auc:.4f}')
print(f'✅ Precision: {precision:.4f}')
print(f'✅ Recall   : {recall:.4f}')
print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# === Predict on Final Test Set and Save Submission ===
y_submit = model.predict(test_selected).ravel()
submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
submission_path = os.path.join(base_path, f'submission_{timestamp}.csv')
submission_csv.to_csv(submission_path)
print(f"✅ Submission saved to: {submission_path}")
print(f"✅ Best model saved to: {model_path}")
