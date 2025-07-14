
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import AUC, Precision, Recall

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import f1_score

# === Set Random Seed for Reproducibility ===
np.random.seed(42)

# === Load Data ===
path = 'Study25/_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# === Scale Data ===
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

# === Stratified Train/Validation Split ===
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=33)
for train_idx, val_idx in sss.split(x, y):
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# === Apply SMOTE only to training set ===
x_train, y_train = SMOTE(random_state=333).fit_resample(x_train, y_train)

# === Build the Model ===
model = Sequential([
    Dense(128, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

path_mcp = 'Study25/_save/keras28_mcp/06_dacon_cancer/'
model.save(path_mcp + 'keras28_dacon_cancer_save.h5')


# === Compile Model ===
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
)


# === Callbacks ===
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

import datetime 
date = datetime.datetime.now()
print(date)  
print(type(date))  
date = date.strftime("%m%d_%H%M")
print(date) 
print(type(date)) 

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_mcp, 'k28_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)

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
y_submit = model.predict(test_csv).ravel()
submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
submission_csv.to_csv(path + 'submission_top.csv')
print("✅ Submission saved!")
