import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import AUC, Precision, Recall

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import os
import datetime

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
scaler = MinMaxScaler()
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
    Dropout(0.2),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

# === Define Save Path and Save Initial Model ===
path_mcp = 'Study25/_save/keras28_mcp/06_dacon_cancer/'
if not os.path.exists(path_mcp):
    os.makedirs(path_mcp)
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

# Create timestamp for file naming
date = datetime.datetime.now().strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_mcp, 'k28_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto', # Keras detects min/max automatically for val_loss
    save_best_only = True,
    filepath = filepath
)

# === Train Model ===
print("--- Starting Model Training ---")
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=32,
    callbacks=[es, lr, mcp],
    verbose=1
)
print("--- Model Training Finished ---")

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
print("\n--- Evaluation Metrics ---")
print(f'✅ loss: {loss:.4f}')
print(f'✅ acc : {accuracy:.4f}')
print(f'✅ AUC : {auc:.4f}')
print(f'✅ Precision: {precision:.4f}')
print(f'✅ Recall   : {recall:.4f}')
print(f'✅ Best F1 on Validation Set: {best_f1:.4f} at threshold {best_threshold:.2f}')
print("------------------------\n")

# === Predict on Final Test Set and Save Submission ===
print("--- Generating Predictions for Submission ---")
y_submit = model.predict(test_csv).ravel()

# Apply the best threshold found on the validation set
submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)

# Save the submission file with a timestamp
submission_filename = f'submission_{date}.csv'
submission_filepath = os.path.join(path, submission_filename)
submission_csv.to_csv(submission_filepath)

print(f"✅ Submission file saved successfully as '{submission_filename}' in the path: {path}")

# # === Predict on Final Test Set and Save Submission ===
# y_submit = model.predict(test_csv).ravel()
# submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
# submission_csv.to_csv(path + 'submission_top.csv')
# print("✅ Submission saved!")
# # === Import Libraries ===
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import f1_score
# from imblearn.over_sampling import SMOTE

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.regularizers import l2
# from keras.metrics import AUC, Precision, Recall
# from keras.optimizers import Adam

# import os
# import datetime

# # === Configuration ===
# class CFG:
#     RANDOM_STATE = 42
#     N_SPLITS = 2  # Number of folds for cross-validation
#     EPOCHS = 500   # Max epochs (EarlyStopping will stop it sooner)
#     BATCH_SIZE = 32
#     PATIENCE_ES = 25  # Patience for EarlyStopping
#     PATIENCE_LR = 10  # Patience for ReduceLROnPlateau
#     LEARNING_RATE = 0.0005 # Optimized learning rate

# # === Set Random Seed for Reproducibility ===
# np.random.seed(CFG.RANDOM_STATE)
# tf.random.set_seed(CFG.RANDOM_STATE)

# # === Path and Directory Setup ===
# DATA_PATH = 'Study25/_data/dacon/cancer/'
# # This is the path for all your saved models, as in your example
# SAVE_PATH_MCP = 'Study25/_save/keras28_mcp/06_dacon_cancer/'
# if not os.path.exists(SAVE_PATH_MCP):
#     os.makedirs(SAVE_PATH_MCP)

# # === Load Data ===
# train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
# test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(DATA_PATH + 'sample_submission.csv', index_col=0)

# # === Feature Engineering (Optional but Recommended) ===
# def create_features(df):
#     epsilon = 1e-6
#     df['T4_T3_ratio'] = df['T4_Result'] / (df['T3_Result'] + epsilon)
#     df['TSH_T4_ratio'] = df['TSH_Result'] / (df['T4_Result'] + epsilon)
#     risk_columns = ['Family_Background', 'Radiation_History', 'Smoke', 'Weight_Risk', 'Diabetes']
#     df['Combined_Risk'] = df[risk_columns].sum(axis=1)
#     return df

# print("✅ Applying feature engineering...")
# train_csv = create_features(train_csv)
# test_csv = create_features(test_csv)


# # === Separate Features and Target Label ===
# x = train_csv.drop(['Cancer'], axis=1)
# y = train_csv['Cancer']

# # === Preprocessing (One-Hot Encode and Align) ===
# print("✅ Preprocessing data...")
# categorical_cols = x.select_dtypes(include='object').columns
# x = pd.get_dummies(x, columns=categorical_cols)
# test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
# x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)


# # === Generate Timestamp for this Training Run ===
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# print(f"✅ Starting training run with timestamp: {date}")


# # === Cross-Validation and Model Training ===
# skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)

# oof_predictions = np.zeros(x.shape[0])
# test_predictions = []

# for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
#     print(f"\n===== FOLD {fold+1} / {CFG.N_SPLITS} =====")
    
#     # --- Data for this fold ---
#     x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # --- Scaling (Fit on train, transform train/val) ---
#     scaler = MinMaxScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_val = scaler.transform(x_val)
    
#     # --- SMOTE (Apply only to training data of the fold) ---
#     smote = SMOTE(random_state=CFG.RANDOM_STATE)
#     x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    
#     # --- Build the Model ---
#     model = Sequential([
#         Dense(64, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
    
#     # --- Save the UNTRAINED model structure (as requested) ---
#     initial_save_path = os.path.join(SAVE_PATH_MCP, f'k28_{date}_fold{fold+1}_initial_model.h5')
#     model.save(initial_save_path)
#     print(f"Untrained model for fold {fold+1} saved to: {initial_save_path}")
    
#     # --- Compile Model ---
#     optimizer = Adam(learning_rate=CFG.LEARNING_RATE)
#     model.compile(
#         loss='binary_crossentropy', 
#         optimizer=optimizer,
#         metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
#     )
    
#     # --- Callbacks with your custom naming convention ---
#     es = EarlyStopping(
#         monitor='val_loss', patience=CFG.PATIENCE_ES, mode='min', restore_best_weights=True
#     )
#     lr = ReduceLROnPlateau(
#         monitor='val_loss', factor=0.5, patience=CFG.PATIENCE_LR, mode='min', verbose=1
#     )
    
#     # Dynamically create the filename for ModelCheckpoint for this fold
#     filename = f'fold{fold+1:02d}_' + '{epoch:04d}-{val_loss:.4f}.h5'
#     filepath = os.path.join(SAVE_PATH_MCP, f'k28_{date}_{filename}')
    
#     mcp = ModelCheckpoint(
#         monitor='val_loss',
#         mode='min',
#         save_best_only=True,
#         filepath=filepath,
#         verbose=0
#     )
    
#     # --- Train Model ---
#     model.fit(
#         x_train_resampled, y_train_resampled,
#         validation_data=(x_val, y_val),
#         epochs=CFG.EPOCHS,
#         batch_size=CFG.BATCH_SIZE,
#         callbacks=[es, lr, mcp],
#         verbose=1
#     )
    
#     # --- Evaluate and Predict ---
#     # Because restore_best_weights=True, the 'model' object already has the best weights.
#     # We predict directly from it. The saved file from mcp is our backup.
#     val_preds_proba = model.predict(x_val).ravel()
#     oof_predictions[val_idx] = val_preds_proba
    
#     # Predict on test set
#     test_scaled = scaler.transform(test_csv)
#     test_fold_preds = model.predict(test_scaled).ravel()
#     test_predictions.append(test_fold_preds)

# # === Post-Training Analysis ===

# # --- Find Best Threshold using OOF Predictions ---
# print("\n===== Finding Best Threshold from OOF Predictions =====")
# thresholds = np.arange(0.1, 0.9, 0.01)
# f1_scores = [f1_score(y, (oof_predictions > t).astype(int)) for t in thresholds]
# best_idx = np.argmax(f1_scores)
# best_threshold = thresholds[best_idx]
# best_f1 = f1_scores[best_idx]

# print(f"✅ Overall OOF F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

# # === Create Submission File ===
# print("\n===== Creating Submission File =====")

# # Average predictions from all folds for a more robust result
# final_test_predictions = np.mean(test_predictions, axis=0)

# # Apply the best threshold found from OOF predictions
# submission_csv['Cancer'] = (final_test_predictions > best_threshold).astype(int)
# final_submission_path = os.path.join(DATA_PATH, f'submission_tf_cv_{date}.csv')
# submission_csv.to_csv(final_submission_path)

# print(f"✅ Submission saved to '{final_submission_path}'")
# print(f"Distribution of predictions: \n{submission_csv['Cancer'].value_counts(normalize=True)}")