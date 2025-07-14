# # === Import Libraries ===
# import pandas as pd
# import numpy as np

# # Keras: deep learning components
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.regularizers import l2
# from keras.metrics import AUC, Precision, Recall

# # Scikit-learn: preprocessing and utilities
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import f1_score
# from imblearn.over_sampling import SMOTE

# # === Load Data ===
# path = 'Study25/_data/dacon/cancer/'

# train_csv = pd.read_csv(path + 'train.csv', index_col=0)           # Training data
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)             # Test data (no labels)
# submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)  # Submission format

# # === Separate Features and Target Label ===
# x = train_csv.drop(['Cancer'], axis=1)  # Features (inputs)
# y = train_csv['Cancer']                 # Label (target: 0 or 1)

# # === One-Hot Encode Categorical Columns ===
# categorical_cols = x.select_dtypes(include='object').columns

# # One-hot encode using pandas get_dummies
# x = pd.get_dummies(x, columns=categorical_cols)
# test_csv = pd.get_dummies(test_csv, columns=categorical_cols)

# # Align columns to ensure same structure
# x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # === Scale Data to 0-1 range for better training ===
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# test_csv = scaler.transform(test_csv)

# # === Stratified Train/Validation Split to keep label balance ===
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=191)
# for train_idx, val_idx in sss.split(x, y):
#     x_train, x_val = x[train_idx], x[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # === Calculate Class Weights to handle imbalance (if class 1 is rare) ===
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
# class_weights_dict = dict(enumerate(class_weights))

# # === Build the Neural Network Model ===
# model = Sequential()

# # Input Layer + Regularization
# model.add(Dense(128, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Hidden Layer 1
# model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Hidden Layer 2
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Output Layer for Binary Classification
# model.add(Dense(1, activation='sigmoid'))  # sigmoid outputs probability between 0 and 1

# #==============above is original


# # === Compile the Model ===
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
# )

# # === Define Callbacks ===
# es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# # === Train the Model ===
# model.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs=1000,
#     batch_size=32,
#     callbacks=[es, lr, mc],
#     verbose=1,
# )

# path_save = 'Study25/_save/keras26/'
# model.save(path + 'dacon_cancer_save.h5')

# # === Evaluate on Validation Data ===
# loss, accuracy, auc = model.evaluate(x_val, y_val, verbose=0)
# y_val_pred = model.predict(x_val).ravel()

# # === Find Best Threshold for Highest F1 Score ===
# best_threshold, best_f1 = 0.5, 0
# for threshold in np.arange(0.3, 0.7, 0.01):
#     preds = (y_val_pred > threshold).astype(int)
#     f1 = f1_score(y_val, preds)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

# # === Print Metrics ===
# print(f'✅ loss: {loss:.4f}')
# print(f'✅ acc : {accuracy:.4f}')
# print(f'✅ AUC : {auc:.4f}')
# print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# # === Predict on Final Test Set ===
# y_submit = model.predict(test_csv).ravel()
# submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
# submission_csv.to_csv(path + 'submission_best1.csv')
# print("✅ Submission saved!")

# ✅ loss: 0.5638
# ✅ acc : 0.8795
# ✅ AUC : 0.6911
# ✅ Best F1: 0.4686 at threshold 0.61


#-========above original 




# print(train_csv.info())
# print(train_csv.shape)
# print(train_csv)        # [87159 rows x 15 columns]

# print(test_csv.info())
# # print(test_csv.shape)
# # print(test_csv)     #[46204 rows x 14 columns]

# print(train_csv.columns)
# #Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#     #    'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#     #    'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#     #    'Cancer'],
#     #   dtype='object')
    
# print(test_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'],
#       dtype='object')

# === Import Libraries ===
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import f1_score
# from imblearn.over_sampling import SMOTE
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.regularizers import l2


# # === Load Data ===
# path = 'Study25/_data/dacon/cancer/'

# train_csv = pd.read_csv(path + 'train.csv', index_col=0)           # Training data
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)             # Test data (no labels)
# submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)  # Submission format

# # === Separate Features and Target Label ===
# x = train_csv.drop(['Cancer'], axis=1)  # Features (inputs)
# y = train_csv['Cancer']                 # Label (target: 0 or 1)

# # === One-Hot Encode Categorical Columns ===
# categorical_cols = x.select_dtypes(include='object').columns

# # One-hot encode using pandas get_dummies
# x = pd.get_dummies(x, columns=categorical_cols)
# test_csv = pd.get_dummies(test_csv, columns=categorical_cols)

# # Align columns to ensure same structure
# x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# # === Scale Data to 0-1 range for better training ===
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# test_csv = scaler.transform(test_csv)

# # === SMOTE for balancing classes ===
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# x_resampled, y_resampled = smote.fit_resample(x, y)

# # === Stratified Train/Validation Split to keep label balance ===
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_idx, val_idx in sss.split(x_resampled, y_resampled):
#     x_train, x_val = x_resampled[train_idx], x_resampled[val_idx]
#     y_train, y_val = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]

# # === Build the Neural Network Model ===
# model = Sequential()

# # Input Layer + Regularization
# # Input Layer + Regularization
# model.add(Dense(128, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# # Hidden Layer 1
# model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# # Hidden Layer 2
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# # Output Layer for Binary Classification
# model.add(Dense(1, activation='sigmoid'))  # sigmoid outputs probability between 0 and 1


# # === Compile the Model ===
# # optimizer = Adam(learning_rate=0.0001)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # === Define Callbacks ===
# es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# # === Train the Model ===
# hist = model.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     epochs=1000,
#     batch_size=64,
#     callbacks=[es, lr, mc],
#     verbose=1
# )

# path_save = 'Study25/_save/keras26/'
# model.save(path + 'dacon_cancer_save.h5')

# # === Evaluate the Model ===
# loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
# y_val_pred = model.predict(x_val).ravel()

# # === Find Best Threshold for Highest F1 Score ===
# best_threshold, best_f1 = 0.5, 0
# for threshold in np.arange(0.3, 0.7, 0.01):
#     preds = (y_val_pred > threshold).astype(int)
#     f1 = f1_score(y_val, preds)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

# # === Print Metrics ===
# print(f'✅ loss: {loss:.4f}')
# print(f'✅ acc : {accuracy:.4f}')
# print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# # === Predict on Final Test Set ===
# y_submit = model.predict(test_csv).ravel()
# submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
# submission_csv.to_csv(path + 'submission_best.csv')
# print("✅ Submission saved!")


# Dropout: 0.2–0.5
# Hidden layers: 2–5
# Neurons: 64–512
# Optimizers: Adam, RMSprop, AdamW
# Batch size: 16–128
# Learning rate: 1e-4 to 1e-2

# === Import Libraries ===
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.metrics import AUC, Precision, Recall

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

# === Set Random Seed for Reproducibility ===
np.random.seed(42)

# === Load Data ===

path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
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
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

# === Compile Model ===
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
)

model.save(path + 'dacon_cancer.h5')

# === Callbacks ===
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# === Train Model ===
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=32,
    callbacks=[es, lr],
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
submission_csv.to_csv(path + 'submission_top1.csv')
print("✅ Submission saved!")
