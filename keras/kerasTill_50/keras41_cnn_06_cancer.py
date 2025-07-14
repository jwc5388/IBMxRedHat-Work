
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
import time

# === Set Random Seed for Reproducibility ===
np.random.seed(42)

# === Load Data ===
path = './Study25/_data/dacon/cancer/'
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

# === Compile Model ===
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
)


start = time.time()
##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,
          
          )
end = time.time()

import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)


if gpus:
    print('GPU exists!')
else:
    print('GPU doesnt exist')


print("걸린시간: ", end - start )
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', acc)
# cpu =  876.7733821868896

# gpu = 