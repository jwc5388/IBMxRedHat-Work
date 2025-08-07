
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam


import time


# 1. Load Data
path = 'Study25/_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

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
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 6. Train-test split (after scaling)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=33
)

# 7. Build model
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 8. Compile and train
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])


es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 20,
                   verbose = 1,
                   restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor = 'val_loss', mode = 'auto', patience = 10,
                        verbose = 1,
                        factor = 0.9)

# 0.1 / 0.05/ 0.025/ 0.0125/ 0.00625 ##factor 0.5

start = time.time()
hist = model.fit(x_train, y_train, epochs = 10000, batch_size =32, verbose = 1, validation_split=0.1,
                 callbacks = [es, rlr])




loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)

acc = accuracy_score(y_test, result)
print("loss:", loss)
print('acc result:', acc)

