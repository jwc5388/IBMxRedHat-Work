from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time

dataset = load_wine()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)
#(178,13), (178,)
print(np.unique(y, return_counts=True))
print(x)
print(y)

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

y = pd.get_dummies(y).values


x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=42, stratify=y)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))


model = Sequential()
model.add(Dense(32, input_dim = 13, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation = 'softmax'))


path_mcp = 'Study25/_save/keras28_mcp/09_wine/'
model.save(path_mcp + 'keras28_wine_save.h5')

model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

es = EarlyStopping(
    monitor= 'val_loss',
    mode='min',
    patience= 30,
    restore_best_weights=True
    
)

import datetime 
date = datetime.datetime.now()
print(date)  #2025-06-02 13:00:52.507403
print(type(date))   #<class 'datetime.datetime'>
#시간을 string으로 만들어라
date = date.strftime("%m%d_%H%M")
print(date) #0602_1305
print(type(date))   # <class 'str'>

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_mcp, 'k28_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath
)


hist = model.fit(x_train, y_train, epochs = 300, batch_size=8, validation_split = 0.2, callbacks=[es, mcp], verbose = 1)


loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('acc:', acc)

# y_pred_probs = model.predict(x_val)
# Predict class probabilities on test set

y_pred_probs = model.predict(x_test)

# Convert softmax outputs to predicted class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# # Calculate F1 Score
# f1 = f1_score(y_test, y_pred, average='macro')  # or 'weighted' if class imbalance
# print(f'✅ F1 Score (macro): {f1:.4f}')


