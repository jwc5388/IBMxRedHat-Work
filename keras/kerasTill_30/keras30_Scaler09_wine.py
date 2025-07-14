from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time

dataset = load_wine()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)
#(178,13), (178,)
print(np.unique(y, return_counts=True))
print(x)
print(y)


y = pd.get_dummies(y).values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y)

# scaler = MinMaxScaler()
# loss: 0.00014172220835462213
# acc: 1.0

# scaler = StandardScaler()
# loss: 0.029121505096554756
# acc: 0.9722222089767456

# scaler = RobustScaler()
# loss: 0.06810205429792404
# acc: 0.9722222089767456

scaler = MaxAbsScaler()
# loss: 0.02491731569170952
# acc: 1.0

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

hist = model.fit(x_train, y_train, epochs = 300, batch_size=8, validation_split = 0.2, callbacks=[es], verbose = 1)


loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('acc:', acc)



y_pred_probs = model.predict(x_test)

r2 = r2_score(y_test, y_pred_probs)
print('r2 score:', r2)

# Convert softmax outputs to predicted class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# # Calculate F1 Score
# f1 = f1_score(y_test, y_pred, average='macro')  # or 'weighted' if class imbalance
# print(f'✅ F1 Score (macro): {f1:.4f}')


