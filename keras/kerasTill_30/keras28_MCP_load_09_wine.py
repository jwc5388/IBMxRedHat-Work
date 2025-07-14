from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential, load_model
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



path_mcp = 'Study25/_save/keras28_mcp/09_wine/'
model = load_model(path_mcp + 'keras28_wine_save.h5')

model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['acc'])


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


