from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


"""import ssl
ssl.create_default_https_context = ssl._create_unverified_context
"""


datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts = True))
print(pd.value_counts(y))


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


y = pd.get_dummies(y).values

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size= 0.8, random_state=42, stratify= y)

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))



path_mcp = 'Study25/_save/keras28_mcp/10_covtype/'
model = load_model(path_mcp + 'keras28_covtype_save.h5')


model.compile(
    loss = 'categorical_crossentropy',
    optimizer= 'adam',
    metrics = ['acc']
)

loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)



