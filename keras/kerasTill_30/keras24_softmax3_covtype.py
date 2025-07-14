from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time


datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts = True))

