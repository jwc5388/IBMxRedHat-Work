import numpy as np
import os
import time
import datetime
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import Model
from keras.layers import Model
from keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping, ModelCheckpoint, RefuceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

if os.path.exists('/workspace/TensorJae/Study25/'):
    BASE_PATH = '/workspace/TensorJae/Study25/'

else: BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHar/Study25/')

basepath = os.path.join(BASE_PATH)
path = basepath + '_data/diabetes/'

date = datetime.datetime.now().strftime("%m%d_%H%M")

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8)

def build_model(drop=0.5, optimizer = 'adam', activation = 'relu', node1 = 128, 
                node2 = 64, node3= 2, node4 = 16, node5 = 8):
    inputs = Input(shape =(10,))
    x = Dense(nod1)