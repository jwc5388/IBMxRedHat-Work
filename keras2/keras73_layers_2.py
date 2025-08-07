import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Conv1D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np
import time


import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101
from keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169
from keras.applications import DenseNet201, InceptionV3, InceptionResNetV2
from keras.layers import Dense, Flatten, AveragePooling2D


# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


vgg16 = VGG16(
    include_top=False,
    input_shape=(32,32,3),

)

vgg16.trainable =True         #가중치 안동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# trainable = True        #30, 30
# trainable = False       #30, 4


import pandas as pd
pd.set_option('max_colwidth', 10) #None 길이 다나와, 10: 10개만 나와
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer type', 'Layer name', 'Layer trainable'])