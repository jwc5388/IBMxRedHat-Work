import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine, fetch_covtype, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1 data
dataset = load_digits()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))       
# print(pd.value_counts(y))           



print(y)




#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8, shuffle=True, stratify=y)

#######################################SMOTE 적용#############################################
import sklearn as sk
from imblearn.over_sampling import SMOTE, RandomOverSampler

print('sklearn version:', sk.__version__) #sklearn version: 1.6.1

import imblearn
print('imblearn version', imblearn.__version__) #imblearn version 0.13.0 선생님꺼 0.12.4

# smote = SMOTE(random_state= seed,
#               k_neighbors=5,        #default
#               #sampling_strategy='auto', #default
#             # sampling_strategy= 0.75
#               sampling_strategy= {0:5000,1:5000, 2:5000},  #(array([0, 1, 2]), array([50, 57, 33]))
#             # n_jobs = -1,  내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
#               )

ros = RandomOverSampler(random_state= seed,
              #sampling_strategy='auto', #default
            # sampling_strategy= 0.75
              sampling_strategy= {0:5000,1:5000, 2:5000},  #(array([0, 1, 2]), array([50, 57, 33]))
            # n_jobs = -1,  내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
              )


x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))

from imblearn.over_sampling import SMOTE, RandomOverSampler


# exit()
#2 model 
model = Sequential()
model.add(Dense(10, input_shape = (64,)))
model.add(Dense(10, activation= 'softmax'))


model.compile(loss = 'sparse_categorical_crossentropy', #원핫 안했잖아!!!!!!!!!!!!!!!
              optimizer= 'adam',
              metrics = ['acc'])


model.fit(x_train, y_train, epochs = 100, validation_split=0.2)


#4 predict, evaluate
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', result[1])


y_pred = model.predict(x_test)
print(y_pred)

y_pred = np.argmax(y_pred, axis = 1)
print(y_pred)
print(y_pred.shape) #[1 1 2 1 0 1 1 2 2 0 2 2 0 1 1 0 1 0 0 1 0 1 1 1 0 0 0 0]

acc = accuracy_score(y_pred, y_test)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_pred, y_test, average = 'macro')
print('accuracy score:', acc)
print('f1 score:', f1)

