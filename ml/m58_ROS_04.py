import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data


path = basepath + '_data/dacon/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# print(train_csv) #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
# print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+ 'submission.csv', index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

train_csv = train_csv.dropna()


# test_csv = test_csv.dropna()
test_csv = test_csv.fillna(train_csv.mean())

x = train_csv.drop(['count'], axis=1)  #count라는 axis=1 열 삭제, 행은 axis =0
feature_names = x.columns
print(x) #[1459 rows x 9 columns]

y = train_csv['count'] 
print(y) #(1459,)    

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
model.add(Dense(10, input_shape = (13,)))
model.add(Dense(3, activation= 'softmax'))



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
