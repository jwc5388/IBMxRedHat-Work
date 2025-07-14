import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine, load_breast_cancer
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
path = basepath +  '_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

print(train_csv.info()) # [652 rows x 9 columns]
print(test_csv.info()) #  [116 rows x 8 columns]
print(sample_submission_csv.info()) # [116 rows x 1 columns]



x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

print(x.shape, y.shape)         #(652, 8) (652,)
print(np.unique(y, return_counts=True))         #(array([0, 1]), array([424, 228]))

# exit()

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=seed, train_size=0.8, shuffle=True, stratify=y)

#######################################SMOTE 적용#############################################
from imblearn.over_sampling import SMOTE
import sklearn as sk

# print('sklearn version:', sk.__version__) #sklearn version: 1.6.1

# import imblearn
# print('imblearn version', imblearn.__version__) #imblearn version 0.13.0 선생님꺼 0.12.4

smote = SMOTE(random_state= seed,
              k_neighbors=5,        #default
              #sampling_strategy='auto', #default
            # sampling_strategy= 0.75
              sampling_strategy= {0:1000,1:1000},  #(array([0, 1, 2]), array([50, 57, 33]))
            # n_jobs = -1,  내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
              )
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))


# exit()
#2 model 
model = Sequential()
model.add(Dense(10, input_shape = (x.shape[1],)))
model.add(Dense(1, activation= 'sigmoid'))



model.compile(loss = 'binary_crossentropy',
              optimizer= 'adam',
              metrics = ['acc'])


model.fit(x_train, y_train, epochs = 100, validation_split=0.2)


#4 predict, evaluate
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', result[1])


y_pred = model.predict(x_test)
print(y_pred)

# y_pred = np.argmax(y_pred, axis = 1)
# ✅ 수정된 코드: 이진 분류용
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
print(y_pred)
print(y_pred.shape) 

acc = accuracy_score(y_pred, y_test)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_pred, y_test, average = 'macro')
print('accuracy score:', acc)
print('f1 score:', f1)


# accuracy score: 0.648854961832061
# f1 score: 0.39351851851851855


# accuracy score: 0.7633587786259542
# f1 score: 0.7230066161926199