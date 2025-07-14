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


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# 1. Load Data
path = basepath + '_data/kaggle/otto/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
feature_names = x.columns
y = train_csv['target']

# # ✅ LabelEncoder 적용
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))   

# (61878, 93) (61878,)
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955]))
# exit()



#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8, shuffle=True, stratify=y)

#######################################SMOTE 적용#############################################
from imblearn.over_sampling import SMOTE
import sklearn as sk

smote = SMOTE(random_state= seed,
              k_neighbors=5,        #default
              sampling_strategy='auto', #default
            # sampling_strategy= 0.75
              #sampling_strategy= {1:5000, 2:5000,3:5000, 4:5000, 5:5000, 6: 5000, 7:5000},  #(array([0, 1, 2]), array([50, 57, 33]))
            # n_jobs = -1,  내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
              )
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))


# exit()
#2 model 
model = Sequential()
model.add(Dense(10, input_shape = (x.shape[1],)))
model.add(Dense(9, activation= 'softmax'))



model.compile(loss = 'sparse_categorical_crossentropy', #원핫 안했잖아!!!!!!!!!!!!!!!
              optimizer= 'adam',
              metrics = ['acc'])


model.fit(x_train, y_train, epochs = 30, validation_split=0.2, )


#4 predict, evaluate
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', result[1])


y_pred = model.predict(x_test)
print(y_pred)


y_pred = np.argmax(y_pred, axis = 1)
print(y_pred)

acc = accuracy_score(y_pred, y_test)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_pred, y_test, average = 'macro')
print("===============", model.__class__.__name__, "======================")

print('accuracy score:', acc)
print('f1 score:', f1)

# accuracy score: 0.7066903684550744
# f1 score: 0.6680337402764495