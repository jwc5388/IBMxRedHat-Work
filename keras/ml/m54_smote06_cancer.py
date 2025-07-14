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

#1 data
dataset = load_breast_cancer()  
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         #(569, 30) (569,)
print(np.unique(y, return_counts=True))         #(array([0, 1]), array([212, 357]))
# print(pd.value_counts(y))           

# exit()



### data 삭제, 라벨이 2인놈을 10개만 남기고 다 지워라!!!!! ###

# x = x[:-40]
# y = y[:-40]

#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8, shuffle=True, stratify=y)

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


y_pred = (y_pred > 0.5).astype(int).reshape(-1)

print(y_pred)
print(y_pred.shape) 

acc = accuracy_score(y_pred, y_test)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_pred, y_test, average = 'macro')
print('accuracy score:', acc)
print('f1 score:', f1)

# accuracy score: 0.3684210526315789
# f1 score: 0.2692307692307692


#재현smote
# accuracy score: 0.7017543859649122
# f1 score: 0.7013867488443759