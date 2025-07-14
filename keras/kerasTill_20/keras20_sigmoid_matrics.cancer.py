import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, mean_squared_error


#1 data
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

print(type(datasets))

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,)
print(type(x)) # <class 'numpy.ndarray'>

print(x)
    
    
"""[[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
 [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
 [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
 ...
 [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
 [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
 [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]
    """
    
"""[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
 0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 0 0 0 0 0 1]
    """
print(y)



print(np.unique(y, return_counts = True))

print(np.unique(y, return_counts = True))

print(pd.value_counts(y))

# (array([0, 1]), array([212, 357]))

print(pd.value_counts(y))
# 1    357
# 0    212

print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())

##y 데이터 자체가 numpy 벡터 형태 밑은 그렇게 하면 됨

## 


#0,1 갯수 몇개인지 찾아보기
#Numpy에서 0,1 갯수 찾지
#pandas


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333,
                                                    )
print(x_train.shape, x_test.shape) #(398, 30) (171, 30)
print(y_train.shape, y_test.shape) #(398,) (171,)



model = Sequential()
model.add(Dense(32, input_dim = 30, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3 compile and train 
#이진분류이기 때문에 binary_crossentropy

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy']) #['acc']



es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights = True,
    #가중치 제일 좋은거 restore
)

start_time = time.time()


hist = model.fit(x_train, y_train, epochs = 300, batch_size = 2, validation_split=0.2, callbacks = [es]  )

end_time = time.time()
#4 predict evaluate

result = model.evaluate(x_test, y_test)
print(result)
# [0.14747418463230133, 0.9561403393745422]
print('loss:', result[0])
print('accuracy:', round(result[1], 5))
# loss: 0.1550610065460205
# accuracy: 0.9473684430122375



y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict =  np.round(y_predict)
print(y_predict[:10])






exit()
from sklearn.metrics import accuracy_score

y_predict_binary = (y_predict > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_predict_binary)

# accuracy_score = accuracy_score(y_test, y_predict)
# acc = accuracy_score(y_test, y_predict)
print('acc_score: ', accuracy)
print('걸린 시간:', round(end_time - start_time, 2), "sec")










# [0.15922503173351288, 0.9473684430122375]

# [0.1494198441505432, 0.9473684430122375]