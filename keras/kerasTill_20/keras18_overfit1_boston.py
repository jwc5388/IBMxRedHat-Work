#17-1 copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# y data = target data

#1 data
# dataset = load_boston()
# # print(dataset)
# #Describe
# print(dataset.DESCR) #(506,13)
# print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = data
y = target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=333 )
#2 model
model = Sequential()
model.add(Dense(50,input_dim = 13, activation ='relu'))
model.add(Dense(100, activation ='relu'))
model.add(Dense(100, activation ='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')

##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 300, batch_size =1,
          verbose = 2,
          validation_split = 0.2)

print('===========hist=========')
print(hist)
print('===========hist.history=========')
print(hist.history)
print('=================loss=================')
print(hist.history['loss'])

print('=================val_loss=================')
print(hist.history['val_loss'])


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) # 9x6 size 
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.title("Boston loss")
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.legend(loc='upper right') #우측 상단에 label 표시
plt.grid() #격자 표시


plt.show()

#like this, key : value - dictionary form
#epoch 1- end you can see the history
# {'loss': [164.796630859375, 86.1928939819336, 75.24617767333984, 67.33546447753906, 68.33426666259766, 67.75279235839844, 69.83555603027344, 57.092018127441406, 59.492462158203125, 55.933570861816406], 'val_loss': [82.12428283691406, 48.262699127197266, 33.37749099731445, 32.25654220581055, 35.78428268432617, 45.08708190917969, 30.095138549804688, 31.247413635253906, 25.036834716796875, 31.765363693237305]}


#4 evaluate and predict
"""
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

result = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, result)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print()

"""


# 4/4 [==============================] - 0s 959us/step - loss: 20.6386
# loss: 20.638578414916992
# 4/4 [==============================] - 0s 664us/step
# r2 score 0.7506057037186485

# 4/4 [==============================] - 0s 966us/step - loss: 20.5447
# loss: 20.544715881347656
# 4/4 [==============================] - 0s 665us/step
# r2 score 0.7517398777786128




#########after validation split
# loss: 16.210494995117188
# r2 score 0.8347199826085285
