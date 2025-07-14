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

# print(x)
# print(x.shape)  #(506, 13)
# print(y)
# print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=3 )
#2 model
model = Sequential()
model.add(Dense(50,input_dim = 13, activation = 'relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3 compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 200, batch_size =1,
          validation_split = 0.15)


#4 evaluate and predict
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

result = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, result)
print("r2 score", r2)


# 4/4 [==============================] - 0s 959us/step - loss: 20.6386
# loss: 20.638578414916992
# 4/4 [==============================] - 0s 664us/step
# r2 score 0.7506057037186485

# 4/4 [==============================] - 0s 966us/step - loss: 20.5447
# loss: 20.544715881347656
# 4/4 [==============================] - 0s 665us/step
# r2 score 0.7517398777786128