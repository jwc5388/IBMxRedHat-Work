import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1 data

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=3)


#model
model = Sequential()
model.add(Dense(1 , input_dim = 1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#compile and train
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 1)

#4 evaluate and predict

result = model.predict([x])
print("prediction of [x]:", result)

loss = model.evaluate(x_test,y_test)
print("loss:", loss)


import matplotlib.pyplot as plt
plt.scatter(x,y) #plot data
plt.plot(x, result, color = 'red')
plt.show()

