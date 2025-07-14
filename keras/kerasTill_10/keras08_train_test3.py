from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#splits the train and test
from sklearn.model_selection import train_test_split
#70% train, %30 evaluation
#1 data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
# # from index 0 to 6 [:7]
# x_train = x[:7]
# y_train = y[:7]

# #from index 7 to the end
# x_test = x[7:]
# y_test = y[7:]


# print(x_test)

# x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.3, random_state=2)
# default is .75 .25 
#train_size can be abbreviated
#random_state you can give anything
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    # train_size=0.7,  
                                                    # shuffle = False,
                                                    random_state= 2 
                                                    )

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# x_train, x_test , y_train, _test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=12312)

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=12)

print(x_train.shape, x_test.shape)

# exit()

#2 model
model = Sequential()
model.add(Dense(1, input_dim= 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# compile, train

model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=500, batch_size = 1)



loss = model.evaluate(x_test,y_test)
result = model.predict([11])

print("loss:", loss)
print("prediction of [11]:", result)