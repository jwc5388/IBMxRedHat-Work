import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 Data
x = np.array(range(10))
print(x)    #[0 1 2 3 4 5 6 7 8 9] scalar 10 vector . vectors -> tensor
print(x.shape) # (10,)

x = np.array(range(1,10)) #[1 2 3 4 5 6 7 8 9]
print(x)  
print(x.shape) # (9,)

x = np.array(range(1,11)) #[ 1  2  3  4  5  6  7  8  9 10]
print(x)

#2개 이상은 LIST
x = np.array([range(10), range(21,31), range(201,211)]) # [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]
print(x)
print(x.shape)  #(3,10)

#here, each row is a feature series, and we want each column to be a data point. so we transpose. then it changes to (10,3)
# 🔹 1. Data Preparation

# x = np.array([range(10), range(21,31), range(201,211)])
# This creates a 2D array with shape (3, 10):

# [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]
# Each row is a feature series, but you want each column to be a data point (sample). So, you transpose:
# x = x.T
# Now shape becomes (10, 3) — meaning:

# 10 samples (rows)

# 3 features (columns): x1, x2, x3

#### same as transpose!!! use whatever you want 
#전치행렬 =  transpose matrix
x = x.T
print(x)
print(x.shape) #(10,3)



y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습]
#[10,31,211] 예측

#2 model

model = Sequential()
model.add(Dense(10,input_dim = 3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#  2. Target Variable (y)
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# This is the label for each of the 10 samples in x.

# So:

# First sample [0, 21, 201] → 1

# Second sample [1, 22, 202] → 2

# ...

# Last sample [9, 30, 210] → 10

epochs = 500
#3 compile and train
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y, epochs = epochs)

# evaluate and predict

loss = model.evaluate(x,y)

result = model.predict([[10,31,211]])

print("the prediction of [10,31,211]:" , result)    #the prediction of [10,31,211]: [[11.000004]]