#36-2 copy
import numpy as np

from keras.datasets import mnist, fashion_mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train)

print(x_train[0])
print(y_train[0]) 

# exit()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)


#이진인지 다중인지 확인!!!! 
print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))

print(pd.value_counts(y_test))

aaa = 7
print(y_train[aaa])

import matplotlib.pyplot as plt

plt.imshow(x_train[aaa], 'twilight_shifted')
plt.show()
