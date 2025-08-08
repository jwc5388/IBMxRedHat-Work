import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.arange(-5,5,0.1)

def relu(x):
    return np.maximum(0,x)


relu = lambda x: (np.maximum(0,x))

# tanh = lambda x : (np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x))

y = relu(x)




plt.plot(x,y)
plt.grid()
plt.show()




