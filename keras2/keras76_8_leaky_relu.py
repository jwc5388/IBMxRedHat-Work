import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.arange(-5,5,0.1)

alpha = 0.01

# def leaky_relu(x):
#     # return np.maximum(alpha*x,x)
#     return np.where(x>0, x, alpha*x)


leaky_relu = lambda x: np.where(x>0, x, alpha*x)
# tanh = lambda x : (np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x))

y = leaky_relu(x)




plt.plot(x,y)
plt.grid()
plt.show()




