import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.arange(-5,5,0.1)

def silu(x):
    return x * (1/(1+np.exp(-x)))


silu = lambda x: x * (1/(1+np.exp(-x)))


y = silu(x)




plt.plot(x,y)
plt.grid()
plt.show()




