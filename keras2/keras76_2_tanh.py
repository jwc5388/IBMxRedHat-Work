import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.arange(-5,5,0.1)


tanh = lambda x : (np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x))


y = np.tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()




