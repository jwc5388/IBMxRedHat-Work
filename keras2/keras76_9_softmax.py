import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.arange(1,5)

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))


softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

# elu = lambda x: (x>0)*x + (x<0)* (alpha*(np.exp(x)-1))


y = softmax(x)


ratio = y
labels = y
plt.pie(ratio, labels, shadow= True, startangle = 90)
plt.plot(x,y)
plt.grid()
plt.show()




