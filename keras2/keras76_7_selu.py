import numpy as np
import matplotlib.pyplot as plt
import torch


x = np.arange(-5,5,0.1)

lmbda = 1.04394933454

def selu(x, alpha, lmbda):
    return lmbda * ((x>0)*x + (x<=0)*(alpha * (np.exp(x)-1)))


# elu = lambda x: (x>0)*x + (x<0)* (alpha*(np.exp(x)-1))


y = selu(x,1.67, 1.05)




plt.plot(x,y)
plt.grid()
plt.show()




