import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)

print(x)

pf = PolynomialFeatures(degree=3, include_bias=False) #default = True
x_pf = pf.fit_transform(x)
print(x_pf)

# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]