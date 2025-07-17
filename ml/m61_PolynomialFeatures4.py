import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random

plt.rcParams['font.family'] = 'Malgun Gothic'

#1 data
random.seed(777)
np.random.seed(777)
x = 2*np.random.rand(100,1) -1
print(np.min(x), np.max(x))

y = 3*x**2 + 2*x + 1 + np.random.randn(100,1) 

pf = PolynomialFeatures(degree=2, include_bias=False)

x_pf = pf.fit_transform(x)

print(x_pf)


#2 model
model = LinearRegression()
model2 = LinearRegression()

#3훈련
model.fit(x,y)
model2.fit(x_pf, y)

#원래 데이터 그리고
plt.scatter(x,y, color='blue', label = 'Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')



x_test = np.linspace(-1,1,100).reshape(-1,1)
x_test_pf = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot_pf = model2.predict(x_test_pf)
plt.plot(x_test, y_plot, color = 'red', label= '기냥')
plt.plot(x_test, y_plot_pf, color = 'green', label= 'Polynomial Regression')

plt.legend()
plt.grid()
plt.show()