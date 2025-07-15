import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=10.0, size=1000)
#지수분포의 평균(mean) 2.0
print(data)
print(data.shape)
print(np.min(data), np.max(data))

# 0.003020118521868382 16.70037195760141

log_data = np.log1p(data)  # np.expn1p(data)

np.log(data)

plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha = 0.5)
plt.title('original')


plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha = 0.5)
plt.title('log transformed')

plt.show()

#log 변환 전 score
#y 만 log 변환 score
### x 만 log 변환 socre
## x,y, log 변환 score 