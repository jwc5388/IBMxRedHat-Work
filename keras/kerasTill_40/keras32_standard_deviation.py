import numpy as np
from sklearn.preprocessing import StandardScaler

#1 data
data = np.array([[1,2,3,1],
                [4,5,6,2],
                [7,8,9,3],
                [10,11,12,114],
                [13,14,15,115]])

print(data.shape) #(5, 4)

#1 평균

means = np.mean(data, axis = 0)
print('평균:', means)


#2 모집단 분산 (n으로 나누기)
# 
# = (x-평균)^2

population_variance = np.var(data, axis=0)
print('모집단 분산:', population_variance)


#3 표본 분산 (n-1로 나눈다)

variance = np.var(data, axis=0)
print('표본분산:', variance)

#4 표본 표준편차
std1 = np.std(data, axis=0, ddof=1)
print("표준 표준편차:", std1)

#4 standardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("StandardScaler : \n", scaled_data)

# StandardScaler : 
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]