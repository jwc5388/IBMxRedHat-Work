import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T

# print(a)
# [[ 1  9]
#  [ 2  8]
#  [ 3  7]
#  [ 4  6]
#  [ 5  5]
#  [ 6  4]
#  [ 7  3]
#  [ 8  2]
#  [ 9  1]
#  [10  0]]
# print(a.shape)      #(10, 2)
# 10행 2열의 데이터중에 첫번째와 두번째 칼럼을 x로 잡고,
# 두번째 컬럼을 y값으로 잡는다.

def split_2d(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps +1):
        x_subset = a[i: i+timesteps-1]
        x.append(x_subset)
        y_subset = a[i+timesteps-1][1]
        y.append(y_subset)
    return np.array(x), np.array(y)

x, y = split_2d(a, 5)
print(x)
print(x.shape)  #(6, 4, 2)

print(y)
print(y.shape)  #(6,)



'''
def split_2d(dataset, timesteps):
    x, y = [], []
    for a in dataset:
        for b in range(len(a) - timesteps + 1):
            subset = a[b : b + timesteps]
            x.append(subset[:-1])
            y.append(subset[-1])
    return np.array(x), np.array(y)

x,y = split_2d(dataset=a, timesteps=10)
print(x, y)
print(x.shape, y.shape)
'''