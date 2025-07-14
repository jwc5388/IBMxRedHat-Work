import numpy as np

a = np.array(range(1,11))

timesteps = 5

print(a.shape)  #(10,)

def split_1d(dataset, timesteps):
    all = []
    for i in range(len(dataset) - timesteps + 1) :
        subset = dataset[i : (i+timesteps)]
        all.append(subset)
    all = np.array(all)
    x = all[:,:-1]
    y = all[:,-1]
    return x, y


# x = all[:,:-1]   # All rows, except last column
# y = all[:,-1]    # Last column only

x, y = split_1d(a, timesteps=timesteps)
print(x)
print(y)

# bbb = split_x(a, timesteps=timesteps)
# # print(bbb)
# # [[ 1  2  3  4  5]
# #  [ 2  3  4  5  6]
# #  [ 3  4  5  6  7]
# #  [ 4  5  6  7  8]
# #  [ 5  6  7  8  9]
# #  [ 6  7  8  9 10]] (6,5)

# x = bbb[: , :-1]      
# y = bbb[: , -1]
# print(x, y)
# # [[1 2 3 4]
# #  [2 3 4 5]
# #  [3 4 5 6]
# #  [4 5 6 7]
# #  [5 6 7 8]
# #  [6 7 8 9]] [ 5  6  7  8  9 10]
# print(x.shape, y.shape) #(6, 4) (6,)