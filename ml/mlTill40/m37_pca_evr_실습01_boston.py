from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import sklearn as sk
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping


print(data)
x = data
y = target


print(x.shape, y.shape)

# (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 42,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))



print(x_train.shape, x_test.shape) #(404, 13) (102, 13)


x = np.concatenate([x_train, x_test], axis=0)

print(x.shape) #(506, 13)


pca = PCA(n_components=13)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

# print(evr_cumsum)




print('0.95 이상 :', np.argmax(evr_cumsum>= 0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum>= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum>= 0.999)+1)
if np.any(evr_cumsum >= 1.0):
    print('1.0 이상 :', np.argmax(evr_cumsum >= 1.0) + 1)
else:
    print('1.0 이상을 만족하는 주성분 수 없음')


# 0.95 이상 : 8
# 0.99 이상 : 12
# 0.999 이상 : 13
# 1.0 이상 : 1


    

# #1 1.0일떄 몇개?
# # 0.999 이상 몇개?
# # 0.99 이상 몇개?
# # 0.95 이상 몇개?

# count1 = 0
# count2 = 0
# count3 = 0
# count4 =0

# print(evr_cumsum.shape)

# # exit()
# for i in range(len(evr_cumsum)):
#     if evr_cumsum[i]==1.0:
#         count1 += 1
#     if evr_cumsum[i]>=0.999:
#         count2 += 1
#     if evr_cumsum[i]>=0.99:
#         count3 +=1
#     if evr_cumsum[i]>= 0.95:
#         count4 +=1
        
# print(count1)
# print(count2)
# print(count3)
# print(count4)