from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

(x_train, _), (x_test, _ )= mnist.load_data()

print(x_train.shape, x_test.shape)

# (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)

print(x.shape)  #(70000, 28, 28)

x = x.reshape(70000, 28*28) # 70000, 784

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

print(evr_cumsum)



#1 1.0일떄 몇개?
# 0.999 이상 몇개?
# 0.99 이상 몇개?
# 0.95 이상 몇개?

count1 = 0
count2 = 0
count3 = 0
count4 =0

print(evr_cumsum.shape)

# exit()
for i in range(len(evr_cumsum)):
    if evr_cumsum[i]==1.0:
        count1 += 1
    if evr_cumsum[i]>=0.999:
        count2 += 1
    if evr_cumsum[i]>=0.99:
        count3 +=1
    if evr_cumsum[i]>= 0.95:
        count4 +=1
        
print(count1)
print(count2)
print(count3)
print(count4)

print('0.95 이상 :', np.argmax(evr_cumsum>= 0.95)+1)
print('0.99 이상 :', np.argmax(evr_cumsum>= 0.99)+1)
print('0.999 이상 :', np.argmax(evr_cumsum>= 0.999)+1)
print('1.0 이상 :', np.argmax(evr_cumsum>= 1.0)+1)


    
