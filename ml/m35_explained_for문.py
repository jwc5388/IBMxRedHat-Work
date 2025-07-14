#traintestsplit 후  scaling 후 pca


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np


#1.data

dataset = load_iris()
x = dataset['data']
y = dataset.target

print(x.shape, y.shape) #(150, 4) (150,)

#어떤놈들이 scaler는 pca 전에 하는게 좋다 했다




x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=337, stratify=y)

scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




n_features = x.shape[1]
for i in range(1,n_features+1):
    
    pca = PCA(n_components= i)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    model = RandomForestClassifier(random_state=333)

    #3 train
    model.fit(x_train_pca, y_train)

    #4 evalutate
    result = model.score(x_test_pca, y_test)
    
    print(x_train_pca.shape, '의 score:', result)
    
    print(f"n_components={i} → model score: {result:.4f}")


evr = pca.explained_variance_ratio_ #설명가능한 변화율
print('evr:', evr)
print('evr_sum:', sum(evr))

# (150, 4) (150,)
# (120, 1) 의 score: 0.9333333333333333
# n_components=1 → model score: 0.9333
# (120, 2) 의 score: 0.8333333333333334
# n_components=2 → model score: 0.8333
# (120, 3) 의 score: 0.8666666666666667
# n_components=3 → model score: 0.8667
# (120, 4) 의 score: 0.9666666666666667
# n_components=4 → model score: 0.9667

#pca 자체에 미치는 영향률
# evr: [0.73515725 0.22803596 0.0311646  0.00564219]
evr_cumsum = np.cumsum(evr)

print('누적합 :', evr_cumsum)

# 누적합 : [0.73515725 0.96319321 0.99435781 1.        ]

#차례대로 ncomponent 1,2,3,4 인 경우 순서대로


#시각화
import matplotlib.pyplot as plt
plt.plot(evr.cumsum())
plt.grid()
plt.show()