#traintestsplit 후  scaling 후 pca
#LDA n_component 는 y 라벨의 갯수 -1 이하로 만들수 있다.

from sklearn.datasets import load_iris, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#1.data
import numpy as np
dataset = load_digits()
x = dataset['data']
y = dataset.target

print(x.shape, y.shape) #(1797, 64) (1797,)
y_origin = y.copy()
y = np.rint(y).astype(int)
print(y)
print(np.unique(y, return_counts=True))     #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))


# exit()

x_train, x_test, y_train, y_test , y_train_o, y_test_o= train_test_split(x,y, y_origin, train_size=0.8, random_state=337)

scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

##################################PCA##################################

# pca = PCA(n_components=10)
# x_train = pca.fit_transform(x_train)
# pca_EVR = pca.explained_variance_ratio_
# # print(pca_EVR)
# print(np.cumsum(pca_EVR))

# # [0.41116448 0.56033961 0.67693515 0.77515491 0.8400911  0.89743789
# #  0.94979913 0.9917447  0.99923825 1.        ]


###################################LDA###################################

lda = LinearDiscriminantAnalysis(n_components=9)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)
lda_EVR = lda.explained_variance_ratio_
print(lda_EVR)
print(np.cumsum(lda_EVR))

# [0.22877641 0.36618999 0.48697251 0.58320486 0.67487202 0.76053594
#  0.83324689 0.89301551 0.95144932 1.


model = RandomForestRegressor(random_state=42)

model.fit(x_train, y_train_o)
result = model.score(x_test, y_test_o)
print('score:', result)



# score: 0.8243152236338163