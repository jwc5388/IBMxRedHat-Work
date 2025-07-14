from sklearn.datasets import load_breast_cancer
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.cluster import KMeans
import pandas as pd

seed = 123
random.seed(seed)
np.random.seed(seed)

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data
path = basepath +  '_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
# model = KNeighborsClassifier(n_neighbors=5)
model = KMeans(n_clusters=2, init='k-means++', 
               n_init=10, random_state=seed)

y_train_pred = model.fit_predict(x_train)

print(y_train_pred[:10])
print(y_train[:10])

from sklearn.metrics import confusion_matrix

# 클러스터 예측
y_train_pred = model.fit_predict(x_train)

# 혼동 행렬 출력
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:\n", cm)


# 클러스터 예측
y_train_pred = model.fit_predict(x_train)

# confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:\n", cm)

# 정확하게 대응되는 방향 선택
acc1 = accuracy_score(y_train, y_train_pred)
acc2 = accuracy_score(y_train, 1 - y_train_pred)

if acc1 > acc2:
    print("클러스터 라벨은 실제 y와 동일하게 매핑됨")
    y_aligned = y_train_pred
else:
    print("클러스터 라벨은 실제 y와 반대로 매핑됨 (0↔1)")
    y_aligned = 1 - y_train_pred

# 정렬된 예측값으로 최종 정확도
print("최종 Accuracy:", accuracy_score(y_train, y_aligned))


import pandas as pd

df = pd.DataFrame({'Actual': y_train, 'Cluster': y_train_pred})
print(pd.crosstab(df['Actual'], df['Cluster'], rownames=['Actual'], colnames=['Predicted Cluster']))


# Confusion Matrix:
#  [[138  32]
#  [ 11 274]]
# 클러스터 라벨은 실제 y와 동일하게 매핑됨
# 최종 Accuracy: 0.9054945054945055
# Predicted Cluster    0    1
# Actual
# 0                  138   32
# 1                   11  274




exit()


model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score :', acc)
f1 = f1_score(y_test, y_pred)
print('f1_score :', f1)


# Name: Outcome, dtype: int64
# Confusion Matrix:
#  [[105 234]
#  [133  49]]
# Confusion Matrix:
#  [[105 234]
#  [133  49]]
# 클러스터 라벨은 실제 y와 반대로 매핑됨 (0↔1)
# 최종 Accuracy: 0.7044145873320538
# Predicted Cluster    0    1
# Actual                     
# 0                  105  234
# 1                  133   49