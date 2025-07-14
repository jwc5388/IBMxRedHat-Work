from sklearn.datasets import load_breast_cancer
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# 1. Load Data
path = basepath + '_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Encode categorical features
le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# 3. Drop unneeded columns
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# 4. Separate features and target
x = train_csv.drop(['Exited'], axis=1)
feature_names = x.columns
y = train_csv['Exited']

print(x.shape, y.shape)      
print(np.unique(y, return_counts=True))        
# (165034, 10) (165034,)
# (array([0, 1]), array([130113,  34921]))
# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from imblearn.over_sampling import SMOTE
import sklearn as sk

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

# Name: Exited, dtype: int64
# Confusion Matrix:
#  [[44647 59443]
#  [17166 10771]]
# Confusion Matrix:
#  [[44647 59443]
#  [17166 10771]]
# 클러스터 라벨은 실제 y와 반대로 매핑됨 (0↔1)
# 최종 Accuracy: 0.5802525241049179
# Predicted Cluster      0      1
# Actual                         
# 0                  44647  59443
# 1                  17166  10771



exit()


model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score :', acc)
f1 = f1_score(y_test, y_pred)
print('f1_score :', f1)

# ================ KNeighborsClassifier ================
# acc :  0.9736842105263158
# accuracy_score : 0.9736842105263158
# f1_score : 0.9793103448275862