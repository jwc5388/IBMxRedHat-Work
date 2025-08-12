import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

#1 data
# x,y = load_iris(return_X_y=True)
x,y = load_digits(return_X_y=True)

print(x.shape, y.shape)

# pca = PCA(n_components=8)
# x = pca.fit_transform(x)
# print(x.shape)
# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=777, stratify=y)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. model

# model = RandomForestClassifier()
#순서는 지켜야함 scaler 먼저
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
model = make_pipeline(PCA(n_components=8), MinMaxScaler(), SVC())

#3 train
model.fit(x_train, y_train)

#4 evaluate, predict
result = model.score(x_test, y_test)
print('model score:', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy score:', acc)