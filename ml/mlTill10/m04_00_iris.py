import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset['target']

x, y = load_iris(return_X_y=True)
# print(x)
# print(y)

print(x.shape, y.shape) #(150, 4) (150,)



#2 model 구성
# from keras.layers import Dense
# from keras.models import Sequential

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape = (4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

##################
#model = LinearSVC(C=0.3)
##################

##################
# model = LogisticRegression()
##################

##################
# model = DecisionTreeClassifier()
##################

model = RandomForestClassifier()









model.fit(x,y)

result = model.score(x,y)

#sparse_categorical_crossentropy 쓰면 y onehot 안해도 됨!
#3 compile. train
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

# model.fit(x,y, epochs= 100)

# result = model.evaluate(x,y)
print(result)



#Decisiontreesclassifier : 1.0
