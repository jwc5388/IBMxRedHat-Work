import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(150, 4) (150,)

print(y)


print(np.unique(y, return_counts=True))
print(pd.value_counts(y))


######OneHotCoding 반드시 y만 ######*#*#*#*#*#*#*#*#*#*
######반 드 시 y 만 ###################

#1. sklearn 용

from sklearn.preprocessing import OneHotEncoder

#y = np.array([0, 1, 2, 1, 0])
# y.shape  # → (5,)
# After y.reshape(-1, 1):

# y = y.reshape(-1, 1)
# y.shape  # → (5, 1)
# Now y looks like this:

# [[0],
#  [1],
#  [2],
#  [1],
#  [0]]
# y = y.reshape(-1,1)
# encoder = OneHotEncoder() #메트릭스 형태를 받기 때문에 N,1로 reshape하고 해야한다

# y = encoder.fit_transform(y)
#print(type(y)) # scipy.sparse.csr.csr_matrix

# y = y.reshape(-1, 1)
# encoder = OneHotEncoder(sparse=False)  # 또는 sparse=False (버전에 따라) numpy 형태 출력, 디폴트는 true.
# # y = y.toarray() #scipy를 numpy로 변환
# y = encoder.fit_transform(y).astype('float32')  # ⭐ 반드시 float32로
# print(y.shape)
# print(type(y))
# print('sklearn onehot:', y)


#2. pd 용

y= pd.get_dummies(y)
# print("▶ Pandas get_dummies:\n", y_pd)
# print(y_pd)
# print(y_pd.shape)


# # #3. keras 용

# from tensorflow.keras.utils import to_categorical
# y_keras = to_categorical(y)
# print("▶ Keras to_categorical:\n", y_keras)




# exit()


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=33)

#2 model
# 4. Build model
model = Sequential([
    Dense(64, input_dim=4, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3-class classification
])



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor = 'val_loss',
                   mode = min,
                   patience = 10,
                   restore_best_weights= True)

model.fit(x_train, y_train, epochs = 1000, batch_size = 8, validation_split = 0.1, verbose = 1, callbacks = [es])


loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss)
print('acc:', accuracy)

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)

print(y_pred_class)
y_test_class = np.argmax(y_test, axis=1)
print(y_test_class)
acc_score = accuracy_score(y_test_class, y_pred_class)
print("✅ accuracy_score:", acc_score)

######accuracy Score 를 사용해서 출력해볼것!!!!################
from sklearn.metrics import accuracy_score


print(y_pred.shape, y_test.shape)





# loss: 0.03608160465955734
# acc: 1.0