#m36_1에서 뽑은 4가지 결과로 5개의 모델 만들기
#input_shape =
# (70000, 154)
#(70000, 331)
#(70000, 486)
#(70000, 713)


#힌트 num = [154, 331, 486, 713, 784]


from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential

(x_train, y_train), (x_test, y_test )= mnist.load_data()

print(x_train.shape, x_test.shape)

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# (60000, 28, 28) (10000, 28, 28)

num = [154, 331, 486, 713, 784]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


for n in num:
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    model = Sequential()
    model.add(Dense(128, input_shape=(n,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pca, y_train, epochs=3, batch_size=32, verbose=0)
    loss, acc = model.evaluate(x_test_pca, y_test, verbose=0)
    print(f"[{n} components] accuracy: {acc:.4f}")
    
    
#     [154 components] accuracy: 0.9059
# [331 components] accuracy: 0.9076
# [486 components] accuracy: 0.9096
# [713 components] accuracy: 0.9070
# [784 components] accuracy: 0.9041