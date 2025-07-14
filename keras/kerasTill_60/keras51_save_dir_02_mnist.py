from random import shuffle
from re import X
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization,Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

datagen = ImageDataGenerator(
    # rescale  = 1/255.  ,     #0~255 스케일링,
    # horizontal_flip = True,  #수평 뒤집기 <- 데이터 증폭 또는 변환 /상하반전
    # vertical_flip = True, #수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range = 0.1,#평행이동 10%
    height_shift_range = 0.1,
    rotation_range = 15,
    # zoom_range = 1.2,
    # shear_range = 0.7,  #좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode = 'nearest'
    
)

augment_size = 40000 #60000 개를 100000개로 
randidx = np.random.randint(x_train.shape[0], size = augment_size)
# np.random.randint(60000, 40000). 위 아래 둘이 같은거임

# print(randidx) #[53330 57656 34795 ... 41446 40462 25691] 4만개의 데이터
# print(np.min(randidx), np.max(randidx)) #0 59996



#,copy를 붙ㅋ이는 이유는 가끔 xtrain[randidx]까지 건드려서 바꾸는 경우가 있다. 안전하게 하기 위해 .copy를 넣는다.
#copy로 새로운 메모리 할당, 서로 영향 x
x_augmented = x_train[randidx].copy()

y_augmented = y_train[randidx].copy()

# print(x_augmented)
# print(x_augmented.shape)        #(40000, 28, 28)


# print(y_augmented.shape)        #(40000,)


x_augmented = x_augmented.reshape(40000,28,28,1)
# x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

print(x_augmented.shape)        #(40000, 28, 28, 1)


x_augmented = datagen.flow(
    x_augmented, 
    y_augmented,
    batch_size = augment_size,
    shuffle = False,
    save_to_dir= '/workspace/TensorJae/Study25/_data/_save_img/02_mnist/'
    
).next()[0]

exit()

print(x_augmented.shape)    #(40000, 28, 28, 1)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))


print(x_train.shape, y_train.shape)     #(100000, 28, 28, 1) (100000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# 4. Build Model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# 5. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
)

# 7. Train
start = time.time()
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),  # no augmentation on validation set
    epochs=500,
    callbacks=[es, mcp],
    verbose=1
)
end = time.time()

# 8. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print('✅ loss:', loss)
print('✅ acc:', acc)
print('⏱️ time:', end - start)

# 9. Predict
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test.values, axis=1)  # Use .values for pandas df

acc_score = accuracy_score(y_test_labels, y_pred)
print("✅ Final Accuracy Score (sklearn):", acc_score)


# ✅ loss: 0.014231361448764801
# ✅ acc: 0.9958000183105469
# ⏱️ time: 803.4022603034973
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
# ✅ Final Accuracy Score (sklearn): 0.9958