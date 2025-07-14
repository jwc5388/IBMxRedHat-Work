from keras.datasets import cifar100
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# 1. Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Data Augmentation (사전 증강)
augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# 증강 적용
x_augmented = next(datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
))[0]

# 3. Concatenate augmented data
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# 4. One-hot encode labels (use keras's stable function)
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

print(x_train.shape, y_train.shape)  # (90000, 32, 32, 3), (90000, 100)

# 5. Build model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
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
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='softmax'))  # CIFAR-100 이므로 100개 클래스

# 6. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 8. Train (no more augmentation here, since we pre-augmented)
start = time.time()
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    batch_size=64,
    callbacks=[es],
    verbose=1
)
end = time.time()

# 9. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("✅ loss:", loss)
print("✅ acc:", acc)
print("⏱️ time:", end - start)

# 10. Predict & Accuracy Score
result = model.predict(x_test)
y_pred = np.argmax(result, axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_true, y_pred)
print("✅ Final Accuracy Score:", acc_score)


# loss: 2.544921398162842
# acc: 0.34450000524520874
# time: 1230.7274162769318

# loss: 1.7060247659683228
# acc: 0.5497999787330627
# time: 3034.2118003368378
