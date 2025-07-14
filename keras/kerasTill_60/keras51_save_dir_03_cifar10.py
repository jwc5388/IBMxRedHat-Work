from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
import time

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 2. Data Augmentation (Offline)
augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

x_augmented = next(datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir= '/workspace/TensorJae/Study25/_data/_save_img/03_cifar10/', 
))[0]

# 3. Combine original and augmented data
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print("✅ Training data shape:", x_train.shape, y_train.shape)
print("✅ Test data shape:", x_test.shape, y_test.shape)

# 4. Build model
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
model.add(Dense(10, activation='softmax'))

# 5. Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 7. Train
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

# 8. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("✅ loss:", loss)
print("✅ acc:", acc)
print("⏱️ time:", end - start)

# 9. Predict
result = model.predict(x_test)
y_pred = np.argmax(result, axis=1)
y_true = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_true, y_pred)
print("✅ Final Accuracy Score:", acc_score)




# ✅ loss: 0.5149935483932495
# ✅ acc: 0.8289999961853027
# ⏱️ time: 2788.2778754234314


# ✅ loss: 0.6327520608901978
# ✅ acc: 0.7950000166893005
# ⏱️ time: 2047.7361674308777
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step
# ✅ Final Accuracy Score: 0.795