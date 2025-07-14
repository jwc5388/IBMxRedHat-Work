import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# 1. Load digits dataset
digits = load_digits()
X = digits.images  # shape: (1797, 8, 8)
y = digits.target


# print(y.shape)

aaa = 7
print(y[aaa])

import matplotlib.pyplot as plt

plt.imshow(X[aaa], 'twilight_shifted')
plt.show()


exit()

# x = digits.data
# aaa = x[0].reshape(8,8)
# print(aaa)

# print(y[0])

# exit()
print()


# 2. Normalize & reshape
X = X / 16.0  # since pixel range is 0–16
X = X.reshape((-1, 8, 8, 1))  # add channel dim for Conv2D

# 3. One-hot encode labels
y_cat = to_categorical(y, num_classes=10)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 5. Define CNN model
model = Sequential([
    Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(8, 8, 1), padding='same'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 6. Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train with validation and checkpoint saving
checkpoint = ModelCheckpoint('digits_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

# 8. Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print("loss:", loss)
print(f"Test Accuracy: {acc:.4f}")

# 9. Load best model (optional)
# from tensorflow.keras.models import load_model
# model = load_model('digits_model.h5')

# 10. Predict some digits and visualize
predictions = model.predict(X_test[:10])
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test[:10], axis=1)

plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title(f"Pred: {pred_labels[i]} / True: {true_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

#다 보이게 하기 output에
np.set_printoptions(threshold=np.inf)