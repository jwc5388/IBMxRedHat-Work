import numpy as np
import pandas as pd
import random

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler # <--- ADDED: For scaling

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import tensorflow as tf
from imblearn.over_sampling import SMOTE # Already present

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1 data
dataset = fetch_covtype()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)
unique_labels, label_counts = np.unique(y, return_counts=True)
print(f"Original unique labels and counts: {unique_labels}, {label_counts}")

# --- FIX 1: Adjust target labels to be 0-indexed ---
# Keras's sparse_categorical_crossentropy expects 0-indexed labels.
# If your labels are 1, 2, ..., 7, subtract 1 to make them 0, 1, ..., 6.
y = y - 1
unique_labels, label_counts = np.unique(y, return_counts=True)
print(f"Adjusted unique labels and counts (0-indexed): {unique_labels}, {label_counts}")
# Now the output layer size (7) correctly corresponds to indices 0-6.

# print(pd.value_counts(y)) # For checking value counts

# traintestsplit
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8, shuffle=True, stratify=y)

# --- FIX 2: Apply Scaling/Normalization BEFORE SMOTE ---
# Scaling is crucial for neural networks. Apply it *before* SMOTE
# because SMOTE creates new samples based on existing ones, and they should
# already be in the correct scale.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) # Use the same scaler fitted on training data

print(f"Shape be₩fore SMOTE: x_train={x_train_scaled.shape}, y_train={y_train.shape}")
print(f"Unique labels and counts before SMOTE: {np.unique(y_train, return_counts=True)}")

#######################################SMOTE 적용#############################################
smote = SMOTE(random_state=seed,
              k_neighbors=5, # Default 5. Consider adjusting if very small minority classes are present.
              sampling_strategy='auto', # This tries to balance all classes# Use all available CPU cores for SMOTE if supported by your imblearn version
              )

# Apply SMOTE to the SCALED training data
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)

print(f"Shape AFTER SMOTE: x_train={x_train_resampled.shape}, y_train={y_train_resampled.shape}")
print(f"Unique labels and counts AFTER SMOTE: {np.unique(y_train_resampled, return_counts=True)}")

#2 model
# The output layer should match the number of unique classes (which is 7, for 0-6)
model = Sequential()
model.add(Dense(10, input_shape=(54,)))
model.add(Dense(7, activation='softmax')) # 7 classes (0 through 6)

model.compile(loss='sparse_categorical_crossentropy', # Correct for 0-indexed integer labels
              optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=10
)

# Train with the RESAMPLED data
# Use x_train_resampled and y_train_resampled
history = model.fit(x_train_resampled, y_train_resampled,
                    epochs=30,
                    validation_split=0.2, # Validation split will be taken from the RESAMPLED data
                    batch_size=128,
                    callbacks=[es],
                    verbose=1) # Set verbose to 1 to see training progress

#4 predict, evaluate
# Evaluate with the SCALED test data
result = model.evaluate(x_test_scaled, y_test)
print('loss:', result[0])
print('acc:', result[1])

# Make predictions on the SCALED test data
y_pred_proba = model.predict(x_test_scaled)
# print(y_pred_proba) # Probability distributions

y_pred = np.argmax(y_pred_proba, axis=1) # Get the class with the highest probability
# print(y_pred) # Predicted labels (0-indexed)
print(f"y_pred shape: {y_pred.shape}")

acc = accuracy_score(y_test, y_pred) # Compare y_test (0-indexed) with y_pred (0-indexed)
f1 = f1_score(y_test, y_pred, average='macro') # Correct for multi-class with 0-indexed labels
print('accuracy score:', acc)
print('f1 score:', f1)