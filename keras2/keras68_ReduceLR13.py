
import time
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss, accuracy_score

# 1. Load Data
path = 'Study25/_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# 2. Feature & Target 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# 3. Label Encoding + One-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y)
#one hot encoding
y_ohe = to_categorical(y_encoded)

# y_ohepd = pd.get_dummies(y_encoded)

# 4. StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(x), 9))
test_preds = np.zeros((len(test_csv), 9))

start = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(x, y_encoded)):
    print(f"\n📂 Fold {fold+1}")

    X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y_ohe[train_idx], y_ohe[val_idx]
    y_train_label = y_encoded[train_idx]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    x_submit = scaler.transform(test_csv)

    # Class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_label), y=y_train_label)
    class_weight_dict = dict(enumerate(class_weights))

    # Model
    model = Sequential([
        Dense(512, input_dim=93, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(), Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.3),
        Dense(9, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=200,
              batch_size=64,
              callbacks=[es, rl],
              class_weight=class_weight_dict,
              verbose=0)

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(x_submit) / skf.n_splits

    print(f"✅ Fold {fold+1} LogLoss: {log_loss(y_val, val_preds):.4f}, Accuracy: {accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_preds, axis=1)):.4f}")

end = time.time()
print(f"\n⏱️ 전체 학습 시간: {end - start:.2f}초")
print(f"📊 OOF LogLoss: {log_loss(y_ohe, oof_preds):.4f}")
print(f"📊 OOF Accuracy: {accuracy_score(np.argmax(y_ohe, axis=1), np.argmax(oof_preds, axis=1)):.4f}")

# 5. 저장
submission_df = pd.DataFrame(test_preds, columns=submission_csv.columns[1:])
submission_df.insert(0, 'id', submission_csv['id'])
submission_df.to_csv(path + 'otto_submission_kfold_classweight.csv', index=False)
print("✅ 제출 파일 저장 완료")
