# === General ===
import numpy as np
import time

# === Keras / TensorFlow ===
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# === Sklearn Metrics ===
from sklearn.metrics import (
    mean_squared_error,         # For RMSE
    r2_score,                   # For R²
    roc_auc_score,              # For binary AUC
    f1_score,                   # For F1 score
    classification_report       # For classification report
)

# === Optional Scalers & Splits ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


model = Sequential()

# ✅ 회귀용 모델 평가
loss, mae = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

# === R2 & RMSE ===
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# === 출력 ===
print(f'✅ Loss (MSE): {loss:.4f}')
print(f'✅ MAE       : {mae:.4f}')
print(f'✅ RMSE      : {rmse:.4f}')
print(f'✅ R² Score  : {r2:.4f}')
print(f'⏱️ Training Time: {end - start:.2f} sec')




# ✅ 이진 분류용 모델 평가
loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
y_pred_prob = model.predict(x_test)

# === sklearn AUC ===
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred_prob)

# === 출력 ===
print(f'✅ Binary Crossentropy Loss : {loss:.4f}')
print(f'✅ Accuracy                 : {acc:.4f}')
print(f'✅ Keras AUC                : {auc:.4f}')
print(f'✅ Sklearn ROC AUC          : {roc_auc:.4f}')
print(f'⏱️ Training Time            : {end - start:.2f} sec')




# ✅ 다중 클래스 분류 모델 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)  # if one-hot encoded

from sklearn.metrics import f1_score, classification_report
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f'✅ Categorical Crossentropy Loss: {loss:.4f}')
print(f'✅ Accuracy                     : {acc:.4f}')
print(f'✅ Weighted F1 Score            : {f1:.4f}')
print(f'⏱️ Training Time                : {end - start:.2f} sec')

# Optional
print("\n📄 Classification Report:\n")
print(classification_report(y_true, y_pred_classes))


# ✅ 다중 출력 회귀 예측
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')  # 또는 'raw_values'
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'✅ RMSE (avg across outputs): {rmse:.4f}')
print(f'✅ R² Score (avg)           : {r2:.4f}')


# ✅ Summary Table: Metrics by Task Type

# Task Type	Output Layer Example	Loss Function	Metrics (Keras & sklearn)
# 회귀 (Regression)	Dense(1)	'mse', 'mae'	RMSE, MAE, R²
# 이진 분류 (Binary)	Dense(1, activation='sigmoid')	'binary_crossentropy'	Accuracy, AUC, ROC AUC
# 다중 분류 (Multiclass)	Dense(n_classes, activation='softmax')	'categorical_crossentropy'	Accuracy, F1-score, Top-k
# 다중 출력 회귀	Dense(n_outputs)	'mse', 'mae'	Per-output RMSE, R²


# ✅ 그래서 이런 경우엔 다음을 함께 봐야 정확해요:

# Metric	사용 목적	추천 상황
# accuracy	전체 예측 중 맞춘 비율	클래스 균형일 때 OK
# AUC	클래스 구분 능력 (곡선 면적)	불균형 데이터일 때 강력 추천 ✅
# precision	예측한 긍정 중 실제로 긍정	False Positive 줄이고 싶을 때
# recall	실제 긍정 중 모델이 맞춘 비율	False Negative 줄이고 싶을 때
# f1_score	precision & recall 조화 평균	둘 다 중요할 때
# roc_auc_score	AUC 계산 (sklearn용)	평가 기준으로 자주 쓰임