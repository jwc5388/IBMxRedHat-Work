# #불균형한 이진분류



# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, BatchNormalization, Dropout
# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import roc_auc_score
# import time


# # 1. Load Data
# path = 'Study25/_data/kaggle/santander/'

# train_csv = pd.read_csv(path + 'train.csv')
# test_csv = pd.read_csv(path + 'test.csv')
# submission_csv = pd.read_csv(path + 'sample_submission.csv')

# # 2. Feature/Target 분리
# x = train_csv.drop(['ID_code', 'target'], axis=1)
# y = train_csv['target']
# x_submit = test_csv.drop(['ID_code'], axis=1)

# # 3. Train/Test Split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)

# # 4. 스케일링
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# x_submit = scaler.transform(x_submit)

# # 5. 클래스 불균형 보정
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# # 6. 모델 구성
# model = Sequential([
#     Dense(256, input_shape=(200,), activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])

# # 7. 모델 컴파일
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 8. 조기 종료 설정
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=20,
#     restore_best_weights=True
# )

# # 9. 모델 훈련
# start = time.time()
# model.fit(x_train, y_train,
#           validation_split=0.2,
#           epochs=10000,
#           batch_size=32,
#           callbacks=[es],
#           class_weight=class_weight_dict,
#           verbose=1)
# end = time.time()

# # 10. 평가
# loss, acc = model.evaluate(x_test, y_test)
# y_pred_prob = model.predict(x_test)
# roc_auc = roc_auc_score(y_test, y_pred_prob)

# print(f"loss: {loss:.4f}")
# print(f"accuracy: {acc:.4f}")
# print(f"roc_auc: {roc_auc:.4f}")
# print(f"걸린 시간: {end - start:.2f}초")

# # 11. 제출 파일 생성
# y_submit = model.predict(x_submit)
# submission_csv['target'] = y_submit
# submission_csv.to_csv(path + 'submission_0608_final.csv')


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import time

# 0. 시드 고정 (재현성을 위해)
np.random.seed(42)
tf.random.set_seed(42)

# 1. 데이터 로드
path = 'Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 2. 피처/타겟 분리 및 기본 피처 리스트 생성
y = train_csv['target']
x = train_csv.drop(['ID_code', 'target'], axis=1)
x_submit = test_csv.drop(['ID_code'], axis=1)
original_features = [col for col in x.columns]

# 3. 피처 엔지니어링 (행별 통계량 추가)
def feature_engineering(df, feature_list):
    df['sum'] = df[feature_list].sum(axis=1)
    df['mean'] = df[feature_list].mean(axis=1)
    df['std'] = df[feature_list].std(axis=1)
    df['min'] = df[feature_list].min(axis=1)
    df['max'] = df[feature_list].max(axis=1)
    df['skew'] = df[feature_list].skew(axis=1)
    df['kurt'] = df[feature_list].kurtosis(axis=1)
    return df

print("피처 엔지니어링 시작...")
x = feature_engineering(x, original_features)
x_submit = feature_engineering(x_submit, original_features)
print("피처 엔지니어링 완료.")

# 4. Keras 모델 생성 함수 정의
def create_model(input_shape):
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # AUC를 직접 모니터링하도록 메트릭에 추가
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# 5. StratifiedKFold 교차 검증 및 훈련
N_SPLITS = 5 # Fold 개수, 시간 관계상 5로 줄여서 테스트 가능
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(x))          # Out-of-Fold 검증 예측값 저장용
test_preds = np.zeros(len(x_submit))  # 테스트 데이터 예측값 저장용

start_time = time.time()
print(f"{N_SPLITS}-Fold 교차 검증 시작...")

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"\n====== Fold {fold+1} ======")
    
    # --- 데이터 분리 ---
    X_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
    
    # --- 스케일링 (Fold 내에서 fit) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    x_submit_scaled = scaler.transform(x_submit) # 현재 fold의 스케일러로 전체 테스트셋 변환

    # --- 클래스 가중치 계산 ---
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # --- 모델 생성 및 콜백 정의 ---
    model = create_model(X_train.shape[1])
    
    # 조기 종료: val_auc가 25번 동안 개선되지 않으면 종료
    es = EarlyStopping(monitor='val_auc', mode='max', patience=25, restore_best_weights=True)
    
    # 학습률 스케줄러: val_auc가 10번 동안 개선되지 않으면 학습률을 절반으로 줄임
    rlr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=10, verbose=1)

    # --- 모델 학습 ---
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=500, # 최대 에포크
              batch_size=512, # 배치 사이즈 증가로 학습 속도 향상
              callbacks=[es, rlr],
              class_weight=class_weight_dict,
              verbose=1) # 로그를 간소화
    
    # --- 예측 (최고 성능 모델로) ---
    val_preds = model.predict(X_val).flatten() # (N, 1) -> (N,)
    fold_test_preds = model.predict(x_submit_scaled).flatten()
    
    # --- 결과 저장 ---
    oof_preds[val_idx] = val_preds
    test_preds += fold_test_preds / N_SPLITS # 각 Fold 예측을 더한 후 Fold 수로 나눔 (앙상블)

end_time = time.time()

# 6. 최종 평가 및 제출
total_auc = roc_auc_score(y, oof_preds)
print(f"\n========================================")
print(f"전체 Out-of-Fold AUC: {total_auc:.6f}")
print(f"총 훈련 시간 cpu: {end_time - start_time:.2f}초")

# 7. 제출 파일 생성
submission_csv['target'] = test_preds
submission_csv.to_csv(path + 'submission_0608_tf_final.csv', index=False)
print("제출 파일 생성 완료: submission_0608_tf_final.csv")