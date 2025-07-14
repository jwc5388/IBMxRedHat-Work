

# 1. 라이브러리 로딩 및 환경 정보
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, callback
import xgboost as xgb
import random
from imblearn.over_sampling import SMOTE
import datetime
import joblib  # 모델 저장에 사용
seed = 190
random.seed(seed)
np.random.seed(seed)

# 파일 저장을 위한 타임스탬프 경로 설정
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')


# 2. 데이터 로딩
############################################### 3377 222

path = './Study25/_data/dacon/cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

##################### 결측치 확인 ####################
#print(train_csv.info())
#print(train_csv.isnull().sum()) #결측치 없음
#print(test_csv.isna().sum()) #결측치 없음
print(train_csv.describe())

print(train_csv.shape, test_csv.shape)  # (87159, 15) (46204, 14)
##################### train_csv와 test_csv 분리 ###############
# 1. 구분용 칼럼 추가
train_csv['is_train'] = 1
test_csv['is_train'] = 0
print(train_csv.shape, test_csv.shape) # (87159, 16) (46204, 15)

############## 범주형 데이터 라벨인코딩 하기 ############
combined = pd.concat([train_csv,test_csv], axis=0)
print(combined)
print(combined.shape) #[133363 rows x 16 columns]

aaa = pd.get_dummies(combined, columns=['Gender','Country','Race','Family_Background','Radiation_History',
                                        'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes',
                                        ], drop_first=True,
                                        dtype=int,
                                        )
print(aaa) # [133363 rows x 27 columns]
print(aaa.columns)
############## 상관계수 시작 ############
print(aaa.corr())

plt.figure(figsize=(5,12))
sns.heatmap(aaa.corr(), annot=True, cmap='coolwarm',fmt='.2f', cbar=True)
plt.title("Cancer와의 상관계수 히트맵")
plt.show
############## 상관계수 끝 ############
# 지울 칼럼들
# [1] ['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result'] # 상관, 피처임포턴스
# [2] ['Smoke_Smoker','Weight_Risk_obese','Diabetes_Yes'] # 상관관계는 없는데 피처임포턴스

# 그래서 우선 [1]의 컬럼을 5개 삭제한다.
drop_features = ["T3_Result","T4_Result","TSH_Result","Nodule_Size","Age"]
aaa = aaa.drop(columns=drop_features)

# print(aaa) #[133363 rows x 22 columns]

# 4. 다시 분리
train_csv = aaa[aaa['is_train'] == 1].drop(columns='is_train')
test_csv = aaa[aaa['is_train'] == 0].drop(columns='is_train')

print(train_csv.shape, test_csv.shape) # (87159, 21) (46204, 21)
print(train_csv.columns)
print(test_csv['Cancer']) # 전부 NaN
test_csv = test_csv.drop(['Cancer'],axis=1)
print(test_csv.shape) # (46204, 20)

############### x와 y 분리 #############
x = train_csv.drop(['Cancer'],axis=1) 
print(x) # [87159 rows x 20 columns]

y = train_csv['Cancer']
print(y.shape) # (87159,)

print(np.unique(y, return_counts=True)) # (array([0., 1.]), array([76700, 10459], dtype=int64))


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=334,stratify=y)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

print(x_train.shape, x_test.shape) # (78443, 20) (8716, 20)
print(y_train.shape, y_test.shape) # (78443,) (8716,)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state= 190)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(x_train.shape, y_train.shape) # (138060, 20) (138060,)
print(pd.Series(y_train).value_counts())
# 0.0    69030
# 1.0    69030

print(np.unique(y_train, return_counts=True)) #(array([0., 1.]), array([69030, 69030], dtype=int64))
print(np.unique(y_test, return_counts=True))

#2. 모델구성
early_stop = callback.EarlyStopping(
    rounds=200,
    metric_name='logloss',     # 모니터링할 평가 지표 /eval_metric 과 동일하게
    # data_name='validation_0',  # eval_set의 첫 번째 데이터 셋
    save_best=True             # 최적 모델 저장 옵션
    # AttributeError: 'best_iteration' is only defined when early stopping is used.
)

# 1. ModelCheckpoint를 위한 저장 경로 설정
# mcp_save_path = f'xgb_model_mcp_{timestamp}.json'

# mcp = callback.ModelCheckpoint(
#     filepath=mcp_save_path,
#     monitor='validation_0-logloss',
#     save_best_only=True,
#     maximize=False,
#     verbose=1 # 저장될 때 메시지 출력
# )

model = XGBClassifier(
    n_estimators=10000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=2,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=seed,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    callbacks=[early_stop],
)

# 훈련
model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    verbose=10,
)

# ✅ 제일 좋은 모델 저장
model.save_model(f'xgb_best_model_{timestamp}.json')


# 🎯 (1) 모델 전체 저장 (추천)
joblib.dump(model, f'xgb_model_{timestamp}.pkl')  # 저장
# 불러오기: model = joblib.load('xgb_model.pkl')


# 4. 평가, 예측
results = model.score(x_test,y_test)
(print('최종점수 :', results))

y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict[:10])

########### submission.csv 파일 만들기 // count 컬럼 값만 넣어주기
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

submission_csv['Cancer'] = y_submit
print(submission_csv[:10])

filename = f"/Users/jaewoo000/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/submission_{timestamp}.csv"
submission_csv.to_csv(filename)
print(f"Submission saved to: {filename}")

f1 = f1_score(y_test, y_predict)
print("f1 :", f1)

# # ==============================================
# # 1. 라이브러리 로딩 및 환경 정보
# # ==============================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import f1_score
# from xgboost import XGBClassifier
# from xgboost.callback import EarlyStopping
# import random
# from imblearn.over_sampling import SMOTE
# import datetime
# import joblib
# import optuna

# # Seed 고정
# seed = 190
# random.seed(seed)
# np.random.seed(seed)

# # 타임스탬프
# timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# # ==============================================
# # 2. 데이터 로딩
# # ==============================================
# path = './_data/dacon/cancer/'

# train_csv = pd.read_csv(path+'train.csv', index_col=0)
# test_csv = pd.read_csv(path+'test.csv', index_col=0)
# submission_csv = pd.read_csv(path+'sample_submission.csv', index_col=0)

# # train/test 결합 후 전처리
# train_csv['is_train'] = 1
# test_csv['is_train'] = 0
# combined = pd.concat([train_csv, test_csv], axis=0)

# # 원-핫 인코딩
# combined = pd.get_dummies(combined, columns=[
#     'Gender','Country','Race','Family_Background','Radiation_History',
#     'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes'
# ], drop_first=True, dtype=int)

# # 불필요 컬럼 제거
# drop_features = ["T3_Result","T4_Result","TSH_Result","Nodule_Size","Age"]
# combined = combined.drop(columns=drop_features)

# # train, test 재분리
# train_csv = combined[combined['is_train'] == 1].drop(columns='is_train')
# test_csv = combined[combined['is_train'] == 0].drop(columns=['is_train','Cancer'])

# x = train_csv.drop(['Cancer'], axis=1)
# y = train_csv['Cancer']

# # ==============================================
# # 3. Optuna objective 함수 정의
# # ==============================================
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
#         'random_state': seed,
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'use_label_encoder': False,
#     }

#     f1_scores = []
#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

#     for train_idx, valid_idx in skf.split(x, y):
#         x_train_fold, x_valid_fold = x.iloc[train_idx], x.iloc[valid_idx]
#         y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

#         # SMOTE 적용
#         smote = SMOTE(random_state=seed)
#         x_train_fold, y_train_fold = smote.fit_resample(x_train_fold, y_train_fold)

#         model = XGBClassifier(**params)

#         # EarlyStopping 콜백 정의
#         early_stop = EarlyStopping(rounds=200, save_best=True)

#         model.fit(
#             x_train_fold, y_train_fold,
#             eval_set=[(x_valid_fold, y_valid_fold)],
#             callbacks=[early_stop],
#             verbose=False
#         )

#         y_pred = model.predict(x_valid_fold)
#         f1 = f1_score(y_valid_fold, y_pred)
#         f1_scores.append(f1)

#     return np.mean(f1_scores)

# # ==============================================
# # 4. Optuna 최적화 실행
# # ==============================================
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)

# print("Best Trial:")
# trial = study.best_trial
# print("  f1_score: {}".format(trial.value))
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

# # ==============================================
# # 5. 최종 모델 학습 (best params)
# # ==============================================
# best_params = trial.params

# # train_test_split + SMOTE for final training
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=334, stratify=y)
# smote = SMOTE(random_state=seed)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# # EarlyStopping 콜백 정의
# early_stop_final = EarlyStopping(rounds=200, save_best=True)

# model = XGBClassifier(**best_params)

# model.fit(
#     x_train, y_train,
#     eval_set=[(x_test, y_test)],
#     callbacks=[early_stop_final],
#     verbose=10,
# )

# # ==============================================
# # 6. 모델 저장
# # ==============================================
# model.save_model(f'xgb_best_model_{timestamp}_optuna.json')
# joblib.dump(model, f'xgb_model_{timestamp}_optuna.pkl')

# # ==============================================
# # 7. 평가 및 submission 생성
# # ==============================================
# results = model.score(x_test, y_test)
# print('최종점수 :', results)

# y_predict = model.predict(x_test)
# f1 = f1_score(y_test, y_predict)
# print("f1 :", f1)

# # submission
# y_submit = model.predict(test_csv)
# submission_csv['Cancer'] = np.round(y_submit)

# filename = f"/Users/jaewoo000/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/submission_{timestamp}_optuna.csv"
# submission_csv.to_csv(filename)
# print(f"Submission saved to: {filename}")

