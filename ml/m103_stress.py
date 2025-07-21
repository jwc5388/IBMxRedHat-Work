# # import os
# # import pandas as pd
# # import numpy as np
# # import random
# # import datetime
# # import optuna
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # from sklearn.model_selection import train_test_split
# # from sklearn.impute import SimpleImputer
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.linear_model import Ridge
# # from sklearn.metrics import mean_absolute_error
# # from xgboost import XGBRegressor
# # from lightgbm import LGBMRegressor
# # from catboost import CatBoostRegressor
# # import lightgbm as lgb
# # import warnings
# # warnings.filterwarnings('ignore')

# # # Seed 고정
# # seed = 42
# # random.seed(seed)
# # np.random.seed(seed)

# # # 경로 설정
# # BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# # path = os.path.join(BASE_PATH, '_data/dacon/stress/')

# # # 데이터 로딩
# # train = pd.read_csv(path + 'train.csv')
# # test = pd.read_csv(path + 'test.csv')
# # submission = pd.read_csv(path + 'sample_submission.csv')

# # # ✅ 결측치 처리
# # train['medical_history'] = train['medical_history'].fillna('none')
# # test['medical_history'] = test['medical_history'].fillna('none')
# # train['family_medical_history'] = train['family_medical_history'].fillna('none')
# # test['family_medical_history'] = test['family_medical_history'].fillna('none')

# # # ✅ 질환별 이진 피처 생성
# # diseases = ['heart disease', 'high blood pressure', 'diabetes']
# # for disease in diseases:
# #     train[f'med_hist_{disease}'] = train['medical_history'].apply(lambda x: int(disease in x))
# #     test[f'med_hist_{disease}'] = test['medical_history'].apply(lambda x: int(disease in x))
# #     train[f'fam_hist_{disease}'] = train['family_medical_history'].apply(lambda x: int(disease in x))
# #     test[f'fam_hist_{disease}'] = test['family_medical_history'].apply(lambda x: int(disease in x))

# # # ✅ sleep_pattern → One-hot 인코딩
# # train['sleep_pattern'] = train['sleep_pattern'].fillna('unknown')
# # test['sleep_pattern'] = test['sleep_pattern'].fillna('unknown')
# # for sp in train['sleep_pattern'].unique():
# #     train[f'sleep_{sp}'] = (train['sleep_pattern'] == sp).astype(int)
# #     test[f'sleep_{sp}'] = (test['sleep_pattern'] == sp).astype(int)

# # # ✅ edu_level → 순서형 인코딩
# # edu_order = {
# #     'high school diploma': 0,
# #     'bachelors degree': 1,
# #     'graduate degree': 2,
# #     'missing': -1
# # }
# # train['edu_level'] = train['edu_level'].fillna('missing')
# # test['edu_level'] = test['edu_level'].fillna('missing')
# # train['edu_level_ord'] = train['edu_level'].map(edu_order)
# # test['edu_level_ord'] = test['edu_level'].map(edu_order)

# # # ✅ mean_working → 중앙값 대체
# # imp = SimpleImputer(strategy='median')
# # train['mean_working'] = imp.fit_transform(train[['mean_working']])
# # test['mean_working'] = imp.transform(test[['mean_working']])

# # # ✅ 파생 변수 생성
# # for df in [train, test]:
# #     df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
# #     df['bp_diff'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
# #     df['cholesterol_high'] = (df['cholesterol'] > 240).astype(int)
# #     df['glucose_high'] = (df['glucose'] > 126).astype(int)
# #     df['overwork'] = (df['mean_working'] > 52).astype(int)

# # # ✅ 범주형 인코딩
# # categorical_cols = train.select_dtypes(include='object').columns.drop('ID')
# # for col in categorical_cols:
# #     le = LabelEncoder()
# #     le.fit(train[col])
# #     test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'unknown')
# #     le.classes_ = np.append(le.classes_, 'unknown') if 'unknown' not in le.classes_ else le.classes_
# #     train[col] = le.transform(train[col])
# #     test[col] = le.transform(test[col])

# # # ✅ Feature/Target
# # x = train.drop(['ID', 'stress_score', 'medical_history', 'family_medical_history', 'sleep_pattern', 'edu_level'], axis=1)
# # y = train['stress_score']
# # x_test_final = test.drop(['ID', 'medical_history', 'family_medical_history', 'sleep_pattern', 'edu_level'], axis=1)

# # # Train/Val Split
# # x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# # # Scaling
# # scaler = StandardScaler()
# # x_train_scaled = scaler.fit_transform(x_train)
# # x_val_scaled = scaler.transform(x_val)
# # x_test_scaled = scaler.transform(x_test_final)

# # # Base Models
# # xgb = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
# #                    random_state=seed, early_stopping_rounds=50, objective='reg:squarederror')
# # xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

# # lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
# #                           random_state=seed, objective='mae')
# # lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
# #               callbacks=[lgb.early_stopping(50, verbose=False)])

# # cat = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
# #                         random_seed=seed, verbose=0, loss_function='MAE')
# # cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

# # # ✅ Feature Importance 시각화
# # importances = xgb.feature_importances_
# # feat_names = x.columns
# # feat_imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
# # feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)

# # plt.figure(figsize=(10, 8))
# # sns.barplot(x='importance', y='feature', data=feat_imp_df.head(20))
# # plt.title('Top 20 Feature Importances (XGBoost)')
# # plt.tight_layout()
# # plt.show()

# # # Stacking
# # oof_train = np.vstack([
# #     xgb.predict(x_train_scaled),
# #     lgb_model.predict(x_train_scaled),
# #     cat.predict(x_train_scaled)
# # ]).T
# # oof_val = np.vstack([
# #     xgb.predict(x_val_scaled),
# #     lgb_model.predict(x_val_scaled),
# #     cat.predict(x_val_scaled)
# # ]).T
# # oof_test = np.vstack([
# #     xgb.predict(x_test_scaled),
# #     lgb_model.predict(x_test_scaled),
# #     cat.predict(x_test_scaled)
# # ]).T

# # # Optuna Ridge Meta Model
# # def objective(trial):
# #     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
# #     ridge = Ridge(alpha=alpha)
# #     ridge.fit(oof_train, y_train)
# #     preds = ridge.predict(oof_val)
# #     return mean_absolute_error(y_val, preds)

# # study = optuna.create_study(direction="minimize")
# # study.optimize(objective, n_trials=30)
# # best_alpha = study.best_params['alpha']

# # meta_model = Ridge(alpha=best_alpha)
# # meta_model.fit(oof_train, y_train)

# # val_pred = meta_model.predict(oof_val)
# # test_pred = meta_model.predict(oof_test)

# # # 저장
# # submission['stress_score'] = test_pred
# # mae_score = mean_absolute_error(y_val, val_pred)
# # today = datetime.datetime.now().strftime('%Y%m%d')
# # score_str = f"{mae_score:.4f}".replace('.', '_')
# # filename = f'submission_stack_preprocessed_{today}_MAE_{score_str}_{seed}.csv'
# # submission.to_csv(os.path.join(path, filename), index=False)

# # print(f"\n✅ 전처리 완료 & 모델 MAE: {mae_score:.4f}")
# # print(f"📁 파일 저장 완료: {filename}")






















import os
import pandas as pd
import numpy as np
import random
import datetime
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Seed 고정
seed = 444
random.seed(seed)
np.random.seed(seed)

# 경로 설정
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/stress/')

# 데이터 로딩
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

# ✅ 결측치 처리
for col in ['medical_history', 'family_medical_history']:
    train[col] = train[col].fillna('none')
    test[col] = test[col].fillna('none')

# ✅ 질환별 이진 피처 생성
diseases = ['heart disease', 'high blood pressure', 'diabetes']
for disease in diseases:
    train[f'med_hist_{disease}'] = train['medical_history'].apply(lambda x: int(disease in x))
    test[f'med_hist_{disease}'] = test['medical_history'].apply(lambda x: int(disease in x))
    train[f'fam_hist_{disease}'] = train['family_medical_history'].apply(lambda x: int(disease in x))
    test[f'fam_hist_{disease}'] = test['family_medical_history'].apply(lambda x: int(disease in x))

# ✅ sleep_pattern → One-hot 인코딩
train['sleep_pattern'] = train['sleep_pattern'].fillna('unknown')
test['sleep_pattern'] = test['sleep_pattern'].fillna('unknown')
for sp in train['sleep_pattern'].unique():
    train[f'sleep_{sp}'] = (train['sleep_pattern'] == sp).astype(int)
    test[f'sleep_{sp}'] = (test['sleep_pattern'] == sp).astype(int)

# ✅ edu_level → 순서형 인코딩
edu_order = {'high school diploma': 0, 'bachelors degree': 1, 'graduate degree': 2, 'missing': -1}
train['edu_level'] = train['edu_level'].fillna('missing')
test['edu_level'] = test['edu_level'].fillna('missing')
train['edu_level_ord'] = train['edu_level'].map(edu_order)
test['edu_level_ord'] = test['edu_level'].map(edu_order)

# ✅ mean_working → 중앙값 대체
imp = SimpleImputer(strategy='median')
train['mean_working'] = imp.fit_transform(train[['mean_working']])
test['mean_working'] = imp.transform(test[['mean_working']])

# ✅ 파생 변수 생성
for df in [train, test]:
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['bp_diff'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
    df['cholesterol_high'] = (df['cholesterol'] > 240).astype(int)
    df['glucose_high'] = (df['glucose'] > 126).astype(int)
    df['overwork'] = (df['mean_working'] > 52).astype(int)

# ✅ 범주형 인코딩
categorical_cols = train.select_dtypes(include='object').columns.drop('ID')
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train[col])
    test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'unknown')
    le.classes_ = np.append(le.classes_, 'unknown') if 'unknown' not in le.classes_ else le.classes_
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
dfs = []
for gender in [0, 1]:
    print(f"\n👨‍⚕️ 성별 {gender} 모델 학습 시작...")

    train_gender = train[train['gender'] == gender].copy()
    test_gender = test[test['gender'] == gender].copy()

    y = train_gender['stress_score']
    x = train_gender.drop(['ID', 'stress_score', 'medical_history', 'family_medical_history', 'sleep_pattern', 'edu_level'], axis=1)
    x_test_final = test_gender.drop(['ID', 'medical_history', 'family_medical_history', 'sleep_pattern', 'edu_level'], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test_final)

    # 각 모델 학습
    xgb = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                       random_state=seed, early_stopping_rounds=50, objective='reg:squarederror')
    xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)
    xgb_pred_val = xgb.predict(x_val_scaled)
    xgb_pred_test = xgb.predict(x_test_scaled)
    xgb_mae = mean_absolute_error(y_val, xgb_pred_val)

    lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                              random_state=seed, objective='mae')
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_pred_val = lgb_model.predict(x_val_scaled)
    lgb_pred_test = lgb_model.predict(x_test_scaled)
    lgb_mae = mean_absolute_error(y_val, lgb_pred_val)

    cat = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                            random_seed=seed, verbose=0, loss_function='MAE')
    cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)
    cat_pred_val = cat.predict(x_val_scaled)
    cat_pred_test = cat.predict(x_test_scaled)
    cat_mae = mean_absolute_error(y_val, cat_pred_val)

    # 가장 좋은 모델 선택
    best_model_idx = np.argmin([xgb_mae, lgb_mae, cat_mae])
    best_preds_test = [xgb_pred_test, lgb_pred_test, cat_pred_test][best_model_idx]
    best_model_name = ['XGBoost', 'LightGBM', 'CatBoost'][best_model_idx]
    best_score = [xgb_mae, lgb_mae, cat_mae][best_model_idx]

    print(f"✅ 성별 {gender} BEST MODEL: {best_model_name} (MAE: {best_score:.4f})")

    temp = pd.DataFrame({'ID': test_gender['ID'], 'stress_score': best_preds_test})
    dfs.append(temp)

# 최종 제출
final_submission = pd.concat(dfs).sort_values(by='ID')
today = datetime.datetime.now().strftime('%Y%m%d')
filename = f'submission_best_single_model_{today}_{seed}.csv'
final_submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n📁 성별별 베스트 모델 저장 완료: {filename}")