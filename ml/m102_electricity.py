
import os
import pandas as pd
import numpy as np
import random
import datetime
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import tensorflow as tf
import warnings
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler
import json
warnings.filterwarnings(action='ignore')

# Seed 고정
seed = 6054
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 경로 설정
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
buildinginfo = pd.read_csv(path + 'building_info.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
samplesub = pd.read_csv(path + 'sample_submission.csv')

# 결측치 처리
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    for col in ['일조(hr)', '일사(MJ/m2)']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
    return df

train = feature_engineering(train)
test = feature_engineering(test)
train = train.merge(buildinginfo, on='건물번호', how='left')
test = test.merge(buildinginfo, on='건물번호', how='left')

# 💡 [개선 전략] '일사량' 대리 변수 생성
# 1. 훈련 데이터에서 월(month)과 시간(hour)별 평균 일사량 계산
solar_proxy = train.groupby(['month', 'hour'])['일사(MJ/m2)'].mean().reset_index()
solar_proxy.rename(columns={'일사(MJ/m2)': 'expected_solar'}, inplace=True)

# 2. train과 test 데이터에 'expected_solar' 피처를 병합
train = train.merge(solar_proxy, on=['month', 'hour'], how='left')
test = test.merge(solar_proxy, on=['month', 'hour'], how='left')

# 병합 과정에서 발생할 수 있는 소량의 결측치는 0으로 채웁니다.
train['expected_solar'] = train['expected_solar'].fillna(0)
test['expected_solar'] = test['expected_solar'].fillna(0)


train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes

features = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'expected_solar'
]

target = '전력소비량(kWh)'

# Optuna 튜닝 함수들
def tune_xgb(trial, x_train, y_train, x_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'early_stopping_rounds': 50,
        'eval_metric': 'mae',
        'random_state': seed,
        'objective': 'reg:squarederror'
    }
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    pred = model.predict(x_val)
    smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
                    (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def tune_lgb(trial, x_train, y_train, x_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': seed,
        'objective': 'mae'
    }
    model = LGBMRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(x_val)
    smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
                    (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def tune_cat(trial, x_train, y_train, x_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_seed': seed,
        'loss_function': 'MAE',
        'verbose': 0
    }
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, verbose=0)
    pred = model.predict(x_val)
    smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
                    (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape

def objective(trial, oof_train, oof_val, y_train, y_val):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    ridge.fit(oof_train, y_train)
    preds = ridge.predict(oof_val)
    smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
                    (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
    return smape


def process_building_kfold(bno):
    print(f"🏢 건물번호 {bno} KFold 처리 중...")
    param_dir = os.path.join(path, "optuna_params")  # ✅ 추가
    os.makedirs(param_dir, exist_ok=True)   
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()
    x = train_b[features].values
    y = np.log1p(train_b[target].values)
    x_test = test_b[features].values

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    test_preds = []
    val_smapes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        print(f" - Fold {fold+1}")
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        # 기존 코드 지우고 아래로 교체
        model_key = f"{bno}_fold{fold+1}_xgb"
        xgb_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(xgb_param_path):
            with open(xgb_param_path, "r") as f:
                xgb_params = json.load(f)
        else:
            xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            xgb_study.optimize(lambda trial: tune_xgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
            xgb_params = xgb_study.best_params
            with open(xgb_param_path, "w") as f:
                json.dump(xgb_params, f)

        best_xgb = XGBRegressor(**xgb_params)
        best_xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

        model_key = f"{bno}_fold{fold+1}_lgb"
        lgb_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(lgb_param_path):
            with open(lgb_param_path, "r") as f:
                lgb_params = json.load(f)
        else:
            lgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            lgb_study.optimize(lambda trial: tune_lgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
            lgb_params = lgb_study.best_params
            with open(lgb_param_path, "w") as f:
                json.dump(lgb_params, f)

        best_lgb = LGBMRegressor(**lgb_params)
        best_lgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

        model_key = f"{bno}_fold{fold+1}_cat"
        cat_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(cat_param_path):
            with open(cat_param_path, "r") as f:
                cat_params = json.load(f)
        else:
            cat_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            cat_study.optimize(lambda trial: tune_cat(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
            cat_params = cat_study.best_params
            with open(cat_param_path, "w") as f:
                json.dump(cat_params, f)

        best_cat = CatBoostRegressor(**cat_params)
        best_cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50, verbose=0)

        # Stacking용 예측값 생성
        oof_train = np.vstack([
            best_xgb.predict(x_train_scaled),
            best_lgb.predict(x_train_scaled),
            best_cat.predict(x_train_scaled)
        ]).T
        oof_val = np.vstack([
            best_xgb.predict(x_val_scaled),
            best_lgb.predict(x_val_scaled),
            best_cat.predict(x_val_scaled)
        ]).T
        oof_test = np.vstack([
            best_xgb.predict(x_test_scaled),
            best_lgb.predict(x_test_scaled),
            best_cat.predict(x_test_scaled)
        ]).T

        # Ridge 튜닝 및 학습
        model_key = f"{bno}_fold{fold+1}_ridge"
        ridge_param_path = os.path.join(param_dir, f"{model_key}.json")

        if os.path.exists(ridge_param_path):
            with open(ridge_param_path, "r") as f:
                ridge_params = json.load(f)
        else:
            ridge_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            ridge_study.optimize(lambda trial: objective(trial, oof_train, oof_val, y_train, y_val), n_trials=30)
            ridge_params = ridge_study.best_params
            with open(ridge_param_path, "w") as f:
                json.dump(ridge_params, f)

        meta = Ridge(alpha=ridge_params['alpha'])
        meta.fit(oof_train, y_train)

        val_pred = meta.predict(oof_val)
        test_pred = meta.predict(oof_test)

        smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
                        (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))

        val_smapes.append(smape)
        test_preds.append(np.expm1(test_pred))

    # 평균 예측값과 SMAPE
    avg_test_pred = np.mean(test_preds, axis=0)
    avg_smape = np.mean(val_smapes)

    return avg_test_pred.tolist(), avg_smape


# 병렬 처리 실행 (KFold 적용)
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(process_building_kfold)(bno) for bno in train['건물번호'].unique()
)

# 결과 합치기
final_preds = []
val_smapes = []
for preds, smape in results:
    final_preds.extend(preds)
    val_smapes.append(smape)

samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
filename = f"submission_stack_optuna_{today}_SMAPE_일사_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")




#  ------------------------------현재 베스트!!!!!!!!!!!!!!!!!--------------------------------------------

# #아래꺼에 일사 추가. 잘 봐라 


# import os
# import pandas as pd
# import numpy as np
# import random
# import datetime
# import optuna
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgb
# import tensorflow as tf
# import warnings
# from joblib import Parallel, delayed
# from sklearn.model_selection import KFold

# warnings.filterwarnings(action='ignore')

# # Seed 고정
# seed = 707
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # 경로 설정
# BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# buildinginfo = pd.read_csv(path + 'building_info.csv')
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# samplesub = pd.read_csv(path + 'sample_submission.csv')

# # 결측치 처리
# for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
#     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # Feature Engineering
# def feature_engineering(df):
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'])
#     df['hour'] = df['일시'].dt.hour
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['month'] = df['일시'].dt.month
#     df['day'] = df['일시'].dt.day
#     df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
#     df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
#     for col in ['일조(hr)', '일사(MJ/m2)']:
#         if col in df.columns:
#             df[col] = df[col].fillna(0)
#     temp = df['기온(°C)']
#     humidity = df['습도(%)']
#     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
#     return df

# train = feature_engineering(train)
# test = feature_engineering(test)
# train = train.merge(buildinginfo, on='건물번호', how='left')
# test = test.merge(buildinginfo, on='건물번호', how='left')

# # 💡 [개선 전략] '일사량' 대리 변수 생성
# # 1. 훈련 데이터에서 월(month)과 시간(hour)별 평균 일사량 계산
# solar_proxy = train.groupby(['month', 'hour'])['일사(MJ/m2)'].mean().reset_index()
# solar_proxy.rename(columns={'일사(MJ/m2)': 'expected_solar'}, inplace=True)

# # 2. train과 test 데이터에 'expected_solar' 피처를 병합
# train = train.merge(solar_proxy, on=['month', 'hour'], how='left')
# test = test.merge(solar_proxy, on=['month', 'hour'], how='left')

# # 병합 과정에서 발생할 수 있는 소량의 결측치는 0으로 채웁니다.
# train['expected_solar'] = train['expected_solar'].fillna(0)
# test['expected_solar'] = test['expected_solar'].fillna(0)


# train['건물유형'] = train['건물유형'].astype('category').cat.codes
# test['건물유형'] = test['건물유형'].astype('category').cat.codes

# features = [
#     '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
#     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
#     'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'expected_solar'
# ]

# target = '전력소비량(kWh)'

# # Optuna 튜닝 함수들
# def tune_xgb(trial, x_train, y_train, x_val, y_val):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'early_stopping_rounds': 50,
#         'eval_metric': 'mae',
#         'random_state': seed,
#         'objective': 'reg:squarederror'
#     }
#     model = XGBRegressor(**params)
#     model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
#     pred = model.predict(x_val)
#     smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def tune_lgb(trial, x_train, y_train, x_val, y_val):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'random_state': seed,
#         'objective': 'mae'
#     }
#     model = LGBMRegressor(**params)
#     model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
#               callbacks=[lgb.early_stopping(50, verbose=False)])
#     pred = model.predict(x_val)
#     smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def tune_cat(trial, x_train, y_train, x_val, y_val):
#     params = {
#         'iterations': trial.suggest_int('iterations', 300, 1000),
#         'depth': trial.suggest_int('depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
#         'random_seed': seed,
#         'loss_function': 'MAE',
#         'verbose': 0
#     }
#     model = CatBoostRegressor(**params)
#     model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, verbose=0)
#     pred = model.predict(x_val)
#     smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def objective(trial, oof_train, oof_val, y_train, y_val):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(oof_train, y_train)
#     preds = ridge.predict(oof_val)
#     smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape


# def process_building_kfold(bno):
#     print(f"🏢 건물번호 {bno} KFold 처리 중...")
#     train_b = train[train['건물번호'] == bno].copy()
#     test_b = test[test['건물번호'] == bno].copy()
#     x = train_b[features].values
#     y = np.log1p(train_b[target].values)
#     x_test = test_b[features].values

#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
#     test_preds = []
#     val_smapes = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
#         print(f" - Fold {fold+1}")
#         x_train, x_val = x[train_idx], x[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         scaler = StandardScaler()
#         x_train_scaled = scaler.fit_transform(x_train)
#         x_val_scaled = scaler.transform(x_val)
#         x_test_scaled = scaler.transform(x_test)

#         # 각 모델 튜닝 및 훈련
#         xgb_study = optuna.create_study(direction="minimize")
#         xgb_study.optimize(lambda trial: tune_xgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
#         best_xgb = XGBRegressor(**xgb_study.best_params)
#         best_xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

#         lgb_study = optuna.create_study(direction="minimize")
#         lgb_study.optimize(lambda trial: tune_lgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
#         best_lgb = LGBMRegressor(**lgb_study.best_params)
#         best_lgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         cat_study = optuna.create_study(direction="minimize")
#         cat_study.optimize(lambda trial: tune_cat(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
#         best_cat = CatBoostRegressor(**cat_study.best_params)
#         best_cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50, verbose=0)

#         # Stacking용 예측값 생성
#         oof_train = np.vstack([
#             best_xgb.predict(x_train_scaled),
#             best_lgb.predict(x_train_scaled),
#             best_cat.predict(x_train_scaled)
#         ]).T
#         oof_val = np.vstack([
#             best_xgb.predict(x_val_scaled),
#             best_lgb.predict(x_val_scaled),
#             best_cat.predict(x_val_scaled)
#         ]).T
#         oof_test = np.vstack([
#             best_xgb.predict(x_test_scaled),
#             best_lgb.predict(x_test_scaled),
#             best_cat.predict(x_test_scaled)
#         ]).T

#         # Ridge 튜닝 및 학습
#         ridge_study = optuna.create_study(direction="minimize")
#         ridge_study.optimize(lambda trial: objective(trial, oof_train, oof_val, y_train, y_val), n_trials=30)
#         meta = Ridge(alpha=ridge_study.best_params['alpha'])
#         meta.fit(oof_train, y_train)

#         val_pred = meta.predict(oof_val)
#         test_pred = meta.predict(oof_test)

#         smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
#                         (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))

#         val_smapes.append(smape)
#         test_preds.append(np.expm1(test_pred))

#     # 평균 예측값과 SMAPE
#     avg_test_pred = np.mean(test_preds, axis=0)
#     avg_smape = np.mean(val_smapes)

#     return avg_test_pred.tolist(), avg_smape


# # 병렬 처리 실행 (KFold 적용)
# results = Parallel(n_jobs=-1, backend='loky')(
#     delayed(process_building_kfold)(bno) for bno in train['건물번호'].unique()
# )

# # 결과 합치기
# final_preds = []
# val_smapes = []
# for preds, smape in results:
#     final_preds.extend(preds)
#     val_smapes.append(smape)

# samplesub['answer'] = final_preds
# today = datetime.datetime.now().strftime('%Y%m%d')
# avg_smape = np.mean(val_smapes)
# filename = f"submission_stack_optuna_{today}_SMAPE_일사_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")






















#얘는 모두 옵튜나 한거 !! 성능 매우 좋음

# import os
# import pandas as pd
# import numpy as np
# import random
# import datetime
# import optuna
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgb
# import tensorflow as tf
# import warnings
# from joblib import Parallel, delayed

# warnings.filterwarnings(action='ignore')

# # Seed 고정
# seed = 4464
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # 경로 설정
# BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# buildinginfo = pd.read_csv(path + 'building_info.csv')
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# samplesub = pd.read_csv(path + 'sample_submission.csv')

# # 결측치 처리
# for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
#     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # Feature Engineering
# def feature_engineering(df):
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'])
#     df['hour'] = df['일시'].dt.hour
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['month'] = df['일시'].dt.month
#     df['day'] = df['일시'].dt.day
#     df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
#     df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
#     for col in ['일조(hr)', '일사(MJ/m2)']:
#         if col in df.columns:
#             df[col] = df[col].fillna(0)
#     temp = df['기온(°C)']
#     humidity = df['습도(%)']
#     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
#     return df

# train = feature_engineering(train)
# test = feature_engineering(test)
# train = train.merge(buildinginfo, on='건물번호', how='left')
# test = test.merge(buildinginfo, on='건물번호', how='left')
# train['건물유형'] = train['건물유형'].astype('category').cat.codes
# test['건물유형'] = test['건물유형'].astype('category').cat.codes

# features = [
#     '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
#     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
#     'is_working_hours', 'sin_hour', 'cos_hour', 'DI'
# ]

# target = '전력소비량(kWh)'

# # Optuna 튜닝 함수들
# def tune_xgb(trial, x_train, y_train, x_val, y_val):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'early_stopping_rounds': 50,
#         'eval_metric': 'mae',
#         'random_state': seed,
#         'objective': 'reg:squarederror'
#     }
#     model = XGBRegressor(**params)
#     model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
#     pred = model.predict(x_val)
#     smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def tune_lgb(trial, x_train, y_train, x_val, y_val):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'random_state': seed,
#         'objective': 'mae'
#     }
#     model = LGBMRegressor(**params)
#     model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
#               callbacks=[lgb.early_stopping(50, verbose=False)])
#     pred = model.predict(x_val)
#     smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def tune_cat(trial, x_train, y_train, x_val, y_val):
#     params = {
#         'iterations': trial.suggest_int('iterations', 300, 1000),
#         'depth': trial.suggest_int('depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
#         'random_seed': seed,
#         'loss_function': 'MAE',
#         'verbose': 0
#     }
#     model = CatBoostRegressor(**params)
#     model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, verbose=0)
#     pred = model.predict(x_val)
#     smape = np.mean(200 * np.abs(np.expm1(pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def objective(trial, oof_train, oof_val, y_train, y_val):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(oof_train, y_train)
#     preds = ridge.predict(oof_val)
#     smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
#     return smape

# def process_building(bno):
#     print(f"🏢 건물번호 {bno} 처리 중...") 
#     train_b = train[train['건물번호'] == bno].copy()
#     test_b = test[test['건물번호'] == bno].copy()
#     x = train_b[features]
#     y = np.log1p(train_b[target])
#     x_test = test_b[features]

#     x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_val_scaled = scaler.transform(x_val)
#     x_test_scaled = scaler.transform(x_test)

#     xgb_study = optuna.create_study(direction="minimize")
#     xgb_study.optimize(lambda trial: tune_xgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
#     best_xgb = XGBRegressor(**xgb_study.best_params)
#     best_xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

#     lgb_study = optuna.create_study(direction="minimize")
#     lgb_study.optimize(lambda trial: tune_lgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
#     best_lgb = LGBMRegressor(**lgb_study.best_params)
#     best_lgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

#     cat_study = optuna.create_study(direction="minimize")
#     cat_study.optimize(lambda trial: tune_cat(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=20)
#     best_cat = CatBoostRegressor(**cat_study.best_params)
#     best_cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50, verbose=0)

#     oof_train = np.vstack([
#         best_xgb.predict(x_train_scaled),
#         best_lgb.predict(x_train_scaled),
#         best_cat.predict(x_train_scaled)
#     ]).T
#     oof_val = np.vstack([
#         best_xgb.predict(x_val_scaled),
#         best_lgb.predict(x_val_scaled),
#         best_cat.predict(x_val_scaled)
#     ]).T
#     oof_test = np.vstack([
#         best_xgb.predict(x_test_scaled),
#         best_lgb.predict(x_test_scaled),
#         best_cat.predict(x_test_scaled)
#     ]).T

#     ridge_study = optuna.create_study(direction="minimize")
#     ridge_study.optimize(lambda trial: objective(trial, oof_train, oof_val, y_train, y_val), n_trials=30)
#     meta = Ridge(alpha=ridge_study.best_params['alpha'])
#     meta.fit(oof_train, y_train)

#     val_pred = meta.predict(oof_val)
#     test_pred = meta.predict(oof_test)

#     smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
#                     (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))

#     return np.expm1(test_pred).tolist(), smape

# # 병렬 처리 실행
# results = Parallel(n_jobs=-1, backend='loky')(
#     delayed(process_building)(bno) for bno in train['건물번호'].unique()
# )

# # 결과 합치기
# final_preds = []
# val_smapes = []
# for preds, smape in results:
#     final_preds.extend(preds)
#     val_smapes.append(smape)

# samplesub['answer'] = final_preds
# today = datetime.datetime.now().strftime('%Y%m%d')
# avg_smape = np.mean(val_smapes)
# filename = f"submission_stack_optuna_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")
























# 최종모델만 optuna - 모두 옵튜나 한게 성능 더 좋음. 이건 혹시나 해서 갖고있는



# # import os
# # import pandas as pd
# # import numpy as np
# # import random
# # import datetime
# # import optuna
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import Ridge
# # from sklearn.metrics import mean_absolute_error
# # from xgboost import XGBRegressor
# # from lightgbm import LGBMRegressor
# # from catboost import CatBoostRegressor
# # import lightgbm as lgb
# # import tensorflow as tf
# # import warnings
# # warnings.filterwarnings(action='ignore')


# # #best 707
# # # Seed 고정
# # seed = 6054
# # random.seed(seed)
# # np.random.seed(seed)
# # tf.random.set_seed(seed)

# # # 경로 설정
# # if os.path.exists('/workspace/TensorJae/Study25/'):
# #     BASE_PATH = '/workspace/TensorJae/Study25/'
# # else:
# #     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# # path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# # buildinginfo = pd.read_csv(path + 'building_info.csv')
# # train = pd.read_csv(path + 'train.csv')
# # test = pd.read_csv(path + 'test.csv')
# # samplesub = pd.read_csv(path + 'sample_submission.csv')

# # # 결측치 처리
# # for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
# #     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # # Feature Engineering
# # def feature_engineering(df):
# #     df = df.copy()
# #     df['일시'] = pd.to_datetime(df['일시'])
# #     df['hour'] = df['일시'].dt.hour
# #     df['dayofweek'] = df['일시'].dt.dayofweek
# #     df['month'] = df['일시'].dt.month
# #     df['day'] = df['일시'].dt.day
# #     df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
# #     df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
# #     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
# #     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
# #     for col in ['일조(hr)', '일사(MJ/m2)']:
# #         if col in df.columns:
# #             df[col] = df[col].fillna(0)
# #     temp = df['기온(°C)']
# #     humidity = df['습도(%)']
# #     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
# #     return df

# # train = feature_engineering(train)
# # test = feature_engineering(test)
# # train = train.merge(buildinginfo, on='건물번호', how='left')
# # test = test.merge(buildinginfo, on='건물번호', how='left')
# # train['건물유형'] = train['건물유형'].astype('category').cat.codes
# # test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # features = [
# #     '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
# #     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
# #     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
# #     'is_working_hours', 'sin_hour', 'cos_hour', 'DI'
# # ]

# # target = '전력소비량(kWh)'
# # final_preds = []
# # val_smapes = []

# # # Optuna 튜닝 함수
# # def objective(trial, oof_train, oof_val, y_train, y_val):
# #     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
# #     ridge = Ridge(alpha=alpha)
# #     ridge.fit(oof_train, y_train)
# #     preds = ridge.predict(oof_val)
# #     smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
# #                     (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
# #     return smape

# # # 건물별 모델 학습
# # building_ids = train['건물번호'].unique()
# # for bno in building_ids:
# #     print(f"🏢 건물번호 {bno} 처리 중...")

# #     train_b = train[train['건물번호'] == bno].copy()
# #     test_b = test[test['건물번호'] == bno].copy()
# #     x = train_b[features]
# #     y = np.log1p(train_b[target])
# #     x_test_final = test_b[features]

# #     x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

# #     scaler = StandardScaler()
# #     x_train_scaled = scaler.fit_transform(x_train)
# #     x_val_scaled = scaler.transform(x_val)
# #     x_test_final_scaled = scaler.transform(x_test_final)

# #     # Base 모델 학습
# #     xgb = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
# #                        random_state=seed, early_stopping_rounds=50, objective='reg:squarederror')
# #     xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

# #     lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
# #                               random_state=seed, objective='mae')
# #     lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
# #                   callbacks=[lgb.early_stopping(50, verbose=False)])

# #     cat = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
# #                             random_seed=seed, verbose=0, loss_function='MAE')
# #     cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

# #     # 스태킹
# #     oof_train_lvl1 = np.vstack([
# #         xgb.predict(x_train_scaled),
# #         lgb_model.predict(x_train_scaled),
# #         cat.predict(x_train_scaled)
# #     ]).T
# #     oof_val_lvl1 = np.vstack([
# #         xgb.predict(x_val_scaled),
# #         lgb_model.predict(x_val_scaled),
# #         cat.predict(x_val_scaled)
# #     ]).T
# #     oof_test_lvl1 = np.vstack([
# #         xgb.predict(x_test_final_scaled),
# #         lgb_model.predict(x_test_final_scaled),
# #         cat.predict(x_test_final_scaled)
# #     ]).T

# #     # Optuna로 Ridge 튜닝
# #     study = optuna.create_study(direction="minimize")
# #     study.optimize(lambda trial: objective(trial, oof_train_lvl1, oof_val_lvl1, y_train, y_val), n_trials=30)
# #     best_alpha = study.best_params['alpha']
# #     meta_model = Ridge(alpha=best_alpha)
# #     meta_model.fit(oof_train_lvl1, y_train)

# #     val_pred = meta_model.predict(oof_val_lvl1)
# #     test_pred = meta_model.predict(oof_test_lvl1)

# #     val_smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
# #                         (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))
# #     val_smapes.append(val_smape)

# #     pred = np.expm1(test_pred)
# #     final_preds.extend(pred)

# # # 결과 저장
# # samplesub['answer'] = final_preds
# # today = datetime.datetime.now().strftime('%Y%m%d')
# # avg_smape = np.mean(val_smapes)
# # score_str = f"{avg_smape:.4f}".replace('.', '_')
# # filename = f"submission_stack_optuna_{today}_SMAPE_{score_str}_{seed}.csv"
# # samplesub.to_csv(os.path.join(path, filename), index=False)

# # print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# # print(f"📁 저장 완료 → {filename}")



