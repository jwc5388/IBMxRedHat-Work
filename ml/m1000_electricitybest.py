

# # #0  ------------------------------현재 베스트!!!!!!!!!!!!!!!!!--------------------------------------------
# # #모두 옵튜나. + 일사. (왔다갔다 하고 있음) trials 바꿈 출력 지표 추가 
# # #아래꺼에 일사 추가. trials 20 설정이 현재 최고. 이거 뒤바꾸기


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
# from sklearn.metrics import mean_absolute_error, r2_score

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
# def feature_engineering(df, is_train=True):
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'])
#     df['hour'] = df['일시'].dt.hour
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['month'] = df['일시'].dt.month
#     df['day'] = df['일시'].dt.day
#     df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
#     df['is_working_hours'] = df['hour'].between(9, 18).astype(int)
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
#     if is_train:
#         for col in ['일조(hr)', '일사(MJ/m2)']:
#             df[col] = df[col].fillna(0)
#     else:
#         df['일조(hr)'] = 0
#         df['일사(MJ/m2)'] = 0

#     temp = df['기온(°C)']
#     humidity = df['습도(%)']
#     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
#     return df

# # 전처리 적용
# train = feature_engineering(train, is_train=True)
# test = feature_engineering(test, is_train=False)

# # 메타 정보 병합
# train = train.merge(buildinginfo, on='건물번호', how='left')
# test = test.merge(buildinginfo, on='건물번호', how='left')

# # 범주형 인코딩
# train['건물유형'] = train['건물유형'].astype('category').cat.codes
# test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # 파생 변수 생성
# for df in [train, test]:
#     df['cooling_area_ratio'] = df['냉방면적(m2)'] / df['연면적(m2)']
#     df['has_pv'] = (df['태양광용량(kW)'] > 0).astype(int)
#     df['has_ess'] = (df['ESS저장용량(kWh)'] > 0).astype(int)
    
    
# features = [
#     '건물유형', '연면적(m2)', '냉방면적(m2)', 
#     '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
#     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
#     'is_working_hours', 'sin_hour', 'cos_hour', 'DI',
#     'cooling_area_ratio', 'has_pv', 'has_ess'
# ]

# target = '전력소비량(kWh)'
# final_preds = []
# val_smapes = []



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

# # # # 💡 [개선 전략] '일사량' 대리 변수 생성
# # # # 1. 훈련 데이터에서 월(month)과 시간(hour)별 평균 일사량 계산
# # # solar_proxy = train.groupby(['month', 'hour'])['일사(MJ/m2)'].mean().reset_index()
# # # solar_proxy.rename(columns={'일사(MJ/m2)': 'expected_solar'}, inplace=True)

# # # # 2. train과 test 데이터에 'expected_solar' 피처를 병합
# # # train = train.merge(solar_proxy, on=['month', 'hour'], how='left')
# # # test = test.merge(solar_proxy, on=['month', 'hour'], how='left')

# # # # 병합 과정에서 발생할 수 있는 소량의 결측치는 0으로 채웁니다.
# # # train['expected_solar'] = train['expected_solar'].fillna(0)
# # # test['expected_solar'] = test['expected_solar'].fillna(0)


# # train['건물유형'] = train['건물유형'].astype('category').cat.codes
# # test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # features = [
# #     '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
# #     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
# #     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
# #     'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 
# #     # 'expected_solar'
# # ]

# # target = '전력소비량(kWh)'

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
#     maes, r2s, stds = [], [], []   # ✅ 지표 수집용 리스트

#     for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
#         print(f" - Fold {fold+1}")
#         x_train, x_val = x[train_idx], x[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         scaler = StandardScaler()
#         x_train_scaled = scaler.fit_transform(x_train)
#         x_val_scaled = scaler.transform(x_val)
#         x_test_scaled = scaler.transform(x_test)

#         # 모델 튜닝 및 학습
#         xgb_study = optuna.create_study(direction="minimize")
#         xgb_study.optimize(lambda trial: tune_xgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=30)
#         best_xgb = XGBRegressor(**xgb_study.best_params)
#         best_xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

#         lgb_study = optuna.create_study(direction="minimize")
#         lgb_study.optimize(lambda trial: tune_lgb(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=30)
#         best_lgb = LGBMRegressor(**lgb_study.best_params)
#         best_lgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         cat_study = optuna.create_study(direction="minimize")
#         cat_study.optimize(lambda trial: tune_cat(trial, x_train_scaled, y_train, x_val_scaled, y_val), n_trials=30)
#         best_cat = CatBoostRegressor(**cat_study.best_params)
#         best_cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50, verbose=0)

#         # Stacking용 예측값
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

#         # Ridge
#         ridge_study = optuna.create_study(direction="minimize")
#         ridge_study.optimize(lambda trial: objective(trial, oof_train, oof_val, y_train, y_val), n_trials=30)
#         meta = Ridge(alpha=ridge_study.best_params['alpha'])
#         meta.fit(oof_train, y_train)

#         val_pred = meta.predict(oof_val)
#         test_pred = meta.predict(oof_test)

#         smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
#                         (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#         mae = mean_absolute_error(np.expm1(y_val), np.expm1(val_pred))
#         r2 = r2_score(np.expm1(y_val), np.expm1(val_pred))
#         test_std = np.std(test_pred)

#         val_smapes.append(smape)
#         maes.append(mae)
#         r2s.append(r2)
#         stds.append(test_std)
#         test_preds.append(np.expm1(test_pred))

#     # 평균 계산
#     avg_test_pred = np.mean(test_preds, axis=0)
#     avg_smape = np.mean(val_smapes)
#     avg_mae = np.mean(maes)
#     avg_r2 = np.mean(r2s)
#     avg_test_std = np.mean(stds)

#     return avg_test_pred.tolist(), avg_smape, avg_mae, avg_r2, avg_test_std


# # 병렬 처리 실행 (KFold 적용)
# results = Parallel(n_jobs=-1, backend='loky')(
#     delayed(process_building_kfold)(bno) for bno in train['건물번호'].unique()
# )

# # 결과 합치기
# final_preds = []
# val_smapes = []
# maes, r2s, test_stds = [], [], []

# for preds, smape, mae, r2, std in results:
#     final_preds.extend(preds)
#     val_smapes.append(smape)
#     maes.append(mae)
#     r2s.append(r2)
#     test_stds.append(std)

# samplesub['answer'] = final_preds
# today = datetime.datetime.now().strftime('%Y%m%d')

# # ✅ 평균 지표 계산
# avg_smape = np.mean(val_smapes)
# smape_std = np.std(val_smapes)
# avg_mae = np.mean(maes)
# mae_std = np.std(maes)
# avg_r2 = np.mean(r2s)
# r2_std = np.std(r2s)
# avg_std = np.mean(test_stds)

# # ✅ 저장
# filename = f"submission_stack_optuna_{today}_SMAPE_일사_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# # ✅ 최종 성능 출력
# print("\n📊 최종 평가 지표 요약")
# print(f"✅ 평균 SMAPE (전체 건물): {avg_smape:.4f} ± {smape_std:.4f}")
# print(f"📐 평균 MAE: {avg_mae:.4f} ± {mae_std:.4f}")
# print(f"📐 평균 R²: {avg_r2:.4f} ± {r2_std:.4f}")
# print(f"📐 예측값 표준편차 (test 기준): {avg_std:.4f}")
# print(f"📁 저장 완료 → {filename}")
























# -*- coding: utf-8 -*-
# Optuna 파라미터 저장/로드 유지, 표준 컬럼 기반 전처리 일원화 버전

import os
import json
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from optuna.samplers import TPESampler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import tensorflow as tf

warnings.filterwarnings("ignore")

# ==============================
# 0) 시드 / 경로
# ==============================
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
    else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
path = os.path.join(BASE_PATH, "_data/dacon/electricity/")

buildinginfo = pd.read_csv(os.path.join(path, "building_info.csv"))
train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "test.csv"))
samplesub = pd.read_csv(os.path.join(path, "sample_submission.csv"))


# === 0) 옵션: building_info 병합 (있으면 병합, 없으면 넘어감)
have_bi = 'buildinginfo' in globals() or 'building_info' in globals()
if 'buildinginfo' in globals():
    bi = buildinginfo.copy()
else:
    bi = None

if bi is not None:
    # 설비 용량은 '-' → 0, 숫자로 캐스팅
    for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
        if col in bi.columns:
            bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
    # 설비 유무 플래그
    bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
    bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

    # 필요한 컬럼만 추려 병합 (없으면 스킵)
    keep_cols = ['건물번호']
    for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
        if c in bi.columns: keep_cols.append(c)
    bi = bi[keep_cols].drop_duplicates('건물번호')

    train = train.merge(bi, on='건물번호', how='left')
    test  = test.merge(bi, on='건물번호',  how='left')

# === 1) 공통 시간 파생
def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 일시 파싱 (형식: 'YYYYMMDD HH')
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    df['hour']      = df['일시'].dt.hour
    df['day']       = df['일시'].dt.day
    df['month']     = df['일시'].dt.month
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['is_weekend']       = (df['dayofweek'] >= 5).astype(int)
    df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
    # 주기 인코딩
    df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12)
    df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
    df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)
    # DI (있을 때만)
    if {'기온(°C)','습도(%)'}.issubset(df.columns):
        t = df['기온(°C)']
        h = df['습도(%)']
        df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
    else:
        df['DI'] = 0.0
    return df

train = add_time_features_kor(train)
test  = add_time_features_kor(test)

# === 2) expected_solar (train 기준 → 둘 다에 머지)
if '일사(MJ/m2)' in train.columns:
    solar_proxy = (
        train.groupby(['month','hour'])['일사(MJ/m2)']
             .mean().reset_index()
             .rename(columns={'일사(MJ/m2)':'expected_solar'})
    )
    train = train.merge(solar_proxy, on=['month','hour'], how='left')
    test  = test.merge(solar_proxy,  on=['month','hour'], how='left')
else:
    # train에 일사가 없으면 0
    train['expected_solar'] = 0.0
    test['expected_solar']  = 0.0

train['expected_solar'] = train['expected_solar'].fillna(0)
test['expected_solar']  = test['expected_solar'].fillna(0)

# === 3) 일별 온도 통계 (train/test 동일 로직)
def add_daily_temp_stats_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '기온(°C)' not in df.columns:
        for c in ['day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range']:
            df[c] = 0.0
        return df
    grp = df.groupby(['건물번호','month','day'])['기온(°C)']
    stats = grp.agg(day_max_temperature='max',
                    day_mean_temperature='mean',
                    day_min_temperature='min').reset_index()
    df = df.merge(stats, on=['건물번호','month','day'], how='left')
    df['day_temperature_range'] = df['day_max_temperature'] - df['day_min_temperature']
    return df

train = add_daily_temp_stats_kor(train)
test  = add_daily_temp_stats_kor(test)

# === 4) CDH / THI / WCT (train/test 동일)
def add_CDH_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '기온(°C)' not in df.columns:
        df['CDH'] = 0.0
        return df
    def _cdh_1d(x):
        cs = np.cumsum(x - 26)
        return np.concatenate((cs[:11], cs[11:] - cs[:-11])) if len(x) >= 12 else np.zeros_like(x, dtype=float)
    parts = []
    # 시간 순으로 정렬 후 건물별 처리
    for bno, g in df.sort_values('일시').groupby('건물번호'):
        arr = g['기온(°C)'].to_numpy()
        cdh = _cdh_1d(arr)
        parts.append(pd.Series(cdh, index=g.index))
    df['CDH'] = pd.concat(parts).sort_index()
    return df

def add_THI_WCT_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'기온(°C)','습도(%)'}.issubset(df.columns):
        t = df['기온(°C)']; h = df['습도(%)']
        df['THI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
    else:
        df['THI'] = 0.0
    if {'기온(°C)','풍속(m/s)'}.issubset(df.columns):
        t = df['기온(°C)']; w = df['풍속(m/s)'].clip(lower=0)
        df['WCT'] = 13.12 + 0.6125*t - 11.37*(w**0.16) + 0.3965*(w**0.16)*t
    else:
        df['WCT'] = 0.0
    return df

train = add_CDH_kor(train)
test  = add_CDH_kor(test)
train = add_THI_WCT_kor(train)
test  = add_THI_WCT_kor(test)

# === 5) 시간대 전력 통계 (train으로 계산 → 둘 다 merge)
if '전력소비량(kWh)' in train.columns:
    pm = (train
          .groupby(['건물번호','hour','dayofweek'])['전력소비량(kWh)']
          .agg(['mean','std'])
          .reset_index()
          .rename(columns={'mean':'day_hour_mean','std':'day_hour_std'}))
    train = train.merge(pm, on=['건물번호','hour','dayofweek'], how='left')
    test  = test.merge(pm,  on=['건물번호','hour','dayofweek'],  how='left')
else:
    train['day_hour_mean'] = 0.0; train['day_hour_std'] = 0.0
    test['day_hour_mean']  = 0.0; test['day_hour_std']  = 0.0

# === 6) (선택) 이상치 제거: 0 kWh 제거
if '전력소비량(kWh)' in train.columns:
    train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# === 7) 범주형 건물유형 인코딩 (있을 때만)
if '건물유형' in train.columns and '건물유형' in test.columns:
    both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
    cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
    train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
    test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)
# 1) 공통 feature (train/test 둘 다 있는 컬럼만 선택)
feature_candidates = [
    # building_info 관련
    '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
    # 기상/환경 데이터
    '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
    # 시간 기반
    'hour','day','month','dayofweek','is_weekend','is_working_hours',
    'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
    # 파생지표
    'DI','expected_solar',
    'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
    'CDH','THI','WCT',
    'day_hour_mean','day_hour_std'
]

features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# 2) target 명시
target = '전력소비량(kWh)'
if target not in train.columns:
    raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# 3) 최종 입력/타깃 데이터
X = train[features].copy()
y = np.log1p(train[target].values)   # 안정화를 위해 log1p 변환
X_test = test[features].copy()

print(f"[확인] 사용 features 개수: {len(features)}")
print(f"[확인] target: {target}")
print(f"[확인] X shape: {X.shape}, X_test shape: {X_test.shape}, y shape: {y.shape}")

# ------------------------------
# 이하: Optuna 고정, 저장/로드 유지(원래 로직 최대한 보존)
# ------------------------------

# 튜닝 목적함수들 (표준 컬럼 기준 y는 로그 변환 사용)
def smape_exp(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

def tune_xgb(trial, x_train, y_train, x_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "early_stopping_rounds": 50,
        "eval_metric": "mae",
        "random_state": seed,
        "objective": "reg:squarederror",
    }
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    pred = model.predict(x_val)
    return smape_exp(y_val, pred)

def tune_lgb(trial, x_train, y_train, x_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": seed,
        "objective": "mae",
    }
    model = LGBMRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(x_val)
    return smape_exp(y_val, pred)

def tune_cat(trial, x_train, y_train, x_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": seed,
        "loss_function": "MAE",
        "verbose": 0,
    }
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, verbose=0)
    pred = model.predict(x_val)
    return smape_exp(y_val, pred)

def objective_ridge(trial, oof_tr, oof_val, y_tr, y_val):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    ridge.fit(oof_tr, y_tr)
    preds = ridge.predict(oof_val)
    return smape_exp(y_val, preds)

def process_building_kfold(bno):
    print(f"🏢 building {bno} KFold...")
    param_dir = os.path.join(path, "optuna_params")
    os.makedirs(param_dir, exist_ok=True)

    tr_b = train[train["건물번호"] == bno].copy()
    te_b = test[test["건물번호"] == bno].copy()

    X = tr_b[features].values
    y = np.log1p(tr_b[target].values.astype(float))
    X_test = te_b[features].values

    kf = KFold(n_splits=7, shuffle=True, random_state=seed)
    test_preds, val_smapes = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        print(f" - fold {fold}")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_test)

        # XGB
        xgb_key = f"{bno}_fold{fold}_xgb"
        xgb_path = os.path.join(param_dir, f"{xgb_key}.json")
        if os.path.exists(xgb_path):
            with open(xgb_path, "r") as f:
                xgb_params = json.load(f)
        else:
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda t: tune_xgb(t, X_tr_s, y_tr, X_va_s, y_va), n_trials=20)
            xgb_params = st.best_params
            with open(xgb_path, "w") as f:
                json.dump(xgb_params, f)
        xgb = XGBRegressor(**xgb_params)
        xgb.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)

        # LGB
        lgb_key = f"{bno}_fold{fold}_lgb"
        lgb_path = os.path.join(param_dir, f"{lgb_key}.json")
        if os.path.exists(lgb_path):
            with open(lgb_path, "r") as f:
                lgb_params = json.load(f)
        else:
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda t: tune_lgb(t, X_tr_s, y_tr, X_va_s, y_va), n_trials=20)
            lgb_params = st.best_params
            with open(lgb_path, "w") as f:
                json.dump(lgb_params, f)
        lgbm = LGBMRegressor(**lgb_params)
        lgbm.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # CAT
        cat_key = f"{bno}_fold{fold}_cat"
        cat_path = os.path.join(param_dir, f"{cat_key}.json")
        if os.path.exists(cat_path):
            with open(cat_path, "r") as f:
                cat_params = json.load(f)
        else:
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda t: tune_cat(t, X_tr_s, y_tr, X_va_s, y_va), n_trials=20)
            cat_params = st.best_params
            with open(cat_path, "w") as f:
                json.dump(cat_params, f)
        cat = CatBoostRegressor(**cat_params)
        cat.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=50, verbose=0)

        # 스태킹을 위한 oof
        oof_tr = np.vstack([
            xgb.predict(X_tr_s),
            lgbm.predict(X_tr_s),
            cat.predict(X_tr_s)
        ]).T
        oof_va = np.vstack([
            xgb.predict(X_va_s),
            lgbm.predict(X_va_s),
            cat.predict(X_va_s)
        ]).T
        oof_te = np.vstack([
            xgb.predict(X_te_s),
            lgbm.predict(X_te_s),
            cat.predict(X_te_s)
        ]).T

        # Ridge 메타
        ridge_key = f"{bno}_fold{fold}_ridge"
        ridge_path = os.path.join(param_dir, f"{ridge_key}.json")
        if os.path.exists(ridge_path):
            with open(ridge_path, "r") as f:
                ridge_params = json.load(f)
        else:
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda t: objective_ridge(t, oof_tr, oof_va, y_tr, y_va), n_trials=30)
            ridge_params = st.best_params
            with open(ridge_path, "w") as f:
                json.dump(ridge_params, f)

        meta = Ridge(alpha=ridge_params["alpha"])
        meta.fit(oof_tr, y_tr)

        va_pred = meta.predict(oof_va)
        te_pred = meta.predict(oof_te)

        fold_smape = smape_exp(y_va, va_pred)
        val_smapes.append(fold_smape)
        test_preds.append(np.expm1(te_pred))  # 역로그

    avg_test_pred = np.mean(test_preds, axis=0)
    avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
    return avg_test_pred.tolist(), avg_smape


# ==============================
# 12) 병렬 실행
# ==============================
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_building_kfold)(bno) for bno in train["건물번호"].unique()
)

final_preds, val_smapes = [], []
for preds, sm in results:
    final_preds.extend(preds)
    val_smapes.append(sm)

samplesub["answer"] = final_preds
today = datetime.datetime.now().strftime("%Y%m%d")
avg_smape = float(np.mean(val_smapes))
filename = f"submission_stack_optuna_stdcols_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")

























#2얘는 모두 옵튜나 한거 !! 성능 매우 좋음

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
























# # 3 최종모델만 optuna - 모두 옵튜나 한게 성능 더 좋음. 이건 혹시나 해서 갖고있는



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
# warnings.filterwarnings(action='ignore')


# #best 707
# # Seed 고정
# seed = 6054
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # 경로 설정
# if os.path.exists('/workspace/TensorJae/Study25/'):
#     BASE_PATH = '/workspace/TensorJae/Study25/'
# else:
#     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')

# path = os.path.join(BASE_PATH, '_data/dacon/electricity/')
# buildinginfo = pd.read_csv(path + 'building_info.csv')
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# samplesub = pd.read_csv(path + 'sample_submission.csv')


# # 결측치 처리
# for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
#     buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# # Feature Engineering
# def feature_engineering(df, is_train=True):
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'])
#     df['hour'] = df['일시'].dt.hour
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['month'] = df['일시'].dt.month
#     df['day'] = df['일시'].dt.day
#     df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
#     df['is_working_hours'] = df['hour'].between(9, 18).astype(int)
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
#     if is_train:
#         for col in ['일조(hr)', '일사(MJ/m2)']:
#             df[col] = df[col].fillna(0)
#     else:
#         df['일조(hr)'] = 0
#         df['일사(MJ/m2)'] = 0

#     temp = df['기온(°C)']
#     humidity = df['습도(%)']
#     df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
#     return df

# # 전처리 적용
# train = feature_engineering(train, is_train=True)
# test = feature_engineering(test, is_train=False)

# # 메타 정보 병합
# train = train.merge(buildinginfo, on='건물번호', how='left')
# test = test.merge(buildinginfo, on='건물번호', how='left')

# # 범주형 인코딩
# train['건물유형'] = train['건물유형'].astype('category').cat.codes
# test['건물유형'] = test['건물유형'].astype('category').cat.codes

# # 파생 변수 생성
# for df in [train, test]:
#     df['cooling_area_ratio'] = df['냉방면적(m2)'] / df['연면적(m2)']
#     df['has_pv'] = (df['태양광용량(kW)'] > 0).astype(int)
#     df['has_ess'] = (df['ESS저장용량(kWh)'] > 0).astype(int)
    
    
# features = [
#     '건물유형', '연면적(m2)', '냉방면적(m2)', 
#     '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
#     '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#     'hour', 'dayofweek', 'month', 'day', 'is_weekend',
#     'is_working_hours', 'sin_hour', 'cos_hour', 'DI',
#     'cooling_area_ratio', 'has_pv', 'has_ess'
# ]

# target = '전력소비량(kWh)'
# final_preds = []
# val_smapes = []

# # Optuna 튜닝 함수
# # def objective(trial, oof_train, oof_val, y_train, y_val):
# #     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
# #     ridge = Ridge(alpha=alpha)
# #     ridge.fit(oof_train, y_train)
# #     preds = ridge.predict(oof_val)
# #     smape = np.mean(200 * np.abs(np.expm1(preds) - np.expm1(y_val)) /
# #                     (np.abs(np.expm1(preds)) + np.abs(np.expm1(y_val)) + 1e-6))
# #     return smape

# # 건물별 모델 학습
# building_ids = train['건물번호'].unique()
# for bno in building_ids:
#     print(f"🏢 건물번호 {bno} 처리 중...")

#     train_b = train[train['건물번호'] == bno].copy()
#     test_b = test[test['건물번호'] == bno].copy()
#     x = train_b[features]
#     y = np.log1p(train_b[target])
#     x_test_final = test_b[features]

#     x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=seed)

#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_val_scaled = scaler.transform(x_val)
#     x_test_final_scaled = scaler.transform(x_test_final)

#     # Base 모델 학습
#     xgb = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
#                        random_state=seed, early_stopping_rounds=50, objective='reg:squarederror')
#     xgb.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

#     lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
#                               random_state=seed, objective='mae')
#     lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
#                   callbacks=[lgb.early_stopping(50, verbose=False)])

#     cat = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
#                             random_seed=seed, verbose=0, loss_function='MAE')
#     cat.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

#     # 스태킹
#     oof_train_lvl1 = np.vstack([
#         xgb.predict(x_train_scaled),
#         lgb_model.predict(x_train_scaled),
#         cat.predict(x_train_scaled)
#     ]).T
#     oof_val_lvl1 = np.vstack([
#         xgb.predict(x_val_scaled),
#         lgb_model.predict(x_val_scaled),
#         cat.predict(x_val_scaled)
#     ]).T
#     oof_test_lvl1 = np.vstack([
#         xgb.predict(x_test_final_scaled),
#         lgb_model.predict(x_test_final_scaled),
#         cat.predict(x_test_final_scaled)
#     ]).T

#     # Optuna로 Ridge 튜닝
#     # study = optuna.create_study(direction="minimize")
#     # study.optimize(lambda trial: objective(trial, oof_train_lvl1, oof_val_lvl1, y_train, y_val), n_trials=30)
#     # best_alpha = study.best_params['alpha']
#     # meta_model = Ridge(alpha=best_alpha)
#     # meta_model.fit(oof_train_lvl1, y_train)
    
#     meta_model = Ridge(alpha=1.0)
#     meta_model.fit(oof_train_lvl1, y_train)
    
#     #=============================================

#     val_pred = meta_model.predict(oof_val_lvl1)
#     test_pred = meta_model.predict(oof_test_lvl1)

#     val_smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
#                         (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))
#     val_smapes.append(val_smape)

#     pred = np.expm1(test_pred)
#     final_preds.extend(pred)

# # 결과 저장
# samplesub['answer'] = final_preds
# today = datetime.datetime.now().strftime('%Y%m%d')
# avg_smape = np.mean(val_smapes)
# score_str = f"{avg_smape:.4f}".replace('.', '_')
# filename = f"submission_stack_optuna_{today}_SMAPE_{score_str}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")



