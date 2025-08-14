

# ### current best + 홍햄 전처리

# ####################🌟🌟🌟🌟🌟🌟🌟🌟 current BEST 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

# ## 현재 optuna_params_extended  쓰면 최고점


# # -*- coding: utf-8 -*-
# import os
# import json
# import random
# import warnings
# import datetime
# import numpy as np
# import pandas as pd
# import optuna

# from sklearn.model_selection import KFold, TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from joblib import Parallel, delayed
# from optuna.samplers import TPESampler

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgb
# import tensorflow as tf

# warnings.filterwarnings("ignore")

# # ==============================
# # 0) 시드 / 경로
# # ==============================
# seed = 222
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# path = os.path.join(BASE_PATH, "_data/dacon/electricity/")

# buildinginfo = pd.read_csv(os.path.join(path, "building_info.csv"))
# train = pd.read_csv(os.path.join(path, "train.csv"))
# test = pd.read_csv(os.path.join(path, "test.csv"))
# samplesub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

# # === 0) 옵션: building_info 병합 (있으면 병합, 없으면 넘어감)
# have_bi = 'buildinginfo' in globals() or 'building_info' in globals()
# if 'buildinginfo' in globals():
#     bi = buildinginfo.copy()
# else:
#     bi = None

# if bi is not None:
#     for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
#         if col in bi.columns:
#             bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
#     bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
#     bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

#     keep_cols = ['건물번호']
#     for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
#         if c in bi.columns: keep_cols.append(c)
#     bi = bi[keep_cols].drop_duplicates('건물번호')

#     train = train.merge(bi, on='건물번호', how='left')
#     test  = test.merge(bi, on='건물번호',  how='left')

# # === 1) 공통 시간 파생
# # [PATCH] 토/일 분리 플래그 추가(is_saturday, is_sunday)
# def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H', errors='coerce')
#     df['hour']      = df['일시'].dt.hour
#     df['day']       = df['일시'].dt.day
#     df['month']     = df['일시'].dt.month
#     df['dayofweek'] = df['일시'].dt.dayofweek  # 월=0, ..., 일=6

#     # [PATCH] 토/일 분리 + 주말
#     df['is_saturday'] = (df['dayofweek'] == 5).astype(int)
#     df['is_sunday']   = (df['dayofweek'] == 6).astype(int)
#     df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)

#     # 근무시간 여부
#     df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)

#     # 주기형 인코딩
#     df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
#     df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
#     df['sin_month'] = np.sin(2*np.pi*df['month']/12)
#     df['cos_month'] = np.cos(2*np.pi*df['month']/12)
#     df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
#     df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)

#     # 열지수(DI)
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['DI'] = 0.0
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

# # === 1-추가) 한국 공휴일 피처 (대체휴일/선거일 포함)
# try:
#     import holidays
#     def add_kr_holidays(df):
#         df = df.copy()
#         kr_hol = holidays.KR()
#         d = df['일시'].dt.date
#         df['is_holiday'] = d.map(lambda x: int(x in kr_hol))
#         prev_d = (df['일시'] - pd.Timedelta(days=1)).dt.date
#         next_d = (df['일시'] + pd.Timedelta(days=1)).dt.date
#         df['is_pre_holiday']  = prev_d.map(lambda x: int(x in kr_hol))
#         df['is_post_holiday'] = next_d.map(lambda x: int(x in kr_hol))
#         daily = df.groupby(df['일시'].dt.date)['is_holiday'].max()
#         daily_roll7 = daily.rolling(7, min_periods=1).sum()
#         df['holiday_7d_count'] = df['일시'].dt.date.map(daily_roll7)
#         dow = df['dayofweek']
#         df['is_bridge_day'] = (((dow==4) & (df['is_post_holiday']==1)) | ((dow==0) & (df['is_pre_holiday']==1))).astype(int)
#         return df
# except Exception:
#     def add_kr_holidays(df):
#         df = df.copy()
#         for c in ['is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day']:
#             df[c] = 0
#         return df

# train = add_kr_holidays(train)
# test  = add_kr_holidays(test)

# # === 2) expected_solar (train 기준 → 둘 다에 머지)
# if '일사(MJ/m2)' in train.columns:
#     solar_proxy = (
#         train.groupby(['month','hour'])['일사(MJ/m2)']
#              .mean().reset_index()
#              .rename(columns={'일사(MJ/m2)':'expected_solar'})
#     )
#     train = train.merge(solar_proxy, on=['month','hour'], how='left')
#     test  = test.merge(solar_proxy,  on=['month','hour'], how='left')
# else:
#     train['expected_solar'] = 0.0
#     test['expected_solar']  = 0.0

# train['expected_solar'] = train['expected_solar'].fillna(0)
# test['expected_solar']  = test['expected_solar'].fillna(0)

# # === 3) 일별 온도 통계
# def add_daily_temp_stats_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if '기온(°C)' not in df.columns:
#         for c in ['day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range']:
#             df[c] = 0.0
#         return df
#     grp = df.groupby(['건물번호','month','day'])['기온(°C)']
#     stats = grp.agg(day_max_temperature='max',
#                     day_mean_temperature='mean',
#                     day_min_temperature='min').reset_index()
#     df = df.merge(stats, on=['건물번호','month','day'], how='left')
#     df['day_temperature_range'] = df['day_max_temperature'] - df['day_min_temperature']
#     return df

# train = add_daily_temp_stats_kor(train)
# test  = add_daily_temp_stats_kor(test)

# # === 4) CDH / THI / WCT
# def add_CDH_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if '기온(°C)' not in df.columns:
#         df['CDH'] = 0.0
#         return df
#     def _cdh_1d(x):
#         cs = np.cumsum(x - 26)
#         return np.concatenate((cs[:11], cs[11:] - cs[:-11])) if len(x) >= 12 else np.zeros_like(x, dtype=float)
#     parts = []
#     for bno, g in df.sort_values('일시').groupby('건물번호'):
#         arr = g['기온(°C)'].to_numpy()
#         cdh = _cdh_1d(arr)
#         parts.append(pd.Series(cdh, index=g.index))
#     df['CDH'] = pd.concat(parts).sort_index()
#     return df

# def add_THI_WCT_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['THI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['THI'] = 0.0
#     if {'기온(°C)','풍속(m/s)'}.issubset(df.columns):
#         t = df['기온(°C)']; w = df['풍속(m/s)'].clip(lower=0)
#         df['WCT'] = 13.12 + 0.6125*t - 11.37*(w**0.16) + 0.3965*(w**0.16)*t
#     else:
#         df['WCT'] = 0.0
#     return df

# train = add_CDH_kor(train)
# test  = add_CDH_kor(test)
# train = add_THI_WCT_kor(train)
# test  = add_THI_WCT_kor(test)

# # === [PATCH] 이상치 클리핑 유틸(훈련 분위수 고정 → 일관 적용)
# def compute_clip_quantiles(df, columns, lower=0.10, upper=0.90):
#     q = {}
#     for c in columns:
#         if c in df.columns:
#             s = df[c]
#             if c == '습도(%)':
#                 s = s.clip(0, 100)
#             q[c] = (float(s.quantile(lower)), float(s.quantile(upper)))
#     return q

# def apply_clip_quantiles(df, qmap):
#     df = df.copy()
#     for c, (lo, hi) in qmap.items():
#         if c in df.columns:
#             if c == '습도(%)':
#                 df[c] = df[c].clip(0, 100)
#             df[c] = df[c].clip(lo, hi)
#     return df

# # === [PATCH] 이상치 클리핑(train 분위수 기준)
# clip_cols = ['풍속(m/s)', '습도(%)']
# qmap = compute_clip_quantiles(train, clip_cols, lower=0.10, upper=0.90)
# train = apply_clip_quantiles(train, qmap)
# test  = apply_clip_quantiles(test,  qmap)

# # === [PATCH] 강수량 0/1 이진화 (>0 기준)
# if '강수량(mm)' in train.columns:
#     train['강수량(mm)'] = (train['강수량(mm)'] > 0).astype(int)
# if '강수량(mm)' in test.columns:
#     test['강수량(mm)']  = (test['강수량(mm)'] > 0).astype(int)

# # === 5) 시간대 전력 통계(전체 train 집계) - 튜닝/베이스 참고용
# if '전력소비량(kWh)' in train.columns:
#     pm = (train
#           .groupby(['건물번호','hour','dayofweek'])['전력소비량(kWh)']
#           .agg(['mean','std'])
#           .reset_index()
#           .rename(columns={'mean':'day_hour_mean','std':'day_hour_std'}))
#     train = train.merge(pm, on=['건물번호','hour','dayofweek'], how='left')
#     test  = test.merge(pm,  on=['건물번호','hour','dayofweek'],  how='left')
# else:
#     train['day_hour_mean'] = 0.0; train['day_hour_std'] = 0.0
#     test['day_hour_mean']  = 0.0; test['day_hour_std']  = 0.0

# # === 6) 이상치 제거: 0 kWh 제거
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop_number=True) if 'drop_number' in dir(pd.DataFrame) else train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # === 7) 범주형 건물유형 인코딩
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
#     cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
#     train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
#     test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# # ------------------------------
# # Feature Set
# # ------------------------------
# feature_candidates = [
#     # building_info
#     '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
#     # weather/raw
#     '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
#     # time parts & cycles
#     'hour','day','month','dayofweek',
#     'is_saturday','is_sunday',  # [PATCH] 추가
#     'is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     # engineered
#     'DI','expected_solar',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     'CDH','THI','WCT',
#     # target stats (전역 집계 - 폴드에서 덮어씀)
#     'day_hour_mean','day_hour_std',
#     # holidays
#     'is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day'
# ]
# features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# # Target
# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# # 최종 입력/타깃
# X = train[features].values
# y_log = np.log1p(train[target].values.astype(float))
# X_test_raw = test[features].values

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# # 전처리 정합성 점검
# print("len(test) =", len(test))
# print("len(samplesub) =", len(samplesub))
# print("건물 수 train vs test:", train["건물번호"].nunique(), test["건물번호"].nunique())
# counts = test.groupby("건물번호").size()
# bad = counts[counts != 168]
# if len(bad):
#     print("⚠️ 168이 아닌 건물 발견:\n", bad)
# assert len(test) == len(samplesub), f"test:{len(test)} sample:{len(samplesub)}"

# # ------------------------------
# # SMAPE helpers
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# def smape(y, yhat):
#     return np.mean(200*np.abs(yhat - y)/(np.abs(yhat)+np.abs(y)+1e-6))

# # ========== Tweedie 전용 유틸 & 튜닝 ==========
# def log1p_pos(arr):
#     return np.log1p(np.clip(arr, a_min=0, a_max=None))

# def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
#     params = {
#         "objective": "tweedie",
#         "metric": "mae",
#         "boosting_type": "gbdt",
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 64, 512),
#         "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
#         "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
#         "random_state": seed,
#         "verbosity": -1,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr_raw, y_va_raw = y_full_sorted_raw[tr_idx], y_full_sorted_raw[va_idx]

#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)

#         model = LGBMRegressor(**params)
#         model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
#                   callbacks=[lgb.early_stopping(50, verbose=False)])

#         pred_raw = model.predict(X_va_s)
#         y_va_log = log1p_pos(y_va_raw)
#         pred_log = log1p_pos(pred_raw)
#         scores.append(smape_exp(y_va_log, pred_log))
#     return float(np.mean(scores))

# def get_or_tune_tweedie_once(bno, X_full, y_full_raw, order_index, param_dir):
#     os.makedirs(param_dir, exist_ok=True)
#     path_twd = os.path.join(param_dir, f"{bno}_twd.json")
#     X_sorted = X_full[order_index]
#     y_sorted_raw = y_full_raw[order_index]
#     if os.path.exists(path_twd):
#         with open(path_twd, "r") as f:
#             return json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=30)
#         best = st.best_params
#         with open(path_twd, "w") as f:
#             json.dump(best, f)
#         return best

# # ------------------------------
# # 기존 튜닝 함수들 (XGB/LGB/CAT)
# # ------------------------------
# def tune_xgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "eval_metric": "mae",
#         "random_state": seed,
#         "objective": "reg:squarederror",
#         "early_stopping_rounds": 50,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = XGBRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_lgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "random_state": seed,
#         "objective": "mae",
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = LGBMRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_cat_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "iterations": trial.suggest_int("iterations", 300, 1000),
#         "depth": trial.suggest_int("depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
#         "random_seed": seed,
#         "loss_function": "MAE",
#         "verbose": 0,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = CatBoostRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=50, verbose=0)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def get_or_tune_params_once(bno, X_full, y_full, order_index, param_dir):
#     os.makedirs(param_dir, exist_ok=True)
#     paths = {
#         "xgb": os.path.join(param_dir, f"{bno}_xgb.json"),
#         "lgb": os.path.join(param_dir, f"{bno}_lgb.json"),
#         "cat": os.path.join(param_dir, f"{bno}_cat.json"),
#     }
#     params = {}
#     X_sorted = X_full[order_index]
#     y_sorted = y_full[order_index]

#     if os.path.exists(paths["xgb"]):
#         with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["xgb"] = st.best_params
#         with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

#     if os.path.exists(paths["lgb"]):
#         with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["lgb"] = st.best_params
#         with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

#     if os.path.exists(paths["cat"]):
#         with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["cat"] = st.best_params
#         with open(paths["cat"], "w") as f: json.dump(params["cat"], f)

#     return params

# # ------------------------------
# # Ridge 튜닝(메타) - OOF 행렬 기반
# # ------------------------------
# def objective_ridge_on_oof(trial, oof_meta, y_full):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     ridge = Ridge(alpha=alpha)
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     scores = []
#     for tr_idx, va_idx in kf.split(oof_meta):
#         ridge.fit(oof_meta[tr_idx], y_full[tr_idx])
#         preds = ridge.predict(oof_meta[va_idx])
#         scores.append(smape_exp(y_full[va_idx], preds))
#     return float(np.mean(scores))

# # ------------------------------
# # [PATCH-1] 타깃통계(누설 차단) 유틸
# # ------------------------------
# def build_target_stats_fold(base_df, idx, target):
#     base = base_df.iloc[idx]

#     # 건물×hour 평균/표준편차를 한 번에 생성
#     g1 = (base
#           .groupby(["건물번호","hour"])[target]
#           .agg(hour_mean="mean", hour_std="std")
#           .reset_index())

#     # 건물×hour×dow 평균/표준편차/중앙값
#     g2 = base.groupby(["건물번호","hour","dayofweek"])[target]
#     d_mean = g2.mean().rename("day_hour_mean").reset_index()
#     d_std  = g2.std().rename("day_hour_std").reset_index()
#     d_med  = g2.median().rename("day_hour_median").reset_index()

#     # 건물×hour×month 평균
#     g3 = (base
#           .groupby(["건물번호","hour","month"])[target]
#           .mean()
#           .rename("month_hour_mean")
#           .reset_index())

#     return g1, d_mean, d_std, d_med, g3

# def merge_target_stats(df, stats):
#     g1, d_mean, d_std, d_med, g3 = stats
#     out = df.merge(g1, on=["건물번호","hour"], how="left")  # hour_mean + hour_std 둘 다 붙음
#     out = out.merge(d_mean, on=["건물번호","hour","dayofweek"], how="left")
#     out = out.merge(d_std,  on=["건물번호","hour","dayofweek"], how="left")
#     out = out.merge(d_med,  on=["건물번호","hour","dayofweek"], how="left")
#     out = out.merge(g3,     on=["건물번호","hour","month"],    how="left")
#     return out

# # ------------------------------
# # 건물 단위 학습/예측
# # ------------------------------
# def process_building_kfold(bno):
#     print(f"🏢 building {bno} KFold...")
#     param_dir = os.path.join(path, "optuna_params_extended")
#     os.makedirs(param_dir, exist_ok=True)

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     X_full = tr_b[features].values
#     y_full_log = np.log1p(tr_b[target].values.astype(float))
#     y_full_raw = tr_b[target].values.astype(float)
#     X_test = te_b[features].values

#     # 시계열 정렬 인덱스 (튜닝용)
#     order = np.argsort(tr_b['일시'].values)

#     # 베이스 모델 파라미터 (건물당 1회)
#     best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
#     best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

#     # 외부 KFold
#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     base_models = ["xgb", "lgb", "cat", "twd"]
#     n_train_b = len(tr_b); n_test_b = len(te_b)
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     # 폴드 루프
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")

#         # [PATCH-1] 폴드별 타깃통계 재계산→머지 (누설 차단)
#         stats = build_target_stats_fold(tr_b, tr_idx, target)
#         tr_fold = merge_target_stats(tr_b.iloc[tr_idx].copy(), stats)
#         va_fold = merge_target_stats(tr_b.iloc[va_idx].copy(), stats)
#         te_fold = merge_target_stats(te_b.copy(),               stats)

#         # 결측 보정
#         fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]

#         # 1) 우선 존재하는 컬럼만 모아서 전역 평균 계산
#         present = [c for c in fill_cols if c in tr_fold.columns]
#         if len(present) == 0:
#             glob_mean = 0.0
#         else:
#             glob_mean = float(pd.concat([tr_fold[present]], axis=1).stack().mean())

#         # 2) 없는 컬럼은 즉시 생성 후 보정, 있는 컬럼은 결측만 채우기
#         for df_ in (tr_fold, va_fold, te_fold):
#             for c in fill_cols:
#                 if c not in df_.columns:
#                     df_[c] = glob_mean
#                 else:
#                     df_[c] = df_[c].fillna(glob_mean)

#         # 행렬 구성
#         X_tr = tr_fold[features].values
#         X_va = va_fold[features].values
#         X_te = te_fold[features].values
#         y_tr_log, y_va_log = np.log1p(tr_fold[target].values.astype(float)), np.log1p(va_fold[target].values.astype(float))
#         y_tr_raw, y_va_raw = tr_fold[target].values.astype(float), va_fold[target].values.astype(float)

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_te)

#         # XGB (log 타깃)
#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         # LGB (log 타깃)
#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # CAT (log 타깃)
#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)

#         # Tweedie (원시 타깃)
#         twd = LGBMRegressor(**best_twd)
#         twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # OOF 저장(모두 로그 스케일 통일)
#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)
#         pred_raw_va_twd = twd.predict(X_va_s)
#         oof_meta[va_idx, 3] = log1p_pos(pred_raw_va_twd)

#         # 테스트 메타 누적
#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)
#         pred_raw_te_twd = twd.predict(X_te_s)
#         test_meta_accum[:, 3] += log1p_pos(pred_raw_te_twd)

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # ----- 메타(Ridge) 튜닝/학습
#     ridge_key = f"{bno}_ridge"
#     ridge_path = os.path.join(param_dir, f"{ridge_key}.json")
#     if os.path.exists(ridge_path):
#         with open(ridge_path, "r") as f:
#             ridge_params = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full_log), n_trials=30)
#         ridge_params = st.best_params
#         with open(ridge_path, "w") as f:
#             json.dump(ridge_params, f)

#     meta = Ridge(alpha=ridge_params["alpha"])
#     meta.fit(oof_meta, y_full_log)

#     # ----- OOF 성능, Smearing 보정, SMAPE 칼리브레이션
#     oof_pred_log = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full_log, oof_pred_log))

#     # [PATCH-3] Smearing
#     resid = y_full_log - oof_pred_log
#     S = float(np.mean(np.exp(resid)))

#     # 테스트 예측 (로그→원복 + Smearing)
#     te_pred_log = meta.predict(test_meta)
#     te_pred = np.expm1(te_pred_log) * S

#     # [PATCH-4] 단조 칼리브레이션 g(p)=a·p^b (OOF 기반)
#     y_oof = np.expm1(y_full_log)
#     p_oof = np.expm1(oof_pred_log) * S
#     a_grid = np.linspace(0.8, 1.2, 21)
#     b_grid = np.linspace(0.85, 1.15, 31)
#     best = (1.0, 1.0, smape(y_oof, p_oof))
#     for a in a_grid:
#         for b in b_grid:
#             s = smape(y_oof, a*(p_oof**b))
#             if s < best[2]:
#                 best = (a, b, s)
#     a_opt, b_opt, _ = best
#     te_pred = a_opt * (te_pred ** b_opt)

#     return te_pred.tolist(), avg_smape

# # ==============================
# # 12) 병렬 실행 (test 건물 기준) + 순서 매핑
# # ==============================
# bld_list = list(np.sort(test["건물번호"].unique()))
# results = Parallel(n_jobs=-1, backend="loky")(
#     delayed(process_building_kfold)(bno) for bno in bld_list
# )

# preds_full = np.zeros(len(test), dtype=float)
# val_smapes = []
# for bno, (preds, sm) in zip(bld_list, results):
#     idx = (test["건물번호"] == bno).values
#     assert idx.sum() == len(preds), f"building {bno}: test rows={idx.sum()}, preds={len(preds)}"
#     preds_full[idx] = preds
#     if not np.isnan(sm):
#         val_smapes.append(sm)

# assert len(preds_full) == len(samplesub), f"final preds:{len(preds_full)}, sample:{len(samplesub)}"
# samplesub["answer"] = preds_full

# today = datetime.datetime.now().strftime("%Y%m%d")
# avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
# filename = f"submission_stack_PATCHED_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")































#건물별 + global 앙상블 아직 안해봄
# # -*- coding: utf-8 -*-
# import os
# import json
# import random
# import warnings
# import datetime
# import numpy as np
# import pandas as pd
# import optuna

# from sklearn.model_selection import KFold, TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from joblib import Parallel, delayed
# from optuna.samplers import TPESampler

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgb
# import tensorflow as tf

# warnings.filterwarnings("ignore")

# # ==============================
# # 0) 시드 / 경로
# # ==============================
# seed = 222
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# path = os.path.join(BASE_PATH, "_data/dacon/electricity/")

# buildinginfo = pd.read_csv(os.path.join(path, "building_info.csv"))
# train = pd.read_csv(os.path.join(path, "train.csv"))
# test = pd.read_csv(os.path.join(path, "test.csv"))
# samplesub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

# # === 0) 옵션: building_info 병합 (있으면 병합, 없으면 넘어감)
# have_bi = 'buildinginfo' in globals() or 'building_info' in globals()
# if 'buildinginfo' in globals():
#     bi = buildinginfo.copy()
# else:
#     bi = None

# if bi is not None:
#     for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
#         if col in bi.columns:
#             bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
#     bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
#     bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

#     keep_cols = ['건물번호']
#     for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
#         if c in bi.columns: keep_cols.append(c)
#     bi = bi[keep_cols].drop_duplicates('건물번호')

#     train = train.merge(bi, on='건물번호', how='left')
#     test  = test.merge(bi, on='건물번호',  how='left')

# # === 1) 공통 시간 파생
# # [PATCH] 토/일 분리 플래그 추가(is_saturday, is_sunday)
# def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H', errors='coerce')
#     df['hour']      = df['일시'].dt.hour
#     df['day']       = df['일시'].dt.day
#     df['month']     = df['일시'].dt.month
#     df['dayofweek'] = df['일시'].dt.dayofweek  # 월=0, ..., 일=6

#     # [PATCH] 토/일 분리 + 주말
#     df['is_saturday'] = (df['dayofweek'] == 5).astype(int)
#     df['is_sunday']   = (df['dayofweek'] == 6).astype(int)
#     df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)

#     # 근무시간 여부
#     df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)

#     # 주기형 인코딩
#     df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
#     df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
#     df['sin_month'] = np.sin(2*np.pi*df['month']/12)
#     df['cos_month'] = np.cos(2*np.pi*df['month']/12)
#     df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
#     df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)

#     # 열지수(DI)
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['DI'] = 0.0
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

# # === 1-추가) 한국 공휴일 피처 (대체휴일/선거일 포함)
# try:
#     import holidays
#     def add_kr_holidays(df):
#         df = df.copy()
#         kr_hol = holidays.KR()
#         d = df['일시'].dt.date
#         df['is_holiday'] = d.map(lambda x: int(x in kr_hol))
#         prev_d = (df['일시'] - pd.Timedelta(days=1)).dt.date
#         next_d = (df['일시'] + pd.Timedelta(days=1)).dt.date
#         df['is_pre_holiday']  = prev_d.map(lambda x: int(x in kr_hol))
#         df['is_post_holiday'] = next_d.map(lambda x: int(x in kr_hol))
#         daily = df.groupby(df['일시'].dt.date)['is_holiday'].max()
#         daily_roll7 = daily.rolling(7, min_periods=1).sum()
#         df['holiday_7d_count'] = df['일시'].dt.date.map(daily_roll7)
#         dow = df['dayofweek']
#         df['is_bridge_day'] = (((dow==4) & (df['is_post_holiday']==1)) | ((dow==0) & (df['is_pre_holiday']==1))).astype(int)
#         return df
# except Exception:
#     def add_kr_holidays(df):
#         df = df.copy()
#         for c in ['is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day']:
#             df[c] = 0
#         return df

# train = add_kr_holidays(train)
# test  = add_kr_holidays(test)

# # === 2) expected_solar (train 기준 → 둘 다에 머지)
# if '일사(MJ/m2)' in train.columns:
#     solar_proxy = (
#         train.groupby(['month','hour'])['일사(MJ/m2)']
#              .mean().reset_index()
#              .rename(columns={'일사(MJ/m2)':'expected_solar'})
#     )
#     train = train.merge(solar_proxy, on=['month','hour'], how='left')
#     test  = test.merge(solar_proxy,  on=['month','hour'], how='left')
# else:
#     train['expected_solar'] = 0.0
#     test['expected_solar']  = 0.0

# train['expected_solar'] = train['expected_solar'].fillna(0)
# test['expected_solar']  = test['expected_solar'].fillna(0)

# # === 3) 일별 온도 통계
# def add_daily_temp_stats_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if '기온(°C)' not in df.columns:
#         for c in ['day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range']:
#             df[c] = 0.0
#         return df
#     grp = df.groupby(['건물번호','month','day'])['기온(°C)']
#     stats = grp.agg(day_max_temperature='max',
#                     day_mean_temperature='mean',
#                     day_min_temperature='min').reset_index()
#     df = df.merge(stats, on=['건물번호','month','day'], how='left')
#     df['day_temperature_range'] = df['day_max_temperature'] - df['day_min_temperature']
#     return df

# train = add_daily_temp_stats_kor(train)
# test  = add_daily_temp_stats_kor(test)

# # === 4) CDH / THI / WCT
# def add_CDH_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if '기온(°C)' not in df.columns:
#         df['CDH'] = 0.0
#         return df
#     def _cdh_1d(x):
#         cs = np.cumsum(x - 26)
#         return np.concatenate((cs[:11], cs[11:] - cs[:-11])) if len(x) >= 12 else np.zeros_like(x, dtype=float)
#     parts = []
#     for bno, g in df.sort_values('일시').groupby('건물번호'):
#         arr = g['기온(°C)'].to_numpy()
#         cdh = _cdh_1d(arr)
#         parts.append(pd.Series(cdh, index=g.index))
#     df['CDH'] = pd.concat(parts).sort_index()
#     return df

# def add_THI_WCT_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['THI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['THI'] = 0.0
#     if {'기온(°C)','풍속(m/s)'}.issubset(df.columns):
#         t = df['기온(°C)']; w = df['풍속(m/s)'].clip(lower=0)
#         df['WCT'] = 13.12 + 0.6125*t - 11.37*(w**0.16) + 0.3965*(w**0.16)*t
#     else:
#         df['WCT'] = 0.0
#     return df

# train = add_CDH_kor(train)
# test  = add_CDH_kor(test)
# train = add_THI_WCT_kor(train)
# test  = add_THI_WCT_kor(test)

# # === [PATCH] 이상치 클리핑 유틸(훈련 분위수 고정 → 일관 적용)
# def compute_clip_quantiles(df, columns, lower=0.10, upper=0.90):
#     q = {}
#     for c in columns:
#         if c in df.columns:
#             s = df[c]
#             if c == '습도(%)':
#                 s = s.clip(0, 100)
#             q[c] = (float(s.quantile(lower)), float(s.quantile(upper)))
#     return q

# def apply_clip_quantiles(df, qmap):
#     df = df.copy()
#     for c, (lo, hi) in qmap.items():
#         if c in df.columns:
#             if c == '습도(%)':
#                 df[c] = df[c].clip(0, 100)
#             df[c] = df[c].clip(lo, hi)
#     return df

# # === [PATCH] 이상치 클리핑(train 분위수 기준)
# clip_cols = ['풍속(m/s)', '습도(%)']
# qmap = compute_clip_quantiles(train, clip_cols, lower=0.10, upper=0.90)
# train = apply_clip_quantiles(train, qmap)
# test  = apply_clip_quantiles(test,  qmap)

# # === [PATCH] 강수량 0/1 이진화 (>0 기준)
# if '강수량(mm)' in train.columns:
#     train['강수량(mm)'] = (train['강수량(mm)'] > 0).astype(int)
# if '강수량(mm)' in test.columns:
#     test['강수량(mm)']  = (test['강수량(mm)'] > 0).astype(int)

# # === 5) 시간대 전력 통계(전체 train 집계) - 튜닝/베이스 참고용
# if '전력소비량(kWh)' in train.columns:
#     pm = (train
#           .groupby(['건물번호','hour','dayofweek'])['전력소비량(kWh)']
#           .agg(['mean','std'])
#           .reset_index()
#           .rename(columns={'mean':'day_hour_mean','std':'day_hour_std'}))
#     train = train.merge(pm, on=['건물번호','hour','dayofweek'], how='left')
#     test  = test.merge(pm,  on=['건물번호','hour','dayofweek'],  how='left')
# else:
#     train['day_hour_mean'] = 0.0; train['day_hour_std'] = 0.0
#     test['day_hour_mean']  = 0.0; test['day_hour_std']  = 0.0

# # === 6) 이상치 제거: 0 kWh 제거
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # === 7) 범주형 건물유형 인코딩 (정수)
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
#     cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
#     train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
#     test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# # ===== [ADD-1] 건물유형 10개 원핫 추가 =====
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both_type = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype(int).reset_index(drop=True)
#     type_ohe  = pd.get_dummies(both_type, prefix='btype', dtype=int)
#     type_ohe_train = type_ohe.iloc[:len(train)].reset_index(drop=True)
#     type_ohe_test  = type_ohe.iloc[len(train):].reset_index(drop=True)

#     train = pd.concat([train.reset_index(drop=True), type_ohe_train], axis=1)
#     test  = pd.concat([test.reset_index(drop=True),  type_ohe_test],  axis=1)

#     extra_btype_cols = [c for c in train.columns if c.startswith('btype_')]
# else:
#     extra_btype_cols = []

# # ------------------------------
# # Feature Set
# # ------------------------------
# feature_candidates = [
#     # building_info
#     '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
#     # weather/raw
#     '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
#     # time parts & cycles
#     'hour','day','month','dayofweek',
#     'is_saturday','is_sunday',
#     'is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     # engineered
#     'DI','expected_solar',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     'CDH','THI','WCT',
#     # target stats (전역 집계 - 폴드에서 덮어씀)
#     'day_hour_mean','day_hour_std',
#     # holidays
#     'is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day'
# ]
# features = [c for c in feature_candidates if c in train.columns and c in test.columns]
# features += [c for c in extra_btype_cols if c not in features]

# # Target
# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# # 최종 입력/타깃
# X = train[features].values
# y_log = np.log1p(train[target].values.astype(float))
# X_test_raw = test[features].values

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# # 전처리 정합성 점검
# print("len(test) =", len(test))
# print("len(samplesub) =", len(samplesub))
# print("건물 수 train vs test:", train["건물번호"].nunique(), test["건물번호"].nunique())
# counts = test.groupby("건물번호").size()
# bad = counts[counts != 168]
# if len(bad):
#     print("⚠️ 168이 아닌 건물 발견:\n", bad)
# assert len(test) == len(samplesub), f"test:{len(test)} sample:{len(samplesub)}"

# # ------------------------------
# # SMAPE helpers
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# def smape(y, yhat):
#     return np.mean(200*np.abs(yhat - y)/(np.abs(yhat)+np.abs(y)+1e-6))

# # ----- [ADD] SMAPE-직접 최적화 블렌딩 보조 함수 -----
# def _smape_from_logs(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# def _simplex_grid_weights(n_models, step=0.05):
#     # ∑w=1, w>=0 단순체적 격자
#     import itertools
#     m = int(round(1/step))
#     for splits in itertools.product(range(m+1), repeat=n_models-1):
#         s = sum(splits)
#         if s <= m:
#             w = [s_i*step for s_i in splits]
#             w.append(1.0 - sum(w))
#             yield np.array(w, dtype=float)

# def blend_smape_minimizer(oof_meta, y_full_log, test_meta, mode="geo", step=0.05, smear_S=1.0):
#     """
#     mode: "arith"=가중합(로그스케일), "geo"=기하평균(원스케일; 추천)
#     smear_S: 스미어링 계수
#     """
#     n_models = oof_meta.shape[1]
#     best_w, best_s = None, np.inf

#     if mode == "arith":
#         for w in _simplex_grid_weights(n_models, step):
#             pred_log = oof_meta @ w
#             s = _smape_from_logs(y_full_log, pred_log)
#             if s < best_s:
#                 best_s, best_w = s, w
#         te_pred_log = test_meta @ best_w
#         te_pred_raw = np.expm1(te_pred_log) * smear_S
#         return te_pred_raw, best_w, best_s

#     elif mode == "geo":
#         oof_raw = np.expm1(oof_meta).clip(min=0)
#         y_raw   = np.expm1(y_full_log)
#         oof_log_raw = np.log(oof_raw + 1e-9)

#         for w in _simplex_grid_weights(n_models, step):
#             pred_raw = np.exp(oof_log_raw @ w)  # 기하평균
#             s = np.mean(200 * np.abs(pred_raw - y_raw) / (np.abs(pred_raw) + np.abs(y_raw) + 1e-6))
#             if s < best_s:
#                 best_s, best_w = s, w

#         te_log_raw = np.log(np.expm1(test_meta).clip(min=0) + 1e-9)
#         te_pred_raw = np.exp(te_log_raw @ best_w) * smear_S
#         return te_pred_raw, best_w, best_s

#     else:
#         raise ValueError("mode must be 'arith' or 'geo'")

# # ========== Tweedie 전용 유틸 & 튜닝 ==========
# def log1p_pos(arr):
#     return np.log1p(np.clip(arr, a_min=0, a_max=None))

# def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
#     params = {
#         "objective": "tweedie",
#         "metric": "mae",
#         "boosting_type": "gbdt",
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 64, 512),
#         "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
#         "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
#         "random_state": seed,
#         "verbosity": -1,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr_raw, y_va_raw = y_full_sorted_raw[tr_idx], y_full_sorted_raw[va_idx]

#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)

#         model = LGBMRegressor(**params)
#         model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
#                   callbacks=[lgb.early_stopping(50, verbose=False)])

#         pred_raw = model.predict(X_va_s)
#         y_va_log = log1p_pos(y_va_raw)
#         pred_log = log1p_pos(pred_raw)
#         scores.append(smape_exp(y_va_log, pred_log))
#     return float(np.mean(scores))

# def get_or_tune_tweedie_once(bno, X_full, y_full_raw, order_index, param_dir):
#     os.makedirs(param_dir, exist_ok=True)
#     path_twd = os.path.join(param_dir, f"{bno}_twd.json")
#     X_sorted = X_full[order_index]
#     y_sorted_raw = y_full_raw[order_index]
#     if os.path.exists(path_twd):
#         with open(path_twd, "r") as f:
#             return json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=30)
#         best = st.best_params
#         with open(path_twd, "w") as f:
#             json.dump(best, f)
#         return best

# # ------------------------------
# # 기존 튜닝 함수들 (XGB/LGB/CAT)
# # ------------------------------
# def tune_xgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "eval_metric": "mae",
#         "random_state": seed,
#         "objective": "reg:squarederror",
#         "early_stopping_rounds": 50,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = XGBRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_lgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "random_state": seed,
#         "objective": "mae",
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = LGBMRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_cat_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "iterations": trial.suggest_int("iterations", 300, 1000),
#         "depth": trial.suggest_int("depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
#         "random_seed": seed,
#         "loss_function": "MAE",
#         "verbose": 0,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = CatBoostRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=50, verbose=0)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def get_or_tune_params_once(bno, X_full, y_full, order_index, param_dir):
#     os.makedirs(param_dir, exist_ok=True)
#     paths = {
#         "xgb": os.path.join(param_dir, f"{bno}_xgb.json"),
#         "lgb": os.path.join(param_dir, f"{bno}_lgb.json"),
#         "cat": os.path.join(param_dir, f"{bno}_cat.json"),
#     }
#     params = {}
#     X_sorted = X_full[order_index]
#     y_sorted = y_full[order_index]

#     if os.path.exists(paths["xgb"]):
#         with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["xgb"] = st.best_params
#         with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

#     if os.path.exists(paths["lgb"]):
#         with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["lgb"] = st.best_params
#         with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

#     if os.path.exists(paths["cat"]):
#         with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["cat"] = st.best_params
#         with open(paths["cat"], "w") as f: json.dump(params["cat"], f)

#     return params

# # ------------------------------
# # Ridge 튜닝(메타) - OOF 행렬 기반 (참고용)
# # ------------------------------
# def objective_ridge_on_oof(trial, oof_meta, y_full):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     ridge = Ridge(alpha=alpha)
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     scores = []
#     for tr_idx, va_idx in kf.split(oof_meta):
#         ridge.fit(oof_meta[tr_idx], y_full[tr_idx])
#         preds = ridge.predict(oof_meta[va_idx])
#         scores.append(smape_exp(y_full[va_idx], preds))
#     return float(np.mean(scores))

# # ===== [ADD-2] 전역(Global) 스태킹 학습 함수 =====
# def train_global_blender(train_df, test_df, features, target, seed=222):
#     print("🌐 Global blender training...")

#     X_all = train_df[features].values
#     y_all_log = np.log1p(train_df[target].values.astype(float))
#     y_all_raw = train_df[target].values.astype(float)
#     X_te_all = test_df[features].values

#     param_dir = os.path.join(path, "optuna_params_extended")
#     os.makedirs(param_dir, exist_ok=True)
#     p_twd = os.path.join(param_dir, "GLOBAL_twd.json")

#     def _get_or_tune_global(model_key, tuner_fn):
#         pth = os.path.join(param_dir, f"GLOBAL_{model_key}.json")
#         if os.path.exists(pth):
#             with open(pth, "r") as f:
#                 return json.load(f)
#         else:
#             order = np.arange(len(y_all_log))
#             X_sorted = X_all[order]
#             y_sorted = y_all_log[order]
#             st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#             st.optimize(lambda t: tuner_fn(t, X_sorted, y_sorted), n_trials=30)
#             best = st.best_params
#             with open(pth, "w") as f:
#                 json.dump(best, f)
#             return best

#     def _get_or_tune_global_twd():
#         if os.path.exists(p_twd):
#             with open(p_twd, "r") as f:
#                 return json.load(f)
#         else:
#             order = np.arange(len(y_all_raw))
#             X_sorted = X_all[order]
#             y_sorted_raw = y_all_raw[order]
#             st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#             st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=30)
#             best = st.best_params
#             with open(p_twd, "w") as f:
#                 json.dump(best, f)
#             return best

#     best_xgb = _get_or_tune_global("xgb", tune_xgb_tss)
#     best_lgb = _get_or_tune_global("lgb", tune_lgb_tss)
#     best_cat = _get_or_tune_global("cat", tune_cat_tss)
#     best_twd = _get_or_tune_global_twd()

#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)
#     oof_meta = np.zeros((len(train_df), 4), dtype=float)
#     test_meta_accum = np.zeros((len(test_df), 4), dtype=float)

#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), 1):
#         print(f"  - Global fold {fold}")
#         X_tr, X_va = X_all[tr_idx], X_all[va_idx]
#         y_tr_log, y_va_log = y_all_log[tr_idx], y_all_log[va_idx]
#         y_tr_raw, y_va_raw = y_all_raw[tr_idx], y_all_raw[va_idx]

#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_te_all)

#         m_xgb = XGBRegressor(**best_xgb)
#         m_xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)
#         oof_meta[va_idx, 0] = m_xgb.predict(X_va_s)
#         test_meta_accum[:, 0] += m_xgb.predict(X_te_s)

#         m_lgb = LGBMRegressor(**best_lgb)
#         m_lgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         oof_meta[va_idx, 1] = m_lgb.predict(X_va_s)
#         test_meta_accum[:, 1] += m_lgb.predict(X_te_s)

#         m_cat = CatBoostRegressor(**best_cat)
#         m_cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)
#         oof_meta[va_idx, 2] = m_cat.predict(X_va_s)
#         test_meta_accum[:, 2] += m_cat.predict(X_te_s)

#         m_twd = LGBMRegressor(**best_twd)
#         m_twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         pred_raw_va_twd = m_twd.predict(X_va_s)
#         oof_meta[va_idx, 3] = log1p_pos(pred_raw_va_twd)
#         test_meta_accum[:, 3] += log1p_pos(m_twd.predict(X_te_s))

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # 스미어링 S(전역 Ridge 임시로 계산) → 이후 SMAPE-최적 블렌딩 적용
#     tmp = Ridge(alpha=0.1).fit(oof_meta, y_all_log)
#     oof_pred_log_tmp = tmp.predict(oof_meta)
#     S = float(np.mean(np.exp(y_all_log - oof_pred_log_tmp)))

#     # 기하 vs 산술 중 더 좋은 블렌딩 선택
#     te_pred_geo,   w_geo,   s_geo   = blend_smape_minimizer(oof_meta, y_all_log, test_meta, mode="geo",   step=0.05, smear_S=S)
#     te_pred_arith, w_arith, s_arith = blend_smape_minimizer(oof_meta, y_all_log, test_meta, mode="arith", step=0.05, smear_S=S)

#     if s_geo <= s_arith:
#         chosen_mode, chosen_w, chosen_s = "geo", w_geo, s_geo
#         oof_log_raw = np.log(np.expm1(oof_meta).clip(min=0) + 1e-9)
#         p_oof = np.exp(oof_log_raw @ chosen_w) * S         # 원스케일
#         GLOBAL_TEST_RAW = te_pred_geo
#     else:
#         chosen_mode, chosen_w, chosen_s = "arith", w_arith, s_arith
#         p_oof = np.expm1(oof_meta @ chosen_w) * S          # 원스케일
#         GLOBAL_TEST_RAW = te_pred_arith

#     # 단조 캘리브레이션 g(p)=a·p^b (전역 OOF 기준)
#     y_oof = np.expm1(y_all_log)
#     a_grid = np.linspace(0.8, 1.2, 21)
#     b_grid = np.linspace(0.85, 1.15, 31)
#     best = (1.0, 1.0, np.mean(200*np.abs(p_oof - y_oof)/(np.abs(p_oof)+np.abs(y_oof)+1e-6)))
#     for a in a_grid:
#         for b in b_grid:
#             s = np.mean(200*np.abs(a*(p_oof**b) - y_oof)/(np.abs(a*(p_oof**b)) + np.abs(y_oof) + 1e-6))
#             if s < best[2]:
#                 best = (a, b, s)
#     a_opt, b_opt, _ = best

#     GLOBAL_OOF_RAW  = a_opt * (p_oof ** b_opt)
#     GLOBAL_TEST_RAW = a_opt * (GLOBAL_TEST_RAW ** b_opt)

#     print(f"🌐 Global chosen={chosen_mode}, OOF SMAPE={chosen_s:.4f}")
#     # train/test의 원래 인덱스 정렬에 맞게 반환 (호출부에서 인덱스로 슬라이스)
#     return GLOBAL_OOF_RAW, GLOBAL_TEST_RAW

# # ------------------------------
# # 건물 단위 학습/예측 (전역-로컬 앙상블 포함)
# # ------------------------------
# def process_building_kfold(bno, GLOBAL_OOF_RAW, GLOBAL_TEST_RAW):
#     print(f"🏢 building {bno} KFold...")
#     param_dir = os.path.join(path, "optuna_params_extended")
#     os.makedirs(param_dir, exist_ok=True)

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     X_full = tr_b[features].values
#     y_full_log = np.log1p(tr_b[target].values.astype(float))
#     y_full_raw = tr_b[target].values.astype(float)

#     # 시계열 정렬 인덱스 (튜닝용)
#     order = np.argsort(tr_b['일시'].values)

#     # 베이스 모델 파라미터 (건물당 1회)
#     best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
#     best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

#     # 외부 KFold
#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     base_models = ["xgb", "lgb", "cat", "twd"]
#     n_train_b = len(tr_b); n_test_b = len(te_b)
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     # 폴드 루프
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")

#         # [폴드 내 누설 차단] 타깃 통계 재계산→머지
#         def build_target_stats_fold(base_df, idx, target_col):
#             base = base_df.iloc[idx]
#             g1 = (base.groupby(["건물번호","hour"])[target_col]
#                   .agg(hour_mean="mean", hour_std="std")
#                   .reset_index())
#             g2 = base.groupby(["건물번호","hour","dayofweek"])[target_col]
#             d_mean = g2.mean().rename("day_hour_mean").reset_index()
#             d_std  = g2.std().rename("day_hour_std").reset_index()
#             d_med  = g2.median().rename("day_hour_median").reset_index()
#             g3 = (base.groupby(["건물번호","hour","month"])[target_col]
#                   .mean()
#                   .rename("month_hour_mean")
#                   .reset_index())
#             return g1, d_mean, d_std, d_med, g3

#         def merge_target_stats(df, stats):
#             g1, d_mean, d_std, d_med, g3 = stats
#             out = df.merge(g1, on=["건물번호","hour"], how="left")
#             out = out.merge(d_mean, on=["건물번호","hour","dayofweek"], how="left")
#             out = out.merge(d_std,  on=["건물번호","hour","dayofweek"], how="left")
#             out = out.merge(d_med,  on=["건물번호","hour","dayofweek"], how="left")
#             out = out.merge(g3,     on=["건물번호","hour","month"],    how="left")
#             return out

#         stats = build_target_stats_fold(tr_b, tr_idx, target)
#         tr_fold = merge_target_stats(tr_b.iloc[tr_idx].copy(), stats)
#         va_fold = merge_target_stats(tr_b.iloc[va_idx].copy(), stats)
#         te_fold = merge_target_stats(te_b.copy(),               stats)

#         # 결측 보정
#         fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
#         present = [c for c in fill_cols if c in tr_fold.columns]
#         glob_mean = float(pd.concat([tr_fold[present]], axis=1).stack().mean()) if len(present) else 0.0
#         for df_ in (tr_fold, va_fold, te_fold):
#             for c in fill_cols:
#                 if c not in df_.columns:
#                     df_[c] = glob_mean
#                 else:
#                     df_[c] = df_[c].fillna(glob_mean)

#         # 행렬 구성
#         X_tr = tr_fold[features].values
#         X_va = va_fold[features].values
#         X_te = te_fold[features].values
#         y_tr_log, y_va_log = np.log1p(tr_fold[target].values.astype(float)), np.log1p(va_fold[target].values.astype(float))
#         y_tr_raw, y_va_raw = tr_fold[target].values.astype(float), va_fold[target].values.astype(float)

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_te)

#         # XGB (log 타깃)
#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         # LGB (log 타깃)
#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # CAT (log 타깃)
#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)

#         # Tweedie (원시 타깃)
#         twd = LGBMRegressor(**best_twd)
#         twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # OOF 저장(모두 로그 스케일 통일)
#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)
#         pred_raw_va_twd = twd.predict(X_va_s)
#         oof_meta[va_idx, 3] = log1p_pos(pred_raw_va_twd)

#         # 테스트 메타 누적
#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)
#         pred_raw_te_twd = twd.predict(X_te_s)
#         test_meta_accum[:, 3] += log1p_pos(pred_raw_te_twd)

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # ===== 로컬: SMAPE 직접최적화 블렌딩 (기하/산술 중 선택) + 스미어링 + 단조 캘리브레이션 =====
#     # 스미어링 S 계산을 위한 임시 Ridge
#     meta_tmp = Ridge(alpha=0.1).fit(oof_meta, y_full_log)
#     oof_pred_log_ridge = meta_tmp.predict(oof_meta)
#     resid_r = y_full_log - oof_pred_log_ridge
#     S = float(np.mean(np.exp(resid_r)))

#     te_pred_geo,   w_geo,   s_geo   = blend_smape_minimizer(oof_meta, y_full_log, test_meta, mode="geo",   step=0.05, smear_S=S)
#     te_pred_arith, w_arith, s_arith = blend_smape_minimizer(oof_meta, y_full_log, test_meta, mode="arith", step=0.05, smear_S=S)

#     if s_geo <= s_arith:
#         chosen_mode = "geo"
#         chosen_w    = w_geo
#         chosen_s    = s_geo
#         oof_log_raw = np.log(np.expm1(oof_meta).clip(min=0) + 1e-9)
#         p_oof = np.exp(oof_log_raw @ chosen_w) * S  # 원스케일
#         te_pred = te_pred_geo                       # 원스케일
#     else:
#         chosen_mode = "arith"
#         chosen_w    = w_arith
#         chosen_s    = s_arith
#         p_oof = np.expm1(oof_meta @ chosen_w) * S   # 원스케일
#         te_pred = te_pred_arith                     # 원스케일

#     # 단조 캘리브레이션 (OOF 기준)
#     y_oof = np.expm1(y_full_log)
#     a_grid = np.linspace(0.8, 1.2, 21)
#     b_grid = np.linspace(0.85, 1.15, 31)
#     best = (1.0, 1.0, np.mean(200*np.abs(p_oof - y_oof)/(np.abs(p_oof)+np.abs(y_oof)+1e-6)))
#     for a in a_grid:
#         for b in b_grid:
#             s = np.mean(200*np.abs(a*(p_oof**b) - y_oof)/(np.abs(a*(p_oof**b)) + np.abs(y_oof) + 1e-6))
#             if s < best[2]:
#                 best = (a, b, s)
#     a_opt, b_opt, _ = best

#     # 로컬 최종(원스케일)
#     p_oof = a_opt * (p_oof ** b_opt)
#     te_pred = a_opt * (te_pred ** b_opt)

#     # ===== 전역(Global)과 로컬(Local) 알파 앙상블 (OOF SMAPE 최소 α) =====
#     # 전역 OOF/TEST에서 현재 건물 행만 추출 (train/test의 원래 인덱스 기준)
#     g_oof_b = GLOBAL_OOF_RAW[tr_b.index.values]  # 원스케일
#     g_te_b  = GLOBAL_TEST_RAW[te_b.index.values] # 원스케일

#     best_a, best_s = 1.0, np.inf
#     for a in np.linspace(0.0, 1.0, 21):
#         comb = a * p_oof + (1.0 - a) * g_oof_b
#         s = smape(y_oof, comb)
#         if s < best_s:
#             best_s = s
#             best_a = a

#     te_pred_final = best_a * te_pred + (1.0 - best_a) * g_te_b
#     print(f"[Building {bno}] mode={chosen_mode}, α(local)={best_a:.2f}, OOF SMAPE={best_s:.4f}")

#     return te_pred_final.tolist(), float(best_s)

# # ==============================
# # 12) 전역 모델 학습 → 병렬 실행 (test 건물 기준) + 순서 매핑
# # ==============================
# # 전역(Global) 모델 먼저 학습
# GLOBAL_OOF_RAW, GLOBAL_TEST_RAW = train_global_blender(train, test, features, target, seed=seed)

# bld_list = list(np.sort(test["건물번호"].unique()))
# results = Parallel(n_jobs=-1, backend="loky")(
#     delayed(process_building_kfold)(bno, GLOBAL_OOF_RAW, GLOBAL_TEST_RAW) for bno in bld_list
# )

# preds_full = np.zeros(len(test), dtype=float)
# val_smapes = []
# for bno, (preds, sm) in zip(bld_list, results):
#     idx = (test["건물번호"] == bno).values
#     assert idx.sum() == len(preds), f"building {bno}: test rows={idx.sum()}, preds={len(preds)}"
#     preds_full[idx] = preds
#     if not np.isnan(sm):
#         val_smapes.append(sm)

# assert len(preds_full) == len(samplesub), f"final preds:{len(preds_full)}, sample:{len(samplesub)}"
# samplesub["answer"] = preds_full

# today = datetime.datetime.now().strftime("%Y%m%d")
# avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
# filename = f"submission_stack_GLOBAL_LOCAL_ENSEMBLE_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")














# -*- coding: utf-8 -*-
import os
import json
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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
seed = 222
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
    for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
        if col in bi.columns:
            bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
    bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
    bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

    keep_cols = ['건물번호']
    for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
        if c in bi.columns: keep_cols.append(c)
    bi = bi[keep_cols].drop_duplicates('건물번호')

    train = train.merge(bi, on='건물번호', how='left')
    test  = test.merge(bi, on='건물번호',  how='left')

# === 1) 공통 시간 파생
def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H', errors='coerce')
    df['hour']      = df['일시'].dt.hour
    df['day']       = df['일시'].dt.day
    df['month']     = df['일시'].dt.month
    df['dayofweek'] = df['일시'].dt.dayofweek  # 월=0, ..., 일=6

    df['is_saturday'] = (df['dayofweek'] == 5).astype(int)
    df['is_sunday']   = (df['dayofweek'] == 6).astype(int)
    df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)

    df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)

    df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12)
    df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
    df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)

    if {'기온(°C)','습도(%)'}.issubset(df.columns):
        t = df['기온(°C)']; h = df['습도(%)']
        df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
    else:
        df['DI'] = 0.0
    return df

train = add_time_features_kor(train)
test  = add_time_features_kor(test)

# === 1-추가) 한국 공휴일 피처
try:
    import holidays
    def add_kr_holidays(df):
        df = df.copy()
        kr_hol = holidays.KR()
        d = df['일시'].dt.date
        df['is_holiday'] = d.map(lambda x: int(x in kr_hol))
        prev_d = (df['일시'] - pd.Timedelta(days=1)).dt.date
        next_d = (df['일시'] + pd.Timedelta(days=1)).dt.date
        df['is_pre_holiday']  = prev_d.map(lambda x: int(x in kr_hol))
        df['is_post_holiday'] = next_d.map(lambda x: int(x in kr_hol))
        daily = df.groupby(df['일시'].dt.date)['is_holiday'].max()
        daily_roll7 = daily.rolling(7, min_periods=1).sum()
        df['holiday_7d_count'] = df['일시'].dt.date.map(daily_roll7)
        dow = df['dayofweek']
        df['is_bridge_day'] = (((dow==4) & (df['is_post_holiday']==1)) | ((dow==0) & (df['is_pre_holiday']==1))).astype(int)
        return df
except Exception:
    def add_kr_holidays(df):
        df = df.copy()
        for c in ['is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day']:
            df[c] = 0
        return df

train = add_kr_holidays(train)
test  = add_kr_holidays(test)

# === 2) expected_solar (train 기준)
if '일사(MJ/m2)' in train.columns:
    solar_proxy = (
        train.groupby(['month','hour'])['일사(MJ/m2)']
             .mean().reset_index()
             .rename(columns={'일사(MJ/m2)':'expected_solar'})
    )
    train = train.merge(solar_proxy, on=['month','hour'], how='left')
    test  = test.merge(solar_proxy,  on=['month','hour'], how='left')
else:
    train['expected_solar'] = 0.0
    test['expected_solar']  = 0.0

train['expected_solar'] = train['expected_solar'].fillna(0)
test['expected_solar']  = test['expected_solar'].fillna(0)

# === 3) 일별 온도 통계
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

# === 4) CDH / THI / WCT
def add_CDH_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '기온(°C)' not in df.columns:
        df['CDH'] = 0.0
        return df
    def _cdh_1d(x):
        cs = np.cumsum(x - 26)
        return np.concatenate((cs[:11], cs[11:] - cs[:-11])) if len(x) >= 12 else np.zeros_like(x, dtype=float)
    parts = []
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

# === [PATCH] 이상치 클리핑 유틸
def compute_clip_quantiles(df, columns, lower=0.10, upper=0.90):
    q = {}
    for c in columns:
        if c in df.columns:
            s = df[c]
            if c == '습도(%)':
                s = s.clip(0, 100)
            q[c] = (float(s.quantile(lower)), float(s.quantile(upper)))
    return q

def apply_clip_quantiles(df, qmap):
    df = df.copy()
    for c, (lo, hi) in qmap.items():
        if c in df.columns:
            if c == '습도(%)':
                df[c] = df[c].clip(0, 100)
            df[c] = df[c].clip(lo, hi)
    return df

clip_cols = ['풍속(m/s)', '습도(%)']
qmap = compute_clip_quantiles(train, clip_cols, lower=0.10, upper=0.90)
train = apply_clip_quantiles(train, qmap)
test  = apply_clip_quantiles(test,  qmap)

# === [PATCH] 강수량 0/1 이진화
if '강수량(mm)' in train.columns:
    train['강수량(mm)'] = (train['강수량(mm)'] > 0).astype(int)
if '강수량(mm)' in test.columns:
    test['강수량(mm)']  = (test['강수량(mm)'] > 0).astype(int)

# === 5) 시간대 전력 통계(전역 집계: 참고/피처)
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

# === 6) 이상치 제거: 0 kWh 제거
if '전력소비량(kWh)' in train.columns:
    train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# === 7) 건물유형: 정수 인코딩 + 원핫(10종)
if '건물유형' in train.columns and '건물유형' in test.columns:
    both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
    cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
    train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
    test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# 원핫
if '건물유형' in train.columns and '건물유형' in test.columns:
    both_type = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype(int).reset_index(drop=True)
    type_ohe  = pd.get_dummies(both_type, prefix='btype', dtype=int)
    type_ohe_train = type_ohe.iloc[:len(train)].reset_index(drop=True)
    type_ohe_test  = type_ohe.iloc[len(train):].reset_index(drop=True)

    train = pd.concat([train.reset_index(drop=True), type_ohe_train], axis=1)
    test  = pd.concat([test.reset_index(drop=True),  type_ohe_test],  axis=1)

    extra_btype_cols = [c for c in train.columns if c.startswith('btype_')]
else:
    extra_btype_cols = []

# ------------------------------
# Feature Set
# ------------------------------
feature_candidates = [
    # building_info
    '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
    # weather/raw
    '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
    # time parts & cycles
    'hour','day','month','dayofweek',
    'is_saturday','is_sunday',
    'is_weekend','is_working_hours',
    'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
    # engineered
    'DI','expected_solar',
    'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
    'CDH','THI','WCT',
    # target stats
    'day_hour_mean','day_hour_std',
    # holidays
    'is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day'
]
features = [c for c in feature_candidates if c in train.columns and c in test.columns]
features += [c for c in extra_btype_cols if c not in features]

# Target
target = '전력소비량(kWh)'
if target not in train.columns:
    raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# 정합성 출력
X = train[features].values
y_log = np.log1p(train[target].values.astype(float))
X_test_raw = test[features].values

print(f"[확인] 사용 features 개수: {len(features)}")
print(f"[확인] target: {target}")
print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")
print("len(test) =", len(test))
print("len(samplesub) =", len(samplesub))
print("건물 수 train vs test:", train["건물번호"].nunique(), test["건물번호"].nunique())
counts = test.groupby("건물번호").size()
bad = counts[counts != 168]
if len(bad):
    print("⚠️ 168이 아닌 건물 발견:\n", bad)
assert len(test) == len(samplesub), f"test:{len(test)} sample:{len(samplesub)}"

# ------------------------------
# SMAPE helpers & 블렌딩 유틸
# ------------------------------
def smape_exp(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

def smape(y, yhat):
    return np.mean(200*np.abs(yhat - y)/(np.abs(yhat)+np.abs(y)+1e-6))

def log1p_pos(arr):
    return np.log1p(np.clip(arr, a_min=0, a_max=None))

def _smape_from_logs(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

def _simplex_grid_weights(n_models, step=0.05):
    import itertools
    m = int(round(1/step))
    for splits in itertools.product(range(m+1), repeat=n_models-1):
        s = sum(splits)
        if s <= m:
            w = [s_i*step for s_i in splits]
            w.append(1.0 - sum(w))
            yield np.array(w, dtype=float)

def blend_smape_minimizer(oof_meta, y_full_log, test_meta, mode="geo", step=0.05, smear_S=1.0):
    """
    mode: "arith"=가중합(로그스케일), "geo"=기하평균(원스케일)
    smear_S: 스미어링 계수
    """
    n_models = oof_meta.shape[1]
    best_w, best_s = None, np.inf

    if mode == "arith":
        for w in _simplex_grid_weights(n_models, step):
            pred_log = oof_meta @ w
            s = _smape_from_logs(y_full_log, pred_log)
            if s < best_s:
                best_s, best_w = s, w
        te_pred_log = test_meta @ best_w
        te_pred_raw = np.expm1(te_pred_log) * smear_S
        return te_pred_raw, best_w, best_s

    elif mode == "geo":
        oof_raw = np.expm1(oof_meta).clip(min=0)
        y_raw   = np.expm1(y_full_log)
        oof_log_raw = np.log(oof_raw + 1e-9)
        for w in _simplex_grid_weights(n_models, step):
            pred_raw = np.exp(oof_log_raw @ w)
            s = np.mean(200 * np.abs(pred_raw - y_raw) / (np.abs(pred_raw) + np.abs(y_raw) + 1e-6))
            if s < best_s:
                best_s, best_w = s, w
        te_log_raw = np.log(np.expm1(test_meta).clip(min=0) + 1e-9)
        te_pred_raw = np.exp(te_log_raw @ best_w) * smear_S
        return te_pred_raw, best_w, best_s

    else:
        raise ValueError("mode must be 'arith' or 'geo'")

# ========== 튜닝 함수들 (XGB/LGB/CAT/Tweedie) ==========
def tune_xgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eval_metric": "mae",
        "random_state": seed,
        "objective": "reg:squarederror",
        "early_stopping_rounds": 50,
    }
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tss.split(X_full_sorted):
        X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
        y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
        model = XGBRegressor(**params)
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
        pred = model.predict(X_va_s)
        scores.append(smape_exp(y_va, pred))
    return float(np.mean(scores))

def tune_lgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": seed,
        "objective": "mae",
    }
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tss.split(X_full_sorted):
        X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
        y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
        model = LGBMRegressor(**params)
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict(X_va_s)
        scores.append(smape_exp(y_va, pred))
    return float(np.mean(scores))

def tune_cat_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": seed,
        "loss_function": "MAE",
        "verbose": 0,
    }
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tss.split(X_full_sorted):
        X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
        y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
        model = CatBoostRegressor(**params)
        model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=50, verbose=0)
        pred = model.predict(X_va_s)
        scores.append(smape_exp(y_va, pred))
    return float(np.mean(scores))

def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
    params = {
        "objective": "tweedie",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 512),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
        "random_state": seed,
        "verbosity": -1,
    }
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tss.split(X_full_sorted):
        X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
        y_tr_raw, y_va_raw = y_full_sorted_raw[tr_idx], y_full_sorted_raw[va_idx]
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
        model = LGBMRegressor(**params)
        model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred_raw = model.predict(X_va_s)
        y_va_log = log1p_pos(y_va_raw)
        pred_log = log1p_pos(pred_raw)
        scores.append(smape_exp(y_va_log, pred_log))
    return float(np.mean(scores))

def get_or_tune_params_once(bno, X_full, y_full, order_index, param_dir):
    os.makedirs(param_dir, exist_ok=True)
    paths = {
        "xgb": os.path.join(param_dir, f"{bno}_xgb.json"),
        "lgb": os.path.join(param_dir, f"{bno}_lgb.json"),
        "cat": os.path.join(param_dir, f"{bno}_cat.json"),
    }
    params = {}
    X_sorted = X_full[order_index]
    y_sorted = y_full[order_index]

    if os.path.exists(paths["xgb"]):
        with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=30)
        params["xgb"] = st.best_params
        with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

    if os.path.exists(paths["lgb"]):
        with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=30)
        params["lgb"] = st.best_params
        with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

    if os.path.exists(paths["cat"]):
        with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=30)
        params["cat"] = st.best_params
        with open(paths["cat"], "w") as f: json.dump(params["cat"], f)

    return params

# ------------------------------
# [NEW] 건물유형(Type-wise) 스태킹 블렌더
# ------------------------------
def train_type_blender(train_df, test_df, features, target, seed=222):
    """
    각 건물유형 값(정수)별로 XGB/LGB/CAT/TWD OOF/TEST 메타 생성 →
    SMAPE 직접최적화 블렌딩(geo/arith 자동) →
    스미어링 + 단조 캘리브레이션 →
    최종 원스케일 예측을 train/test 전체 인덱스에 맞춰 반환
    """
    assert '건물유형' in train_df.columns and '건물유형' in test_df.columns
    param_dir = os.path.join(path, "optuna_params_extended")
    os.makedirs(param_dir, exist_ok=True)

    TYPE_OOF_RAW  = np.zeros(len(train_df), dtype=float)
    TYPE_TEST_RAW = np.zeros(len(test_df),  dtype=float)

    type_ids = sorted(train_df['건물유형'].dropna().unique().tolist())

    for t in type_ids:
        print(f"🏷️ type={t} blender training...")
        tr_t = train_df[train_df['건물유형'] == t].copy()
        te_t = test_df [test_df ['건물유형'] == t].copy()
        if len(tr_t) == 0:
            continue

        X_all = tr_t[features].values
        y_log = np.log1p(tr_t[target].values.astype(float))
        y_raw = tr_t[target].values.astype(float)
        X_te  = te_t[features].values

        # --- 파라미터 캐시/튜닝(유형별) ---
        def _load_or_tune(name, tuner_fn, Xs, ys):
            pth = os.path.join(param_dir, f"TYPE{t}_{name}.json")
            if os.path.exists(pth):
                with open(pth, "r") as f:
                    return json.load(f)
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda trial: tuner_fn(trial, Xs, ys), n_trials=30)
            best = st.best_params
            with open(pth, "w") as f:
                json.dump(best, f)
            return best

        best_xgb = _load_or_tune("xgb", tune_xgb_tss, X_all, y_log)
        best_lgb = _load_or_tune("lgb", tune_lgb_tss, X_all, y_log)
        best_cat = _load_or_tune("cat", tune_cat_tss, X_all, y_log)

        def _load_or_tune_twd(Xs, ys_raw):
            pth = os.path.join(param_dir, f"TYPE{t}_twd.json")
            if os.path.exists(pth):
                with open(pth, "r") as f:
                    return json.load(f)
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda trial: tune_lgb_tweedie_tss(trial, Xs, ys_raw), n_trials=30)
            best = st.best_params
            with open(pth, "w") as f:
                json.dump(best, f)
            return best
        best_twd = _load_or_tune_twd(X_all, y_raw)

        # --- 8-fold OOF/TEST 메타 ---
        kf = KFold(n_splits=8, shuffle=True, random_state=seed)
        oof_meta = np.zeros((len(tr_t), 4), dtype=float)
        test_meta_accum = np.zeros((len(te_t), 4), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), 1):
            print(f"  - type={t} fold {fold}")
            X_tr, X_va = X_all[tr_idx], X_all[va_idx]
            y_tr_log, y_va_log = y_log[tr_idx], y_log[va_idx]
            y_tr_raw, y_va_raw = y_raw[tr_idx], y_raw[va_idx]

            sc = StandardScaler().fit(X_tr)
            X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va); X_te_s = sc.transform(X_te)

            m_xgb = XGBRegressor(**best_xgb)
            m_xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)
            oof_meta[va_idx, 0] = m_xgb.predict(X_va_s)
            test_meta_accum[:, 0] += m_xgb.predict(X_te_s)

            m_lgb = LGBMRegressor(**best_lgb)
            m_lgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_meta[va_idx, 1] = m_lgb.predict(X_va_s)
            test_meta_accum[:, 1] += m_lgb.predict(X_te_s)

            m_cat = CatBoostRegressor(**best_cat)
            m_cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)
            oof_meta[va_idx, 2] = m_cat.predict(X_va_s)
            test_meta_accum[:, 2] += m_cat.predict(X_te_s)

            m_twd = LGBMRegressor(**best_twd)
            m_twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_meta[va_idx, 3] = log1p_pos(m_twd.predict(X_va_s))
            test_meta_accum[:, 3] += log1p_pos(m_twd.predict(X_te_s))

        test_meta = test_meta_accum / kf.get_n_splits()

        # --- 스미어링 S(임시 Ridge) ---
        tmp = Ridge(alpha=0.1).fit(oof_meta, y_log)
        S = float(np.mean(np.exp(y_log - tmp.predict(oof_meta))))

        # --- SMAPE 직접최적화 블렌딩 (geo/arith 중 선택) ---
        te_geo,   w_geo,   s_geo   = blend_smape_minimizer(oof_meta, y_log, test_meta, mode="geo",   step=0.05, smear_S=S)
        te_arith, w_arith, s_arith = blend_smape_minimizer(oof_meta, y_log, test_meta, mode="arith", step=0.05, smear_S=S)

        if s_geo <= s_arith:
            chosen_w = w_geo
            oof_log_raw = np.log(np.expm1(oof_meta).clip(min=0) + 1e-9)
            p_oof = np.exp(oof_log_raw @ chosen_w) * S
            p_te  = te_geo
        else:
            chosen_w = w_arith
            p_oof = np.expm1(oof_meta @ chosen_w) * S
            p_te  = te_arith

        # --- 단조 캘리브레이션 g(p)=a·p^b (유형 OOF 기준) ---
        y_oof = np.expm1(y_log)
        a_grid = np.linspace(0.8, 1.2, 21)
        b_grid = np.linspace(0.85, 1.15, 31)
        best = (1.0, 1.0, smape(y_oof, p_oof))
        for a in a_grid:
            for b in b_grid:
                s = smape(y_oof, a*(p_oof**b))
                if s < best[2]:
                    best = (a, b, s)
        a_opt, b_opt, _ = best

        # 최종(원스케일) 타입 예측
        TYPE_OOF_RAW[tr_t.index.values] = a_opt * (p_oof ** b_opt)
        TYPE_TEST_RAW[te_t.index.values] = a_opt * (p_te  ** b_opt)

        print(f"🏷️ type={t} chosen_w={chosen_w}, OOF SMAPE={best[2]:.4f}")

    return TYPE_OOF_RAW, TYPE_TEST_RAW

# ------------------------------
# 건물 단위(Local) 스태킹 (기존 개선형)
# ------------------------------
def process_building_kfold(bno):
    print(f"🏢 building {bno} KFold...")
    param_dir = os.path.join(path, "optuna_params_extended")
    os.makedirs(param_dir, exist_ok=True)

    tr_b = train[train["건물번호"] == bno].copy()
    te_b = test[test["건물번호"] == bno].copy()

    X_full = tr_b[features].values
    y_full_log = np.log1p(tr_b[target].values.astype(float))
    y_full_raw = tr_b[target].values.astype(float)

    # 시계열 정렬 인덱스 (튜닝용)
    order = np.argsort(tr_b['일시'].values)

    # 베이스 모델 파라미터 (건물당 1회)
    best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)

    # Tweedie 파라미터 (건물당 1회)
    def get_or_tune_tweedie_once(bno, X_full, y_full_raw, order_index, param_dir):
        os.makedirs(param_dir, exist_ok=True)
        path_twd = os.path.join(param_dir, f"{bno}_twd.json")
        X_sorted = X_full[order_index]
        y_sorted_raw = y_full_raw[order_index]
        if os.path.exists(path_twd):
            with open(path_twd, "r") as f:
                return json.load(f)
        else:
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=30)
            best = st.best_params
            with open(path_twd, "w") as f:
                json.dump(best, f)
            return best
    best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

    # 외부 KFold
    kf = KFold(n_splits=8, shuffle=True, random_state=seed)

    base_models = ["xgb", "lgb", "cat", "twd"]
    n_train_b = len(tr_b); n_test_b = len(te_b)
    oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
    test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

    # 폴드 루프
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        print(f" - fold {fold}")

        # 폴드 내 타깃통계(누설 차단)
        def build_target_stats_fold(base_df, idx, target_col):
            base = base_df.iloc[idx]
            g1 = (base.groupby(["건물번호","hour"])[target_col]
                  .agg(hour_mean="mean", hour_std="std")
                  .reset_index())
            g2 = base.groupby(["건물번호","hour","dayofweek"])[target_col]
            d_mean = g2.mean().rename("day_hour_mean").reset_index()
            d_std  = g2.std().rename("day_hour_std").reset_index()
            d_med  = g2.median().rename("day_hour_median").reset_index()
            g3 = (base.groupby(["건물번호","hour","month"])[target_col]
                  .mean()
                  .rename("month_hour_mean")
                  .reset_index())
            return g1, d_mean, d_std, d_med, g3

        def merge_target_stats(df, stats):
            g1, d_mean, d_std, d_med, g3 = stats
            out = df.merge(g1, on=["건물번호","hour"], how="left")
            out = out.merge(d_mean, on=["건물번호","hour","dayofweek"], how="left")
            out = out.merge(d_std,  on=["건물번호","hour","dayofweek"], how="left")
            out = out.merge(d_med,  on=["건물번호","hour","dayofweek"], how="left")
            out = out.merge(g3,     on=["건물번호","hour","month"],    how="left")
            return out

        stats = build_target_stats_fold(tr_b, tr_idx, target)
        tr_fold = merge_target_stats(tr_b.iloc[tr_idx].copy(), stats)
        va_fold = merge_target_stats(tr_b.iloc[va_idx].copy(), stats)
        te_fold = merge_target_stats(te_b.copy(),               stats)

        # 결측 보정
        fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
        present = [c for c in fill_cols if c in tr_fold.columns]
        glob_mean = float(pd.concat([tr_fold[present]], axis=1).stack().mean()) if len(present) else 0.0
        for df_ in (tr_fold, va_fold, te_fold):
            for c in fill_cols:
                if c not in df_.columns:
                    df_[c] = glob_mean
                else:
                    df_[c] = df_[c].fillna(glob_mean)

        # 행렬 구성
        X_tr = tr_fold[features].values
        X_va = va_fold[features].values
        X_te = te_fold[features].values
        y_tr_log, y_va_log = np.log1p(tr_fold[target].values.astype(float)), np.log1p(va_fold[target].values.astype(float))
        y_tr_raw, y_va_raw = tr_fold[target].values.astype(float), va_fold[target].values.astype(float)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_te)

        # XGB (log 타깃)
        xgb = XGBRegressor(**best_params["xgb"])
        xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

        # LGB (log 타깃)
        lgbm = LGBMRegressor(**best_params["lgb"])
        lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # CAT (log 타깃)
        cat = CatBoostRegressor(**best_params["cat"])
        cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)

        # Tweedie (원시 타깃)
        twd = LGBMRegressor(**best_twd)
        twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # OOF 저장(모두 로그 스케일 통일)
        oof_meta[va_idx, 0] = xgb.predict(X_va_s)
        oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
        oof_meta[va_idx, 2] = cat.predict(X_va_s)
        pred_raw_va_twd = twd.predict(X_va_s)
        oof_meta[va_idx, 3] = log1p_pos(pred_raw_va_twd)

        # 테스트 메타 누적
        test_meta_accum[:, 0] += xgb.predict(X_te_s)
        test_meta_accum[:, 1] += lgbm.predict(X_te_s)
        test_meta_accum[:, 2] += cat.predict(X_te_s)
        pred_raw_te_twd = twd.predict(X_te_s)
        test_meta_accum[:, 3] += log1p_pos(pred_raw_te_twd)

    test_meta = test_meta_accum / kf.get_n_splits()

    # ===== 로컬: SMAPE 직접최적화 블렌딩 (기하/산술 중 선택) + 스미어링 + 단조 캘리브레이션 =====
    meta_tmp = Ridge(alpha=0.1).fit(oof_meta, y_full_log)
    oof_pred_log_ridge = meta_tmp.predict(oof_meta)
    resid_r = y_full_log - oof_pred_log_ridge
    S = float(np.mean(np.exp(resid_r)))

    te_pred_geo,   w_geo,   s_geo   = blend_smape_minimizer(oof_meta, y_full_log, test_meta, mode="geo",   step=0.05, smear_S=S)
    te_pred_arith, w_arith, s_arith = blend_smape_minimizer(oof_meta, y_full_log, test_meta, mode="arith", step=0.05, smear_S=S)

    if s_geo <= s_arith:
        chosen_mode = "geo"
        chosen_w    = w_geo
        chosen_s    = s_geo
        oof_log_raw = np.log(np.expm1(oof_meta).clip(min=0) + 1e-9)
        p_oof = np.exp(oof_log_raw @ chosen_w) * S  # 원스케일
        te_pred = te_pred_geo                       # 원스케일
    else:
        chosen_mode = "arith"
        chosen_w    = w_arith
        chosen_s    = s_arith
        p_oof = np.expm1(oof_meta @ chosen_w) * S   # 원스케일
        te_pred = te_pred_arith                     # 원스케일

    # 단조 캘리브레이션 (OOF 기준)
    y_oof = np.expm1(y_full_log)
    a_grid = np.linspace(0.8, 1.2, 21)
    b_grid = np.linspace(0.85, 1.15, 31)
    best = (1.0, 1.0, smape(y_oof, p_oof))
    for a in a_grid:
        for b in b_grid:
            s = smape(y_oof, a*(p_oof**b))
            if s < best[2]:
                best = (a, b, s)
    a_opt, b_opt, _ = best

    # 로컬 최종(원스케일)
    p_oof = a_opt * (p_oof ** b_opt)
    te_pred = a_opt * (te_pred ** b_opt)

    print(f"[Building {bno}] mode={chosen_mode}, OOF SMAPE={best[2]:.4f}")

    return te_pred.tolist(), float(best[2])

# ==============================
# 12) 건물유형 스태킹 → 건물별 병렬 스태킹 → 최종 앙상블(단순 평균)
# ==============================
# 1) 건물유형(Type-wise) 스태킹
TYPE_OOF_RAW, TYPE_TEST_RAW = train_type_blender(train, test, features, target, seed=seed)

# 2) 건물별(Local) 스태킹 (병렬)
bld_list = list(np.sort(test["건물번호"].unique()))
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_building_kfold)(bno) for bno in bld_list
)

# 3) 결과 취합
preds_full_local = np.zeros(len(test), dtype=float)
val_smapes = []
for bno, (preds, sm) in zip(bld_list, results):
    idx = (test["건물번호"] == bno).values
    assert idx.sum() == len(preds), f"building {bno}: test rows={idx.sum()}, preds={len(preds)}"
    preds_full_local[idx] = preds
    if not np.isnan(sm):
        val_smapes.append(sm)

# 4) 최종 앙상블: 건물별(Local) + 건물유형(Type-wise) → 단순 평균
final_preds = 0.5 * preds_full_local + 0.5 * TYPE_TEST_RAW

assert len(final_preds) == len(samplesub), f"final preds:{len(final_preds)}, sample:{len(samplesub)}"
samplesub["answer"] = final_preds

today = datetime.datetime.now().strftime("%Y%m%d")
avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
filename = f"submission_stack_LOCAL_PLUS_TYPE_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (건물별 OOF 기준): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")