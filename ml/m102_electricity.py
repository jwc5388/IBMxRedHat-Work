

####current best !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# exit()
# import os
# import json
# import random
# import warnings
# import datetime
# import numpy as np
# import pandas as pd
# import optuna

# from sklearn.model_selection import KFold, TimeSeriesSplit  # ← 변경: TSS 추가
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
# seed = 2025
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
#     # 설비 용량은 '-' → 0, 숫자로 캐스팅
#     for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
#         if col in bi.columns:
#             bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
#     # 설비 유무 플래그
#     bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
#     bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

#     # 필요한 컬럼만 추려 병합 (없으면 스킵)
#     keep_cols = ['건물번호']
#     for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
#         if c in bi.columns: keep_cols.append(c)
#     bi = bi[keep_cols].drop_duplicates('건물번호')

#     train = train.merge(bi, on='건물번호', how='left')
#     test  = test.merge(bi, on='건물번호',  how='left')

# # === 1) 공통 시간 파생
# def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
#     df['hour']      = df['일시'].dt.hour
#     df['day']       = df['일시'].dt.day
#     df['month']     = df['일시'].dt.month
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['is_weekend']       = (df['dayofweek'] >= 5).astype(int)
#     df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
#     df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
#     df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
#     df['sin_month'] = np.sin(2*np.pi*df['month']/12)
#     df['cos_month'] = np.cos(2*np.pi*df['month']/12)
#     df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
#     df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['DI'] = 0.0
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

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

# # === 3) 일별 온도 통계 (train/test 동일 로직)
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

# # === 4) CDH / THI / WCT (train/test 동일)
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

# # === 5) 시간대 전력 통계 (train으로 계산 → 둘 다 merge)
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

# # === 6) (선택) 이상치 제거: 0 kWh 제거
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # === 7) 범주형 건물유형 인코딩 (있을 때만)
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
#     cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
#     train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
#     test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# # 1) 공통 feature (train/test 둘 다 있는 컬럼만 선택)
# feature_candidates = [
#     '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
#     '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
#     'hour','day','month','dayofweek','is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     'DI','expected_solar',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     'CDH','THI','WCT',
#     'day_hour_mean','day_hour_std'
# ]
# features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# # 2) target 명시
# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# # 3) 최종 입력/타깃 데이터
# X = train[features].values
# y = np.log1p(train[target].values.astype(float))
# X_test_raw = test[features].values
# ts = train['일시']  # 내부/외부 분할에 참고 가능

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y shape: {y.shape}")

# # ------------------------------
# # SMAPE
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# # ------------------------------
# # 내부 튜닝: TimeSeriesSplit(n_splits=3)로 교체 (시계열 보전)
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
#     """건물당 1회만 튜닝하고 JSON 저장/로드
#        변경: 내부 검증을 TimeSeriesSplit으로(시계열 순서=order_index에 맞춰 정렬)"""
#     os.makedirs(param_dir, exist_ok=True)
#     paths = {
#         "xgb": os.path.join(param_dir, f"{bno}_xgb.json"),
#         "lgb": os.path.join(param_dir, f"{bno}_lgb.json"),
#         "cat": os.path.join(param_dir, f"{bno}_cat.json"),
#     }
#     params = {}

#     # 시계열 정렬
#     X_sorted = X_full[order_index]
#     y_sorted = y_full[order_index]

#     # XGB
#     if os.path.exists(paths["xgb"]):
#         with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["xgb"] = st.best_params
#         with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

#     # LGB
#     if os.path.exists(paths["lgb"]):
#         with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["lgb"] = st.best_params
#         with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

#     # CAT
#     if os.path.exists(paths["cat"]):
#         with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["cat"] = st.best_params
#         with open(paths["cat"], "w") as f: json.dump(params["cat"], f)

#     return params

# # ------------------------------
# # Ridge 튜닝(메타) - OOF 행렬 기반으로 건물당 1회
# # ------------------------------
# def objective_ridge_on_oof(trial, oof_meta, y_full):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     ridge = Ridge(alpha=alpha)
#     # 간단히 5-Fold CV로 OOF 메타 최적화
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     scores = []
#     for tr_idx, va_idx in kf.split(oof_meta):
#         ridge.fit(oof_meta[tr_idx], y_full[tr_idx])
#         preds = ridge.predict(oof_meta[va_idx])
#         scores.append(smape_exp(y_full[va_idx], preds))
#     return float(np.mean(scores))

# def process_building_kfold(bno):
#     print(f"🏢 building {bno} KFold...")
#     param_dir = os.path.join(path, "optuna_params")
#     os.makedirs(param_dir, exist_ok=True)

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     X_full = tr_b[features].values
#     y_full = np.log1p(tr_b[target].values.astype(float))
#     X_test = te_b[features].values

#     # 시계열 정렬 인덱스 (내부 튜닝용)
#     order = np.argsort(tr_b['일시'].values)

#     # 건물당 1회 Optuna 튜닝(TimeSeriesSplit) 후 파라미터 로드
#     best_params = get_or_tune_params_once(bno, X_full, y_full, order, param_dir)

#     # 외부 KFold(그대로 유지)
#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     # ----- 진짜 OOF 스태킹 준비 -----
#     n_train_b = len(tr_b)
#     n_test_b  = len(te_b)
#     base_models = ["xgb", "lgb", "cat"]
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)  # 각 행=훈련행, 열=베이스모델
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     # 폴드 루프: base 학습 → 검증셋 예측을 OOF에 채우고, 테스트는 폴드 평균을 누적
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")
#         X_tr, X_va = X_full[tr_idx], X_full[va_idx]
#         y_tr, y_va = y_full[tr_idx], y_full[va_idx]

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_test)

#         # 베이스 모델들 학습 (튜닝 파라미터 사용, ES는 기존 방식 유지)
#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)

#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)],
#                  callbacks=[lgb.early_stopping(50, verbose=False)])

#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va),
#                 early_stopping_rounds=50, verbose=0)

#         # 검증셋 OOF 예측 저장
#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)

#         # 테스트 메타 입력 누적(폴드 평균)
#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # ----- 메타(Ridge) 건물당 1회 튜닝/학습: OOF 행렬 기반 -----
#     ridge_key = f"{bno}_ridge"
#     ridge_path = os.path.join(param_dir, f"{ridge_key}.json")

#     if os.path.exists(ridge_path):
#         with open(ridge_path, "r") as f:
#             ridge_params = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full), n_trials=30)
#         ridge_params = st.best_params
#         with open(ridge_path, "w") as f:
#             json.dump(ridge_params, f)

#     meta = Ridge(alpha=ridge_params["alpha"])
#     meta.fit(oof_meta, y_full)

#     # OOF 성능(전체 기준) 및 테스트 예측
#     oof_pred = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full, oof_pred))

#     te_pred_log = meta.predict(test_meta)
#     te_pred = np.expm1(te_pred_log)

#     return te_pred.tolist(), avg_smape

# # ==============================
# # 12) 병렬 실행
# # ==============================
# results = Parallel(n_jobs=-1, backend="loky")(
#     delayed(process_building_kfold)(bno) for bno in train["건물번호"].unique()
# )

# final_preds, val_smapes = [], []
# for preds, sm in results:
#     final_preds.extend(preds)
#     val_smapes.append(sm)

# samplesub["answer"] = final_preds
# today = datetime.datetime.now().strftime("%Y%m%d")
# avg_smape = float(np.mean(val_smapes))
# filename = f"submission_stack_optuna_stdcols_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")




















# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# # ===== 0) 경로 =====
# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# PATH = os.path.join(BASE_PATH, "_data/dacon/electricity/")
# OUT_DIR = os.path.join(PATH, "holiday_dates")
# os.makedirs(OUT_DIR, exist_ok=True)

# # ===== 1) 휴무일 탐지 함수 =====
# def detect_holidays_simple(file_path, name):
#     df = pd.read_csv(file_path)
#     df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H', errors='coerce')
#     df['date'] = df['일시'].dt.date
#     df['dow'] = df['일시'].dt.dayofweek  # 0=월, 6=일
    
#     # 1) 주말
#     # weekend_days = set(df.loc[df['dow'] >= 5, 'date'])

#     # 2) 한국 공휴일
#     try:
#         import holidays
#         years = sorted({d.year for d in df['date']})
#         hol = holidays.KR(years=years)
#         official_holidays = {pd.Timestamp(d).date() for d in hol.keys()}
#     except ImportError:
#         official_holidays = set()

#     # 합치기
#     all_holidays = sorted(official_holidays)

#     # CSV 저장
#     out_path = os.path.join(OUT_DIR, f'holiday_dates_{name}.csv')
#     pd.DataFrame({'date': all_holidays}).to_csv(out_path, index=False)
#     print(f"[{name}] 휴무/공휴일 후보 {len(all_holidays)}개 저장 → {out_path}")

# # ===== 2) 실행 =====
# detect_holidays_simple(os.path.join(PATH, "train.csv"), "train")
# detect_holidays_simple(os.path.join(PATH, "test.csv"), "test")
# # exit()




















## 휴무 추정일만 추가

# import os
# import json
# import random
# import warnings
# import datetime
# import numpy as np
# import pandas as pd
# import optuna

# from sklearn.model_selection import KFold, TimeSeriesSplit  # ← 내부 튜닝용
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from joblib import Parallel, delayed
# from optuna.samplers import TPESampler

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgb
# import tensorflow as tf

# # -------------------------
# # NEW: 공휴일 라이브러리
# # -------------------------
# from datetime import timedelta
# import holidays

# warnings.filterwarnings("ignore")

# # ==============================
# # 0) 시드 / 경로 / 옵션
# # ==============================
# seed = 2025
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# path = os.path.join(BASE_PATH, "_data/dacon/electricity/")

# # ---- NEW: 휴일/휴무 처리 스위치
# DROP_HOLIDAY = False           # True면 train에서 공휴일 제거
# DROP_CLOSURE = False           # True면 train에서 휴무 추정일 제거
# WEIGHT_HOLIDAY = 0.6           # 공휴일 가중치
# WEIGHT_CLOSURE = 0.3          # 휴무 추정일 가중치

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
#     # 설비 용량은 '-' → 0, 숫자로 캐스팅
#     for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
#         if col in bi.columns:
#             bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
#     # 설비 유무 플래그
#     bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
#     bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

#     # 필요한 컬럼만 추려 병합 (없으면 스킵)
#     keep_cols = ['건물번호']
#     for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
#         if c in bi.columns: keep_cols.append(c)
#     bi = bi[keep_cols].drop_duplicates('건물번호')

#     train = train.merge(bi, on='건물번호', how='left')
#     test  = test.merge(bi, on='건물번호',  how='left')

# # === 1) 공통 시간 파생
# def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
#     df['hour']      = df['일시'].dt.hour
#     df['day']       = df['일시'].dt.day
#     df['month']     = df['일시'].dt.month
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['is_weekend']       = (df['dayofweek'] >= 5).astype(int)
#     df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
#     df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
#     df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
#     df['sin_month'] = np.sin(2*np.pi*df['month']/12)
#     df['cos_month'] = np.cos(2*np.pi*df['month']/12)
#     df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
#     df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['DI'] = 0.0
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

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

# # -------------------------
# # NEW: 캘린더 플래그 + 휴무 추정일 라벨
# # -------------------------
# def add_calendar_flags(df, country='KR'):
#     df = df.copy()
#     df['date'] = df['일시'].dt.date
#     years = sorted({d.year for d in df['일시']})
#     hol = holidays.country_holidays(country=country, years=years)
#     is_holiday = df['date'].apply(lambda d: d in hol).astype(int)
#     prev_is_holiday = df['date'].apply(lambda d: (d - timedelta(days=1)) in hol).astype(int)
#     next_is_holiday = df['date'].apply(lambda d: (d + timedelta(days=1)) in hol).astype(int)
#     df['is_holiday'] = is_holiday
#     df['is_holiday_prev'] = prev_is_holiday
#     df['is_holiday_next'] = next_is_holiday
#     return df

# def tag_closure_like_days(df, target_col='전력소비량(kWh)'):
#     df = df.copy()
#     if target_col not in df.columns:
#         df['is_closure_like_day'] = 0
#         return df
#     df['date'] = df['일시'].dt.date
#     daily = (df.groupby(['건물번호','date'])[target_col]
#                .sum().rename('day_total').reset_index())
#     tmp = df[['건물번호','date','dayofweek']].drop_duplicates()
#     daily = daily.merge(tmp, on=['건물번호','date'], how='left')
#     base = (daily.groupby(['건물번호','dayofweek'])['day_total']
#                   .median().rename('base_median').reset_index())
#     p10  = (daily.groupby(['건물번호'])['day_total']
#                   .quantile(0.10).rename('p10').reset_index())
#     daily = (daily.merge(base, on=['건물번호','dayofweek'], how='left')
#                   .merge(p10, on='건물번호', how='left'))
#     daily['closure_like'] = ((daily['day_total'] < 0.4*daily['base_median']) |
#                              (daily['day_total'] <= daily['p10'])).astype(int)
#     df = df.merge(daily[['건물번호','date','closure_like']],
#                   on=['건물번호','date'], how='left')
#     df['is_closure_like_day'] = df['closure_like'].fillna(0).astype(int)
#     df.drop(columns=['closure_like'], inplace=True)
#     return df

# # 적용
# train = add_calendar_flags(train, country='KR')
# test  = add_calendar_flags(test,  country='KR')
# train = tag_closure_like_days(train, target_col='전력소비량(kWh)')
# test  = tag_closure_like_days(test,  target_col='전력소비량(kWh)')

# # === 5) 시간대 전력 통계 (train으로 계산 → 둘 다 merge)
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

# # === 6) (선택) 이상치 제거: 0 kWh 제거
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # -------------------------
# # NEW: 휴일/휴무 드롭 또는 가중치 부여
# # -------------------------
# if DROP_HOLIDAY:
#     train = train.loc[train['is_holiday'] == 0].reset_index(drop=True)
# if DROP_CLOSURE:
#     train = train.loc[train['is_closure_like_day'] == 0].reset_index(drop=True)

# train['_sample_weight'] = 1.0
# if not DROP_HOLIDAY:
#     train.loc[train['is_holiday'] == 1, '_sample_weight'] = WEIGHT_HOLIDAY
# if not DROP_CLOSURE:
#     train.loc[train['is_closure_like_day'] == 1, '_sample_weight'] = WEIGHT_CLOSURE

# # === 7) 범주형 건물유형 인코딩 (있을 때만)
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
#     cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
#     train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
#     test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# # 1) 공통 feature (train/test 둘 다 있는 컬럼만 선택)
# feature_candidates = [
#     '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
#     '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
#     'hour','day','month','dayofweek','is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     'DI','expected_solar',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     'CDH','THI','WCT',
#     'day_hour_mean','day_hour_std',
#     # NEW: 캘린더/휴무 플래그
#     'is_holiday','is_holiday_prev','is_holiday_next','is_closure_like_day'
# ]
# features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# # 2) target
# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# # 3) 최종 입력/타깃 데이터
# X = train[features].values
# y = np.log1p(train[target].values.astype(float))
# W = train.get('_sample_weight', pd.Series(1.0, index=train.index)).values  # NEW
# X_test_raw = test[features].values
# ts = train['일시']

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y shape: {y.shape}")

# # ------------------------------
# # SMAPE
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# # ------------------------------
# # 내부 튜닝: TimeSeriesSplit(n_splits=3)
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

#     # XGB
#     if os.path.exists(paths["xgb"]):
#         with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["xgb"] = st.best_params
#         with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

#     # LGB
#     if os.path.exists(paths["lgb"]):
#         with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=30)
#         params["lgb"] = st.best_params
#         with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

#     # CAT
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

# def process_building_kfold(bno):
#     print(f"🏢 building {bno} KFold...")
#     param_dir = os.path.join(path, "optuna_params")
#     os.makedirs(param_dir, exist_ok=True)

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     X_full = tr_b[features].values
#     y_full = np.log1p(tr_b[target].values.astype(float))
#     W_full = tr_b.get('_sample_weight', pd.Series(1.0, index=tr_b.index)).values  # NEW
#     X_test = te_b[features].values

#     order = np.argsort(tr_b['일시'].values)
#     best_params = get_or_tune_params_once(bno, X_full, y_full, order, param_dir)

#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     n_train_b = len(tr_b)
#     n_test_b  = len(te_b)
#     base_models = ["xgb", "lgb", "cat"]
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")
#         X_tr, X_va = X_full[tr_idx], X_full[va_idx]
#         y_tr, y_va = y_full[tr_idx], y_full[va_idx]
#         w_tr = W_full[tr_idx]  # NEW

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_test)

#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr, sample_weight=w_tr, eval_set=[(X_va_s, y_va)], verbose=False)

#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr, sample_weight=w_tr,
#                  eval_set=[(X_va_s, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr, sample_weight=w_tr,
#                 eval_set=(X_va_s, y_va), early_stopping_rounds=50, verbose=0)

#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)

#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)

#     test_meta = test_meta_accum / kf.get_n_splits()

#     ridge_key = f"{bno}_ridge"
#     ridge_path = os.path.join(param_dir, f"{ridge_key}.json")

#     if os.path.exists(ridge_path):
#         with open(ridge_path, "r") as f:
#             ridge_params = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full), n_trials=30)
#         ridge_params = st.best_params
#         with open(ridge_path, "w") as f:
#             json.dump(ridge_params, f)

#     meta = Ridge(alpha=ridge_params["alpha"])
#     meta.fit(oof_meta, y_full)

#     oof_pred = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full, oof_pred))

#     te_pred_log = meta.predict(test_meta)
#     te_pred = np.expm1(te_pred_log)

#     return te_pred.tolist(), avg_smape

# # ==============================
# # 12) 병렬 실행
# # ==============================
# results = Parallel(n_jobs=-1, backend="loky")(
#     delayed(process_building_kfold)(bno) for bno in train["건물번호"].unique()
# )

# final_preds, val_smapes = [], []
# for preds, sm in results:
#     final_preds.extend(preds)
#     val_smapes.append(sm)

# samplesub["answer"] = final_preds
# today = datetime.datetime.now().strftime("%Y%m%d")
# avg_smape = float(np.mean(val_smapes))
# filename = f"submission_stack_optuna_cal_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")

# exit()








### 다 추가하고, 공휴일 처리까지 된거



# # exit()
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

# from datetime import timedelta
# import holidays

# warnings.filterwarnings("ignore")

# # ==============================
# # 0) 시드 / 경로 / 옵션
# # ==============================
# seed = 2025
# random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# path = os.path.join(BASE_PATH, "_data/dacon/electricity/")

# # 휴일/휴무 처리 스위치 (원하는 값으로 바꿔써)
# DROP_HOLIDAY  = True     # True면 train에서 공휴일 제거
# DROP_CLOSURE  = True     # True면 train에서 휴무 추정일 제거
# WEIGHT_HOLIDAY = 0.7     # 공휴일 가중치 (DROP_HOLIDAY=False일 때만 적용)
# WEIGHT_CLOSURE = 0.5     # 휴무 추정일 가중치 (DROP_CLOSURE=False일 때만)

# # ==============================
# # 안전 로그 변환 유틸 (음수 방지)
# # ==============================
# def log1p_pos(a):
#     a = np.asarray(a, dtype=float)
#     return np.log1p(np.maximum(a, 0.0))

# # ==============================
# # 데이터 로드
# # ==============================
# buildinginfo = pd.read_csv(os.path.join(path, "building_info.csv"))
# train = pd.read_csv(os.path.join(path, "train.csv"))
# test  = pd.read_csv(os.path.join(path, "test.csv"))
# samplesub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

# # === building_info 병합
# bi = buildinginfo.copy() if 'buildinginfo' in globals() else None
# if bi is not None:
#     for col in ['태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','연면적(m2)','냉방면적(m2)']:
#         if col in bi.columns:
#             bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
#     bi['태양광_유무'] = (bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int) if '태양광용량(kW)' in bi.columns else 0
#     bi['ESS_유무']  = (bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int) if 'ESS저장용량(kWh)' in bi.columns else 0

#     keep_cols = ['건물번호']
#     for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
#         if c in bi.columns: keep_cols.append(c)
#     bi = bi[keep_cols].drop_duplicates('건물번호')
#     train = train.merge(bi, on='건물번호', how='left')
#     test  = test.merge(bi,  on='건물번호',  how='left')

# # ==============================
# # 1) 시간 파생
# # ==============================
# def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
#     df['hour']      = df['일시'].dt.hour
#     df['day']       = df['일시'].dt.day
#     df['month']     = df['일시'].dt.month
#     df['dayofweek'] = df['일시'].dt.dayofweek
#     df['is_weekend']       = (df['dayofweek'] >= 5).astype(int)
#     df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
#     df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
#     df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
#     df['sin_month'] = np.sin(2*np.pi*df['month']/12)
#     df['cos_month'] = np.cos(2*np.pi*df['month']/12)
#     df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
#     df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
#     else:
#         df['DI'] = 0.0
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

# # 2) expected_solar
# if '일사(MJ/m2)' in train.columns:
#     solar_proxy = (train.groupby(['month','hour'])['일사(MJ/m2)']
#                         .mean().reset_index().rename(columns={'일사(MJ/m2)':'expected_solar'}))
#     train = train.merge(solar_proxy, on=['month','hour'], how='left')
#     test  = test.merge(solar_proxy,  on=['month','hour'], how='left')
# else:
#     train['expected_solar'] = 0.0; test['expected_solar'] = 0.0

# train['expected_solar'] = train['expected_solar'].fillna(0)
# test['expected_solar']  = test['expected_solar'].fillna(0)

# # 3) 일별 온도 통계
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

# train = add_daily_temp_stats_kor(train); test = add_daily_temp_stats_kor(test)

# # 4) CDH / THI / WCT
# def add_CDH_kor(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if '기온(°C)' not in df.columns:
#         df['CDH'] = 0.0; return df
#     def _cdh_1d(x):
#         cs = np.cumsum(x - 26)
#         return np.concatenate((cs[:11], cs[11:] - cs[:-11])) if len(x) >= 12 else np.zeros_like(x, dtype=float)
#     parts = []
#     for bno, g in df.sort_values('일시').groupby('건물번호'):
#         arr = g['기온(°C)'].to_numpy()
#         parts.append(pd.Series(_cdh_1d(arr), index=g.index))
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

# train = add_CDH_kor(train); test = add_CDH_kor(test)
# train = add_THI_WCT_kor(train); test = add_THI_WCT_kor(test)

# # -------------------------
# # 캘린더 플래그 + 휴무 추정일
# # -------------------------
# def add_calendar_flags(df, country='KR'):
#     df = df.copy()
#     df['date'] = df['일시'].dt.date
#     years = sorted({d.year for d in df['일시']})
#     hol = holidays.country_holidays(country=country, years=years)
#     df['is_holiday']       = df['date'].apply(lambda d: d in hol).astype(int)
#     df['is_holiday_prev']  = df['date'].apply(lambda d: (d - timedelta(days=1)) in hol).astype(int)
#     df['is_holiday_next']  = df['date'].apply(lambda d: (d + timedelta(days=1)) in hol).astype(int)
#     return df

# def tag_closure_like_days(df, target_col='전력소비량(kWh)'):
#     df = df.copy()
#     if target_col not in df.columns:
#         df['is_closure_like_day'] = 0; return df
#     df['date'] = df['일시'].dt.date
#     daily = df.groupby(['건물번호','date'])[target_col].sum().rename('day_total').reset_index()
#     tmp = df[['건물번호','date','dayofweek']].drop_duplicates()
#     daily = daily.merge(tmp, on=['건물번호','date'], how='left')
#     base = daily.groupby(['건물번호','dayofweek'])['day_total'].median().rename('base_median').reset_index()
#     p10  = daily.groupby(['건물번호'])['day_total'].quantile(0.10).rename('p10').reset_index()
#     daily = daily.merge(base, on=['건물번호','dayofweek'], how='left').merge(p10, on='건물번호', how='left')
#     daily['closure_like'] = ((daily['day_total'] < 0.4*daily['base_median']) |
#                              (daily['day_total'] <= daily['p10'])).astype(int)
#     df = df.merge(daily[['건물번호','date','closure_like']], on=['건물번호','date'], how='left')
#     df['is_closure_like_day'] = daily_val = df['closure_like'].fillna(0).astype(int)
#     df.drop(columns=['closure_like'], inplace=True)
#     return df

# train = add_calendar_flags(train, 'KR'); test = add_calendar_flags(test, 'KR')
# train = tag_closure_like_days(train, '전력소비량(kWh)')
# test  = tag_closure_like_days(test,  '전력소비량(kWh)')  # test에는 0으로만 채워짐(타깃 없음)

# # 5) 시간대 전력 통계
# if '전력소비량(kWh)' in train.columns:
#     pm = (train.groupby(['건물번호','hour','dayofweek'])['전력소비량(kWh)']
#                .agg(['mean','std']).reset_index()
#                .rename(columns={'mean':'day_hour_mean','std':'day_hour_std'}))
#     train = train.merge(pm, on=['건물번호','hour','dayofweek'], how='left')
#     test  = test.merge(pm,  on=['건물번호','hour','dayofweek'], how='left')
# else:
#     train['day_hour_mean']=0.0; train['day_hour_std']=0.0
#     test['day_hour_mean']=0.0;  test['day_hour_std']=0.0

# # 6) 0 kWh 제거(선택)
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # 휴일/휴무 드롭 또는 가중치
# if DROP_HOLIDAY:
#     train = train.loc[train['is_holiday'] == 0].reset_index(drop=True)
# if DROP_CLOSURE:
#     train = train.loc[train['is_closure_like_day'] == 0].reset_index(drop=True)

# train['_sample_weight'] = 1.0
# if not DROP_HOLIDAY:
#     train.loc[train['is_holiday'] == 1, '_sample_weight'] = WEIGHT_HOLIDAY
# if not DROP_CLOSURE:
#     train.loc[train['is_closure_like_day'] == 1, '_sample_weight'] = WEIGHT_CLOSURE

# # 범주형 인코딩
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
#     cat_map = {cat:i for i,cat in enumerate(both.cat.categories)}
#     train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
#     test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# # ==============================
# # 최종 피처/타깃
# # ==============================
# feature_candidates = [
#     '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
#     '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
#     'hour','day','month','dayofweek','is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     'DI','expected_solar',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     'CDH','THI','WCT',
#     'day_hour_mean','day_hour_std',
#     'is_holiday','is_holiday_prev','is_holiday_next','is_closure_like_day'
# ]
# features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# X = train[features].values
# y_raw = train[target].values.astype(float)
# y_log = log1p_pos(y_raw)
# W = train.get('_sample_weight', pd.Series(1.0, index=train.index)).values
# X_test_raw = test[features].values
# ts = train['일시']

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# # ==============================
# # SMAPE (로그 스케일용)
# # ==============================
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# # ==============================
# # 튜닝 함수들 (TSS)
# # ==============================
# def tune_xgb_tss(trial, X_full_sorted, y_full_log, seed=seed):
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
#     scores=[]
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_log[tr_idx], y_full_log[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = XGBRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_lgb_tss(trial, X_full_sorted, y_full_log, seed=seed):
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
#     scores=[]
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_log[tr_idx], y_full_log[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = LGBMRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_cat_tss(trial, X_full_sorted, y_full_log, seed=seed):
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
#     scores=[]
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_log[tr_idx], y_full_log[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = CatBoostRegressor(**params)
#         model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=50, verbose=0)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# # Tweedie는 raw 타깃 기준으로 튜닝 → 평가/스태킹은 log1p_pos로 변환
# def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_raw, seed=seed):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 31, 256),
#         "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
#         "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
#         "objective": "tweedie",
#         "metric": "mae",
#         "random_state": seed,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores=[]
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr_raw, y_va_raw = y_full_raw[tr_idx], y_full_raw[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = LGBMRegressor(**params)
#         model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])
#         pred_raw = model.predict(X_va_s)
#         y_va_log  = log1p_pos(y_va_raw)
#         pred_log  = log1p_pos(pred_raw)
#         scores.append(smape_exp(y_va_log, pred_log))
#     return float(np.mean(scores))

# # ==============================
# # 파라미터 저장/로드 (건물당 1회)
# # ==============================
# def get_or_tune_params_once(bno, X_full, y_full_log, order_index, param_dir):
#     os.makedirs(param_dir, exist_ok=True)
#     paths = {
#         "xgb": os.path.join(param_dir, f"{bno}_xgb.json"),
#         "lgb": os.path.join(param_dir, f"{bno}_lgb.json"),
#         "cat": os.path.join(param_dir, f"{bno}_cat.json"),
#         "twd": os.path.join(param_dir, f"{bno}_twd.json"),
#     }
#     params = {}

#     X_sorted = X_full[order_index]
#     y_sorted_log = y_full_log[order_index]
#     y_sorted_raw = np.expm1(y_sorted_log)

#     # XGB
#     if os.path.exists(paths["xgb"]):
#         with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted_log), n_trials=30)
#         params["xgb"] = st.best_params; open(paths["xgb"], "w").write(json.dumps(params["xgb"]))

#     # LGB (MAE)
#     if os.path.exists(paths["lgb"]):
#         with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted_log), n_trials=30)
#         params["lgb"] = st.best_params; open(paths["lgb"], "w").write(json.dumps(params["lgb"]))

#     # CAT
#     if os.path.exists(paths["cat"]):
#         with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted_log), n_trials=30)
#         params["cat"] = st.best_params; open(paths["cat"], "w").write(json.dumps(params["cat"]))

#     # LGB Tweedie (raw target)
#     if os.path.exists(paths["twd"]):
#         with open(paths["twd"], "r") as f: params["twd"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=30)
#         params["twd"] = st.best_params; open(paths["twd"], "w").write(json.dumps(params["twd"]))

#     return params

# # ==============================
# # 메타 튜닝 (Ridge)
# # ==============================
# def objective_ridge_on_oof(trial, oof_meta, y_full_log):
#     alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
#     ridge = Ridge(alpha=alpha)
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     scores = []
#     for tr_idx, va_idx in kf.split(oof_meta):
#         ridge.fit(oof_meta[tr_idx], y_full_log[tr_idx])
#         preds = ridge.predict(oof_meta[va_idx])
#         scores.append(smape_exp(y_full_log[va_idx], preds))
#     return float(np.mean(scores))

# # ==============================
# # 건물별 학습
# # ==============================
# def process_building_kfold(bno):
#     print(f"🏢 building {bno} KFold...")
#     param_dir = os.path.join(path, "optuna_params"); os.makedirs(param_dir, exist_ok=True)

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     X_full = tr_b[features].values
#     y_full_raw = tr_b[target].values.astype(float)
#     y_full_log = log1p_pos(y_full_raw)
#     W_full = tr_b.get('_sample_weight', pd.Series(1.0, index=tr_b.index)).values
#     X_test = te_b[features].values

#     order = np.argsort(tr_b['일시'].values)
#     best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)

#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     n_train_b = len(tr_b)
#     n_test_b  = len(te_b)
#     base_models = ["xgb", "lgb", "cat", "twd"]
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")
#         X_tr, X_va = X_full[tr_idx], X_full[va_idx]
#         y_tr_log, y_va_log = y_full_log[tr_idx], y_full_log[va_idx]
#         y_tr_raw, y_va_raw = y_full_raw[tr_idx], y_full_raw[va_idx]
#         w_tr = W_full[tr_idx]

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr); X_va_s = sc.transform(X_va); X_te_s = sc.transform(X_test)

#         # XGB (log target)
#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr_log, sample_weight=w_tr, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         # LGB (log target)
#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr_log, sample_weight=w_tr,
#                  eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # CAT (log target)
#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr_log, sample_weight=w_tr,
#                 eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)

#         # Tweedie (raw target)
#         twd_params = best_params["twd"].copy()
#         twd_params.update({"objective":"tweedie","metric":"mae","random_state":seed})
#         twd = LGBMRegressor(**twd_params)
#         twd.fit(X_tr_s, y_tr_raw, sample_weight=w_tr,
#                 eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # OOF (log scale)
#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)
#         oof_meta[va_idx, 3] = log1p_pos(twd.predict(X_va_s))

#         # TEST meta 누적 (log scale)
#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)
#         test_meta_accum[:, 3] += log1p_pos(twd.predict(X_te_s))

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # 메타 튜닝/학습
#     ridge_key = f"{bno}_ridge"
#     ridge_path = os.path.join(param_dir, f"{ridge_key}.json")
#     if os.path.exists(ridge_path):
#         with open(ridge_path, "r") as f:
#             ridge_params = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full_log), n_trials=30)
#         ridge_params = st.best_params; open(ridge_path, "w").write(json.dumps(ridge_params))

#     meta = Ridge(alpha=ridge_params["alpha"])
#     meta.fit(oof_meta, y_full_log)

#     oof_pred_log = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full_log, oof_pred_log))

#     te_pred_log = meta.predict(test_meta)
#     te_pred = np.expm1(te_pred_log)

#     return te_pred.tolist(), avg_smape

# # ==============================
# # 병렬 실행
# # ==============================
# results = Parallel(n_jobs=-1, backend="loky")(
#     delayed(process_building_kfold)(bno) for bno in train["건물번호"].unique()
# )

# final_preds, val_smapes = [], []
# for preds, sm in results:
#     final_preds.extend(preds)
#     val_smapes.append(sm)

# samplesub["answer"] = final_preds
# today = datetime.datetime.now().strftime("%Y%m%d")
# avg_smape = float(np.mean(val_smapes))
# filename = f"submission_stack_optuna_tweedie_fix_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")





import os
import json
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold, TimeSeriesSplit  # ← 변경: TSS 추가
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
seed = 2025
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
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    df['hour']      = df['일시'].dt.hour
    df['day']       = df['일시'].dt.day
    df['month']     = df['일시'].dt.month
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['is_weekend']       = (df['dayofweek'] >= 5).astype(int)
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
    '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
    '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
    'hour','day','month','dayofweek','is_weekend','is_working_hours',
    'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
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

# 3) 최종 입력/타깃 데이터 (기존 유지)
X = train[features].values
y_log = np.log1p(train[target].values.astype(float))   # 기존 베이스 모델용 (로그 타깃)
X_test_raw = test[features].values
ts = train['일시']  # 내부/외부 분할에 참고 가능

print(f"[확인] 사용 features 개수: {len(features)}")
print(f"[확인] target: {target}")
print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# ------------------------------
# SMAPE
# ------------------------------
def smape_exp(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# ========== Tweedie 전용 유틸 & 튜닝 추가 (NEW) ==========
def log1p_pos(arr):
    """음수 안전 로그 변환 (Tweedie 예측을 스태킹용 로그로 변환)"""
    return np.log1p(np.clip(arr, a_min=0, a_max=None))

def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
    """Tweedie는 원시 타깃으로 학습 → 예측을 log1p로 변환해 평가"""
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
        model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        pred_raw = model.predict(X_va_s)
        y_va_log = log1p_pos(y_va_raw)
        pred_log = log1p_pos(pred_raw)
        scores.append(smape_exp(y_va_log, pred_log))
    return float(np.mean(scores))

def get_or_tune_tweedie_once(bno, X_full, y_full_raw, order_index, param_dir):
    """Tweedie 전용 파라미터 (기존 함수와 분리, 기존 로직 보존)"""
    os.makedirs(param_dir, exist_ok=True)
    path_twd = os.path.join(param_dir, f"{bno}_twd.json")

    # 시계열 정렬
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
# ========================================================

# ------------------------------
# 기존 튜닝 함수들 (XGB/LGB/CAT) - 원본 유지
# ------------------------------
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

def get_or_tune_params_once(bno, X_full, y_full, order_index, param_dir):
    """건물당 1회만 튜닝하고 JSON 저장/로드 (기존 3모델)"""
    os.makedirs(param_dir, exist_ok=True)
    paths = {
        "xgb": os.path.join(param_dir, f"{bno}_xgb.json"),
        "lgb": os.path.join(param_dir, f"{bno}_lgb.json"),
        "cat": os.path.join(param_dir, f"{bno}_cat.json"),
    }
    params = {}

    # 시계열 정렬
    X_sorted = X_full[order_index]
    y_sorted = y_full[order_index]

    # XGB
    if os.path.exists(paths["xgb"]):
        with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=30)
        params["xgb"] = st.best_params
        with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

    # LGB
    if os.path.exists(paths["lgb"]):
        with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=30)
        params["lgb"] = st.best_params
        with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

    # CAT
    if os.path.exists(paths["cat"]):
        with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=30)
        params["cat"] = st.best_params
        with open(paths["cat"], "w") as f: json.dump(params["cat"], f)

    return params

# ------------------------------
# Ridge 튜닝(메타) - OOF 행렬 기반으로 건물당 1회
# ------------------------------
def objective_ridge_on_oof(trial, oof_meta, y_full):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in kf.split(oof_meta):
        ridge.fit(oof_meta[tr_idx], y_full[tr_idx])
        preds = ridge.predict(oof_meta[va_idx])
        scores.append(smape_exp(y_full[va_idx], preds))
    return float(np.mean(scores))

def process_building_kfold(bno):
    print(f"🏢 building {bno} KFold...")
    param_dir = os.path.join(path, "optuna_params")
    os.makedirs(param_dir, exist_ok=True)

    tr_b = train[train["건물번호"] == bno].copy()
    te_b = test[test["건물번호"] == bno].copy()

    X_full = tr_b[features].values
    y_full_log = np.log1p(tr_b[target].values.astype(float))  # 기존 3모델용
    y_full_raw = tr_b[target].values.astype(float)            # Tweedie용
    X_test = te_b[features].values

    # 시계열 정렬 인덱스 (내부 튜닝용)
    order = np.argsort(tr_b['일시'].values)

    # 기존 3모델 파라미터
    best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
    # Tweedie 파라미터 (NEW)
    best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

    # 외부 KFold(그대로 유지)
    kf = KFold(n_splits=8, shuffle=True, random_state=seed)

    # ----- 진짜 OOF 스태킹 준비 -----
    n_train_b = len(tr_b)
    n_test_b  = len(te_b)
    base_models = ["xgb", "lgb", "cat", "twd"]   # ← Tweedie 추가
    oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)  # 각 행=훈련행, 열=베이스모델
    test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

    # 폴드 루프
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        print(f" - fold {fold}")
        X_tr, X_va = X_full[tr_idx], X_full[va_idx]
        y_tr_log, y_va_log = y_full_log[tr_idx], y_full_log[va_idx]
        y_tr_raw, y_va_raw = y_full_raw[tr_idx], y_full_raw[va_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_test)

        # XGB (log 타깃)
        xgb = XGBRegressor(**best_params["xgb"])
        xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

        # LGB (log 타깃, objective=MAE)
        lgbm = LGBMRegressor(**best_params["lgb"])
        lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])

        # CAT (log 타깃)
        cat = CatBoostRegressor(**best_params["cat"])
        cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log),
                early_stopping_rounds=50, verbose=0)

        # Tweedie (원시 타깃)  ← NEW
        twd = LGBMRegressor(**best_twd)
        twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

        # 검증셋 OOF 예측 저장 (모두 로그 스케일로 통일)
        oof_meta[va_idx, 0] = xgb.predict(X_va_s)
        oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
        oof_meta[va_idx, 2] = cat.predict(X_va_s)

        pred_raw_va_twd = twd.predict(X_va_s)
        oof_meta[va_idx, 3] = log1p_pos(pred_raw_va_twd)   # ← 변환

        # 테스트 메타 입력 누적(폴드 평균) - 로그 스케일
        test_meta_accum[:, 0] += xgb.predict(X_te_s)
        test_meta_accum[:, 1] += lgbm.predict(X_te_s)
        test_meta_accum[:, 2] += cat.predict(X_te_s)

        pred_raw_te_twd = twd.predict(X_te_s)
        test_meta_accum[:, 3] += log1p_pos(pred_raw_te_twd)  # ← 변환

    test_meta = test_meta_accum / kf.get_n_splits()

    # ----- 메타(Ridge) 건물당 1회 튜닝/학습: OOF 행렬 기반 -----
    ridge_key = f"{bno}_ridge"
    ridge_path = os.path.join(param_dir, f"{ridge_key}.json")

    if os.path.exists(ridge_path):
        with open(ridge_path, "r") as f:
            ridge_params = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full_log), n_trials=30)
        ridge_params = st.best_params
        with open(ridge_path, "w") as f:
            json.dump(ridge_params, f)

    meta = Ridge(alpha=ridge_params["alpha"])
    meta.fit(oof_meta, y_full_log)

    # OOF 성능(전체 기준) 및 테스트 예측
    oof_pred = meta.predict(oof_meta)
    avg_smape = float(smape_exp(y_full_log, oof_pred))

    te_pred_log = meta.predict(test_meta)
    te_pred = np.expm1(te_pred_log)

    return te_pred.tolist(), avg_smape

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