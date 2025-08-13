

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







############진짜 현재 최고

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

# # 3) 최종 입력/타깃 데이터 (기존 유지)
# X = train[features].values
# y_log = np.log1p(train[target].values.astype(float))   # 기존 베이스 모델용 (로그 타깃)
# X_test_raw = test[features].values
# ts = train['일시']  # 내부/외부 분할에 참고 가능

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# # ------------------------------
# # SMAPE
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# # ========== Tweedie 전용 유틸 & 튜닝 추가 (NEW) ==========
# def log1p_pos(arr):
#     """음수 안전 로그 변환 (Tweedie 예측을 스태킹용 로그로 변환)"""
#     return np.log1p(np.clip(arr, a_min=0, a_max=None))

# def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
#     """Tweedie는 원시 타깃으로 학습 → 예측을 log1p로 변환해 평가"""
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
#     """Tweedie 전용 파라미터 (기존 함수와 분리, 기존 로직 보존)"""
#     os.makedirs(param_dir, exist_ok=True)
#     path_twd = os.path.join(param_dir, f"{bno}_twd.json")

#     # 시계열 정렬
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
# # ========================================================

# # ------------------------------
# # 기존 튜닝 함수들 (XGB/LGB/CAT) - 원본 유지
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
#     """건물당 1회만 튜닝하고 JSON 저장/로드 (기존 3모델)"""
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
#     y_full_log = np.log1p(tr_b[target].values.astype(float))  # 기존 3모델용
#     y_full_raw = tr_b[target].values.astype(float)            # Tweedie용
#     X_test = te_b[features].values

#     # 시계열 정렬 인덱스 (내부 튜닝용)
#     order = np.argsort(tr_b['일시'].values)

#     # 기존 3모델 파라미터
#     best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
#     # Tweedie 파라미터 (NEW)
#     best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

#     # 외부 KFold(그대로 유지)
#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     # ----- 진짜 OOF 스태킹 준비 -----
#     n_train_b = len(tr_b)
#     n_test_b  = len(te_b)
#     base_models = ["xgb", "lgb", "cat", "twd"]   # ← Tweedie 추가
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)  # 각 행=훈련행, 열=베이스모델
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     # 폴드 루프
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")
#         X_tr, X_va = X_full[tr_idx], X_full[va_idx]
#         y_tr_log, y_va_log = y_full_log[tr_idx], y_full_log[va_idx]
#         y_tr_raw, y_va_raw = y_full_raw[tr_idx], y_full_raw[va_idx]

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_test)

#         # XGB (log 타깃)
#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         # LGB (log 타깃, objective=MAE)
#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)],
#                  callbacks=[lgb.early_stopping(50, verbose=False)])

#         # CAT (log 타깃)
#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log),
#                 early_stopping_rounds=50, verbose=0)

#         # Tweedie (원시 타깃)  ← NEW
#         twd = LGBMRegressor(**best_twd)
#         twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
#                 callbacks=[lgb.early_stopping(50, verbose=False)])

#         # 검증셋 OOF 예측 저장 (모두 로그 스케일로 통일)
#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)

#         pred_raw_va_twd = twd.predict(X_va_s)
#         oof_meta[va_idx, 3] = log1p_pos(pred_raw_va_twd)   # ← 변환

#         # 테스트 메타 입력 누적(폴드 평균) - 로그 스케일
#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)

#         pred_raw_te_twd = twd.predict(X_te_s)
#         test_meta_accum[:, 3] += log1p_pos(pred_raw_te_twd)  # ← 변환

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # ----- 메타(Ridge) 건물당 1회 튜닝/학습: OOF 행렬 기반 -----
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

#     # OOF 성능(전체 기준) 및 테스트 예측
#     oof_pred = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full_log, oof_pred))

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










####################🌟🌟🌟🌟🌟🌟🌟🌟 current BEST 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟



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

# === 1-추가) 한국 공휴일 피처 (대체휴일/선거일 포함)
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

# === 5) 시간대 전력 통계(전체 train 집계) - 튜닝/베이스 참고용
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

# === 7) 범주형 건물유형 인코딩
if '건물유형' in train.columns and '건물유형' in test.columns:
    both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
    cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
    train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
    test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# ------------------------------
# Feature Set
# ------------------------------
feature_candidates = [
    # building_info
    '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
    # weather/raw
    '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
    # time parts & cycles
    'hour','day','month','dayofweek','is_weekend','is_working_hours',
    'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
    # engineered
    'DI','expected_solar',
    'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
    'CDH','THI','WCT',
    # target stats (전역 집계 - 폴드에서 덮어씀)
    'day_hour_mean','day_hour_std',
    # holidays
    'is_holiday','is_pre_holiday','is_post_holiday','holiday_7d_count','is_bridge_day'
]
features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# # 타깃 통계 누설 차단용 컬럼(폴드별 재계산 값)을 features에 보장
# for c in ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]:
#     if c not in features:
#         features.append(c)

# Target
target = '전력소비량(kWh)'
if target not in train.columns:
    raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# 최종 입력/타깃
X = train[features].values
y_log = np.log1p(train[target].values.astype(float))
X_test_raw = test[features].values

print(f"[확인] 사용 features 개수: {len(features)}")
print(f"[확인] target: {target}")
print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# 전처리 정합성 점검
print("len(test) =", len(test))
print("len(samplesub) =", len(samplesub))
print("건물 수 train vs test:", train["건물번호"].nunique(), test["건물번호"].nunique())
counts = test.groupby("건물번호").size()
bad = counts[counts != 168]
if len(bad):
    print("⚠️ 168이 아닌 건물 발견:\n", bad)
assert len(test) == len(samplesub), f"test:{len(test)} sample:{len(samplesub)}"

# ------------------------------
# SMAPE helpers
# ------------------------------
def smape_exp(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

def smape(y, yhat):
    return np.mean(200*np.abs(yhat - y)/(np.abs(yhat)+np.abs(y)+1e-6))

# ========== Tweedie 전용 유틸 & 튜닝 ==========
def log1p_pos(arr):
    return np.log1p(np.clip(arr, a_min=0, a_max=None))

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
        model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        pred_raw = model.predict(X_va_s)
        y_va_log = log1p_pos(y_va_raw)
        pred_log = log1p_pos(pred_raw)
        scores.append(smape_exp(y_va_log, pred_log))
    return float(np.mean(scores))

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

# ------------------------------
# 기존 튜닝 함수들 (XGB/LGB/CAT)
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
# Ridge 튜닝(메타) - OOF 행렬 기반
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

# ------------------------------
# [PATCH-1] 타깃통계(누설 차단) 유틸
# ------------------------------
def build_target_stats_fold(base_df, idx, target):
    base = base_df.iloc[idx]

    # 건물×hour 평균/표준편차를 한 번에 생성
    g1 = (base
          .groupby(["건물번호","hour"])[target]
          .agg(hour_mean="mean", hour_std="std")
          .reset_index())

    # 건물×hour×dow 평균/표준편차/중앙값
    g2 = base.groupby(["건물번호","hour","dayofweek"])[target]
    d_mean = g2.mean().rename("day_hour_mean").reset_index()
    d_std  = g2.std().rename("day_hour_std").reset_index()
    d_med  = g2.median().rename("day_hour_median").reset_index()

    # 건물×hour×month 평균
    g3 = (base
          .groupby(["건물번호","hour","month"])[target]
          .mean()
          .rename("month_hour_mean")
          .reset_index())

    return g1, d_mean, d_std, d_med, g3

def merge_target_stats(df, stats):
    g1, d_mean, d_std, d_med, g3 = stats
    out = df.merge(g1, on=["건물번호","hour"], how="left")  # hour_mean + hour_std 둘 다 붙음
    out = out.merge(d_mean, on=["건물번호","hour","dayofweek"], how="left")
    out = out.merge(d_std,  on=["건물번호","hour","dayofweek"], how="left")
    out = out.merge(d_med,  on=["건물번호","hour","dayofweek"], how="left")
    out = out.merge(g3,     on=["건물번호","hour","month"],    how="left")
    return out

# ------------------------------
# 건물 단위 학습/예측
# ------------------------------
def process_building_kfold(bno):
    print(f"🏢 building {bno} KFold...")
    param_dir = os.path.join(path, "optuna_params")
    os.makedirs(param_dir, exist_ok=True)

    tr_b = train[train["건물번호"] == bno].copy()
    te_b = test[test["건물번호"] == bno].copy()

    X_full = tr_b[features].values
    y_full_log = np.log1p(tr_b[target].values.astype(float))
    y_full_raw = tr_b[target].values.astype(float)
    X_test = te_b[features].values

    # 시계열 정렬 인덱스 (튜닝용)
    order = np.argsort(tr_b['일시'].values)

    # 베이스 모델 파라미터 (건물당 1회)
    best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
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

        # [PATCH-1] 폴드별 타깃통계 재계산→머지 (누설 차단)
        stats = build_target_stats_fold(tr_b, tr_idx, target)
        tr_fold = merge_target_stats(tr_b.iloc[tr_idx].copy(), stats)
        va_fold = merge_target_stats(tr_b.iloc[va_idx].copy(), stats)
        te_fold = merge_target_stats(te_b.copy(),               stats)

        # 결측 보정
        fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]

        # 1) 우선 존재하는 컬럼만 모아서 전역 평균 계산
        present = [c for c in fill_cols if c in tr_fold.columns]
        if len(present) == 0:
            # 극단적으로 아무것도 없을 때 대비(거의 없음): 0.0 사용
            glob_mean = 0.0
        else:
            glob_mean = float(pd.concat([tr_fold[present]], axis=1).stack().mean())

        # 2) 없는 컬럼은 즉시 생성 후 보정, 있는 컬럼은 결측만 채우기
        for df_ in (tr_fold, va_fold, te_fold):
            for c in fill_cols:
                if c not in df_.columns:
                    df_[c] = glob_mean
                else:
                    df_[c] = df_[c].fillna(glob_mean)

        # 행렬 구성

        # [추가] 폴드에서 방금 생성된 통계 컬럼들만 선택적으로 확장
        fold_only_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
        features_fold = features + [c for c in fold_only_cols if c in tr_fold.columns]
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

    # ----- 메타(Ridge) 튜닝/학습
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

    # ----- OOF 성능, Smearing 보정, SMAPE 칼리브레이션
    oof_pred_log = meta.predict(oof_meta)
    avg_smape = float(smape_exp(y_full_log, oof_pred_log))

    # [PATCH-3] Smearing
    resid = y_full_log - oof_pred_log
    S = float(np.mean(np.exp(resid)))

    # 테스트 예측 (로그→원복 + Smearing)
    te_pred_log = meta.predict(test_meta)
    te_pred = np.expm1(te_pred_log) * S

    # [PATCH-4] 단조 칼리브레이션 g(p)=a·p^b (OOF 기반)
    y_oof = np.expm1(y_full_log)
    p_oof = np.expm1(oof_pred_log) * S
    a_grid = np.linspace(0.8, 1.2, 21)
    b_grid = np.linspace(0.85, 1.15, 31)
    best = (1.0, 1.0, smape(y_oof, p_oof))
    for a in a_grid:
        for b in b_grid:
            s = smape(y_oof, a*(p_oof**b))
            if s < best[2]:
                best = (a, b, s)
    a_opt, b_opt, _ = best
    te_pred = a_opt * (te_pred ** b_opt)

    return te_pred.tolist(), avg_smape

# ==============================
# 12) 병렬 실행 (test 건물 기준) + 순서 매핑
# ==============================
bld_list = list(np.sort(test["건물번호"].unique()))
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_building_kfold)(bno) for bno in bld_list
)

preds_full = np.zeros(len(test), dtype=float)
val_smapes = []
for bno, (preds, sm) in zip(bld_list, results):
    idx = (test["건물번호"] == bno).values
    assert idx.sum() == len(preds), f"building {bno}: test rows={idx.sum()}, preds={len(preds)}"
    preds_full[idx] = preds
    if not np.isnan(sm):
        val_smapes.append(sm)

assert len(preds_full) == len(samplesub), f"final preds:{len(preds_full)}, sample:{len(samplesub)}"
samplesub["answer"] = preds_full

today = datetime.datetime.now().strftime("%Y%m%d")
avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
filename = f"submission_stack_PATCHED_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")


























# ### 내 옵튜나 파라미터로 돌리는 전처리 4등꺼 추가된 코드 #########
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
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

# # === 1.5) 기상 결측 다단계 보간 (건물×월×시 → 건물×시 → 시 → 전체)
# def fill_weather_missing(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if "강수량(mm)" in df.columns:
#         df["강수량(mm)"] = df["강수량(mm)"].fillna(0)
#     for col in ["풍속(m/s)", "습도(%)"]:
#         if col not in df.columns: 
#             continue
#         lvl1 = df.groupby(["건물번호","month","hour"])[col].transform("mean")
#         x = df[col].copy().fillna(lvl1)

#         mask = x.isna()
#         if mask.any():
#             lvl2 = df.groupby(["건물번호","hour"])[col].transform("mean")
#             x = x.where(~mask, lvl2)

#         x = pd.Series(x, index=df.index); mask = x.isna()
#         if mask.any():
#             lvl3 = df.groupby(["hour"])[col].transform("mean")
#             x = x.where(~mask, lvl3)

#         df[col] = pd.Series(x, index=df.index).fillna(df[col].mean())
#     return df

# train = fill_weather_missing(train)
# test  = fill_weather_missing(test)

# # === 2) expected_solar (train 기준 → 둘 다 merge)
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

# # === 4) CDH 교정 / THI 교정+구간화 / WCT 단위 보정
# def add_CDH_fixed(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.sort_values(["건물번호","일시"]).copy()
#     if '기온(°C)' not in df.columns:
#         df['CDH'] = 0.0
#         return df
#     def _cdh_1d(x):
#         z  = x - 26.0
#         cs = np.concatenate([[0.0], np.cumsum(z)])
#         win = 12
#         head = np.cumsum(z)[:min(win-1, len(z))]
#         if len(z) >= win:
#             tail = cs[win:] - cs[:-win]
#             return np.concatenate([head, tail])
#         else:
#             return head
#     parts = []
#     for _, g in df.groupby('건물번호', sort=False):
#         arr = g['기온(°C)'].to_numpy()
#         cdh = _cdh_1d(arr)
#         parts.append(pd.Series(cdh, index=g.index))
#     df['CDH'] = pd.concat(parts).sort_index()
#     return df

# def add_THI_binned_and_WCT(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     # THI (섭씨 기반) + 구간화
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         thi = t - (0.55 - 0.0055*h) * (t - 14.5)
#         df['DI']  = thi                          # 연속값(참고용)
#         df['THI'] = pd.cut(thi, [0,68,75,80,200], labels=[1,2,3,4]).astype(int)
#     else:
#         df['DI'] = 0.0
#         df['THI'] = 1
#     # WCT (풍속 m/s → km/h)
#     if {'기온(°C)','풍속(m/s)'}.issubset(df.columns):
#         t = df['기온(°C)']; w = (df['풍속(m/s)']*3.6).clip(lower=0)
#         df['WCT'] = 13.12 + 0.6125*t - 11.37*(w**0.16) + 0.3965*(w**0.16)*t
#     else:
#         df['WCT'] = 0.0
#     return df

# train = add_CDH_fixed(train)
# test  = add_CDH_fixed(test)
# train = add_THI_binned_and_WCT(train)
# test  = add_THI_binned_and_WCT(test)

# # === 5) 타깃 통계 확장 (건물×hour, 건물×hour×요일(median 포함), 건물×hour×month)
# def add_target_stats(train_df, test_df, target="전력소비량(kWh)"):
#     tr = train_df.copy(); te = test_df.copy()

#     g1 = tr.groupby(["건물번호","hour"])[target]
#     h_mean = g1.mean().rename("hour_mean").reset_index()
#     h_std  = g1.std().rename("hour_std").reset_index()

#     g2 = tr.groupby(["건물번호","hour","dayofweek"])[target]
#     d_mean = g2.mean().rename("day_hour_mean").reset_index()
#     d_std  = g2.std().rename("day_hour_std").reset_index()
#     d_med  = g2.median().rename("day_hour_median").reset_index()

#     g3 = tr.groupby(["건물번호","hour","month"])[target]
#     m_mean = g3.mean().rename("month_hour_mean").reset_index()

#     def merge_all(df):
#         df = df.merge(h_mean, on=["건물번호","hour"], how="left")
#         df = df.merge(h_std,  on=["건물번호","hour"], how="left")
#         df = df.merge(d_mean, on=["건물번호","hour","dayofweek"], how="left")
#         df = df.merge(d_std,  on=["건물번호","hour","dayofweek"], how="left")
#         df = df.merge(d_med,  on=["건물번호","hour","dayofweek"], how="left")
#         df = df.merge(m_mean, on=["건물번호","hour","month"],  how="left")
#         return df

#     tr = merge_all(tr)
#     te = merge_all(te)

#     # NaN 보정(희소 조합 대비)
#     fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
#     for c in fill_cols:
#         glob = float(tr[c].mean())
#         tr[c] = tr[c].fillna(glob)
#         te[c] = te[c].fillna(glob)

#     return tr, te

# train, test = add_target_stats(train, test, target="전력소비량(kWh)")

# # === 6) KMeans 군집 + 군집×시간 통계
# from sklearn.cluster import KMeans
# def add_cluster_features(train_df, test_df, target="전력소비량(kWh)", n_clusters=5, seed=42):
#     tmp = train_df.copy()
#     tmp["is_weekend_tmp"] = (tmp["dayofweek"] >= 5).astype(int)
#     wk = tmp[tmp["is_weekend_tmp"]==0].pivot_table(values=target, index="건물번호", columns="hour", aggfunc="mean")
#     we = tmp[tmp["is_weekend_tmp"]==1].pivot_table(values=target, index="건물번호", columns="hour", aggfunc="mean")
#     patt = pd.merge(wk, we, left_index=True, right_index=True, how="left").fillna(method="ffill").fillna(method="bfill")

#     kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
#     labs = kmeans.fit_predict(patt.values)
#     cl = pd.DataFrame({"건물번호": patt.index, "cluster": labs})

#     tr = train_df.merge(cl, on="건물번호", how="left")
#     te = test_df.merge(cl,  on="건물번호", how="left")

#     grp = tr.groupby(["cluster","hour","dayofweek"])[target].mean().rename("cluster_day_hour_mean").reset_index()
#     tr = tr.merge(grp, on=["cluster","hour","dayofweek"], how="left")
#     te = te.merge(grp,  on=["cluster","hour","dayofweek"],  how="left")

#     return tr, te

# train, test = add_cluster_features(train, test, n_clusters=5, seed=seed)

# # === 7) 기온/습도 이동평균(4일·7일) - test는 train-tail 붙여 경계 안정화
# def add_rollings(train_df, test_df, windows=(96,168)):
#     tr = train_df.sort_values(["건물번호","일시"]).copy()
#     te = test_df.sort_values(["건물번호","일시"]).copy()

#     for w in windows:
#         tr[f"기온_{w//24}일_이동평균"] = tr.groupby("건물번호")["기온(°C)"].transform(lambda s: s.rolling(w,1).mean())
#         tr[f"습도_{w//24}일_이동평균"] = tr.groupby("건물번호")["습도(%)"].transform(lambda s: s.rolling(w,1).mean())

#     out=[]
#     for b, te_part in te.groupby("건물번호"):
#         tr_part = tr[tr["건물번호"]==b]
#         cat = pd.concat([tr_part, te_part]).sort_values("일시")
#         for w in windows:
#             cat[f"기온_{w//24}일_이동평균"] = cat["기온(°C)"].rolling(w,1).mean()
#             cat[f"습도_{w//24}일_이동평균"] = cat["습도(%)"].rolling(w,1).mean()
#         out.append(cat.loc[te_part.index, :])
#     te = pd.concat(out).sort_index()
#     return tr, te

# train, test = add_rollings(train, test, windows=(96,168))

# # === 8) (선택) 이상치 제거: 0 kWh 제거 (원 코드 유지)
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # === 9) 범주형 건물유형 인코딩 (있을 때만)
# if '건물유형' in train.columns and '건물유형' in test.columns:
#     both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
#     cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
#     train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
#     test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)

# # ------------------------------
# # Feature Set
# # ------------------------------
# # 기존 후보 + 확장 통계/군집/롤링까지 포함
# feature_candidates = [
#     # building_info
#     '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
#     # weather/raw
#     '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
#     # time parts & cycles
#     'hour','day','month','dayofweek','is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     # engineered
#     'expected_solar','DI','THI','WCT','CDH',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     # target stats (기존 + 확장)
#     'day_hour_mean','day_hour_std',           # 기존(요일 기준)
#     'hour_mean','hour_std','day_hour_median','month_hour_mean',  # 확장
#     # cluster
#     'cluster','cluster_day_hour_mean',
#     # rollings
#     '기온_4일_이동평균','습도_4일_이동평균','기온_7일_이동평균','습도_7일_이동평균'
# ]
# # train/test 공통 컬럼만 사용
# features = [c for c in feature_candidates if (c in train.columns and c in test.columns)]

# # Target
# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# # 최종 입력/타깃
# X = train[features].values
# y_log = np.log1p(train[target].values.astype(float))   # 기존 3모델용 (로그 타깃)
# y_raw = train[target].values.astype(float)             # Tweedie용
# X_test_raw = test[features].values
# ts = train['일시']

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# # ------------------------------
# # SMAPE helpers
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# def log1p_pos(arr):
#     return np.log1p(np.clip(arr, a_min=0, a_max=None))

# # ========== Tweedie 전용 튜닝 ==========
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
#     y_full_log = np.log1p(tr_b[target].values.astype(float))  # 기존 3모델용
#     y_full_raw = tr_b[target].values.astype(float)            # Tweedie용
#     X_test = te_b[features].values

#     order = np.argsort(tr_b['일시'].values)

#     best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
#     best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     base_models = ["xgb", "lgb", "cat", "twd"]
#     n_train_b = len(tr_b); n_test_b  = len(te_b)
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")
#         X_tr, X_va = X_full[tr_idx], X_full[va_idx]
#         y_tr_log, y_va_log = y_full_log[tr_idx], y_full_log[va_idx]
#         y_tr_raw, y_va_raw = y_full_raw[tr_idx], y_full_raw[va_idx]

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_test)

#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)],
#                  callbacks=[lgb.early_stopping(50, verbose=False)])

#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log),
#                 early_stopping_rounds=50, verbose=0)

#         twd = LGBMRegressor(**best_twd)
#         twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
#                 callbacks=[lgb.early_stopping(50, verbose=False)])

#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)
#         oof_meta[va_idx, 3] = log1p_pos(twd.predict(X_va_s))

#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)
#         test_meta_accum[:, 3] += log1p_pos(twd.predict(X_te_s))

#     test_meta = test_meta_accum / kf.get_n_splits()

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

#     oof_pred = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full_log, oof_pred))

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




# # -*- coding: utf-8 -*-
# import os
# import json
# import random
# import warnings
# import datetime
# import numpy as np
# import pandas as pd
# import optuna  # Ridge(메타)만 튜닝 용도

# from sklearn.model_selection import KFold
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
#     return df

# train = add_time_features_kor(train)
# test  = add_time_features_kor(test)

# # === 1.5) 기상 결측 다단계 보간 (건물×월×시 → 건물×시 → 시 → 전체)
# def fill_weather_missing(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if "강수량(mm)" in df.columns:
#         df["강수량(mm)"] = df["강수량(mm)"].fillna(0)
#     for col in ["풍속(m/s)", "습도(%)"]:
#         if col not in df.columns:
#             continue
#         lvl1 = df.groupby(["건물번호","month","hour"])[col].transform("mean")
#         x = df[col].copy().fillna(lvl1)

#         mask = x.isna()
#         if mask.any():
#             lvl2 = df.groupby(["건물번호","hour"])[col].transform("mean")
#             x = x.where(~mask, lvl2)

#         x = pd.Series(x, index=df.index); mask = x.isna()
#         if mask.any():
#             lvl3 = df.groupby(["hour"])[col].transform("mean")
#             x = x.where(~mask, lvl3)

#         df[col] = pd.Series(x, index=df.index).fillna(df[col].mean())
#     return df

# train = fill_weather_missing(train)
# test  = fill_weather_missing(test)

# # === 2) expected_solar (train 기준 → 둘 다 merge)
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

# # === 4) CDH / THI(구간화) / WCT
# def add_CDH_fixed(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.sort_values(["건물번호","일시"]).copy()
#     if '기온(°C)' not in df.columns:
#         df['CDH'] = 0.0
#         return df
#     def _cdh_1d(x):
#         z  = x - 26.0
#         cs = np.concatenate([[0.0], np.cumsum(z)])
#         win = 12
#         head = np.cumsum(z)[:min(win-1, len(z))]
#         if len(z) >= win:
#             tail = cs[win:] - cs[:-win]
#             return np.concatenate([head, tail])
#         else:
#             return head
#     parts = []
#     for _, g in df.groupby('건물번호', sort=False):
#         arr = g['기온(°C)'].to_numpy()
#         cdh = _cdh_1d(arr)
#         parts.append(pd.Series(cdh, index=g.index))
#     df['CDH'] = pd.concat(parts).sort_index()
#     return df

# def add_THI_binned_and_WCT(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     if {'기온(°C)','습도(%)'}.issubset(df.columns):
#         t = df['기온(°C)']; h = df['습도(%)']
#         thi = t - (0.55 - 0.0055*h) * (t - 14.5)
#         df['DI']  = thi
#         df['THI'] = pd.cut(thi, [0,68,75,80,200], labels=[1,2,3,4]).astype(int)
#     else:
#         df['DI'] = 0.0
#         df['THI'] = 1
#     if {'기온(°C)','풍속(m/s)'}.issubset(df.columns):
#         t = df['기온(°C)']; w = (df['풍속(m/s)']*3.6).clip(lower=0)
#         df['WCT'] = 13.12 + 0.6125*t - 11.37*(w**0.16) + 0.3965*(w**0.16)*t
#     else:
#         df['WCT'] = 0.0
#     return df

# train = add_CDH_fixed(train)
# test  = add_CDH_fixed(test)
# train = add_THI_binned_and_WCT(train)
# test  = add_THI_binned_and_WCT(test)

# # === 5) 타깃 통계 (건물×hour / 요일 / month)
# def add_target_stats(train_df, test_df, target="전력소비량(kWh)"):
#     tr = train_df.copy(); te = test_df.copy()

#     g1 = tr.groupby(["건물번호","hour"])[target]
#     h_mean = g1.mean().rename("hour_mean").reset_index()
#     h_std  = g1.std().rename("hour_std").reset_index()

#     g2 = tr.groupby(["건물번호","hour","dayofweek"])[target]
#     d_mean = g2.mean().rename("day_hour_mean").reset_index()
#     d_std  = g2.std().rename("day_hour_std").reset_index()
#     d_med  = g2.median().rename("day_hour_median").reset_index()

#     g3 = tr.groupby(["건물번호","hour","month"])[target]
#     m_mean = g3.mean().rename("month_hour_mean").reset_index()

#     def merge_all(df):
#         df = df.merge(h_mean, on=["건물번호","hour"], how="left")
#         df = df.merge(h_std,  on=["건물번호","hour"], how="left")
#         df = df.merge(d_mean, on=["건물번호","hour","dayofweek"], how="left")
#         df = df.merge(d_std,  on=["건물번호","hour","dayofweek"], how="left")
#         df = df.merge(d_med,  on=["건물번호","hour","dayofweek"], how="left")
#         df = df.merge(m_mean, on=["건물번호","hour","month"],  how="left")
#         return df

#     tr = merge_all(tr)
#     te = merge_all(te)

#     fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
#     for c in fill_cols:
#         glob = float(tr[c].mean())
#         tr[c] = tr[c].fillna(glob)
#         te[c] = te[c].fillna(glob)

#     return tr, te

# train, test = add_target_stats(train, test, target="전력소비량(kWh)")

# # === 6) KMeans 군집 + 군집×시간 통계
# from sklearn.cluster import KMeans
# def add_cluster_features(train_df, test_df, target="전력소비량(kWh)", n_clusters=5, seed=42):
#     tmp = train_df.copy()
#     tmp["is_weekend_tmp"] = (tmp["dayofweek"] >= 5).astype(int)
#     wk = tmp[tmp["is_weekend_tmp"]==0].pivot_table(values=target, index="건물번호", columns="hour", aggfunc="mean")
#     we = tmp[tmp["is_weekend_tmp"]==1].pivot_table(values=target, index="건물번호", columns="hour", aggfunc="mean")
#     patt = pd.merge(wk, we, left_index=True, right_index=True, how="left").fillna(method="ffill").fillna(method="bfill")

#     kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
#     labs = kmeans.fit_predict(patt.values)
#     cl = pd.DataFrame({"건물번호": patt.index, "cluster": labs})

#     tr = train_df.merge(cl, on="건물번호", how="left")
#     te = test_df.merge(cl,  on="건물번호", how="left")

#     grp = tr.groupby(["cluster","hour","dayofweek"])[target].mean().rename("cluster_day_hour_mean").reset_index()
#     tr = tr.merge(grp, on=["cluster","hour","dayofweek"], how="left")
#     te = te.merge(grp,  on=["cluster","hour","dayofweek"],  how="left")

#     return tr, te

# train, test = add_cluster_features(train, test, n_clusters=5, seed=seed)

# # === 7) 기온/습도 이동평균(4일·7일) - 안전 슬라이싱 버전
# def add_rollings(train_df, test_df, windows=(96,168)):
#     tr = train_df.sort_values(["건물번호","일시"]).copy()
#     te = test_df.sort_values(["건물번호","일시"]).copy()

#     for w in windows:
#         tr[f"기온_{w//24}일_이동평균"] = tr.groupby("건물번호")["기온(°C)"].transform(lambda s: s.rolling(w,1).mean())
#         tr[f"습도_{w//24}일_이동평균"] = tr.groupby("건물번호")["습도(%)"].transform(lambda s: s.rolling(w,1).mean())

#     out = []
#     for b, te_part in te.groupby("건물번호"):
#         tr_part = tr[tr["건물번호"] == b]
#         # 인덱스 충돌 방지 + 위치기반 슬라이싱
#         cat = pd.concat([tr_part, te_part], axis=0, ignore_index=True)
#         for w in windows:
#             cat[f"기온_{w//24}일_이동평균"] = cat["기온(°C)"].rolling(w,1).mean()
#             cat[f"습도_{w//24}일_이동평균"] = cat["습도(%)"].rolling(w,1).mean()
#         tail = cat.iloc[-len(te_part):].copy()
#         # 원래 te_part 인덱스로 복구해 전체 te 정렬 유지
#         tail.index = te_part.index
#         out.append(tail)

#     te_fixed = pd.concat(out, axis=0).sort_index()
#     return tr, te_fixed

# train, test = add_rollings(train, test, windows=(96,168))

# # === 8) (선택) 이상치 제거: 0 kWh 제거
# if '전력소비량(kWh)' in train.columns:
#     train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# # === 9) 범주형 건물유형 인코딩 (있을 때만)
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
#     'hour','day','month','dayofweek','is_weekend','is_working_hours',
#     'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
#     # engineered
#     'expected_solar','DI','THI','WCT','CDH',
#     'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
#     # target stats
#     'day_hour_mean','day_hour_std','hour_mean','hour_std','day_hour_median','month_hour_mean',
#     # cluster
#     'cluster','cluster_day_hour_mean',
#     # rollings
#     '기온_4일_이동평균','습도_4일_이동평균','기온_7일_이동평균','습도_7일_이동평균'
# ]
# features = [c for c in feature_candidates if (c in train.columns and c in test.columns)]

# # Target
# target = '전력소비량(kWh)'
# if target not in train.columns:
#     raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# # 최종 입력/타깃
# X = train[features].values
# y_log = np.log1p(train[target].values.astype(float))   # 베이스 모델용 (로그 타깃)
# X_test_raw = test[features].values
# ts = train['일시']

# print(f"[확인] 사용 features 개수: {len(features)}")
# print(f"[확인] target: {target}")
# print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y_log shape: {y_log.shape}")

# # 전처리 정합성 점검
# print("len(test) =", len(test))
# print("len(samplesub) =", len(samplesub))
# print("건물 수 train vs test:", train["건물번호"].nunique(), test["건물번호"].nunique())
# # 건물별 행 수(모두 168이어야 정상)
# counts = test.groupby("건물번호").size()
# bad = counts[counts != 168]
# if len(bad):
#     print("⚠️ 168이 아닌 건물 발견:\n", bad)
# # 길이 사전점검 (여기서 바로 실패시키면 원인 추적 쉬움)
# assert len(test) == len(samplesub), f"test:{len(test)} sample:{len(samplesub)}"

# # ------------------------------
# # SMAPE helpers
# # ------------------------------
# def smape_exp(y_true_log, y_pred_log):
#     y_true = np.expm1(y_true_log)
#     y_pred = np.expm1(y_pred_log)
#     return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# # ------------------------------
# # 고정 하이퍼파라미터
# # ------------------------------
# xgb_params = dict(
#     random_state=seed,
#     n_estimators=4682,
#     max_leaves=101,
#     min_child_weight=7.581207558322951,
#     learning_rate=0.08979034933474227,
#     subsample=0.8905280196300354,
#     colsample_bylevel=1.0,
#     colsample_bytree=0.9523645407001878,
#     reg_alpha=0.006919296411231538,
#     reg_lambda=0.0998936254543762,
#     n_jobs=-1,
# )
# lgbm_params = dict(
#     random_state=seed,
#     n_estimators=15000,
#     num_leaves=8,
#     min_child_samples=12,
#     learning_rate=0.17010396907527026,
#     colsample_bytree=0.9605563464803123,
#     reg_alpha=0.1110993344544235,
#     reg_lambda=0.7948637803974561,
#     verbose=-1,
#     n_jobs=-1,
# )
# cat_params = dict(
#     learning_rate=0.14059048492476106,
#     loss_function="RMSE",
#     random_seed=seed,
#     verbose=False,
#     n_estimators=10000,
#     early_stopping_rounds=100,
#     objective="MAE",
# )

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

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     # test가 없으면 스킵
#     if len(te_b) == 0:
#         return [], np.nan

#     X_full = tr_b[features].values
#     y_full_log = np.log1p(tr_b[target].values.astype(float))
#     X_test = te_b[features].values

#     # 외부 KFold
#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     base_models = ["xgb", "lgb", "cat"]
#     n_train_b = len(tr_b); n_test_b  = len(te_b)
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")
#         X_tr, X_va = X_full[tr_idx], X_full[va_idx]
#         y_tr_log, y_va_log = y_full_log[tr_idx], y_full_log[va_idx]

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_test)

#         # 고정 파라미터로 베이스 모델 학습
#         xgb = XGBRegressor(**xgb_params)
#         xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         lgbm = LGBMRegressor(**lgbm_params)
#         lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)],
#                  callbacks=[lgb.early_stopping(50, verbose=False)])

#         cat = CatBoostRegressor(**cat_params)
#         cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log),
#                 early_stopping_rounds=100, verbose=False)

#         # OOF 저장
#         oof_meta[va_idx, 0] = xgb.predict(X_va_s)
#         oof_meta[va_idx, 1] = lgbm.predict(X_va_s)
#         oof_meta[va_idx, 2] = cat.predict(X_va_s)

#         # Test 메타 누적
#         test_meta_accum[:, 0] += xgb.predict(X_te_s)
#         test_meta_accum[:, 1] += lgbm.predict(X_te_s)
#         test_meta_accum[:, 2] += cat.predict(X_te_s)

#     test_meta = test_meta_accum / kf.get_n_splits()

#     # ----- 메타(Ridge) 건물당 튜닝/학습 -----
#     param_dir = os.path.join(path, "optuna_params_fixed")
#     os.makedirs(param_dir, exist_ok=True)
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

#     # OOF 성능 및 테스트 예측
#     oof_pred = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full_log, oof_pred))

#     te_pred_log = meta.predict(test_meta)
#     te_pred = np.expm1(te_pred_log)

#     return te_pred.tolist(), avg_smape

# # ==============================
# # 12) 병렬 실행 (test 건물 기준) + 순서 매핑
# # ==============================
# bld_list = list(np.sort(test["건물번호"].unique()))
# results = Parallel(n_jobs=-1, backend="loky")(
#     delayed(process_building_kfold)(bno) for bno in bld_list
# )

# # test 원래 순서에 직접 매핑
# preds_full = np.zeros(len(test), dtype=float)
# val_smapes = []
# for bno, (preds, sm) in zip(bld_list, results):
#     idx = (test["건물번호"] == bno).values
#     # 건물별 길이 정합 체크 (각 건물 168행 가정)
#     assert idx.sum() == len(preds), f"building {bno}: test rows={idx.sum()}, preds={len(preds)}"
#     preds_full[idx] = preds
#     if not np.isnan(sm):
#         val_smapes.append(sm)

# # 최종 길이/정렬 정합성
# assert len(preds_full) == len(samplesub), f"final preds:{len(preds_full)}, sample:{len(samplesub)}"

# samplesub["answer"] = preds_full
# today = datetime.datetime.now().strftime("%Y%m%d")
# avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
# filename = f"submission_stack_FIXED_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
# samplesub.to_csv(os.path.join(path, filename), index=False)

# print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
# print(f"📁 저장 완료 → {filename}")