#얘로 옵튜나한게 최고. 근데 내 다른 모델/전처리에다가 적용시킨거!

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

# ------------------------------
# [탐색 범위 교체] LightGBM Tweedie (원시 타깃)
# ------------------------------
def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
    params = {
        "objective": "tweedie",
        "metric": "mae",
        "boosting_type": "gbdt",

        "n_estimators": trial.suggest_int("n_estimators", 1000, 12000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

        "num_leaves": trial.suggest_int("num_leaves", 32, 1024),
        "max_depth": trial.suggest_int("max_depth", -1, 16),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),

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

        model = LGBMRegressor(**params, n_jobs=-1)
        model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

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
        st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=60)
        best = st.best_params
        with open(path_twd, "w") as f:
            json.dump(best, f)
        return best

# ------------------------------
# [탐색 범위 교체] XGBoost / LightGBM / CatBoost (log 타깃)
# ------------------------------
def tune_xgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1200, 8000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),

        "eval_metric": "mae",
        "random_state": seed,
        "objective": "reg:squarederror",
        "early_stopping_rounds": 100,
        # "tree_method": "hist",  # 가능하면 활성화
    }
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tss.split(X_full_sorted):
        X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
        y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
        model = XGBRegressor(**params, n_jobs=-1)
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
        pred = model.predict(X_va_s)
        scores.append(smape_exp(y_va, pred))
    return float(np.mean(scores))

def tune_lgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
    params = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",

        "n_estimators": trial.suggest_int("n_estimators", 3000, 20000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

        "num_leaves": trial.suggest_int("num_leaves", 32, 1024),
        "max_depth": trial.suggest_int("max_depth", -1, 16),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),

        "random_state": seed,
        "verbosity": -1,
    }
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tss.split(X_full_sorted):
        X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
        y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
        model = LGBMRegressor(**params, n_jobs=-1)
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict(X_va_s)
        scores.append(smape_exp(y_va, pred))
    return float(np.mean(scores))

def tune_cat_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
    params = {
        "iterations": trial.suggest_int("iterations", 3000, 20000),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),

        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),

        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),

        "loss_function": "MAE",
        "random_seed": seed,
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
        model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=200, verbose=0)
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
        st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=60)
        params["xgb"] = st.best_params
        with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

    if os.path.exists(paths["lgb"]):
        with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=60)
        params["lgb"] = st.best_params
        with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

    if os.path.exists(paths["cat"]):
        with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=60)
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

    g1 = (base
          .groupby(["건물번호","hour"])[target]
          .agg(hour_mean="mean", hour_std="std")
          .reset_index())

    g2 = base.groupby(["건물번호","hour","dayofweek"])[target]
    d_mean = g2.mean().rename("day_hour_mean").reset_index()
    d_std  = g2.std().rename("day_hour_std").reset_index()
    d_med  = g2.median().rename("day_hour_median").reset_index()

    g3 = (base
          .groupby(["건물번호","hour","month"])[target]
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

    order = np.argsort(tr_b['일시'].values)

    best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
    best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

    kf = KFold(n_splits=8, shuffle=True, random_state=seed)

    base_models = ["xgb", "lgb", "cat", "twd"]
    n_train_b = len(tr_b); n_test_b = len(te_b)
    oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
    test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        print(f" - fold {fold}")

        # 폴드별 타깃통계 재계산→머지 (누설 차단)
        stats = build_target_stats_fold(tr_b, tr_idx, target)
        tr_fold = merge_target_stats(tr_b.iloc[tr_idx].copy(), stats)
        va_fold = merge_target_stats(tr_b.iloc[va_idx].copy(), stats)
        te_fold = merge_target_stats(te_b.copy(),               stats)

        # 결측 보정
        fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
        present = [c for c in fill_cols if c in tr_fold.columns]
        if len(present) == 0:
            glob_mean = 0.0
        else:
            glob_mean = float(pd.concat([tr_fold[present]], axis=1).stack().mean())
        for df_ in (tr_fold, va_fold, te_fold):
            for c in fill_cols:
                if c not in df_.columns:
                    df_[c] = glob_mean
                else:
                    df_[c] = df_[c].fillna(glob_mean)

        # 행렬
        features_fold = features + [c for c in fill_cols if c in tr_fold.columns]
        X_tr = tr_fold[features_fold].values
        X_va = va_fold[features_fold].values
        X_te = te_fold[features_fold].values
        y_tr_log, y_va_log = np.log1p(tr_fold[target].values.astype(float)), np.log1p(va_fold[target].values.astype(float))
        y_tr_raw, y_va_raw = tr_fold[target].values.astype(float), va_fold[target].values.astype(float)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_te)

        # XGB (log)
        xgb = XGBRegressor(**best_params["xgb"])
        xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

        # LGB (log)
        lgbm = LGBMRegressor(**best_params["lgb"])
        lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # CAT (log)
        cat = CatBoostRegressor(**best_params["cat"])
        cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)

        # Tweedie (raw)
        twd = LGBMRegressor(**best_twd)
        twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # OOF 저장(로그 스케일)
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
        st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full_log), n_trials=50)
        ridge_params = st.best_params
        with open(ridge_path, "w") as f:
            json.dump(ridge_params, f)

    meta = Ridge(alpha=ridge_params["alpha"])
    meta.fit(oof_meta, y_full_log)

    # ----- OOF 성능, Smearing 보정, SMAPE 칼리브레이션
    oof_pred_log = meta.predict(oof_meta)
    avg_smape = float(smape_exp(y_full_log, oof_pred_log))

    resid = y_full_log - oof_pred_log
    S = float(np.mean(np.exp(resid)))

    te_pred_log = meta.predict(test_meta)
    te_pred = np.expm1(te_pred_log) * S

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
#     'hour','day','month','dayofweek','is_weekend','is_working_hours',
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

# # ------------------------------
# # [탐색 범위 교체] LightGBM Tweedie (원시 타깃)
# # ------------------------------
# def tune_lgb_tweedie_tss(trial, X_full_sorted, y_full_sorted_raw, seed=seed):
#     params = {
#         "objective": "tweedie",
#         "metric": "mae",
#         "boosting_type": "gbdt",

#         "n_estimators": trial.suggest_int("n_estimators", 1000, 12000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

#         "num_leaves": trial.suggest_int("num_leaves", 32, 1024),
#         "max_depth": trial.suggest_int("max_depth", -1, 16),

#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
#         "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),

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

#         model = LGBMRegressor(**params, n_jobs=-1)
#         model.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)],
#                   callbacks=[lgb.early_stopping(100, verbose=False)])

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
#         st.optimize(lambda t: tune_lgb_tweedie_tss(t, X_sorted, y_sorted_raw), n_trials=60)
#         best = st.best_params
#         with open(path_twd, "w") as f:
#             json.dump(best, f)
#         return best

# # ------------------------------
# # [탐색 범위 교체] XGBoost / LightGBM / CatBoost (log 타깃)
# # ------------------------------
# def tune_xgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 1200, 8000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

#         "max_depth": trial.suggest_int("max_depth", 3, 12),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
#         "gamma": trial.suggest_float("gamma", 0.0, 5.0),

#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),

#         "eval_metric": "mae",
#         "random_state": seed,
#         "objective": "reg:squarederror",
#         "early_stopping_rounds": 100,
#         # "tree_method": "hist",  # 가능하면 활성화
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = XGBRegressor(**params, n_jobs=-1)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_lgb_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "objective": "mae",
#         "metric": "mae",
#         "boosting_type": "gbdt",

#         "n_estimators": trial.suggest_int("n_estimators", 3000, 20000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),

#         "num_leaves": trial.suggest_int("num_leaves", 32, 1024),
#         "max_depth": trial.suggest_int("max_depth", -1, 16),

#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
#         "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),

#         "random_state": seed,
#         "verbosity": -1,
#     }
#     tss = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for tr_idx, va_idx in tss.split(X_full_sorted):
#         X_tr, X_va = X_full_sorted[tr_idx], X_full_sorted[va_idx]
#         y_tr, y_va = y_full_sorted[tr_idx], y_full_sorted[va_idx]
#         sc = StandardScaler().fit(X_tr)
#         X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
#         model = LGBMRegressor(**params, n_jobs=-1)
#         model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)],
#                   callbacks=[lgb.early_stopping(100, verbose=False)])
#         pred = model.predict(X_va_s)
#         scores.append(smape_exp(y_va, pred))
#     return float(np.mean(scores))

# def tune_cat_tss(trial, X_full_sorted, y_full_sorted, seed=seed):
#     params = {
#         "iterations": trial.suggest_int("iterations", 3000, 20000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),

#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),

#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
#         "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),

#         "loss_function": "MAE",
#         "random_seed": seed,
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
#         model.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=200, verbose=0)
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
#         st.optimize(lambda t: tune_xgb_tss(t, X_sorted, y_sorted), n_trials=60)
#         params["xgb"] = st.best_params
#         with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)

#     if os.path.exists(paths["lgb"]):
#         with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_lgb_tss(t, X_sorted, y_sorted), n_trials=60)
#         params["lgb"] = st.best_params
#         with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)

#     if os.path.exists(paths["cat"]):
#         with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
#     else:
#         st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
#         st.optimize(lambda t: tune_cat_tss(t, X_sorted, y_sorted), n_trials=60)
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

#     g1 = (base
#           .groupby(["건물번호","hour"])[target]
#           .agg(hour_mean="mean", hour_std="std")
#           .reset_index())

#     g2 = base.groupby(["건물번호","hour","dayofweek"])[target]
#     d_mean = g2.mean().rename("day_hour_mean").reset_index()
#     d_std  = g2.std().rename("day_hour_std").reset_index()
#     d_med  = g2.median().rename("day_hour_median").reset_index()

#     g3 = (base
#           .groupby(["건물번호","hour","month"])[target]
#           .mean()
#           .rename("month_hour_mean")
#           .reset_index())

#     return g1, d_mean, d_std, d_med, g3

# def merge_target_stats(df, stats):
#     g1, d_mean, d_std, d_med, g3 = stats
#     out = df.merge(g1, on=["건물번호","hour"], how="left")
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
#     param_dir = os.path.join(path, "optuna_params")
#     os.makedirs(param_dir, exist_ok=True)

#     tr_b = train[train["건물번호"] == bno].copy()
#     te_b = test[test["건물번호"] == bno].copy()

#     X_full = tr_b[features].values
#     y_full_log = np.log1p(tr_b[target].values.astype(float))
#     y_full_raw = tr_b[target].values.astype(float)
#     X_test = te_b[features].values

#     order = np.argsort(tr_b['일시'].values)

#     best_params = get_or_tune_params_once(bno, X_full, y_full_log, order, param_dir)
#     best_twd = get_or_tune_tweedie_once(bno, X_full, y_full_raw, order, param_dir)

#     kf = KFold(n_splits=8, shuffle=True, random_state=seed)

#     base_models = ["xgb", "lgb", "cat", "twd"]
#     n_train_b = len(tr_b); n_test_b = len(te_b)
#     oof_meta = np.zeros((n_train_b, len(base_models)), dtype=float)
#     test_meta_accum = np.zeros((n_test_b, len(base_models)), dtype=float)

#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
#         print(f" - fold {fold}")

#         # 폴드별 타깃통계 재계산→머지 (누설 차단)
#         stats = build_target_stats_fold(tr_b, tr_idx, target)
#         tr_fold = merge_target_stats(tr_b.iloc[tr_idx].copy(), stats)
#         va_fold = merge_target_stats(tr_b.iloc[va_idx].copy(), stats)
#         te_fold = merge_target_stats(te_b.copy(),               stats)

#         # 결측 보정
#         fill_cols = ["hour_mean","hour_std","day_hour_mean","day_hour_std","day_hour_median","month_hour_mean"]
#         present = [c for c in fill_cols if c in tr_fold.columns]
#         if len(present) == 0:
#             glob_mean = 0.0
#         else:
#             glob_mean = float(pd.concat([tr_fold[present]], axis=1).stack().mean())
#         for df_ in (tr_fold, va_fold, te_fold):
#             for c in fill_cols:
#                 if c not in df_.columns:
#                     df_[c] = glob_mean
#                 else:
#                     df_[c] = df_[c].fillna(glob_mean)

#         # 행렬
#         features_fold = features + [c for c in fill_cols if c in tr_fold.columns]
#         X_tr = tr_fold[features_fold].values
#         X_va = va_fold[features_fold].values
#         X_te = te_fold[features_fold].values
#         y_tr_log, y_va_log = np.log1p(tr_fold[target].values.astype(float)), np.log1p(va_fold[target].values.astype(float))
#         y_tr_raw, y_va_raw = tr_fold[target].values.astype(float), va_fold[target].values.astype(float)

#         sc = StandardScaler()
#         X_tr_s = sc.fit_transform(X_tr)
#         X_va_s = sc.transform(X_va)
#         X_te_s = sc.transform(X_te)

#         # XGB (log)
#         xgb = XGBRegressor(**best_params["xgb"])
#         xgb.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], verbose=False)

#         # LGB (log)
#         lgbm = LGBMRegressor(**best_params["lgb"])
#         lgbm.fit(X_tr_s, y_tr_log, eval_set=[(X_va_s, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # CAT (log)
#         cat = CatBoostRegressor(**best_params["cat"])
#         cat.fit(X_tr_s, y_tr_log, eval_set=(X_va_s, y_va_log), early_stopping_rounds=50, verbose=0)

#         # Tweedie (raw)
#         twd = LGBMRegressor(**best_twd)
#         twd.fit(X_tr_s, y_tr_raw, eval_set=[(X_va_s, y_va_raw)], callbacks=[lgb.early_stopping(50, verbose=False)])

#         # OOF 저장(로그 스케일)
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
#         st.optimize(lambda t: objective_ridge_on_oof(t, oof_meta, y_full_log), n_trials=50)
#         ridge_params = st.best_params
#         with open(ridge_path, "w") as f:
#             json.dump(ridge_params, f)

#     meta = Ridge(alpha=ridge_params["alpha"])
#     meta.fit(oof_meta, y_full_log)

#     # ----- OOF 성능, Smearing 보정, SMAPE 칼리브레이션
#     oof_pred_log = meta.predict(oof_meta)
#     avg_smape = float(smape_exp(y_full_log, oof_pred_log))

#     resid = y_full_log - oof_pred_log
#     S = float(np.mean(np.exp(resid)))

#     te_pred_log = meta.predict(test_meta)
#     te_pred = np.expm1(te_pred_log) * S

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