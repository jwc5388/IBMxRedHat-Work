# -*- coding: utf-8 -*-
# Time-based CV + Fold-safe 집계 + 주간 랙/롤링 + 최근 가중치 + 시간기반 Optuna + 보정/클리핑
# seed bagging 제외

import os
import json
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import optuna

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
bi = buildinginfo.copy() if 'buildinginfo' in globals() else buildinginfo.copy()

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
    # 월말/월초 플래그(작은 이득)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    return df

train = add_time_features_kor(train)
test  = add_time_features_kor(test)

# === 2) 일별 온도 통계 (train/test 동일 로직)
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

# === 3) CDH / THI / WCT (train/test 동일)
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

# === 4) 시간 기반 주간 랙/롤링: train은 shift로, test는 train의 과거시점 매핑
def add_weekly_lags_train(df_train: pd.DataFrame):
    df = df_train.sort_values(['건물번호','일시']).copy()
    df['lag168'] = df.groupby('건물번호')['전력소비량(kWh)'].shift(168)
    df['rm168']  = (df.groupby('건물번호')['전력소비량(kWh)']
                    .transform(lambda s: s.shift(1).rolling(168, min_periods=24).mean()))
    med = df.groupby('건물번호')['전력소비량(kWh)'].transform('median')
    df['lag168'] = df['lag168'].fillna(med)
    df['rm168']  = df['rm168'].fillna(med)
    return df

def add_weekly_lags_test(df_train_with_lag: pd.DataFrame, df_test: pd.DataFrame):
    # test의 (건물, t)의 lag168 = train의 (건물, t-168h) 값을 lookup
    train_map = df_train_with_lag.set_index(['건물번호','일시'])['전력소비량(kWh)']
    # 롤링 평균도 t-1 시점 롤링을 train에서 lookup (보수적)
    train_rm_map = df_train_with_lag.set_index(['건물번호','일시'])['rm168'] if 'rm168' in df_train_with_lag.columns else None

    df_te = df_test.copy()
    t_minus_168 = df_te['일시'] - pd.to_timedelta(168, unit='h')
    key = list(zip(df_te['건물번호'].values, t_minus_168.values))
    lag_vals = train_map.reindex(key).values
    df_te['lag168'] = lag_vals

    if train_rm_map is not None:
        # rm168은 t-1 시점의 롤링평균을 대신 사용(엄격한 누수 방지)
        t_minus_1 = df_te['일시'] - pd.to_timedelta(1, unit='h')
        key_rm = list(zip(df_te['건물번호'].values, t_minus_1.values))
        rm_vals = train_rm_map.reindex(key_rm).values
        df_te['rm168'] = rm_vals

    # 결측 대체: 건물별 train 중앙값
    med_map = df_train_with_lag.groupby('건물번호')['전력소비량(kWh)'].median().to_dict()
    df_te['lag168'] = df_te.apply(lambda r: med_map.get(r['건물번호'], 0.0) if pd.isna(r['lag168']) else r['lag168'], axis=1)
    if 'rm168' in df_te.columns:
        df_te['rm168'] = df_te.apply(lambda r: med_map.get(r['건물번호'], 0.0) if pd.isna(r['rm168']) else r['rm168'], axis=1)
    else:
        df_te['rm168'] = df_te['lag168']

    return df_te

train = add_weekly_lags_train(train)
test  = add_weekly_lags_test(train, test)

# === 5) 특징 후보
feature_candidates = [
    '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
    '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
    'hour','day','month','dayofweek','is_weekend','is_working_hours',
    'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
    'DI','day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
    'CDH','THI','WCT',
    'is_month_end','is_month_start',
    # fold-safe로 대체할 예정인 자리표시자(머지 시 생성)
    # 'day_hour_mean','day_hour_std','expected_solar'
    # 주간 랙/롤링
    'lag168','rm168'
]

# 공통 컬럼만 일단 확보(이후 fold-safe 머지로 추가 컬럼 들어옴)
features_base = [c for c in feature_candidates if c in train.columns and c in test.columns]

target = '전력소비량(kWh)'
assert target in train.columns, "train에 target이 없습니다."

# ------------------------------
# SMAPE (log domain 입력)
# ------------------------------
def smape_exp(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# ------------------------------
# 시간기반 폴드 생성(자동): 끝에서 9일을 3등분(각 3일 검증), 그 이전 전체를 각 fold의 train으로 사용
# ------------------------------
def make_time_folds(df_b):
    df_b = df_b.sort_values('일시')
    days = np.sort(df_b['일시'].dt.floor('D').unique())
    if len(days) < 15:
        # 데이터가 짧으면 마지막 6일을 3개 창(2일씩)으로
        val_len = 2
        total_val_days = min(6, len(days))
    else:
        val_len = 3
        total_val_days = min(9, len(days))
    last_days = days[-total_val_days:]
    folds = []
    for i in range(0, total_val_days, val_len):
        val_days = last_days[i:i+val_len]
        if len(val_days) < val_len: break
        vs, ve = val_days[0], val_days[-1] + np.timedelta64(23, 'h')  # 포함 끝시각
        tr_mask = df_b['일시'] < vs
        va_mask = (df_b['일시'] >= vs) & (df_b['일시'] <= ve)
        tr_idx = df_b.index[tr_mask].to_numpy()
        va_idx = df_b.index[va_mask].to_numpy()
        if len(tr_idx) and len(va_idx):
            folds.append((tr_idx, va_idx))
    return folds

# ------------------------------
# Fold-safe 집계: 시간대 전력 통계 & expected_solar
# ------------------------------
def fold_safe_hour_stats(df_train_fold):
    pm = (df_train_fold
          .groupby(['건물번호','hour','dayofweek'])[target]
          .agg(['mean','std']).reset_index()
          .rename(columns={'mean':'day_hour_mean','std':'day_hour_std'}))
    return pm

def fold_safe_expected_solar(df_train_fold):
    if '일사(MJ/m2)' in df_train_fold.columns:
        solar_proxy = (df_train_fold.groupby(['month','hour'])['일사(MJ/m2)']
                       .mean().reset_index()
                       .rename(columns={'일사(MJ/m2)':'expected_solar'}))
    else:
        solar_proxy = pd.DataFrame({'month':[], 'hour':[], 'expected_solar':[]})
    return solar_proxy

def merge_fold_safe(df_any, pm, solar_proxy):
    d = df_any.copy()
    # 기존 열 제거 후 재머지(충돌 방지)
    d = d.drop(columns=['day_hour_mean','day_hour_std','expected_solar'], errors='ignore')
    d = d.merge(pm, on=['건물번호','hour','dayofweek'], how='left')
    if solar_proxy.shape[0] > 0:
        d = d.merge(solar_proxy, on=['month','hour'], how='left')
    else:
        d['expected_solar'] = 0.0
    d[['day_hour_mean','day_hour_std','expected_solar']] = d[['day_hour_mean','day_hour_std','expected_solar']].fillna(0.0)
    return d

# ------------------------------
# 최근 가중치
# ------------------------------
def make_recency_weight(df_b, idx, decay=0.005):
    tmax = df_b['일시'].max()
    dt_h = (tmax - df_b.loc[idx, '일시']).dt.total_seconds() / 3600.0
    return np.exp(-decay * dt_h)

# ------------------------------
# 예측 후 보정 & 클리핑
# ------------------------------
def build_calib_table(df_va, pred_log):
    df = df_va.copy()
    df['pred'] = np.expm1(pred_log)
    df['ratio'] = (df[target] + 1e-6) / (df['pred'] + 1e-6)
    tab = df.groupby(['hour','dayofweek'])['ratio'].median().to_dict()
    return tab

def apply_calib(df_te, pred_log, tab):
    base = np.expm1(pred_log)
    ratios = df_te.apply(lambda r: tab.get((r['hour'], r['dayofweek']), 1.0), axis=1).values
    return base * ratios

def per_building_clip(df_b, y_pred):
    lo = df_b[target].quantile(0.01) if target in df_b.columns else 0.0
    hi = df_b[target].quantile(0.99) if target in df_b.columns else np.percentile(y_pred, 99)
    return np.clip(y_pred, max(0.0, lo*0.5), hi*1.2)

# ------------------------------
# 시간기반 Optuna 튜닝(건물별 1회 저장/로드)
# ------------------------------
def tune_xgb_timecv(trial, df_b, features, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth":    trial.suggest_int("max_depth", 3, 8),
        "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":    trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": seed,
        "early_stopping_rounds": 50,
    }
    folds = make_time_folds(df_b)
    smapes = []
    for tr_idx, va_idx in folds:
        df_tr = df_b.loc[tr_idx].copy()
        df_va = df_b.loc[va_idx].copy()
        pm = fold_safe_hour_stats(df_tr)
        solar_proxy = fold_safe_expected_solar(df_tr)
        df_tr = merge_fold_safe(df_tr, pm, solar_proxy)
        df_va = merge_fold_safe(df_va, pm, solar_proxy)

        X_tr = df_tr[features].values
        y_tr = np.log1p(df_tr[target].values.astype(float))
        X_va = df_va[features].values
        y_va = np.log1p(df_va[target].values.astype(float))

        w_tr = make_recency_weight(df_tr, df_tr.index)
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        smapes.append(smape_exp(y_va, pred))
    return float(np.mean(smapes))

def tune_lgb_timecv(trial, df_b, features, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth":    trial.suggest_int("max_depth", 3, 8),
        "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":    trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "objective": "mae",
        "random_state": seed,
    }
    folds = make_time_folds(df_b)
    smapes = []
    for tr_idx, va_idx in folds:
        df_tr = df_b.loc[tr_idx].copy()
        df_va = df_b.loc[va_idx].copy()
        pm = fold_safe_hour_stats(df_tr)
        solar_proxy = fold_safe_expected_solar(df_tr)
        df_tr = merge_fold_safe(df_tr, pm, solar_proxy)
        df_va = merge_fold_safe(df_va, pm, solar_proxy)

        X_tr = df_tr[features].values
        y_tr = np.log1p(df_tr[target].values.astype(float))
        X_va = df_va[features].values
        y_va = np.log1p(df_va[target].values.astype(float))

        w_tr = make_recency_weight(df_tr, df_tr.index)
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict(X_va)
        smapes.append(smape_exp(y_va, pred))
    return float(np.mean(smapes))

def tune_cat_timecv(trial, df_b, features, seed):
    params = {
        "iterations":   trial.suggest_int("iterations", 300, 1000),
        "depth":        trial.suggest_int("depth", 3, 8),
        "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed":  seed,
        "loss_function":"MAE",
        "verbose": 0,
    }
    folds = make_time_folds(df_b)
    smapes = []
    for tr_idx, va_idx in folds:
        df_tr = df_b.loc[tr_idx].copy()
        df_va = df_b.loc[va_idx].copy()
        pm = fold_safe_hour_stats(df_tr)
        solar_proxy = fold_safe_expected_solar(df_tr)
        df_tr = merge_fold_safe(df_tr, pm, solar_proxy)
        df_va = merge_fold_safe(df_va, pm, solar_proxy)

        X_tr = df_tr[features].values
        y_tr = np.log1p(df_tr[target].values.astype(float))
        X_va = df_va[features].values
        y_va = np.log1p(df_va[target].values.astype(float))

        w_tr = make_recency_weight(df_tr, df_tr.index)
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
        pred = model.predict(X_va)
        smapes.append(smape_exp(y_va, pred))
    return float(np.mean(smapes))

def get_or_tune_params_once_timecv(bno, df_b, features, param_dir):
    """건물당 1회만 시간기반 CV로 튜닝하고 JSON 저장/로드"""
    os.makedirs(param_dir, exist_ok=True)
    paths = {
        "xgb": os.path.join(param_dir, f"{bno}_xgb_time.json"),
        "lgb": os.path.join(param_dir, f"{bno}_lgb_time.json"),
        "cat": os.path.join(param_dir, f"{bno}_cat_time.json"),
    }
    params = {}
    # XGB
    if os.path.exists(paths["xgb"]):
        with open(paths["xgb"], "r") as f: params["xgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_xgb_timecv(t, df_b, features, seed), n_trials=30)
        params["xgb"] = st.best_params
        with open(paths["xgb"], "w") as f: json.dump(params["xgb"], f)
    # LGB
    if os.path.exists(paths["lgb"]):
        with open(paths["lgb"], "r") as f: params["lgb"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_lgb_timecv(t, df_b, features, seed), n_trials=30)
        params["lgb"] = st.best_params
        with open(paths["lgb"], "w") as f: json.dump(params["lgb"], f)
    # CAT
    if os.path.exists(paths["cat"]):
        with open(paths["cat"], "r") as f: params["cat"] = json.load(f)
    else:
        st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        st.optimize(lambda t: tune_cat_timecv(t, df_b, features, seed), n_trials=30)
        params["cat"] = st.best_params
        with open(paths["cat"], "w") as f: json.dump(params["cat"], f)
    return params

# ------------------------------
# 메타(Ridge) 튜닝도 시간기반
# ------------------------------
def objective_ridge_time(trial, oof_tr, oof_val, y_tr, y_val):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    ridge = Ridge(alpha=alpha)
    ridge.fit(oof_tr, y_tr)
    preds = ridge.predict(oof_val)
    return smape_exp(y_val, preds)

# ------------------------------
# 건물 단위 처리 (시간기반 CV)
# ------------------------------
def process_building_timecv(bno):
    print(f"🏢 building {bno} Time-CV...")
    param_dir = os.path.join(path, "optuna_params_time")
    os.makedirs(param_dir, exist_ok=True)

    tr_b = train[train["건물번호"] == bno].copy()
    te_b = test[test["건물번호"] == bno].copy()

    # 빌딩별 공통 특징 세트(머지 전 기본)
    features = features_base.copy()

    # 시간기반 튜닝(건물당 1회)
    best_params = get_or_tune_params_once_timecv(bno, tr_b, features, param_dir)

    folds = make_time_folds(tr_b)
    test_preds, val_smapes = [], []

    # Ridge도 건물당 1회만: 첫 fold에서 튜닝 → 저장, 이후 로드
    ridge_key = f"{bno}_ridge_time"
    ridge_path = os.path.join(param_dir, f"{ridge_key}.json")
    ridge_params_cached = None
    if os.path.exists(ridge_path):
        with open(ridge_path, "r") as f:
            ridge_params_cached = json.load(f)

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f" - fold {fold}")
        df_tr = tr_b.loc[tr_idx].copy()
        df_va = tr_b.loc[va_idx].copy()
        df_te = te_b.copy()

        # Fold-safe 집계 생성
        pm = fold_safe_hour_stats(df_tr)
        solar_proxy = fold_safe_expected_solar(df_tr)

        # Fold-safe 머지
        df_tr = merge_fold_safe(df_tr, pm, solar_proxy)
        df_va = merge_fold_safe(df_va, pm, solar_proxy)
        df_te = merge_fold_safe(df_te, pm, solar_proxy)

        # 특징 컬럼(머지 후 추가된 열 포함)
        cols_now = features + [c for c in ['day_hour_mean','day_hour_std','expected_solar'] if c in df_tr.columns and c not in features]
        X_tr = df_tr[cols_now].values
        y_tr = np.log1p(df_tr[target].values.astype(float))
        X_va = df_va[cols_now].values
        y_va = np.log1p(df_va[target].values.astype(float))
        X_te = df_te[cols_now].values

        # 최근 가중치
        w_tr = make_recency_weight(df_tr, df_tr.index)

        # Base 모델 학습
        xgb = XGBRegressor(**best_params["xgb"])
        xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

        lgbm = LGBMRegressor(**best_params["lgb"])
        lgbm.fit(X_tr, y_tr, sample_weight=w_tr,
                 eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])

        cat = CatBoostRegressor(**best_params["cat"])
        cat.fit(X_tr, y_tr, sample_weight=w_tr,
                eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)

        # 스태킹 입력
        oof_tr = np.vstack([xgb.predict(X_tr), lgbm.predict(X_tr), cat.predict(X_tr)]).T
        oof_va = np.vstack([xgb.predict(X_va), lgbm.predict(X_va), cat.predict(X_va)]).T
        oof_te = np.vstack([xgb.predict(X_te), lgbm.predict(X_te), cat.predict(X_te)]).T

        # Ridge 메타: 건물당 1회 튜닝/저장
        if ridge_params_cached is None and fold == 1:
            st = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
            st.optimize(lambda t: objective_ridge_time(t, oof_tr, oof_va, y_tr, y_va), n_trials=30)
            ridge_params_cached = st.best_params
            with open(ridge_path, "w") as f:
                json.dump(ridge_params_cached, f)

        meta = Ridge(alpha=ridge_params_cached["alpha"])
        meta.fit(oof_tr, y_tr)

        va_pred_log = meta.predict(oof_va)
        te_pred_log = meta.predict(oof_te)

        # 검증 성능
        fold_smape = smape_exp(y_va, va_pred_log)

        # 예측 후 보정(검증으로 테이블) → 테스트에 적용
        calib_tab = build_calib_table(df_va, va_pred_log)
        te_pred = apply_calib(df_te, te_pred_log, calib_tab)

        # 퍼센타일 클리핑
        te_pred = per_building_clip(tr_b, te_pred)

        val_smapes.append(fold_smape)
        test_preds.append(te_pred)

    avg_test_pred = np.mean(test_preds, axis=0)
    avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
    return avg_test_pred.tolist(), avg_smape

# ==============================
# 병렬 실행
# ==============================
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_building_timecv)(bno) for bno in train["건물번호"].unique()
)

final_preds, val_smapes = [], []
for preds, sm in results:
    final_preds.extend(preds)
    val_smapes.append(sm)

samplesub["answer"] = final_preds
today = datetime.datetime.now().strftime("%Y%m%d")
avg_smape = float(np.mean(val_smapes))
filename = f"submission_timecv_foldsafe_weeklylag_recency_calibclip_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")