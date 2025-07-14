import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold # train_test_split 대신 KFold 사용
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors # 3D 기술자 추가
import warnings
import copy
from datetime import datetime
import random

# --- 기본 설정 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')

# --- 경로 설정 ---
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') \
    else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/drugs/')

# --- 데이터 로드 ---
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')


# ==============================================================================
# [변경] 1. 특징 공학 고도화: 2D/3D/물리화학적 속성을 모두 포함하는 함수로 변경
# ==============================================================================
def get_all_descriptors(smiles, seed=42):
    """
    SMILES 문자열로부터 2D, 3D, 물리화학적 기술자를 모두 추출하는 함수.
    """
    # 2D/3D 기술자 계산 시 오류가 많으므로 robust한 예외 처리 필수
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None # 오류 발생 시 None 반환 후 후처리

        # 1. 200개의 물리화학적 기술자
        all_descriptors = [desc[1](mol) for desc in Descriptors._descList]

        # 2. Morgan Fingerprint
        morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]

        # 3. MACCS Keys
        maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]

        # 4. 3D 기술자 (WHIM)
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=seed, maxAttempts=1000, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d)
            whim_descriptors = list(rdMolDescriptors.GetWHIM(mol_3d))
        except Exception: # 3D 구조 최적화 실패 시
            whim_descriptors = [0] * 114

        return all_descriptors + morgan + maccs + whim_descriptors
    except:
        return None

print("고도화된 특징 추출을 시작합니다... (시간이 다소 소요될 수 있습니다)")
# 특징 추출 적용 (오류 발생 시 None으로 채우고, 이후 평균값으로 대체)
train['features'] = train['Canonical_Smiles'].apply(get_all_descriptors)
test['features'] = test['Canonical_Smiles'].apply(get_all_descriptors)

# 특징 추출 실패한 행(None) 처리
train.dropna(subset=['features'], inplace=True)
test['features'].fillna(pd.Series([np.zeros(len(train['features'].iloc[0]))] * len(test)), inplace=True)

x_train_raw = np.array(train['features'].tolist(), dtype=np.float32)
x_test_raw = np.array(test['features'].tolist(), dtype=np.float32)
y_train_full = train['Inhibition'].values

# np.inf, np.nan 값을 0으로 대체
x_train_raw = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
x_test_raw = np.nan_to_num(x_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

print(f"특징 추출 완료. 학습 데이터 형태: {x_train_raw.shape}")
# ==============================================================================


# --- 평가지표 ---
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def normalized_rmse(y_true, y_pred):
    y_true_range = np.max(y_true) - np.min(y_true)
    if y_true_range == 0: return 0
    return np.sqrt(mean_squared_error(y_true, y_pred)) / y_true_range

def pearson_correlation(y_true, y_pred):
    return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)

def competition_score(y_true, y_pred):
    # NRMSE 계산 시 분모가 0이 되는 경우 방지
    y_true_range = np.max(y_true) - np.min(y_true)
    if y_true_range == 0: return pearson_correlation(y_true, y_pred)
    
    nrmse = np.sqrt(mean_squared_error(y_true, y_pred)) / y_true_range
    return 0.5 * (1 - min(nrmse, 1)) + 0.5 * pearson_correlation(y_true, y_pred)

# --- 모델 정의 ---
# n_estimators(iterations)는 early_stopping으로 조절되므로 충분히 큰 값으로 설정
base_models = {
    "LightGBM": lgb.LGBMRegressor(n_estimators=2000, random_state=seed, n_jobs=-1,
                                 learning_rate=0.05, num_leaves=31, max_depth=7, verbose = -1),
    "XGBoost": xgb.XGBRegressor(n_estimators=2000, random_state=seed, tree_method='gpu_hist',
                                learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8),
    "CatBoost": cat.CatBoostRegressor(iterations=2000, random_seed=seed, verbose=0,
                                      learning_rate=0.05, depth=6, l2_leaf_reg=3)
}


# ==============================================================================
# [변경] 2. K-Fold 교차 검증 및 모델 블렌딩으로 학습/예측 로직 변경
# ==============================================================================
# K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Out-of-Fold(OOF) 예측값과 테스트 데이터 예측값을 저장할 배열 초기화
oof_preds = np.zeros((x_train_raw.shape[0], len(base_models)))
test_preds = np.zeros((x_test_raw.shape[0], len(base_models)))

# 모델별로 K-Fold 학습 및 예측 수행
for model_idx, (name, model) in enumerate(base_models.items()):
    print(f"\n===== {name} 모델 K-Fold 학습 및 예측 시작 =====")
    
    # 각 Fold의 테스트 예측값을 임시 저장할 리스트
    fold_test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_raw)):
        print(f"  Fold {fold+1}/{kf.n_splits} 학습 중...")
        
        # 1. 데이터 분할
        x_train_fold, x_val_fold = x_train_raw[train_idx], x_train_raw[val_idx]
        y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

        # 2. 스케일링 (매 Fold 마다 독립적으로 수행하여 데이터 누수 방지)
        scaler = StandardScaler()
        x_train_fold = scaler.fit_transform(x_train_fold)
        x_val_fold = scaler.transform(x_val_fold)
        x_test_scaled = scaler.transform(x_test_raw) # 테스트 데이터도 동일하게 변환
        
        # 3. 모델 학습 (Early Stopping 적용)
        m = copy.deepcopy(model)
        if name == "LightGBM":
            m.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        elif name == "XGBoost":
             m.fit(x_train_fold, y_train_fold,
            eval_set=[(x_val_fold, y_val_fold)],
             early_stopping_rounds=100,
            verbose=False)
        elif name == "CatBoost":
             m.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
                  early_stopping_rounds=100, verbose=0)
        else:
            m.fit(x_train_fold, y_train_fold)
            
        # 4. 예측
        # 검증 데이터 예측 (OOF)
        oof_preds[val_idx, model_idx] = m.predict(x_val_fold)
        # 테스트 데이터 예측
        fold_test_preds.append(m.predict(x_test_scaled))
        
    # 5. 테스트 예측값 평균
    # 각 Fold에서 예측한 테스트 결과의 평균을 해당 모델의 최종 예측값으로 사용
    test_preds[:, model_idx] = np.mean(fold_test_preds, axis=0)
    
    # 모델별 OOF 점수 출력
    model_oof_score = competition_score(y_train_full, oof_preds[:, model_idx])
    print(f"→ {name} 모델 OOF Score: {model_oof_score:.4f}")
    

# ==============================================================================
# [추가] 모델별 상세 OOF 성능 지표 출력
# ==============================================================================
print("\n📊 모델별 최종 OOF 성능 비교")
print("-" * 80)
for model_idx, name in enumerate(base_models.keys()):
    y_true = y_train_full
    y_pred = oof_preds[:, model_idx]
    
    _rmse = rmse(y_true, y_pred)
    _nrmse = normalized_rmse(y_true, y_pred)
    _pearson = pearson_correlation(y_true, y_pred)
    _score = competition_score(y_true, y_pred)
    
    print(f"{name:20} | RMSE: {_rmse:.4f} | NRMSE: {_nrmse:.4f} | Pearson: {_pearson:.4f} | Score: {_score:.4f}")
print("-" * 80)
# ==============================================================================

# --- 최종 예측: 모델 예측값 블렌딩 ---
# 각 모델의 OOF 점수를 가중치로 사용 (성능 좋은 모델에 더 큰 가중치)
oof_scores = np.array([competition_score(y_train_full, oof_preds[:, i]) for i in range(len(base_models))])
weights = oof_scores / oof_scores.sum()

print("\n--- 모델별 가중치 ---")
for name, weight in zip(base_models.keys(), weights):
    print(f"{name}: {weight:.4f}")

# 가중 평균을 사용하여 최종 예측값 생성
final_predictions = np.sum(test_preds * weights, axis=1)

# Blended OOF Score 계산
blended_oof_preds = np.sum(oof_preds * weights, axis=1)
blended_score = competition_score(y_train_full, blended_oof_preds)
print(f"\n🏆 최종 Blended OOF Score: {blended_score:.4f}")
# ==============================================================================


# --- 제출 파일 생성 ---
submission['Inhibition'] = final_predictions
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_{timestamp}_score_{blended_score:.4f}.csv"
submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n예측 결과 저장 완료 → {filename}")


# --- 결과 시각화 (OOF 예측값 활용) ---
plt.figure(figsize=(10, 6))
plt.scatter(y_train_full, blended_oof_preds, alpha=0.5)
plt.plot([min(y_train_full), max(y_train_full)], [min(y_train_full), max(y_train_full)], 'r--')
plt.xlabel('실제값 (Inhibition)')
plt.ylabel('OOF 예측값 (Blended)')
plt.title(f'Blended Model OOF Performance (Score: {blended_score:.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()