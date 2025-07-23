import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from datetime import datetime
import random
import warnings
import copy

# Seed 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')

# 경로 및 데이터 로드
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/drugs/')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')
# from rdkit.Chem import Draw
# smi = "COC(=O)[C@@H](OC(C)(C)C)c1c(C)nc2C(=O)N(Cc3ccc(OC)cc3)CCc2c1Cl"
# mol = Chem.MolFromSmiles(smi)
# img = Draw.MolToImage(mol)
# img.show() 

# exit()

# # ✅ 여기 붙여
# from rdkit import Chem

# failures = []
# for i, s in enumerate(test['Canonical_Smiles']):
#     mol = Chem.MolFromSmiles(s)
#     if mol is None:
#         failures.append((i, s))

# print(f"⚠️ 실패한 SMILES 개수: {len(failures)}")
# for idx, smile in failures:
#     print(f"  - index: {idx}, SMILES: {smile}")

# exit()

# Feature Extraction (NaN 반환 유지)
# 실패 로그 저장용
train_nan_info = []
test_nan_info = []

def get_molecule_descriptors(smiles, index=None, is_test=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if is_test:
                test_nan_info.append((index, smiles, "MolFromSmiles=None"))
            else:
                train_nan_info.append((index, smiles, "MolFromSmiles=None"))
            return [np.nan] * 2233

        basic = [ Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
                 Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol), 
                 Descriptors.FractionCSP3(mol), Descriptors.NumAliphaticRings(mol), Lipinski.NumAromaticHeterocycles(mol),
                 Lipinski.NumSaturatedHeterocycles(mol), Lipinski.NumAliphaticHeterocycles(mol), Descriptors.HeavyAtomCount(mol),
                 Descriptors.RingCount(mol), Descriptors.NOCount(mol), Descriptors.NHOHCount(mol), Descriptors.NumRadicalElectrons(mol) ]
        morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
        maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
        features = basic + morgan + maccs

        if np.any(pd.isna(features)):
            if is_test:
                test_nan_info.append((index, smiles, "NaN in descriptor"))
            else:
                train_nan_info.append((index, smiles, "NaN in descriptor"))
        return features

    except Exception as e:
        if is_test:
            test_nan_info.append((index, smiles, f"Exception: {e}"))
        else:
            train_nan_info.append((index, smiles, f"Exception: {e}"))
        return [np.nan] * 2233
    


# 각각 인덱스를 전달하며 결측치 추적
train['features'] = [get_molecule_descriptors(smi, idx, is_test=False) for idx, smi in enumerate(train['Canonical_Smiles'])]
test['features'] = [get_molecule_descriptors(smi, idx, is_test=True) for idx, smi in enumerate(test['Canonical_Smiles'])]

x_raw = np.array(train['features'].tolist())
x_test_raw = np.array(test['features'].tolist())

# 📢 요약 리포트 출력
print(f"✅ Train 결측치 총합: {np.isnan(x_raw).sum()}")
print(f"✅ Test  결측치 총합: {np.isnan(x_test_raw).sum()}")
print(f"⚠️ Train NaN 포함 샘플 수: {len(train_nan_info)}")
print(f"⚠️ Test  NaN 포함 샘플 수: {len(test_nan_info)}")

print("\n📌 Train NaN 포함된 SMILES:")
for idx, smi, reason in train_nan_info[:10]:  # 많으면 자르기
    print(f" - index: {idx}, SMILES: {smi}, reason: {reason}")

print("\n📌 Test NaN 포함된 SMILES:")
for idx, smi, reason in test_nan_info[:10]:
    print(f" - index: {idx}, SMILES: {smi}, reason: {reason}")
    
    
    
y = train['Inhibition'].values


# [수정] 제안된 방식으로 결측치 처리
print("결측치를 '정보'로 활용하여 처리합니다...")
# 1. 결측 여부를 나타내는 'is_missing' 특징 추가
train['is_missing'] = np.isnan(x_raw).any(axis=1).astype(int)
test['is_missing'] = np.isnan(x_test_raw).any(axis=1).astype(int)
x_raw = np.c_[x_raw, train['is_missing']]
x_test_raw = np.c_[x_test_raw, test['is_missing']]

# 2. 결측값을 0으로 대체
imputer = SimpleImputer(strategy='constant', fill_value=0)
x_raw = imputer.fit_transform(x_raw)
x_test_raw = imputer.transform(x_test_raw)



# 평가 지표 정의
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def normalized_rmse(y_true, y_pred): return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
def pearson_correlation(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0: return 0.0
    return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)
def competition_score(y_true, y_pred): return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)
def create_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=seed,
        tree_method='hist',
        predictor='auto',
        objective='reg:squarederror',
        early_stopping_rounds=50,     # ✅ XGBoost만 모델 생성자에 설정
        eval_metric='rmse'
    )

def create_lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=seed,
        objective='regression'
    )

def create_cat_model():
    return cat.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=seed,
        loss_function='RMSE',
        verbose=0
    )

# [추가] LassoCV를 이용한 안정적인 특징 선택
print("\nLassoCV를 사용하여 특징을 선택합니다...")
scaler_fs = RobustScaler()
x_scaled_fs = scaler_fs.fit_transform(x_raw)
lasso = LassoCV(cv=5, random_state=seed, n_jobs=-1).fit(x_scaled_fs, y)
selector = SelectFromModel(lasso, prefit=True)
x_selected = selector.transform(x_raw)
x_test_selected = selector.transform(x_test_raw)
print(f"Lasso selected {x_selected.shape[1]} features.")

# [수정] K-Fold 교차 검증을 사용한 모델 학습 및 평가
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# OOF(Out-of-Fold) 예측 및 테스트 예측을 저장할 배열 초기화
oof_preds = np.zeros((len(x_selected), 3)) # 3개 모델
test_preds = np.zeros((len(x_test_selected), 3))

base_models = {
    "XGBoost": create_xgb_model(),
    "LightGBM": create_lgb_model(),
    "CatBoost": create_cat_model()
}

print("\nK-Fold 교차 검증을 시작합니다...")
for i, (name, model) in enumerate(base_models.items()):
    print(f"  Training {name} model...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_selected, y)):
        # 데이터 분할
        x_train_fold, x_val_fold = x_selected[train_idx], x_selected[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 스케일링 (Fold 내부에서 수행하여 데이터 누수 방지)
        scaler = RobustScaler()
        x_train_fold_scaled = scaler.fit_transform(x_train_fold)
        x_val_fold_scaled = scaler.transform(x_val_fold)

        # 모델 학습
        m = copy.deepcopy(model)
        if name == "LightGBM":
            m.fit(x_train_fold_scaled, y_train_fold, eval_set=[(x_val_fold_scaled, y_val_fold)], callbacks=[lgb.early_stopping(50, verbose=False)])
        elif name == "CatBoost":
            m.fit(x_train_fold_scaled, y_train_fold, eval_set=[(x_val_fold_scaled, y_val_fold)], early_stopping_rounds=50, verbose=0)
        elif name == "XGBoost": m.fit(x_train_fold_scaled, y_train_fold,
              eval_set=[(x_val_fold_scaled, y_val_fold)],
              verbose=0)
        # OOF 예측값 저장
        oof_preds[val_idx, i] = m.predict(x_val_fold_scaled)
        
        # 테스트 데이터 예측값 누적
        x_test_fold_scaled = scaler.transform(x_test_selected)
        test_preds[:, i] += m.predict(x_test_fold_scaled) / kf.n_splits

# --- 메타 모델 학습 및 최종 예측 ---
print("\n메타 모델을 학습하고 최종 예측을 수행합니다...")
meta_model = Ridge()
# OOF 예측값으로 메타 모델 학습
meta_model.fit(oof_preds, y)

# 테스트 데이터에 대한 최종 예측
y_pred_test = meta_model.predict(test_preds)

# OOF 점수 계산 (신뢰할 수 있는 검증 점수)
# ✅ OOF 점수 계산 및 출력
y_oof_pred = meta_model.predict(oof_preds)
from sklearn.metrics import r2_score
oof_r2 = r2_score(y, y_oof_pred)

oof_rmse = rmse(y, y_oof_pred)
oof_nrmse = normalized_rmse(y, y_oof_pred)
oof_pearson = pearson_correlation(y, y_oof_pred)
oof_score = competition_score(y, y_oof_pred)

print("\n📊 최종 OOF 성능")
print(f"  - RMSE       : {oof_rmse:.4f}")
print(f"  - NRMSE      : {oof_nrmse:.4f}")
print(f"  - R2 Score   : {oof_r2:.4f}")
print(f"  - Pearson    : {oof_pearson:.4f}")
print(f"  - Comp Score : {oof_score:.4f}")

# 예측 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_kfold_final_{timestamp}.csv"
submission['Inhibition'] = y_pred_test
submission.to_csv(os.path.join(path, filename), index=False)
print(f"✅ 예측 결과 저장 완료 → {filename}")

