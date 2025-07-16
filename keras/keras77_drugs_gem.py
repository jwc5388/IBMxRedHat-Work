# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import KFold # train_test_split 대신 KFold 사용
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import Ridge
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cat
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors # 3D 기술자 추가
# import warnings
# import copy
# from datetime import datetime
# import random

# # --- 기본 설정 ---
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# warnings.filterwarnings('ignore')

# # --- 경로 설정 ---
# BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') \
#     else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# path = os.path.join(BASE_PATH, '_data/dacon/drugs/')

# # --- 데이터 로드 ---
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# submission = pd.read_csv(path + 'sample_submission.csv')


# # ==============================================================================
# # [변경] 1. 특징 공학 고도화: 2D/3D/물리화학적 속성을 모두 포함하는 함수로 변경
# # ==============================================================================
# def get_all_descriptors(smiles, seed=42):
#     """
#     SMILES 문자열로부터 2D, 3D, 물리화학적 기술자를 모두 추출하는 함수.
#     """
#     # 2D/3D 기술자 계산 시 오류가 많으므로 robust한 예외 처리 필수
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None # 오류 발생 시 None 반환 후 후처리

#         # 1. 200개의 물리화학적 기술자
#         all_descriptors = [desc[1](mol) for desc in Descriptors._descList]

#         # 2. Morgan Fingerprint
#         morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]

#         # 3. MACCS Keys
#         maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]

#         # 4. 3D 기술자 (WHIM)
#         mol_3d = Chem.AddHs(mol)
#         AllChem.EmbedMolecule(mol_3d, randomSeed=seed, maxAttempts=1000, useRandomCoords=True)
#         try:
#             AllChem.MMFFOptimizeMolecule(mol_3d)
#             whim_descriptors = list(rdMolDescriptors.GetWHIM(mol_3d))
#         except Exception: # 3D 구조 최적화 실패 시
#             whim_descriptors = [0] * 114

#         return all_descriptors + morgan + maccs + whim_descriptors
#     except:
#         return None

# print("고도화된 특징 추출을 시작합니다... (시간이 다소 소요될 수 있습니다)")
# # 특징 추출 적용 (오류 발생 시 None으로 채우고, 이후 평균값으로 대체)
# train['features'] = train['Canonical_Smiles'].apply(get_all_descriptors)
# test['features'] = test['Canonical_Smiles'].apply(get_all_descriptors)

# # 특징 추출 실패한 행(None) 처리
# train.dropna(subset=['features'], inplace=True)
# test['features'].fillna(pd.Series([np.zeros(len(train['features'].iloc[0]))] * len(test)), inplace=True)

# x_train_raw = np.array(train['features'].tolist(), dtype=np.float32)
# x_test_raw = np.array(test['features'].tolist(), dtype=np.float32)
# y_train_full = train['Inhibition'].values

# # np.inf, np.nan 값을 0으로 대체
# x_train_raw = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
# x_test_raw = np.nan_to_num(x_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

# print(f"특징 추출 완료. 학습 데이터 형태: {x_train_raw.shape}")
# # ==============================================================================


# # --- 평가지표 ---
# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# def normalized_rmse(y_true, y_pred):
#     y_true_range = np.max(y_true) - np.min(y_true)
#     if y_true_range == 0: return 0
#     return np.sqrt(mean_squared_error(y_true, y_pred)) / y_true_range

# def pearson_correlation(y_true, y_pred):
#     return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)

# def competition_score(y_true, y_pred):
#     # NRMSE 계산 시 분모가 0이 되는 경우 방지
#     y_true_range = np.max(y_true) - np.min(y_true)
#     if y_true_range == 0: return pearson_correlation(y_true, y_pred)
    
#     nrmse = np.sqrt(mean_squared_error(y_true, y_pred)) / y_true_range
#     return 0.5 * (1 - min(nrmse, 1)) + 0.5 * pearson_correlation(y_true, y_pred)

# # --- 모델 정의 ---
# # n_estimators(iterations)는 early_stopping으로 조절되므로 충분히 큰 값으로 설정
# base_models = {
#     "LightGBM": lgb.LGBMRegressor(n_estimators=2000, random_state=seed, n_jobs=-1,
#                                  learning_rate=0.05, num_leaves=31, max_depth=7, verbose = -1),
#     "XGBoost": xgb.XGBRegressor(n_estimators=2000, random_state=seed, tree_method='gpu_hist',
#                                 learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8),
#     "CatBoost": cat.CatBoostRegressor(iterations=2000, random_seed=seed, verbose=0,
#                                       learning_rate=0.05, depth=6, l2_leaf_reg=3)
# }


# # ==============================================================================
# # [변경] 2. K-Fold 교차 검증 및 모델 블렌딩으로 학습/예측 로직 변경
# # ==============================================================================
# # K-Fold 설정
# kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# # Out-of-Fold(OOF) 예측값과 테스트 데이터 예측값을 저장할 배열 초기화
# oof_preds = np.zeros((x_train_raw.shape[0], len(base_models)))
# test_preds = np.zeros((x_test_raw.shape[0], len(base_models)))

# # 모델별로 K-Fold 학습 및 예측 수행
# for model_idx, (name, model) in enumerate(base_models.items()):
#     print(f"\n===== {name} 모델 K-Fold 학습 및 예측 시작 =====")
    
#     # 각 Fold의 테스트 예측값을 임시 저장할 리스트
#     fold_test_preds = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_raw)):
#         print(f"  Fold {fold+1}/{kf.n_splits} 학습 중...")
        
#         # 1. 데이터 분할
#         x_train_fold, x_val_fold = x_train_raw[train_idx], x_train_raw[val_idx]
#         y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

#         # 2. 스케일링 (매 Fold 마다 독립적으로 수행하여 데이터 누수 방지)
#         scaler = StandardScaler()
#         x_train_fold = scaler.fit_transform(x_train_fold)
#         x_val_fold = scaler.transform(x_val_fold)
#         x_test_scaled = scaler.transform(x_test_raw) # 테스트 데이터도 동일하게 변환
        
#         # 3. 모델 학습 (Early Stopping 적용)
#         m = copy.deepcopy(model)
#         if name == "LightGBM":
#             m.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
#                   callbacks=[lgb.early_stopping(100, verbose=False)])
#         elif name == "XGBoost":
#              m.fit(x_train_fold, y_train_fold,
#             eval_set=[(x_val_fold, y_val_fold)],
#              early_stopping_rounds=100,
#             verbose=False)
#         elif name == "CatBoost":
#              m.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
#                   early_stopping_rounds=100, verbose=0)
#         else:
#             m.fit(x_train_fold, y_train_fold)
            
#         # 4. 예측
#         # 검증 데이터 예측 (OOF)
#         oof_preds[val_idx, model_idx] = m.predict(x_val_fold)
#         # 테스트 데이터 예측
#         fold_test_preds.append(m.predict(x_test_scaled))
        
#     # 5. 테스트 예측값 평균
#     # 각 Fold에서 예측한 테스트 결과의 평균을 해당 모델의 최종 예측값으로 사용
#     test_preds[:, model_idx] = np.mean(fold_test_preds, axis=0)
    
#     # 모델별 OOF 점수 출력
#     model_oof_score = competition_score(y_train_full, oof_preds[:, model_idx])
#     print(f"→ {name} 모델 OOF Score: {model_oof_score:.4f}")
    

# # ==============================================================================
# # [추가] 모델별 상세 OOF 성능 지표 출력
# # ==============================================================================
# print("\n📊 모델별 최종 OOF 성능 비교")
# print("-" * 80)
# for model_idx, name in enumerate(base_models.keys()):
#     y_true = y_train_full
#     y_pred = oof_preds[:, model_idx]
    
#     _rmse = rmse(y_true, y_pred)
#     _nrmse = normalized_rmse(y_true, y_pred)
#     _pearson = pearson_correlation(y_true, y_pred)
#     _score = competition_score(y_true, y_pred)
    
#     print(f"{name:20} | RMSE: {_rmse:.4f} | NRMSE: {_nrmse:.4f} | Pearson: {_pearson:.4f} | Score: {_score:.4f}")
# print("-" * 80)
# # ==============================================================================

# # --- 최종 예측: 모델 예측값 블렌딩 ---
# # 각 모델의 OOF 점수를 가중치로 사용 (성능 좋은 모델에 더 큰 가중치)
# oof_scores = np.array([competition_score(y_train_full, oof_preds[:, i]) for i in range(len(base_models))])
# weights = oof_scores / oof_scores.sum()

# print("\n--- 모델별 가중치 ---")
# for name, weight in zip(base_models.keys(), weights):
#     print(f"{name}: {weight:.4f}")

# # 가중 평균을 사용하여 최종 예측값 생성
# final_predictions = np.sum(test_preds * weights, axis=1)

# # Blended OOF Score 계산
# blended_oof_preds = np.sum(oof_preds * weights, axis=1)
# blended_score = competition_score(y_train_full, blended_oof_preds)
# print(f"\n🏆 최종 Blended OOF Score: {blended_score:.4f}")
# # ==============================================================================


# # --- 제출 파일 생성 ---
# submission['Inhibition'] = final_predictions
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# filename = f"submission_{timestamp}_score_{blended_score:.4f}.csv"
# submission.to_csv(os.path.join(path, filename), index=False)
# print(f"\n예측 결과 저장 완료 → {filename}")


# # --- 결과 시각화 (OOF 예측값 활용) ---
# plt.figure(figsize=(10, 6))
# plt.scatter(y_train_full, blended_oof_preds, alpha=0.5)
# plt.plot([min(y_train_full), max(y_train_full)], [min(y_train_full), max(y_train_full)], 'r--')
# plt.xlabel('실제값 (Inhibition)')
# plt.ylabel('OOF 예측값 (Blended)')
# plt.title(f'Blended Model OOF Performance (Score: {blended_score:.4f})')
# plt.grid(True)
# plt.tight_layout()
# plt.show()




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from datetime import datetime
import random
import warnings
import copy

# 설정
seed = 700
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')

# 경로
BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') \
    else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/drugs/')

# 데이터 로드
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

# --- Feature Extraction ---
def get_molecule_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 2233
        basic = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol), Descriptors.NumAliphaticRings(mol),
            Lipinski.NumAromaticHeterocycles(mol), Lipinski.NumSaturatedHeterocycles(mol),
            Lipinski.NumAliphaticHeterocycles(mol), Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol), Descriptors.NOCount(mol), Descriptors.NHOHCount(mol),
            Descriptors.NumRadicalElectrons(mol)
        ]
        morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
        maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
        return basic + morgan + maccs
    except:
        return [0] * 2233

# 피처 생성
train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

x_raw = np.array(train['features'].tolist())
y = train['Inhibition'].values
x_test_raw = np.array(test['features'].tolist())

# 데이터 분할 (피처 선택용)
x_train_raw, x_val_raw, y_train, y_val = train_test_split(x_raw, y, test_size=0.2, random_state=seed)

# 정규화
scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train_raw)
x_val_scaled = scaler.transform(x_val_raw)
x_test_scaled_full = scaler.transform(x_test_raw) # 전체 테스트셋도 미리 스케일링

# 평가 지표
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def normalized_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def pearson_correlation(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return np.clip(corr, 0, 1)

def competition_score(y_true, y_pred):
    nrmse = normalized_rmse(y_true, y_pred)
    pearson = pearson_correlation(y_true, y_pred)
    return 0.5 * (1 - min(nrmse, 1)) + 0.5 * pearson

# --- XGBoost 중요도 계산 ---
xgb_model_for_fs = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed,
    tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,  early_stopping_rounds = 10
)

# 피처 선택을 위해 초기 모델 학습
xgb_model_for_fs.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                    #  callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
                    # early_stopping_rounds=10,
                     verbose=0)

importances = xgb_model_for_fs.feature_importances_

# ######################################################################
# ###  [수정된 부분] 청크 단위 재귀적 피처 제거 (기존 threshold 방식 대체) ###
# ######################################################################
print("\n🚀 Starting Fast Feature Selection (Chunked RFE)...")
K = 333  # 분석할 상위 피처 개수
step = 20  # 한 번에 제거할 피처 개수 (조정 가능)
min_features_to_keep = 20  # 최소로 남길 피처 개수

# 중요도 순으로 정렬된 상위 K개 피처의 '원래 인덱스'
top_k_indices = np.argsort(importances)[::-1][:K]

# 상위 K개 피처만으로 임시 데이터셋 생성
x_train_topk = x_train_scaled[:, top_k_indices]
x_val_topk = x_val_scaled[:, top_k_indices]

# 변수 초기화
best_score_fs = -np.inf  # fs: feature selection
best_indices_subset = None # 최종 선택될 '원래 인덱스'

# Top-K 피처셋 내에서, 중요도가 낮은 피처를 step씩 제거하며 최적 조합 탐색
for num_features in range(K, min_features_to_keep - 1, -step):
    # 현재 단계에서 사용할 피처들의 인덱스 (top-k 내에서의 인덱스)
    current_indices_in_topk = list(range(num_features))
    
    selected_train = x_train_topk[:, current_indices_in_topk]
    selected_val = x_val_topk[:, current_indices_in_topk]

    temp_model = xgb.XGBRegressor(**xgb_model_for_fs.get_params())
    temp_model.fit(selected_train, y_train, eval_set=[(selected_val, y_val)],
                   callbacks=[xgb.callback.EarlyStopping(rounds=30)], verbose=0)

    score = competition_score(y_val, temp_model.predict(selected_val))
    print(f"Features: {num_features:3d} | Score: {score:.4f}")

    if score > best_score_fs:
        best_score_fs = score
        # 성능이 가장 좋았을 때의 '원래 인덱스'들을 저장
        best_indices_subset = top_k_indices[:num_features]

print(f"\n🔥 Best score after feature selection: {best_score_fs:.4f} with {len(best_indices_subset)} features.")

# 최종 데이터 준비
# 전체 데이터에 스케일링 및 최종 피처 선택 적용
x_scaled_full = scaler.transform(x_raw)
x_selected_full = x_scaled_full[:, best_indices_subset]
x_test_selected = x_test_scaled_full[:, best_indices_subset]

# 최종 모델 학습을 위한 데이터 분할
x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
    x_selected_full, y, test_size=0.2, random_state=seed
)
# ######################################################################
# ###                        수정된 부분 끝                          ###
# ######################################################################


# --- 모델 정의 및 학습 ---
base_models = {
    "XGBoost": xgb.XGBRegressor(**xgb_model_for_fs.get_params()),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed, ),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=seed),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=seed),
    # "CatBoost": cat.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
    #     l2_leaf_reg=3, random_seed=seed, verbose=0)
}

trained_models = {}
best_score = -np.inf
best_model_name = None

for name, model in base_models.items():
    print(f"\n{name} 모델 학습 중...")
    m = copy.deepcopy(model)
    # if name == "CatBoost":
    #     m.fit(x_train_final, y_train_final, eval_set=(x_val_final, y_val_final), early_stopping_rounds=50, verbose=0)
    if name == "XGBoost":
        m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)],
              callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)], verbose=0)
    elif name == "LightGBM":
        m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)],
              callbacks=[lgb.early_stopping(50)])
    else:
        m.fit(x_train_final, y_train_final)

    y_pred = m.predict(x_val_final)
    score = competition_score(y_val_final, y_pred)
    print(f"→ Score: {score:.4f}")
    trained_models[name] = m
    if score > best_score:
        best_score = score
        best_model_name = name
# stacking에 학습되지 않은 원래 모델 객체를 넣기 위해 base model 다시 정의
stacking_estimators = [
    ("XGBoost", xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed,
        tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0
    )),
    ("LightGBM", lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
        random_state=seed
    )),
    ("GradientBoosting", GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8,
        random_state=seed
    )),
    ("RandomForest", RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=seed
    )),
]

# stacking 모델 재정의
stacking_model = StackingRegressor(
    estimators=stacking_estimators,
    final_estimator=Ridge(),
    n_jobs=-1
)

# fit 호출
stacking_model.fit(x_train_final, y_train_final)
y_pred_stack = stacking_model.predict(x_val_final)
stack_score = competition_score(y_val_final, y_pred_stack)
print(f"→ Stacking | Score: {stack_score:.4f}")

# 최종 선택
final_model = stacking_model if stack_score > best_score else trained_models[best_model_name]

# 전체 데이터로 재학습 및 예측
final_model.fit(x_selected_full, y)
y_pred_test = final_model.predict(x_test_selected)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_final_{timestamp}.csv"
submission['Inhibition'] = y_pred_test
submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n✅ 예측 결과 저장 완료 → {filename}")

# --- [복원된 부분] 모든 모델 성능 비교 ---
print("\n📊 모델별 성능 비교 (정확한 유효성 검사 세트를 사용)")
for name, model in trained_models.items():
    y_pred_compare = model.predict(x_val_final)
    print(f"{name:20} | RMSE: {rmse(y_val_final, y_pred_compare):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_compare):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_compare):.4f} | Score: {competition_score(y_val_final, y_pred_compare):.4f}")

print(f"{'StackingRegressor':20} | RMSE: {rmse(y_val_final, y_pred_stack):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_stack):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_stack):.4f} | Score: {competition_score(y_val_final, y_pred_stack):.4f}")