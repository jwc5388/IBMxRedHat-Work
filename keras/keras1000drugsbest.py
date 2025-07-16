# ✅ 수정 사항 반영:
# - early_stopping_rounds 및 callbacks 전부 제거 → 모델 정의부로 이동
# - XGBoost, LightGBM, CatBoost 모두 반영 완료

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
from datetime import datetime
import random
import warnings
import copy

seed = 42
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')

BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
path = os.path.join(BASE_PATH, '_data/dacon/drugs/')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

def get_molecule_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [0] * 2233
        basic = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol), Descriptors.FractionCSP3(mol),
            Descriptors.NumAliphaticRings(mol), Lipinski.NumAromaticHeterocycles(mol),
            Lipinski.NumSaturatedHeterocycles(mol), Lipinski.NumAliphaticHeterocycles(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol), Descriptors.NOCount(mol),
            Descriptors.NHOHCount(mol), Descriptors.NumRadicalElectrons(mol)
        ]
        morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
        maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
        return basic + morgan + maccs
    except:
        return [0] * 2233

train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

x_raw = np.array(train['features'].tolist())
y = train['Inhibition'].values
x_test_raw = np.array(test['features'].tolist())

scaler = RobustScaler()
x_scaled = scaler.fit_transform(x_raw)
x_test_scaled = scaler.transform(x_test_raw)

x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=seed)

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def normalized_rmse(y_true, y_pred): return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
def pearson_correlation(y_true, y_pred): return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)
def competition_score(y_true, y_pred): return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

def create_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed,
        # tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
        early_stopping_rounds=50, eval_metric="rmse"
    )

def create_lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
        random_state=seed, early_stopping=50
    )

def create_cat_model():
    return cat.CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=seed, verbose=0,
        early_stopping_rounds=50
    )

xgb_model = create_xgb_model()
xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

booster = xgb_model.get_booster()
score_dict = booster.get_score(importance_type='gain')
total_gain = sum(score_dict.values())
score_list = [score_dict.get(f"f{i}", 0) / total_gain for i in range(x_train.shape[1])]
thresholds = np.sort(score_list)

max_score = -np.inf
best_selection = None
for threshold in thresholds:
    selection = SelectFromModel(xgb_model, threshold=threshold, prefit=True)
    selected_train = selection.transform(x_train)
    selected_val = selection.transform(x_val)
    if selected_train.shape[1] == 0: continue
    temp_model = create_xgb_model()
    temp_model.fit(selected_train, y_train, eval_set=[(selected_val, y_val)], verbose=0)
    score = competition_score(y_val, temp_model.predict(selected_val))
    if score > max_score:
        max_score = score
        best_selection = selection

print(f"\n🔥 Best score after feature selection: {max_score:.4f}")

x_selected = best_selection.transform(x_scaled)
x_test_selected = best_selection.transform(x_test_scaled)
x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_selected, y, test_size=0.2, random_state=seed)

base_models = {
    "XGBoost": create_xgb_model(),
    "LightGBM": create_lgb_model(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=seed),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=seed),
    "CatBoost": create_cat_model()
}

trained_models = {}
best_score = -np.inf
best_model_name = None

for name, model in base_models.items():
    print(f"\n{name} 모델 학습 중...")
    m = copy.deepcopy(model)
    if name in ["CatBoost", "XGBoost", "LightGBM"]:
        m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)], verbose=0)
    else:
        m.fit(x_train_final, y_train_final)

    y_pred = m.predict(x_val_final)
    score = competition_score(y_val_final, y_pred)
    print(f"→ Score: {score:.4f}")
    trained_models[name] = m
    if score > best_score:
        best_score = score
        best_model_name = name

stacking_model = StackingRegressor(
    estimators=[(k.lower(), v) for k, v in trained_models.items()],
    final_estimator=Ridge(), n_jobs=-1
)

stacking_model.fit(x_train_final, y_train_final)
y_pred_stack = stacking_model.predict(x_val_final)
stack_score = competition_score(y_val_final, y_pred_stack)
print(f"→ Stacking | Score: {stack_score:.4f}")

final_model = stacking_model if stack_score > best_score else trained_models[best_model_name]
final_model.fit(x_selected, y)
y_pred_test = final_model.predict(x_test_selected)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"submission_final_{timestamp}.csv"
submission['Inhibition'] = y_pred_test
submission.to_csv(os.path.join(path, filename), index=False)
print(f"\n✅ 예측 결과 저장 완료 → {filename}")

print("\n📊 모델별 성능 비교")
for name, model in trained_models.items():
    y_pred_compare = model.predict(x_val_final)
    print(f"{name:20} | RMSE: {rmse(y_val_final, y_pred_compare):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_compare):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_compare):.4f} | Score: {competition_score(y_val_final, y_pred_compare):.4f}")

print(f"{'StackingRegressor':20} | RMSE: {rmse(y_val_final, y_pred_stack):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_stack):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_stack):.4f} | Score: {stack_score:.4f}")


	# 1.	RDKit 기반 피처 생성 (basic + Morgan + MACCS)
	# 2.	정규화 (StandardScaler) 적용 후 feature importance 계산
	# 3.	XGBoost gain 기준 feature importance → SelectFromModel으로 최적 threshold 탐색
	# 4.	최적 threshold 기준으로 불필요한 feature 제거
	# 5.	그 결과를 반영하여 XGBoost, LightGBM, CatBoost 등 개별 모델 학습
	# 6.	스태킹 모델 학습 및 최종 성능 비교
	# 7.	테스트셋 예측값 제출 저장까지 포함
 
 
 
 
 
 
 
 
 
 #2 Modified version with faster feature deletion
 
 # ✅ 수정 사항 반영:
# - early_stopping_rounds 및 callbacks 전부 제거 → 모델 정의부로 이동
# - XGBoost, LightGBM, CatBoost 모두 반영 완료
# - feature selection loop → 상위 30개 feature 기준으로만 탐색
# - train/val split 인덱스 고정으로 검증 누수 방지

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.feature_selection import SelectFromModel
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cat
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem
# from datetime import datetime
# import random
# import warnings
# import copy

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# warnings.filterwarnings('ignore')

# BASE_PATH = '/workspace/TensorJae/Study25/' if os.path.exists('/workspace/TensorJae/Study25/') else os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
# path = os.path.join(BASE_PATH, '_data/dacon/drugs/')
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# submission = pd.read_csv(path + 'sample_submission.csv')

# def get_molecule_descriptors(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None: return [0] * 2233
#         basic = [
#             Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
#             Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
#             Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol), Descriptors.FractionCSP3(mol),
#             Descriptors.NumAliphaticRings(mol), Lipinski.NumAromaticHeterocycles(mol),
#             Lipinski.NumSaturatedHeterocycles(mol), Lipinski.NumAliphaticHeterocycles(mol),
#             Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol), Descriptors.NOCount(mol),
#             Descriptors.NHOHCount(mol), Descriptors.NumRadicalElectrons(mol)
#         ]
#         morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
#         maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
#         return basic + morgan + maccs
#     except:
#         return [0] * 2233

# train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
# test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)

# x_raw = np.array(train['features'].tolist())
# y = train['Inhibition'].values
# x_test_raw = np.array(test['features'].tolist())

# scaler = RobustScaler()
# x_scaled = scaler.fit_transform(x_raw)
# x_test_scaled = scaler.transform(x_test_raw)

# # ✅ 고정 인덱스를 통한 split
# train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
# x_train, x_val = x_scaled[train_idx], x_scaled[val_idx]
# y_train, y_val = y[train_idx], y[val_idx]

# def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
# def normalized_rmse(y_true, y_pred): return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
# def pearson_correlation(y_true, y_pred): return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)
# def competition_score(y_true, y_pred): return 0.5 * (1 - min(normalized_rmse(y_true, y_pred), 1)) + 0.5 * pearson_correlation(y_true, y_pred)

# def create_xgb_model():
#     return xgb.XGBRegressor(
#         n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
#         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=seed,
#         tree_method='hist', predictor='auto',
#         early_stopping_rounds=50, eval_metric="rmse"
#     )

# def create_lgb_model():
#     return lgb.LGBMRegressor(
#         n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
#         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1,
#         random_state=seed, early_stopping=50
#     )

# def create_cat_model():
#     return cat.CatBoostRegressor(
#         iterations=500, learning_rate=0.05, depth=6,
#         l2_leaf_reg=3, random_seed=seed, verbose=0,
#         early_stopping_rounds=50
#     )

# xgb_model = create_xgb_model()
# xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

# booster = xgb_model.get_booster()
# score_dict = booster.get_score(importance_type='gain')
# total_gain = sum(score_dict.values())
# score_list = [score_dict.get(f"f{i}", 0) / total_gain for i in range(x_train.shape[1])]

# # ✅ 시간 단축: 상위 30개 feature만 threshold로 사용
# thresholds = np.sort(score_list)[-30:]

# max_score = -np.inf
# best_selection = None
# for threshold in thresholds:
#     selection = SelectFromModel(xgb_model, threshold=threshold, prefit=True)
#     selected_train = selection.transform(x_train)
#     selected_val = selection.transform(x_val)
#     if selected_train.shape[1] == 0: continue
#     temp_model = create_xgb_model()
#     temp_model.fit(selected_train, y_train, eval_set=[(selected_val, y_val)], verbose=0)
#     score = competition_score(y_val, temp_model.predict(selected_val))
#     if score > max_score:
#         max_score = score
#         best_selection = selection

# print(f"\n🔥 Best score after feature selection: {max_score:.4f}")

# x_selected = best_selection.transform(x_scaled)
# x_test_selected = best_selection.transform(x_test_scaled)

# x_train_final = x_selected[train_idx]
# x_val_final = x_selected[val_idx]
# y_train_final = y[train_idx]
# y_val_final = y[val_idx]

# base_models = {
#     "XGBoost": create_xgb_model(),
#     "LightGBM": create_lgb_model(),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
#         min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=seed),
#     "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5,
#         min_samples_leaf=2, random_state=seed),
#     "CatBoost": create_cat_model()
# }

# trained_models = {}
# best_score = -np.inf
# best_model_name = None

# for name, model in base_models.items():
#     print(f"\n{name} 모델 학습 중...")
#     m = copy.deepcopy(model)
#     if name in ["CatBoost", "XGBoost", "LightGBM"]:
#         m.fit(x_train_final, y_train_final, eval_set=[(x_val_final, y_val_final)], verbose=0)
#     else:
#         m.fit(x_train_final, y_train_final)

#     y_pred = m.predict(x_val_final)
#     score = competition_score(y_val_final, y_pred)
#     print(f"→ Score: {score:.4f}")
#     trained_models[name] = m
#     if score > best_score:
#         best_score = score
#         best_model_name = name

# stacking_model = StackingRegressor(
#     estimators=[(k.lower(), v) for k, v in trained_models.items()],
#     final_estimator=Ridge(), n_jobs=-1
# )

# stacking_model.fit(x_train_final, y_train_final)
# y_pred_stack = stacking_model.predict(x_val_final)
# stack_score = competition_score(y_val_final, y_pred_stack)
# print(f"→ Stacking | Score: {stack_score:.4f}")

# final_model = stacking_model if stack_score > best_score else trained_models[best_model_name]
# final_model.fit(x_selected, y)
# y_pred_test = final_model.predict(x_test_selected)

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# filename = f"submission_final_{timestamp}.csv"
# submission['Inhibition'] = y_pred_test
# submission.to_csv(os.path.join(path, filename), index=False)
# print(f"\n✅ 예측 결과 저장 완료 → {filename}")

# print("\n📊 모델별 성능 비교")
# for name, model in trained_models.items():
#     y_pred_compare = model.predict(x_val_final)
#     print(f"{name:20} | RMSE: {rmse(y_val_final, y_pred_compare):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_compare):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_compare):.4f} | Score: {competition_score(y_val_final, y_pred_compare):.4f}")

# print(f"{'StackingRegressor':20} | RMSE: {rmse(y_val_final, y_pred_stack):.4f} | NRMSE: {normalized_rmse(y_val_final, y_pred_stack):.4f} | Pearson: {pearson_correlation(y_val_final, y_pred_stack):.4f} | Score: {stack_score:.4f}")