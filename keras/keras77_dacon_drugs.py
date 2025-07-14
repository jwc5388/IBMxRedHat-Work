# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# import xgboost as xgb
# import lightgbm as lgb

# from rdkit import Chem
# from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem

# import warnings
# warnings.filterwarnings('ignore')

# # 데이터 로드
# path = './Study25/_data/dacon/drugs/'
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')
# submission = pd.read_csv(path + 'sample_submission.csv')

# print(f"훈련 데이터 : {train.shape}")
# print(f"테스트 데이터 : {test.shape}")

# # 훈련 데이터 : (1681, 3)
# # 테스트 데이터 : (100, 2)


# exit()


# # 분자 특성 추출 함수
# def get_molecule_descriptors(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return [0] * 2232

#         basic = [
#             Descriptors.MolWt(mol),
#             Descriptors.MolLogP(mol),
#             Descriptors.NumHAcceptors(mol),
#             Descriptors.NumHDonors(mol),
#             Descriptors.TPSA(mol),
#             Descriptors.NumRotatableBonds(mol),
#             Descriptors.NumAromaticRings(mol),
#             Descriptors.NumHeteroatoms(mol),
#             Descriptors.FractionCSP3(mol),
#             Descriptors.NumAliphaticRings(mol),
#             Lipinski.NumAromaticHeterocycles(mol),
#             Lipinski.NumSaturatedHeterocycles(mol),
#             Lipinski.NumAliphaticHeterocycles(mol),
#             Descriptors.HeavyAtomCount(mol),
#             Descriptors.RingCount(mol),
#             Descriptors.NOCount(mol),
#             Descriptors.NHOHCount(mol),
#             Descriptors.NumRadicalElectrons(mol),
#         ]

#         morgan = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
#         maccs = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]

#         return basic + morgan + maccs
#     except:
#         return [0] * 2232

# # 훈련용 특성 추출
# train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)
# x_train_list = train['features'].tolist()

# # 길이 맞추기
# max_len = max(len(x) for x in x_train_list)
# x_train_list = [x + [0] * (max_len - len(x)) for x in x_train_list]
# x_train = np.array(x_train_list)
# y_train = train['Inhibition'].values

# # 테스트용 특성 추출
# test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)
# x_test_list = test['features'].tolist()
# x_test_list = [x + [0] * (max_len - len(x)) for x in x_test_list]
# x_test = np.array(x_test_list)

# print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

# # 스케일링
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # 데이터 분할
# x_train_final, x_val, y_train_final, y_val = train_test_split(
#     x_train, y_train, test_size=0.2, random_state=42
# )

# # 평가 지표 함수
# def normalized_rmse(y_true, y_pred):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     return rmse / (np.max(y_true) - np.min(y_true))

# def pearson_correlation(y_true, y_pred):
#     corr = np.corrcoef(y_true, y_pred)[0, 1]
#     return np.clip(corr, 0, 1)

# def competition_score(y_true, y_pred):
#     nrmse = min(normalized_rmse(y_true, y_pred), 1)
#     pearson = pearson_correlation(y_true, y_pred)
#     return 0.5 * (1 - nrmse) + 0.5 * pearson

# # 모델 학습 및 검증
# def train_and_evaluate_model(model, x_train, y_train, x_val, y_val):
#     model.fit(x_train, y_train)
#     y_val_pred = model.predict(x_val)
#     val_nrmse = normalized_rmse(y_val, y_val_pred)
#     val_pearson = pearson_correlation(y_val, y_val_pred)
#     val_score = competition_score(y_val, y_val_pred)
#     print(f"검증 NRMSE: {val_nrmse:.4f}")
#     print(f"검증 Pearson: {val_pearson:.4f}")
#     print(f"검증 점수: {val_score:.4f}")
#     return model, val_score

# # 모델 정의
# models = {
#     "XGBoost": xgb.XGBRegressor(
#         n_estimators=500, learning_rate=0.05, max_depth=6,
#         subsample=0.8, colsample_bytree=0.8, gamma=0,
#         reg_alpha=0.1, reg_lambda=1, random_state=42
#     ),
#     "LightGBM": lgb.LGBMRegressor(
#         n_estimators=500, learning_rate=0.05, num_leaves=31,
#         max_depth=6, subsample=0.8, colsample_bytree=0.8,
#         reg_alpha=0.1, reg_lambda=1, random_state=42
#     ),
#     "GradientBoosting": GradientBoostingRegressor(
#         n_estimators=300, learning_rate=0.05, max_depth=5,
#         min_samples_split=5, min_samples_leaf=2, subsample=0.8,
#         random_state=42
#     ),
#     "RandomForest": RandomForestRegressor(
#         n_estimators=300, max_depth=10, min_samples_split=5,
#         min_samples_leaf=2, random_state=42
#     )
# }

# # 모델 학습 루프
# best_score = -np.inf
# best_model_name = None
# trained_models = {}

# for name, model in models.items():
#     print(f"\n{name} 모델 학습 중...")
#     trained_model, val_score = train_and_evaluate_model(
#         model, x_train_final, y_train_final, x_val, y_val
#     )
#     trained_models[name] = trained_model
#     if val_score > best_score:
#         best_score = val_score
#         best_model_name = name

# print(f"\n최고 성능 모델: {best_model_name}, 검증 점수: {best_score:.4f}")

# # 테스트 데이터 예측 및 저장
# final_model = models[best_model_name]
# final_model.fit(x_train, y_train)

# y_pred_test = final_model.predict(x_test)
# submission['Inhibition'] = y_pred_test
# submission.to_csv('improved_submission.csv', index=False)
# print("예측 결과 저장: improved_submission.csv")

# # 검증 결과 시각화
# plt.figure(figsize=(10, 6))
# y_val_pred = trained_models[best_model_name].predict(x_val)
# plt.scatter(y_val, y_val_pred, alpha=0.5)
# plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
# plt.xlabel('실제값')
# plt.ylabel('예측값')
# plt.title(f'{best_model_name} 모델 검증 성능')
# plt.show()
# plt.savefig('model_performance.png')
# print("모델 성능 시각화 저장: model_performance.png")

# # 특성 중요도 시각화
# if hasattr(final_model, 'feature_importances_'):
#     n_features = 20
#     importances = final_model.feature_importances_
#     indices = np.argsort(importances)[::-1][:n_features]
#     plt.figure(figsize=(12, 8))
#     plt.title('상위 특성 중요도')
#     plt.bar(range(n_features), importances[indices])
#     plt.xticks(range(n_features), indices, rotation=90)
#     plt.tight_layout()
#     plt.savefig('feature_importance.png')
#     print("특성 중요도 시각화 저장: feature_importance.png")


# 전체 코드: Mordred + Morgan Fingerprint + 앙상블 예측

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

# 1. 데이터 로드
path = basepath + '_data/dacon/drugs/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

# 2. 피처 추출 함수 정의
calc = Calculator(descriptors, ignore_3D=True)

def get_combined_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mordred_desc = list(calc(mol).values())
        morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        return mordred_desc + morgan_fp
    except:
        return None

# 3. 분자 특성 추출
train['features'] = train['Canonical_Smiles'].progress_apply(get_combined_features)
test['features'] = test['Canonical_Smiles'].progress_apply(get_combined_features)

train = train[train['features'].notnull()].reset_index(drop=True)
test = test[test['features'].notnull()].reset_index(drop=True)

x_train = np.array(train['features'].tolist())
y_train = train['Inhibition'].values
x_test = np.array(test['features'].tolist())

# 4. 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. 평가 함수 정의
def normalized_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def pearson_correlation(y_true, y_pred):
    return np.clip(np.corrcoef(y_true, y_pred)[0, 1], 0, 1)

def competition_score(y_true, y_pred):
    nrmse = min(normalized_rmse(y_true, y_pred), 1)
    pearson = pearson_correlation(y_true, y_pred)
    return 0.5 * (1 - nrmse) + 0.5 * pearson

# 6. 학습용 데이터 분할
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print("x_train NaN 개수:", np.isnan(x_train).sum())
print("x_val NaN 개수:", np.isnan(x_val).sum())

# exit()
from catboost import CatBoostRegressor
# 7. 모델 정의
models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42),
    "CatBoost": CatBoostRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42),

    # "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
}

# 8. 모델 학습 및 검증
val_preds = {}
test_preds = {}

for name, model in models.items():
    print(f"\n[{name}] 학습 중...")
    model.fit(x_train_final, y_train_final)
    y_val_pred = model.predict(x_val)
    score = competition_score(y_val, y_val_pred)
    print(f"{name} 검증 점수: {score:.4f}")
    
    val_preds[name] = y_val_pred
    test_preds[name] = model.predict(x_test)

# 9. 앙상블 예측
y_val_ensemble = np.mean(list(val_preds.values()), axis=0)
y_test_ensemble = np.mean(list(test_preds.values()), axis=0)

final_score = competition_score(y_val, y_val_ensemble)
print(f"\n✅ 앙상블 최종 점수: {final_score:.4f}")
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_ensemble))
print(f"📉 앙상블 RMSE: {rmse_val:.4f}")

# 10. 제출 저장
submission['Inhibition'] = y_test_ensemble
submission.to_csv(path + 'ensemble_submission.csv', index=False)
print("📁 예측 결과 저장 완료: ensemble_submission.csv")

# 11. 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_ensemble, alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('앙상블 모델 검증 성능')
plt.tight_layout()
plt.savefig('ensemble_model_performance.png')
print("📊 모델 성능 시각화 저장 완료: ensemble_model_performance.png")


# def is_valid_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return False
#         Chem.SanitizeMol(mol)
#         return True
#     except:
#         return False

# def sanitize_and_augment_smiles(smiles, n_aug=5):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return []
#     try:
#         Chem.SanitizeMol(mol)
#     except:
#         return []
#     augmented_smiles = []
#     for _ in range(n_aug):
#         rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
#         augmented_smiles.append(rand_smiles)
#     return augmented_smiles

# def morgan_fp(smiles, radius=2, n_bits=2048):
#     mol = Chem.MolFromSmiles(smiles)
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#     arr = np.zeros((n_bits,), dtype=int)
#     DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# def maccs_fp(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     fp = MACCSkeys.GenMACCSKeys(mol)
#     arr = np.zeros((fp.GetNumBits(),), dtype=int)
#     DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# def calc_descriptors(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     return {
#         'MolWt': Descriptors.MolWt(mol),
#         'LogP': Descriptors.MolLogP(mol),
#         'TPSA': Descriptors.TPSA(mol),
#         'HBD': Lipinski.NumHDonors(mol),
#         'HBA': Lipinski.NumHAcceptors(mol),
#         'RotatableBonds': Lipinski.NumRotatableBonds(mol),
#         'RingCount': Lipinski.RingCount(mol)
#     }

# def smiles_to_graph(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None

#     atom_features = []
#     for atom in mol.GetAtoms():
#         atom_features.append([atom.GetAtomicNum()])
#     x = torch.tensor(atom_features, dtype=torch.float)

#     edge_index = []
#     for bond in mol.GetBonds():
#         start = bond.GetBeginAtomIdx()
#         end = bond.GetEndAtomIdx()
#         edge_index.append([start, end])
#         edge_index.append([end, start])
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

#     return Data(x=x, edge_index=edge_index)

# # --- 3. 데이터 정제 및 증강 ---
# train = train[train['Canonical_Smiles'].apply(is_valid_smiles)].reset_index(drop=True)
# test = test[test['Canonical_Smiles'].apply(is_valid_smiles)].reset_index(drop=True)
# train = train.drop_duplicates(subset='Canonical_Smiles').reset_index(drop=True)

# aug_smiles_list = []
# aug_targets_list = []

# for idx, row in train.iterrows():
#     smiles = row['Canonical_Smiles']
#     target = row['Inhibition']
#     augmented = sanitize_and_augment_smiles(smiles, n_aug=5)
#     augmented.append(smiles)
#     for sm in augmented:
#         aug_smiles_list.append(sm)
#         aug_targets_list.append(target)

# aug_train_df = pd.DataFrame({'Canonical_Smiles': aug_smiles_list, 'Inhibition': aug_targets_list})
