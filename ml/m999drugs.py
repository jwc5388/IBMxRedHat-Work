# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from rdkit import Chem

# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, BatchNorm

# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from torch.optim.lr_scheduler import CosineAnnealingLR

# import torch

# import os

# if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
#     BASE_PATH = '/workspace/TensorJae/Study25/'
# else:                                                 # 로컬인 경우
#     BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
# basepath = os.path.join(BASE_PATH)

# # CUDA 사용 가능 여부 확인
# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# print(f"CUDA 사용 가능 여부: {USE_CUDA}")
# print(f"현재 사용 중인 디바이스: {DEVICE}")


# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')
    
# print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# # ================== 하이퍼파라미터 ==================
# class Config:
#     DATA_PATH = basepath + '_data/dacon/drugs/'
#     N_SPLITS = 5
#     RANDOM_STATE = 42
#     EPOCHS = 300
#     PATIENCE = 30
#     BATCH_SIZE = 128
#     LR = 0.0005
#     AUGMENTATION_FACTOR = 4

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"사용 디바이스: {device}")

# # ================== 데이터 로딩 ==================
# train_df = pd.read_csv(os.path.join(Config.DATA_PATH, 'train.csv'))
# test_df = pd.read_csv(os.path.join(Config.DATA_PATH, 'test.csv'))
# submission = pd.read_csv(os.path.join(Config.DATA_PATH, 'sample_submission.csv'))

# train_df['Inhibition_log'] = np.log1p(train_df['Inhibition'])
# y_mean = train_df['Inhibition_log'].mean()
# y_std = train_df['Inhibition_log'].std()
# train_df['target'] = (train_df['Inhibition_log'] - y_mean) / y_std

# # ================== SMILES to Graph ==================
# def one_hot_encoding(x, permitted_list):
#     if x not in permitted_list:
#         x = permitted_list[-1]
#     return [int(x == s) for s in permitted_list]

# def atom_features(atom):
#     return one_hot_encoding(atom.GetAtomicNum(), [5,6,7,8,9,15,16,17,35,53,999]) + \
#            one_hot_encoding(atom.GetDegree(), [0,1,2,3,4,5]) + \
#            one_hot_encoding(atom.GetTotalNumHs(), [0,1,2,3,4]) + \
#            one_hot_encoding(atom.GetImplicitValence(), [0,1,2,3,4,5]) + \
#            [atom.GetIsAromatic()]

# def bond_features(bond):
#     bt = bond.GetBondType()
#     return [int(bt == Chem.rdchem.BondType.SINGLE),
#             int(bt == Chem.rdchem.BondType.DOUBLE),
#             int(bt == Chem.rdchem.BondType.TRIPLE),
#             int(bt == Chem.rdchem.BondType.AROMATIC),
#             int(bond.IsInRing())]

# def smiles_to_data(smiles, target=None):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None: return None
#     atoms = [atom_features(atom) for atom in mol.GetAtoms()]
#     edges, edge_attrs = [], []
#     for bond in mol.GetBonds():
#         i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         feat = bond_features(bond)
#         edges.extend([[i, j], [j, i]])
#         edge_attrs.extend([feat, feat])
#     x = torch.tensor(atoms, dtype=torch.float)
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#     if target is not None:
#         data.y = torch.tensor([target], dtype=torch.float)
#     return data

# sample_mol = Chem.MolFromSmiles('CN(C)C(=O)c1ccccc1')
# IN_CHANNELS = len(atom_features(sample_mol.GetAtomWithIdx(0)))
# EDGE_CHANNELS = len(bond_features(sample_mol.GetBondWithIdx(0)))

# # ================== 모델 정의 ==================
# class OptimizedMPNN(nn.Module):
#     def __init__(self, in_channels, edge_channels, hidden_channels=256, out_channels=1, dropout=0.3):
#         super().__init__()
#         self.conv1 = NNConv(in_channels, hidden_channels,
#                             nn.Sequential(nn.Linear(edge_channels, 128), nn.ReLU(),
#                                           nn.Linear(128, in_channels * hidden_channels)), aggr='mean')
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = NNConv(hidden_channels, hidden_channels,
#                             nn.Sequential(nn.Linear(edge_channels, 256), nn.ReLU(),
#                                           nn.Linear(256, hidden_channels * hidden_channels)), aggr='mean')
#         self.bn2 = BatchNorm(hidden_channels)
#         self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
#         self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
#         self.fc3 = nn.Linear(hidden_channels // 2, out_channels)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
#         x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
#         x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         return self.fc3(x)

# # ================== 학습/평가 함수 ==================
# def train_epoch(model, loader, optimizer):
#     model.train()
#     total_loss = 0
#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         out = model(batch)
#         loss = F.mse_loss(out.view(-1), batch.y.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * batch.num_graphs
#     return total_loss / len(loader.dataset)

# def evaluate(model, loader):
#     model.eval()
#     preds, targets = [], []
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             out = model(batch)
#             preds.extend(out.view(-1).cpu().numpy())
#             if batch.y is not None:
#                 targets.extend(batch.y.view(-1).cpu().numpy())
#     if len(targets) == 0:
#         return np.array(preds)
#     else:
#         return np.array(preds), np.array(targets)

# # ================== K-Fold ==================
# kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
# oof_preds = np.zeros(len(train_df))
# all_test_preds = np.zeros(len(test_df))

# for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
#     print(f"\n== Fold {fold+1} ==")
#     train_graphs = []
#     for i in tqdm(train_idx):
#         smi = train_df.iloc[i]['Canonical_Smiles']
#         y = train_df.iloc[i]['target']
#         g = smiles_to_data(smi, y)
#         if g: train_graphs.append(g)
#         for _ in range(Config.AUGMENTATION_FACTOR):
#             try:
#                 mol = Chem.MolFromSmiles(smi)
#                 r_smi = Chem.MolToSmiles(mol, doRandom=True)
#                 g_aug = smiles_to_data(r_smi, y)
#                 if g_aug: train_graphs.append(g_aug)
#             except: continue
#     val_graphs = [smiles_to_data(train_df.iloc[i]['Canonical_Smiles'], train_df.iloc[i]['target']) for i in val_idx]
#     val_graphs = [g for g in val_graphs if g is not None]

#     train_loader = DataLoader(train_graphs, batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_graphs, batch_size=Config.BATCH_SIZE)

#     model = OptimizedMPNN(IN_CHANNELS, EDGE_CHANNELS).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

#     best_rmse = float('inf')
#     patience = 0

#     for epoch in range(1, Config.EPOCHS+1):
#         loss = train_epoch(model, train_loader, optimizer)
#         val_preds, val_targets = evaluate(model, val_loader)
#         val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
#         scheduler.step()

#         if val_rmse < best_rmse:
#             best_rmse = val_rmse
#             patience = 0
#             best_val_preds = val_preds
#             torch.save(model.state_dict(), f'model_fold{fold+1}.pt')
#         else:
#             patience += 1
#             if patience >= Config.PATIENCE:
#                 break

#     oof_preds[val_idx] = best_val_preds

#     test_graphs = [smiles_to_data(s) for s in test_df['Canonical_Smiles']]
#     test_graphs = [g for g in test_graphs if g is not None]
#     test_loader = DataLoader(test_graphs, batch_size=Config.BATCH_SIZE)

#     model.load_state_dict(torch.load(f'model_fold{fold+1}.pt'))
#     fold_test_preds = evaluate(model, test_loader)
#     all_test_preds += fold_test_preds / Config.N_SPLITS

# # ================== 예측값 복원 ==================
# oof_preds_inv = np.expm1(oof_preds * y_std + y_mean)
# test_preds_inv = np.expm1(all_test_preds * y_std + y_mean)
# oof_preds_inv[oof_preds_inv < 0] = 0
# test_preds_inv[test_preds_inv < 0] = 0

# # ================== 평가 및 저장 ==================
# rmse = np.sqrt(mean_squared_error(train_df['Inhibition'], oof_preds_inv))
# norm_rmse = rmse / (train_df['Inhibition'].max() - train_df['Inhibition'].min())
# pearson = np.corrcoef(train_df['Inhibition'], oof_preds_inv)[0, 1]
# final_score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * pearson

# print(f"\nRMSE: {rmse:.4f}")
# print(f"Normalized RMSE: {norm_rmse:.4f}")
# print(f"Pearson Correlation: {pearson:.4f}")
# print(f"Final Score: {final_score:.4f}")

# submission['Inhibition'] = test_preds_inv
# submission.to_csv('submission_optimized_mpnn.csv', index=False)
# print("제출 저장 완료: submission_optimized_mpnn.csv")






import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.AtomPairs import Pairs, Torsions # 이 줄을 추가합니다.
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Leaderboard metrics
def normalized_rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = float(np.max(y_true) - np.min(y_true))
    return 0.0 if denom == 0.0 else rmse / denom
def pearson_corr_clip01(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    sy, sp = y_true.std(ddof=0), y_pred.std(ddof=0)
    if sy == 0.0 or sp == 0.0: return 0.0
    cov = np.cov(y_true, y_pred, bias=True)[0, 1]
    r = cov / (sy * sp)
    return float(np.clip(r, 0.0, 1.0))
def leaderboard_score(y_true, y_pred):
    A, B = normalized_rmse(y_true, y_pred), pearson_corr_clip01(y_true, y_pred)
    return A, B, 0.5 * ((1.0 - A) + B)
def print_metrics(name, y_true, y_pred):
    A, B, S = leaderboard_score(y_true, y_pred)
    print(f"{name:>10s} | A(NRMSE)={A:.6f}  B(Pearson)={B:.6f}  Score={S:.6f}")
    return A, B, S

# 1) 시드 고정
seed = 8915
def seed_everything(s):
    random.seed(s); np.random.seed(s); os.environ['PYTHONHASHSEED'] = str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_everything(seed)

# 2) 데이터 로드
path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/dacon/drugs'
train = pd.read_csv(os.path.join(path, "train.csv"))
test  = pd.read_csv(os.path.join(path, "test.csv"))
y     = train['Inhibition'].values

# 3) Scaffold split
def scaffold_split(df, smiles_col='Canonical_Smiles', n_folds=5):
    scaffolds = {}
    for i, smi in enumerate(df[smiles_col]):
        scaf = MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        scaffolds.setdefault(scaf, []).append(i)
    groups = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    folds = [[] for _ in range(n_folds)]
    for idx, (_, idxs) in enumerate(groups): folds[idx % n_folds].extend(idxs)
    return folds
folds = scaffold_split(train, n_folds=5)

# 4) SMILES 증강
def smiles_augment(smiles, n_aug=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [smiles]*n_aug
    out = {Chem.MolToSmiles(mol, canonical=True)}
    while len(out) < n_aug: out.add(Chem.MolToSmiles(mol, doRandom=True))
    return list(out)

# 5) 전역 피처
def global_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(10, dtype=np.float32)
    vals = [ Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol), Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol), sum(int(a.GetIsAromatic()) for a in mol.GetAtoms())/max(1,mol.GetNumAtoms()) ]
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ==============================================================================
# [추가] 다양한 분자 지문 생성 함수
# ==============================================================================
def morgan_fp(smi, radius=2, nBits=2048):
    arr = np.zeros((nBits,), dtype=np.uint8)
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
def atom_pair_fp(smi, nBits=2048):
    arr = np.zeros((nBits,), dtype=np.uint8)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return arr
    # ↳ RDKit 버전 호환: rdMolDescriptors의 해시드 AtomPair 사용
    fp = rdMD.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def topological_torsion_fp(smi, nBits=2048):
    arr = np.zeros((nBits,), dtype=np.uint8)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return arr
    # ↳ RDKit 버전 호환: rdMolDescriptors의 해시드 TopologicalTorsion 사용
    fp = rdMD.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
# ==============================================================================

# 7) SMILES→Graph (GNN 특징 강화)
def mol_to_graph(smi, lbl=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    atom_features = [[
        atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(),
        atom.GetHybridization(), atom.GetIsAromatic(), atom.GetChiralTag(),
        atom.IsInRing(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
    ] for atom in mol.GetAtoms()]
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_features = [bond.GetBondTypeAsDouble(), bond.GetIsConjugated(), bond.IsInRing()]
        edge_attr.extend([bond_features, bond_features])
    if not edge_index: return None
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        global_feat=torch.tensor(global_descriptors(smi), dtype=torch.float32).view(1,-1),
        y=None if lbl is None else torch.tensor([lbl], dtype=torch.float32)
    )

# 8) GNN 정의 (강화된 특징에 맞게 수정)
class MPNN(nn.Module):
    def __init__(self):
        super().__init__()
        atom_feat_dim, bond_feat_dim = 9, 3
        self.ec1 = nn.Sequential(nn.Linear(bond_feat_dim, 128), nn.SiLU(), nn.Linear(128, atom_feat_dim * 128))
        self.conv1 = NNConv(atom_feat_dim, 128, self.ec1, aggr='mean')
        self.ec2 = nn.Sequential(nn.Linear(bond_feat_dim, 128), nn.SiLU(), nn.Linear(128, 128 * 128))
        self.conv2 = NNConv(128, 128, self.ec2, aggr='mean')
        self.ec3 = nn.Sequential(nn.Linear(bond_feat_dim, 128), nn.SiLU(), nn.Linear(128, 128 * 128))
        self.conv3 = NNConv(128, 128, self.ec3, aggr='mean')
        self.gp = nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.SiLU())
        self.fc = nn.Sequential(nn.Linear(128 + 32, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, d):
        x = F.silu(self.conv1(d.x, d.edge_index, d.edge_attr))
        x = F.silu(self.conv2(x, d.edge_index, d.edge_attr))
        x = F.silu(self.conv3(x, d.edge_index, d.edge_attr))
        x = global_mean_pool(x, d.batch)
        g = self.gp(d.global_feat.to(x.device)).expand(x.size(0), -1)
        return self.fc(torch.cat([x, g], 1)).squeeze()

# 9) Loss & routines
def weighted_mse(p,t):
    w = torch.ones_like(t); w[t<5]=2.5; w[t>95]=2.5
    return (w*(p-t)**2).mean()
def train_epoch(m,ldr,opt,dev):
    m.train(); tot,cnt=0.0,0
    for b in ldr:
        b=b.to(dev); opt.zero_grad(); loss=weighted_mse(m(b),b.y.view(-1)); loss.backward(); opt.step()
        tot+=loss.item()*b.num_graphs; cnt+=b.num_graphs
    return tot/cnt
def eval_epoch(m,ldr,dev):
    m.eval(); ps,ts=[],[]
    with torch.no_grad():
        for b in ldr:
            b=b.to(dev); ps.append(m(b).cpu().numpy()); ts.append(b.y.view(-1).cpu().numpy())
    return np.concatenate(ps), np.concatenate(ts)

# ==============================================================================
# [수정] 다양한 Fingerprint를 포함한 Classical features 생성
# ==============================================================================
X_desc = np.vstack([global_descriptors(s) for s in train['Canonical_Smiles']])
X_fp_morgan = np.vstack([morgan_fp(s) for s in train['Canonical_Smiles']])
X_fp_ap = np.vstack([atom_pair_fp(s) for s in train['Canonical_Smiles']])
X_fp_tt = np.vstack([topological_torsion_fp(s) for s in train['Canonical_Smiles']])
X_all  = np.hstack([X_desc, X_fp_morgan, X_fp_ap, X_fp_tt])
# ==============================================================================

# 11) OOF buffers & params
N = len(train)
oof_gnn, oof_ridge, oof_rf, oof_cb = [np.zeros(N, dtype=np.float32) for _ in range(4)]
fold_states = []
cb_params = dict(iterations=1500, depth=8, learning_rate=0.04, l2_leaf_reg=5, subsample=0.8, random_seed=seed, verbose=0, early_stopping_rounds=50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 12) Scaffold OOF loop
for fold_idx in range(5):
    print(f"\n===== FOLD {fold_idx+1}/5 =====")
    tr_idx = [i for f in range(5) if f!=fold_idx for i in folds[f]]
    va_idx = folds[fold_idx]
    # GNN
    print("  Training GNN..."); tr_graphs = []
    for i in tqdm(tr_idx, desc="  Creating train graphs"):
        smi,lab = train.loc[i,'Canonical_Smiles'], y[i]
        for aug in smiles_augment(smi,5): 
            g=mol_to_graph(aug,lab)
            if g: tr_graphs.append(g)
    va_graphs = [g for g in [mol_to_graph(train.loc[i,'Canonical_Smiles'],y[i]) for i in va_idx] if g]
    tr_ld, va_ld = DataLoader(tr_graphs, batch_size=64, shuffle=True), DataLoader(va_graphs, batch_size=128)
    model = MPNN().to(device); opt = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=5,factor=0.5); best_rmse,best_st = np.inf, None
    for ep in range(60):
        train_epoch(model,tr_ld,opt,device); p,t = eval_epoch(model,va_ld,device)
        rmse = np.sqrt(mean_squared_error(t,p)); sch.step(rmse)
        if rmse<best_rmse: best_rmse,best_st = rmse,model.state_dict()
    fold_states.append(best_st); model.load_state_dict(best_st); p,_ = eval_epoch(model,va_ld,device)
    oof_gnn[va_idx] = p[:len(va_idx)]
    # Classical
    print("  Training classical models...")
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]; y_tr, y_va = y[tr_idx], y[va_idx]
    sc = StandardScaler().fit(X_tr)
    oof_ridge[va_idx] = Ridge(alpha=5.0,random_state=seed).fit(sc.transform(X_tr),y_tr).predict(sc.transform(X_va))
    oof_rf[va_idx] = RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=3,random_state=seed,n_jobs=-1).fit(X_tr,y_tr).predict(X_va)
    cb = CatBoostRegressor(**cb_params); cb.fit(X_tr,y_tr,eval_set=(X_va,y_va),verbose=False)
    oof_cb[va_idx] = cb.predict(X_va)

# ==============================================================================
# [수정] 메타 모델 입력에 원본 특징 추가
# ==============================================================================
print("\nTraining meta-model with original features...")
meta_X = np.hstack([oof_gnn.reshape(-1,1), oof_ridge.reshape(-1,1), oof_rf.reshape(-1,1), oof_cb.reshape(-1,1), X_desc])
# ==============================================================================
meta = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.02, random_seed=seed, verbose=0, early_stopping_rounds=50)
meta.fit(meta_X, y)

# 14) 검증 지표
val_preds = meta.predict(meta_X)
A, B, LB = leaderboard_score(y, val_preds)
print("\n===== Validation (Meta on OOF) =====")
print(f"✅ Normalized RMSE (A): {A:.6f}"); print(f"✅ Pearson Corr (B):    {B:.6f}"); print(f"✅ Leaderboard Score:   {LB:.6f}")
print("----- Base models OOF -----"); print_metrics("GNN", y, oof_gnn); print_metrics("Ridge", y, oof_ridge)
print_metrics("RF", y, oof_rf); print_metrics("CatBoost", y, oof_cb)

# 15) Retrain classical on full data
sc_full = StandardScaler().fit(X_all)
ridge_f = Ridge(alpha=5.0,random_state=seed).fit(sc_full.transform(X_all),y)
rf_f    = RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=3,random_state=seed,n_jobs=-1).fit(X_all,y)
cb_f    = CatBoostRegressor(**cb_params).fit(X_all,y,verbose=False)

# 16) Test 예측
print("\nPredicting on test data...")
test_gnn = []
for smi in tqdm(test['Canonical_Smiles'],desc="  GNN TTA"):
    preds=[]
    for st in fold_states:
        m = MPNN().to(device); m.load_state_dict(st); m.eval()
        tta=[]
        for aug in smiles_augment(smi,20):
            g=mol_to_graph(aug)
            if not g: continue
            g=g.to(device); g.batch=torch.zeros(g.x.size(0),dtype=torch.long,device=device)
            with torch.no_grad(): tta.append(m(g).cpu().item())
        if tta: preds.append(np.mean(tta))
    test_gnn.append(np.mean(preds) if preds else 0.0)

# ==============================================================================
# [수정] Test 데이터에도 다양한 Fingerprint 적용 및 메타 모델 입력 구성
# ==============================================================================
X_desc_test = np.vstack([global_descriptors(s) for s in test['Canonical_Smiles']])
X_fp_morgan_test = np.vstack([morgan_fp(s) for s in test['Canonical_Smiles']])
X_fp_ap_test = np.vstack([atom_pair_fp(s) for s in test['Canonical_Smiles']])
X_fp_tt_test = np.vstack([topological_torsion_fp(s) for s in test['Canonical_Smiles']])
X_test_all  = np.hstack([X_desc_test, X_fp_morgan_test, X_fp_ap_test, X_fp_tt_test])

ridge_pred = ridge_f.predict(sc_full.transform(X_test_all))
rf_pred    = rf_f.predict(X_test_all)
cb_pred    = cb_f.predict(X_test_all)
meta_test   = np.hstack([np.array(test_gnn).reshape(-1,1), ridge_pred.reshape(-1,1), rf_pred.reshape(-1,1), cb_pred.reshape(-1,1), X_desc_test])
# ==============================================================================
final_preds = np.clip(meta.predict(meta_test),0,100)

sub_path = os.path.join(path,"submission_strat1_2_3_5.csv")
pd.DataFrame({"ID": test['ID'], "Inhibition": final_preds}).to_csv(sub_path, index=False)
print(f"\n✅ {os.path.basename(sub_path)} 저장 완료")
metrics_txt = os.path.join(path, "metrics_strat1_2_3.txt")
with open(metrics_txt, "w") as f:
    f.write("Validation (Meta on OOF)\n"); f.write(f"Normalized RMSE (A): {A:.6f}\n")
    f.write(f"Pearson Corr (B):    {B:.6f}\n"); f.write(f"Leaderboard Score:   {LB:.6f}\n")
print(f"✅ 검증 지표 저장: {os.path.basename(metrics_txt)}")