import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, BatchNorm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# CUDA 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"CUDA 사용 가능 여부: {USE_CUDA}")
print(f"현재 사용 중인 디바이스: {DEVICE}")


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# ================== 하이퍼파라미터 ==================
class Config:
    DATA_PATH = basepath + '_data/dacon/drugs/'
    N_SPLITS = 5
    RANDOM_STATE = 42
    EPOCHS = 300
    PATIENCE = 30
    BATCH_SIZE = 128
    LR = 0.0005
    AUGMENTATION_FACTOR = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# ================== 데이터 로딩 ==================
train_df = pd.read_csv(os.path.join(Config.DATA_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(Config.DATA_PATH, 'test.csv'))
submission = pd.read_csv(os.path.join(Config.DATA_PATH, 'sample_submission.csv'))

train_df['Inhibition_log'] = np.log1p(train_df['Inhibition'])
y_mean = train_df['Inhibition_log'].mean()
y_std = train_df['Inhibition_log'].std()
train_df['target'] = (train_df['Inhibition_log'] - y_mean) / y_std

# ================== SMILES to Graph ==================
def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]

def atom_features(atom):
    return one_hot_encoding(atom.GetAtomicNum(), [5,6,7,8,9,15,16,17,35,53,999]) + \
           one_hot_encoding(atom.GetDegree(), [0,1,2,3,4,5]) + \
           one_hot_encoding(atom.GetTotalNumHs(), [0,1,2,3,4]) + \
           one_hot_encoding(atom.GetImplicitValence(), [0,1,2,3,4,5]) + \
           [atom.GetIsAromatic()]

def bond_features(bond):
    bt = bond.GetBondType()
    return [int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE),
            int(bt == Chem.rdchem.BondType.AROMATIC),
            int(bond.IsInRing())]

def smiles_to_data(smiles, target=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edges.extend([[i, j], [j, i]])
        edge_attrs.extend([feat, feat])
    x = torch.tensor(atoms, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data

sample_mol = Chem.MolFromSmiles('CN(C)C(=O)c1ccccc1')
IN_CHANNELS = len(atom_features(sample_mol.GetAtomWithIdx(0)))
EDGE_CHANNELS = len(bond_features(sample_mol.GetBondWithIdx(0)))

# ================== 모델 정의 ==================
class OptimizedMPNN(nn.Module):
    def __init__(self, in_channels, edge_channels, hidden_channels=256, out_channels=1, dropout=0.3):
        super().__init__()
        self.conv1 = NNConv(in_channels, hidden_channels,
                            nn.Sequential(nn.Linear(edge_channels, 128), nn.ReLU(),
                                          nn.Linear(128, in_channels * hidden_channels)), aggr='mean')
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = NNConv(hidden_channels, hidden_channels,
                            nn.Sequential(nn.Linear(edge_channels, 256), nn.ReLU(),
                                          nn.Linear(256, hidden_channels * hidden_channels)), aggr='mean')
        self.bn2 = BatchNorm(hidden_channels)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

# ================== 학습/평가 함수 ==================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend(out.view(-1).cpu().numpy())
            if batch.y is not None:
                targets.extend(batch.y.view(-1).cpu().numpy())
    if len(targets) == 0:
        return np.array(preds)
    else:
        return np.array(preds), np.array(targets)

# ================== K-Fold ==================
kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
oof_preds = np.zeros(len(train_df))
all_test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"\n== Fold {fold+1} ==")
    train_graphs = []
    for i in tqdm(train_idx):
        smi = train_df.iloc[i]['Canonical_Smiles']
        y = train_df.iloc[i]['target']
        g = smiles_to_data(smi, y)
        if g: train_graphs.append(g)
        for _ in range(Config.AUGMENTATION_FACTOR):
            try:
                mol = Chem.MolFromSmiles(smi)
                r_smi = Chem.MolToSmiles(mol, doRandom=True)
                g_aug = smiles_to_data(r_smi, y)
                if g_aug: train_graphs.append(g_aug)
            except: continue
    val_graphs = [smiles_to_data(train_df.iloc[i]['Canonical_Smiles'], train_df.iloc[i]['target']) for i in val_idx]
    val_graphs = [g for g in val_graphs if g is not None]

    train_loader = DataLoader(train_graphs, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=Config.BATCH_SIZE)

    model = OptimizedMPNN(IN_CHANNELS, EDGE_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_rmse = float('inf')
    patience = 0

    for epoch in range(1, Config.EPOCHS+1):
        loss = train_epoch(model, train_loader, optimizer)
        val_preds, val_targets = evaluate(model, val_loader)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        scheduler.step()

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience = 0
            best_val_preds = val_preds
            torch.save(model.state_dict(), f'model_fold{fold+1}.pt')
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                break

    oof_preds[val_idx] = best_val_preds

    test_graphs = [smiles_to_data(s) for s in test_df['Canonical_Smiles']]
    test_graphs = [g for g in test_graphs if g is not None]
    test_loader = DataLoader(test_graphs, batch_size=Config.BATCH_SIZE)

    model.load_state_dict(torch.load(f'model_fold{fold+1}.pt'))
    fold_test_preds = evaluate(model, test_loader)
    all_test_preds += fold_test_preds / Config.N_SPLITS

# ================== 예측값 복원 ==================
oof_preds_inv = np.expm1(oof_preds * y_std + y_mean)
test_preds_inv = np.expm1(all_test_preds * y_std + y_mean)
oof_preds_inv[oof_preds_inv < 0] = 0
test_preds_inv[test_preds_inv < 0] = 0

# ================== 평가 및 저장 ==================
rmse = np.sqrt(mean_squared_error(train_df['Inhibition'], oof_preds_inv))
norm_rmse = rmse / (train_df['Inhibition'].max() - train_df['Inhibition'].min())
pearson = np.corrcoef(train_df['Inhibition'], oof_preds_inv)[0, 1]
final_score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * pearson

print(f"\nRMSE: {rmse:.4f}")
print(f"Normalized RMSE: {norm_rmse:.4f}")
print(f"Pearson Correlation: {pearson:.4f}")
print(f"Final Score: {final_score:.4f}")

submission['Inhibition'] = test_preds_inv
submission.to_csv('submission_optimized_mpnn.csv', index=False)
print("제출 저장 완료: submission_optimized_mpnn.csv")
