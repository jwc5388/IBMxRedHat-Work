import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

# 1. 데이터 로드
path = basepath + '_data/dacon/drugs/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

# ======================= 1. Mordred + Morgan Feature ==========================
calc = Calculator(descriptors, ignore_3D=True)

def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mordred_desc = list(calc(mol).values())
        morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        return mordred_desc + morgan_fp
    except:
        return None


train['features'] = train['Canonical_Smiles'].apply(get_features)
test['features'] = test['Canonical_Smiles'].apply(get_features)

train = train[train['features'].notnull()].reset_index(drop=True)
test = test[test['features'].notnull()].reset_index(drop=True)

x_mordred = np.array(train['features'].tolist())
y = train['Inhibition'].values
x_test_mordred = np.array(test['features'].tolist())

# ========================== 2. PCA ============================
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_mordred)
x_test_scaled = scaler.transform(x_test_mordred)

pca = PCA(n_components=300)
x_pca = pca.fit_transform(x_scaled)
x_test_pca = pca.transform(x_test_scaled)

# ========================== 3. GNN =============================
def smiles_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y_tensor = torch.tensor([label], dtype=torch.float) if label is not None else None
    return Data(x=x, edge_index=edge_index, y=y_tensor)

train_graphs = [smiles_to_graph(s, l) for s, l in zip(train['Canonical_Smiles'], y)]
train_graphs = [g for g in train_graphs if g is not None]

test_graphs = [smiles_to_graph(s) for s in test['Canonical_Smiles']]
test_graphs = [g for g in test_graphs if g is not None]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = torch.nn.Linear(128, 64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
gnn_model = GCN().to(device)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

gnn_model.train()
for epoch in range(5):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gnn_model(batch)
        loss = loss_fn(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()

# ====== 4. GNN Embedding 추출 (train + test) ======
gnn_model.eval()

def extract_embeddings(loader):
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = gnn_model(batch)
            embeddings.append(out.cpu().numpy())
    return np.vstack(embeddings)

x_gnn = extract_embeddings(train_loader)
x_test_gnn = extract_embeddings(test_loader)

# ====== 5. Concat: [PCA features | GNN features] ======
x_final = np.concatenate([x_pca, x_gnn], axis=1)
x_test_final = np.concatenate([x_test_pca, x_test_gnn], axis=1)

# ====== 6. Regressor (e.g., GradientBoosting) ======
x_train_final, x_val, y_train_final, y_val = train_test_split(x_final, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(x_train_final, y_train_final)

y_pred = model.predict(x_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"✅ Final RMSE: {rmse:.4f}")

# ====== 7. Inference & Save ======
y_test_pred = model.predict(x_test_final)

# submission = pd.read_csv('sample_submission.csv')
submission['Inhibition'] = y_test_pred
submission.to_csv('gnn_pca_submission.csv', index=False)
print("📁 Saved: gnn_pca_submission.csv")