import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 시드 고정
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.expanduser(f'~/_save/jena/{timestamp}/')
os.makedirs(SAVE_DIR, exist_ok=True)
DATA_PATH = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/jena/jena_climate_2009_2016.csv'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model.pth')

# --- 데이터 로딩 및 파싱 ---
df = pd.read_csv(DATA_PATH)
df['Date Time'] = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S")
start_pred = pd.to_datetime("31.12.2016 00:10:00", format="%d.%m.%Y %H:%M:%S")
df_train_val = df[df['Date Time'] < start_pred].copy()

# --- 파생 특성 추가 ---
def add_time_features(df):
    df['hour'] = df['Date Time'].dt.hour
    df['dayofyear'] = df['Date Time'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

df_train_val = add_time_features(df_train_val)

# --- 풍향 deg → sin, cos ---
def deg_to_sin_cos(deg):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

target_column = 'wd (deg)'
df_train_val['wd_sin'], df_train_val['wd_cos'] = deg_to_sin_cos(df_train_val[target_column])

# --- 특성 컬럼 정리 및 정규화 ---
feature_columns = [col for col in df_train_val.columns if col not in ['Date Time', 'wd (deg)', 'hour', 'dayofyear']]
scaler = StandardScaler().fit(df_train_val[feature_columns])
df_train_val[feature_columns] = scaler.transform(df_train_val[feature_columns])

# --- Dataset 정의 ---
TIMESTEPS = 144

class JenaRNN_Dataset(Dataset):
    def __init__(self, df, timesteps):
        self.x = []
        self.y = []
        data = df[feature_columns].values
        y1 = df['wd_sin'].values
        y2 = df['wd_cos'].values
        for i in range(len(data) - timesteps):
            self.x.append(data[i:i+timesteps])
            self.y.append([y1[i+timesteps], y2[i+timesteps]])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = JenaRNN_Dataset(df_train_val, TIMESTEPS)
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# --- 모델 정의 ---
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return torch.tanh(self.fc(out))

input_size = len(feature_columns)
model = RNNModel(input_size=input_size).to(DEVICE)

# --- 손실함수 및 옵티마이저 ---
def angular_loss(y_true, y_pred):
    y_true_angle = torch.atan2(y_true[:, 0], y_true[:, 1])
    y_pred_angle = torch.atan2(y_pred[:, 0], y_pred[:, 1])
    return torch.mean(1 - torch.cos(y_true_angle - y_pred_angle))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 학습 루프 ---
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = angular_loss(yb, preds)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = angular_loss(yb, preds)
            val_loss += loss.item() * xb.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# --- 모델 저장 ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ 모델 저장 완료: {MODEL_SAVE_PATH}")