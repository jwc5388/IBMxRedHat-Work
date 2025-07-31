# === Imports ===
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import os

# === Seed & Device ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')   # Apple Silicon
else:
    DEVICE = torch.device('cpu')

print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# === 1) Data ===
path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/kaggle/netflix/'
save_path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_save/torch/'

train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# === 2) Dataset ===
class Custom_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, timesteps: int = 30):
        super().__init__()
        X = df.iloc[:, 1:4].values.astype(np.float32)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        y = df['Close'].values.astype(np.float32)

        self.X = X
        self.y = y
        self.timesteps = timesteps

    def __len__(self):
        return len(self.X) - self.timesteps

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.timesteps]
        y = self.y[idx + self.timesteps]
        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# timesteps 설정
timesteps = 30
custom_dataset = Custom_Dataset(df=train_csv, timesteps=timesteps)
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True, drop_last=True)

# === 3) 모델 정의 ===
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=3, timesteps=30):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn(x)        # (B, T, H)
        x = x[:, -1, :]           # 마지막 시점 출력 (B, H)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)       # (B,)

# 모델 초기화
model = RNN(input_size=3, hidden_size=64, num_layers=3, timesteps=timesteps).to(DEVICE)

# === 4) 저장된 모델 로드 및 평가 ===
model.load_state_dict(torch.load(save_path + 't25_netflix.pth', map_location=DEVICE))
model.eval()

criterion = nn.MSELoss()
y_predict = []
y_true = []
total_loss = 0.0

with torch.no_grad():
    for x_test, y_test in train_loader:
        x_test = x_test.to(DEVICE).float()
        y_test = y_test.to(DEVICE).float()

        y_pred = model(x_test)
        loss = criterion(y_pred, y_test)
        total_loss += loss.item()

        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())

# === 5) 결과 정리 ===
y_predict = np.concatenate(y_predict).flatten()
y_true = np.concatenate(y_true).flatten()

r2 = r2_score(y_true, y_predict)

print('✅ 평균 손실(MSE):', total_loss / len(train_loader))
print('✅ R² 점수:', r2)