# import pandas as pd
# import numpy as np
# import torch 
# import torch.nn as nn
# import torch.optim as optim
# import random
# import tqdm

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')
    
# print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# #1 data
# path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/kaggle/netflix/'

# train_csv = pd.read_csv(path + 'train.csv')     #[967 rows x 6 columns] (n,)
# test_csv = pd.read_csv(path + 'test.csv')
# submission_csv = pd.read_csv(path + 'sample_submission.csv')


# print(train_csv)
# print(train_csv.info())
# print(train_csv.describe())

# # import matplotlib.pyplot as plt

# # data = train_csv.iloc[:,1:4]

# # data['종가'] = train_csv['Close']
# # print(data)


# # hist = data.hist()
# # plt.show()


# from torch.utils.data.dataset import Dataset, TensorDataset
# from torch.utils.data import DataLoader

# class Custom_Dataset(Dataset):
#     def __init__(self, df, timesteps = 40):
#         super().__init__()
#         self.train_csv = train_csv
#         self.x = self.train_csv.iloc[:,1:4].values
#         self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0)- np.min(self.x, axis=0))   # minmaxScaler

#         self.y = self.train_csv['Close'].values
#         self.timesteps = timesteps

#     # (10,1) -> (8,3,1) 전체-timestep +1
#     #(967,3) -> (n,30,3)
#     def __len__(self):
#         return len(self.x) - self.timesteps    #행 - Timesteps
    
#     def __getitem__(self, idx):
#         x = self.x[idx : idx+ self.timesteps]         # x[idx: idx + 타임스탭스]
#         y = self.y[idx + self.timesteps]        # y[idx + 타임스탭스]
#         return x,y 


# custom_dataset = Custom_Dataset(df = train_csv, timesteps=30)

# train_loader = DataLoader(custom_dataset, batch_size=32)


# for batch_idx, (xb, yb) in enumerate(train_loader):
#     print('========배치', batch_idx)
#     print('x:', xb.shape)       #x: torch.Size([32, 30, 3])
#     print('y:', yb.shape)       #y: torch.Size([32, 30])


# # exit()


# #2 model 
# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(input_size = 3,
#                           hidden_size=64,
#                           num_layers=3,
#                           batch_first=True) # (n, 30, 64)
        
#         self.fc1 = nn.Linear(in_features=30*64, out_features=32)
#         self.fc2 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, _ = self.rnn(x)

#         # x = torch.reshape(x, x.shape[0], )
#         # x = torch.reshape(-1, 30*64)
#         x = x[:, -1, :]
#         x = self.fc1(x)
#         self.relu(x)
#         x = self.fc2(x)
#         return x
    
# model = RNN().to(DEVICE)


# from torch.optim import Adam
# optim = Adam(params=model.parameters(), lr = 0.001)
# from tqdm import tqdm 

# for epoch in range(1,201):
#     iterator = tqdm(train_loader)
#     for x,y in iterator:
#         optim.zero_grad()
#         hypothesis = model(x.type(torch.FloatTensor).to(DEVICE))
#         loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))
#         loss.backward()
#         optim.step()
#         # iterator.set_description()

# === Imports ===
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # 이 이름을 그대로 사용 (아래서 optimizer 변수로 구분)
import random
from tqdm import tqdm  # ← tqdm는 이렇게 가져와야 바로 tqdm(...) 사용 가능

from torch.utils.data import Dataset, DataLoader

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

train_csv = pd.read_csv(path + 'train.csv')      # [967 x 6] 가정
test_csv = pd.read_csv(path + 'test.csv')
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.head())
print(train_csv.info())
print(train_csv.describe(include='all'))

# === 2) Dataset ===
class Custom_Dataset(Dataset):
    """
    슬라이딩 윈도우로 (timesteps, features=3) → 다음 시점 Close(스칼라)를 예측
    """
    def __init__(self, df: pd.DataFrame, timesteps: int = 30):
        super().__init__()
        # 인자로 전달된 df 사용 (이전 코드에선 self.train_csv 고정 → 수정)
        X = df.iloc[:, 1:4].values.astype(np.float32)  # (N, 3)
        # 간단 MinMax (열별). 분모 0 대비 epsilon 추가
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

        y = df['Close'].values.astype(np.float32)      # (N,)

        self.X = X
        self.y = y
        self.timesteps = timesteps

    def __len__(self):
        # 마지막 타깃을 만들 수 없는 timesteps만큼 제외
        return len(self.X) - self.timesteps

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.timesteps]     # (timesteps, 3)
        y = self.y[idx + self.timesteps]           # scalar
        # Tensor로 변환
        x = torch.from_numpy(x)                    # float32 (timesteps, 3)
        y = torch.tensor(y, dtype=torch.float32)   # float32  ()
        return x, y

# timesteps=30 세팅
custom_dataset = Custom_Dataset(df=train_csv, timesteps=30)

# 학습용 DataLoader: 셔플 권장, 마지막 배치 크기 고정하려면 drop_last=True
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True, drop_last=True)

# 배치 모양 확인
for batch_idx, (xb, yb) in enumerate(train_loader):
    print('======== 배치', batch_idx)
    print('x:', xb.shape)  # torch.Size([32, 30, 3])
    print('y:', yb.shape)  # torch.Size([32])
    break  # 한 배치만 확인하고 종료

# === 3) Model ===
class RNN(nn.Module):
    """
    입력: (B, T=30, F=3)
    RNN 출력: (B, T, H=64)
    여기서는 '마지막 시점 출력'만 사용 → (B, 64) → MLP → 스칼라
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=3, timesteps=30):
        super().__init__()
        self.timesteps = timesteps
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # 마지막 시점만 사용하므로 in_features=hidden_size
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, F)
        x, _ = self.rnn(x)          # (B, T, H)
        x = x[:, -1, :]             # (B, H) 마지막 시점
        x = self.fc1(x)             # (B, 32)
        x = self.relu(x)            # ← 반환값 다시 받기 (이전 코드 bug fix)
        x = self.fc2(x)             # (B, 1)
        return x.squeeze(1)         # (B,)

model = RNN(input_size=3, hidden_size=64, num_layers=3, timesteps=30).to(DEVICE)


save_path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_save/torch/'
# torch.save(model.state_dict(), save_path+ 't25_netflix.pth')

#4 

"""

# === 4) Train ===


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50  # 필요에 따라 조절
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for xb, yb in pbar:
        xb = xb.to(DEVICE)                              # (B, 30, 3)
        yb = yb.to(DEVICE)                              # (B,)

        optimizer.zero_grad()
        preds = model(xb)                               # (B,)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    epoch_loss = running_loss / (len(train_loader.dataset) // 32 * 32)  # drop_last=True니까 배치*32로 근사
    print(f"[Epoch {epoch}] mean loss: {epoch_loss:.6f}")

save_path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_save/torch/'
torch.save(model.state_dict(), save_path+ 't25_netflix.pth')
# from torchinfo import summary
# summary(model.cpu(), (2,3,1))
# # from torchsummary import summary
# # summary(model.cpu(), (3, 1))   # 일시적으로 cpu에서 summary 실행
# model.to(DEVICE)               # 다시 원래 디바이스로 복귀
"""