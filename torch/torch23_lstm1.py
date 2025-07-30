#7-6 copy


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import random

##########################랜덤 고정###########################

SEED = 337
import random
random.seed(SEED) #python random fix
np.random.seed(SEED) #numpy random fix
torch.manual_seed(SEED) #torch random fix
torch.cuda.manual_seed(SEED) #torch cuda random fix

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')
    
# print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
 
x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],])        # (7, 3)
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)


x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  #(7, 3, 1)
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

print(x.shape, y.size()) #torch.Size([7, 3, 1]) torch.Size([7])

from torch.utils.data import TensorDataset, DataLoader

train_set = TensorDataset(x,y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

aaa = iter(train_loader)
bbb = next(aaa)

print(bbb)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layer1 = nn.LSTM(
            input_size=1,   #feature 개수. 텐서플로에서는 input_dim
            hidden_size=32, #output_node 의 갯수, 텐서플로에서는 unit
            num_layers=1,   # default , RNN 은닉층의 레이어의 갯수
            batch_first=True,   #default False 
            #원래 이건데 (N,3,1) False 옵션을 주면 (3,N,1)
            #그래서 다시 True 주면 원위치된다.  머리쓰기 귀찮으니까 그냥 이 옵션 반드시 넣는다.
            #(N,,3,32)  
            )
        # self.rnn_layer1 = nn.RNN(1, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    # def forward(self, x, h0=None):
    #     if h0 == None:
    #         h0 = torch.zeros(1,x.size(0), 32).to(DEVICE)
    #         #(num_layers, batch_size, hidden_size)

    #     x, hidden_state = self.rnn_layer1(x, h0)

    def forward(self, x):
        # (batch, seq_len, input_size)
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        c0 = torch.zeros(1, x.size(0), 32).to(x.device)
        x, _ = self.lstm_layer1(x, (h0, c0))  # output: (batch, seq_len, hidden_size)
        x = self.relu(x)
        x = x[:, -1, :]  # 마지막 시점의 hidden state
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    # def forward(self, x, h0, c0):
    #     # # x, _ = self.lstm_layer1(x)
    #     if h0 is None:
    #         h0 = torch.zeros(1, x.size(0), 32).to(DEVICE)
    #     if c0 is None:
    #         c0 = torch.zeros(1, x.size(0), 32).to(DEVICE)
    #     # x, (hn,cn) = self.rnn_layer1(x, (h0, c0))

    #     x, _ = self.rnn_layer1(x, h0)
    #     x = self.relu(x)
        
    #     # x = x.reshape(-1, 3*32)
    #     x = x[:, -1,:]          #가장 마지막 시점의 출력만

    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.relu(x)
    #     x = self.fc3(x)
    #     return x
    
model = LSTM().to(DEVICE)

from torchinfo import summary
summary(model.cpu(),( 2,3,1))
# from torchsummary import summary
# summary(model.cpu(), (3, 1))   # 일시적으로 cpu에서 summary 실행
model.to(DEVICE)               # 다시 원래 디바이스로 복귀


# exit()
# 3. 손실함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 훈련 함수 정의
def train(model, loader):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        output = model(xb).squeeze()  # (batch, 1) → (batch)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# 5. 평가 함수 정의
def evaluate(model, loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# 6. 학습 루프 실행
EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader)
    val_loss = evaluate(model, train_loader)  # 평가용 test_loader 없으므로 train 기준
    if epoch % 20 == 0 or epoch == 1:
        print(f"[{epoch}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

# 7. 최종 예측 결과 확인
model.eval()
with torch.no_grad():
    pred = model(x).squeeze().cpu().numpy()
    target = y.cpu().numpy()
    print("\n[예측 결과]")
    print("예측값:", np.round(pred, 2))
    print("정답값:", target)