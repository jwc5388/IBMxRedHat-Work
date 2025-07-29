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

SEED = 42
import random
random.seed(SEED) #python random fix
np.random.seed(SEED) #numpy random fix
torch.manual_seed(SEED) #torch random fix
torch.cuda.manual_seed(SEED) #torch cuda random fix

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

dataset = fetch_california_housing()
x = dataset.data
y = dataset.target


print(x.shape)
print(y.shape)

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=SEED,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train = torch.tensor(x_train, dtype=torch.float).to(DEVICE)
# x_test = torch.tensor(x_test,dtype=torch.float).to(DEVICE)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# y_train = torch.tensor(y_train,dtype=torch.float).to(DEVICE)
# y_test = torch.tensor(y_test,dtype=torch.float).to(DEVICE)

print(x_train.dtype)
print(x_train.shape, y_train.shape)
print(type(x_train))

# torch.float32
# torch.Size([398, 30]) torch.Size([398])
# <class 'torch.Tensor'>


#2 model 
# model = nn.Sequential(
#     nn.Linear(8, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.SiLU(),
#     nn.Linear(16,1)
# ).to(DEVICE)



class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # or super().__init__()
        super(Model, self).__init__()   #nn.Module 에 있는 Model 과 self 다 쓰겠다.
        ### 모델에 대한 정의부분을 구현 ###
        self.linear1 = nn.Linear(input_dim, 64) 
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):           #정의 구현 사제단!
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x
    

model = Model(8 ,1).to(DEVICE)  #여기서 init 함수 실행 
        

#3 compile, train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.008)


def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss= criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1500

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs:{}, loss:{}'.format(epoch, loss)) #verbose

print('==================================')

def evaluate(model, criterion, x,y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
        
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)

print('최종 Loss:', last_loss)

y_predict = model(x_test)
print(type(y_predict))  #<class 'torch.Tensor'>


#이진분류는 .round
#ㄷ다중분류는 argmax

y_pred = y_predict.detach().cpu().numpy()
y_true = y_test.detach().cpu().numpy()


r2 = r2_score(y_true, y_pred)
# acc = accuracy_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy().round())

print('r2 score:', r2)



# 최종 Loss: 0.2719866633415222
# <class 'torch.Tensor'>
# r2 score: 0.7924413681030273