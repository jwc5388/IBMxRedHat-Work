#11-2 copy


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
import warnings
##########################랜덤 고정###########################

# warnings.filterwarnings('ignore')

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

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


print(x.shape)
print(y.shape)

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.9, shuffle=True, random_state=SEED,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
# x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# y_train = torch.tensor(y_train,dtype=torch.float).to(DEVICE)
# y_test = torch.tensor(y_test,dtype=torch.float).to(DEVICE)

print(x_train.dtype)
print(x_train.shape, y_train.shape)
print(type(x_train))



######### torch dataset 만들기
from torch.utils.data import TensorDataset # xy 합치기
from torch.utils.data import DataLoader    # batch 정의!!!



##### 1. x, y 합쳐
train_set = TensorDataset(x_train, y_train) # tuple 형태
test_set = TensorDataset(x_test, y_test)
print(train_set)        #<torch.utils.data.dataset.TensorDataset object at 0x17a962cb0>
print(type(train_set))  #<class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set))   #455
print(train_set[0])
# print(train_set[1])

print(train_set[0][0])  #첫번째 x
print(train_set[0][1])  #첫번째 y
# (tensor([-1.4408, -0.4353, -1.3621, -1.1391,  0.7806,  0.7189,  2.8231, -0.1191,
#          1.0927,  2.4582, -0.2638, -0.0161, -0.4704, -0.4748,  0.8384,  3.2510,
#          8.4389,  3.3920,  2.6212,  2.0612, -1.2329, -0.4763, -1.2479, -0.9740,
#          0.7229,  1.1867,  4.6728,  0.9320,  2.0972,  1.8865], device='mps:0'), tensor([1.], device='mps:0')
# exit()


#######2 batch를 정의한다
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
print(len(train_loader))    #52
print(train_loader)         #<torch.utils.data.dataloader.DataLoader object at 0x309e25240>
# print(train_loader[0][0])# error

# print(train_loader[0])#/error



# for x_batch, y_batch in train_loader:
#     print(x_batch)
#     print(y_batch)
#     break

# exit()

print('===============================================')

########iterator 데이터 확인하기#########
#1 for 문으로 확인
# for aaa in train_loader:
#     print(aaa)
#     break       #첫번쨰 배치 출력
#2 next() 사용
# bbb = iter(train_loader)
# bbb._next()

# aaa = next(bbb)
# print(aaa)
# exit()




# torch.float32
# torch.Size([398, 30]) torch.Size([398])
# <class 'torch.Tensor'>


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
        self.sig = nn.Sigmoid()
        
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
        x = self.sig(x)
        return x
    

model = Model(30 ,1).to(DEVICE)
        

#3 compile, train
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.008)


def train(model, criterion, optimizer, loader):
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss= criterion(hypothesis,y_batch)
        
        loss.backward()
        optimizer.step()
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()
    return total_loss/ len(loader)

epochs = 500

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs:{}, loss:{}'.format(epoch, loss)) #verbose
    
    
# exit()
print('==================================')

def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader: 
        with torch.no_grad():
            y_pred = model(x_batch)
            loss2 = criterion(y_pred, y_batch)
            total_loss += loss2.item()
        
    return total_loss/len(loader) 

last_loss = evaluate(model, criterion, test_loader)

print('최종 Loss:', last_loss)

y_predict = model(x_test)
print(type(y_predict))  #<class 'torch.Tensor'>



y_pred = np.round(y_predict.detach().cpu().numpy())
y_true = y_test.detach().cpu().numpy()

accuracy = accuracy_score(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
# acc = accuracy_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy().round())

print('accuracy score:', accuracy)
# print('r2 score:', r2)

# 최종 Loss: 2.2243926525115967
# <class 'torch.Tensor'>
# accuracy score: 0.9122807017543859