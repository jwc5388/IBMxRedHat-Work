import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)



import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

#1 data
path = basepath +  '_data/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col= 0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())


x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=337, stratify=y
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
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
).to(DEVICE)


#3 compile, train
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)


def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss= criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 200

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
#다중분류는 argmax



acc = accuracy_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy().round())
print('acc:', acc)


# 최종 Loss: 25.896615982055664
# <class 'torch.Tensor'>
# acc: 0.732824427480916