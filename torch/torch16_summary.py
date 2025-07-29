#실습!!!!
#드랍아웃 적용해보아요

from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, r2_score
import torchvision.transforms as tr


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)


SEED = 42
import random
random.seed(SEED) #python random fix
np.random.seed(SEED) #numpy random fix
torch.manual_seed(SEED) #torch random fix
torch.cuda.manual_seed(SEED) #torch cuda random fix


import os

if os.path.exists('/workspace/TensorJae/Study25/'):   # GPU 서버인 경우
    BASE_PATH = '/workspace/TensorJae/Study25/'
else:                                                 # 로컬인 경우
    BASE_PATH = os.path.expanduser('~/Desktop/IBM:RedHat/Study25/')
    
basepath = os.path.join(BASE_PATH)

path = basepath + '_data/torch/'



transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5),(0.5))]) #크기 늘리고 to tensor
##########################tr.Normalize((0.5),(0.5))##########################
# Z-score normalization(정규화의 표준화???)
#(x-평균)/표준편차
#(x-0.5)/0.5 위 식처럼 해야하는데 통상 평균 0.5, 표편 0.5로 계산하면
#-1~1 사이의 범위가 나오니 이미지 전처리에서는 통상 0.5 0.5 한다.
#############################################################################
# #to.Tensor = 토치텐서바꾸기 + minmaxScaler


train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)
print(len(train_dataset))   #60000
# print(train_dataset[0][0])  #
print(train_dataset[0][1])  #5

img_tensor, label = train_dataset[0]
print(label)        #5
print(img_tensor.shape)     #torch.Size([1, 56, 56]) torch 데이터는 컬러 가로 세로
print(img_tensor.min(), img_tensor.max())   #tensor(0.) tensor(0.9922) 전처리가 돼있다


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(len(train_loader))    #1875
# exit()

class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        # super(self, DNN).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride= 1, ),    #(1,56,56) -> (64, 54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),   #(n, 64, 27,27)
            nn.Dropout(0.2)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), #(n,32,25,25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),        #(n,32,12,12)
            nn.Dropout(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1),     #(n,16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),    #(n,16,5,5)
            nn.Dropout(0.2)
        )
        
        self.flatten = nn.Flatten()
        
        self.hidden_layer4 = nn.Sequential(         #플래튼에서 받아라
            nn.Linear(16*5*5, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32,10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        # x = x.view(x.shape[0], -1)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        
        return x


model = CNN(1).to(DEVICE)   #torch에서는 channel 만 Input으로 넣어줌, 나머지는 알아서 맞춰줌

print(model)



# CNN(
#   (hidden_layer1): Sequential(
#     (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Dropout(p=0.2, inplace=False)
#   )
#   (hidden_layer2): Sequential(
#     (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#     (3): Dropout(p=0.2, inplace=False)
#   )
#   (hidden_layer3): Sequential(
#     (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#     (3): Dropout(p=0.2, inplace=False)
#   )
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (hidden_layer4): Sequential(
#     (0): Linear(in_features=400, out_features=64, bias=True)
#     (1): ReLU()
#   )
#   (hidden_layer5): Sequential(
#     (0): Linear(in_features=64, out_features=32, bias=True)
#     (1): ReLU()
#   )
#   (output_layer): Linear(in_features=32, out_features=10, bias=True)
#   (dropout): Dropout(p=0.2, inplace=False)
# )

# from torchsummary import summary 
# summary(model, [1,56,56])

from torchinfo import summary
summary(model, [1,56,56])




exit()























criterion= nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4)

def train(model, criterion, optimizer, loader):
    #model.train()
    epoch_loss = 0
    epoch_acc = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        
        loss= criterion(hypothesis, y_batch)
        
        
        loss.backward()
        optimizer.step() # w = x -lr*로스를 웨이트로 미분한값 
        
        y_predict = torch.argmax(hypothesis,1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss/len(loader), epoch_acc / len(loader)



def evaluate(model, criterion, loader):
    model.eval()
    #model.train()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss= criterion(hypothesis, y_batch)
            
            
            y_predict = torch.argmax(hypothesis,1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc

        return epoch_loss/len(loader), epoch_acc / len(loader)



epochs = 10

for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch : {epoch}, loss : {loss:.4f}, acc : {acc:.3f}, \
          val_loss:{val_loss:.4f}, val_acc:{val_acc:.3f}')
    
loss, acc = evaluate(model, criterion, test_loader)
print('======================================')


print('최종 loss:', loss)
print('최종 acc:', acc.item())

# 최종 loss: 0.171147753732999
# 최종 acc: 0.9480830430984497