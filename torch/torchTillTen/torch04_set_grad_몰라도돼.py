from re import DEBUG
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

import torch
print(torch.backends.mps.is_available())      # 하드웨어적으로 가능 여부
print(torch.backends.mps.is_built())  

import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

#1 data
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]).transpose()
y = np.array([1,2,3,4,5,6,7,7,9,10])


x = torch.FloatTensor(x).to(DEVICE)   #차원을 늘려준다 

y= y.reshape(-1,1)
y = torch.FloatTensor(y).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())   #torch.Size([10, 2]) torch.Size([10, 1])


x_mean = torch.mean(x)
x_std = torch.std(x)

##################### standard scaling ########################
x = (x -torch.mean(x)) / torch.std(x)
###############################################################
print('스케일링 후:', x)

# exit()

# model = Sequential()
# model.add(Dense(1, input_dim = 1)) #아웃풋, 인풋
# model = nn.Linear(1,1).to(DEVICE)  #인풋 아웃풋 y = wx + b
model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
    
).to(DEVICE)

#3 compile, train
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr = 0.01)    #정사하강법이라는 옵티마이저

print(model.parameters)
exit()

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드, 드랍아웃 배치노멀라이제이션 적용
    optimizer.zero_grad()   #기울기 초기화. 
                            #각 배치마다 기울기를 초기화(0으로) 하여, 기울기 누적에 의한 문제 해결
    hypothesis = model(x)   #y = xw + b 
    loss = criterion(hypothesis, y) #loss = mse() = 시그마(y-hypothesis)^2/n
    
    loss.backward() #기울기(gradient) 값까지만 계산.
    optimizer.step() #가중치 갱신
    return loss.item() 
    
    
epochs = 5000
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {} '.format(epoch, loss))
    
    
print('==========================================================')
#4 평가, 예측

# model.evaluate(x,y)
def evaluate(model, criterion, x, y):
    model.eval()        #[평가모드] 드랍아웃, 배치노말을 쓰지 않겠다.
    # with torch.no_grad():     #gradient 갱신을 하지 않겠다
    #     y_predict = model(x)
    #     loss2 = criterion(y, y_predict) #loss 의 최종값
    
    torch.set_grad_enabled(False)
    y_predict = model(x)
    loss2 = criterion(y, y_predict) #loss 의 최종값
    torch.set_grad_enabled(True)
    
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 Loss: ', loss2)



#10, 1.3의 예측값
x_pred = (torch.Tensor([[10, 1.3]]).to(DEVICE) - x_mean)/ x_std
result = model(x_pred)
print('10, 1.3 의 예측값:', result)
print('10, 1.3 의 예측값:', result.item())