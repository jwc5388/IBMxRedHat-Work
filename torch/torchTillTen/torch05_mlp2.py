import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#2.4.1 - 12.4



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

#1 data
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], [9,8,7,6,5,4,3,2,1,0]]).T #(2,5) again here, gave (10,3) data has to be (3,10)
y = np.array([1,2,3,4,5,6,7,8,9,10])

# transpose switches row and column
print(x.shape, y.shape) #(10, 3) (10,)
# x = torch.FloatTensor(x).to(DEVICE)
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

# x = torch.tensor(x, dtype = float).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

# y = torch.tensor(y).unsqueeze(1).to(DEVICE)

x_mean = torch.mean(x)
x_std = torch.std(x)

x = (x - torch.mean(x))/ torch.std(x)

model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,5),
    nn.Linear(5,5),
    nn.Linear(5,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 3000
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {} '.format(epoch, loss))

def evaulate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaulate(model, criterion, x,y)

print('최종 loss:', loss2)

x_pred = (torch.Tensor([[11,2.0,-1]]).to(DEVICE) - x_mean)/ x_std
result = model(x_pred)

print('11, 2.0, -1 의 예측값:', result)
print('11, 2.0, -1 의 예측값:', result.item())

