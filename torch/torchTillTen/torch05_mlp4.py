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


x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1],[9,8,7,6,5,4,3,2,1,0]])

print(x.shape)
print(y.shape)

#1 is 3
x = x.T     #(10,1)
y = y.T     #(10,3)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

# x = torch.tensor(x, dtype = float).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)

x_mean = torch.mean(x)
x_std = torch.std(x)

x = (x - torch.mean(x))/ torch.std(x)

model = nn.Sequential(
    nn.Linear(1,16),
    nn.Linear(16,8),
    nn.Linear(8,3),
    # nn.Linear(2,1),
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

x_pred = (torch.Tensor([[10]]).to(DEVICE) - x_mean)/ x_std
result = model(x_pred)

print('10 의 예측값:', result)
print('10 의 예측값:', result.detach().cpu().numpy())
# print('10 의 예측값:', result.item())

