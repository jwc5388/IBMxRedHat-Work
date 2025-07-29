import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

#1 data
x = np.array([range(10), range(21,31), range(201,211)]) #(3,10)

y = np.array([[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]) #(2,10)

x = x.T #(10,3)
y = y.T # (10,2) ##############check notes#################!!!!!!!!!!! multiple columns possible!!!!

x= torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)


x_mean = torch.mean(x)
x_std = torch.std(x)

x = (x-torch.mean(x))/torch.std(x)

model = nn.Sequential(
    nn.Linear(3,16),
    nn.Linear(16,8),
    nn.Linear(8,2),
    # nn.Linear(2,1),
).to(DEVICE)


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train(model, criterion,optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 5000

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x,y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

def evaluate(model, criterion, x,y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
        
    return loss2.item()

loss2 = evaluate(model, criterion, x,y)

print('final loss:', loss2)

x_pred = (torch.Tensor([[10,31,211]]).to(DEVICE)- x_mean)/x_std

result = model(x_pred)
print('10, 31, 211 의 예측값:', result)
# print('10,31,211 의 예측값:', result.item())
print('10, 31, 211 의 예측값:', result.detach())
print('10, 31, 211 의 예측값:', result.detach().cpu().numpy())
print('10,31,211 의 예측값:', result[0][0].item(), result[0][1].item())
print('10,31,211 의 예측값:', result.cpu().detach().numpy())  # 예: [[5.4, 4.6]]
print('10,31,211 의 예측값:', result.squeeze().tolist())  # 예: [5.4, 4.6]