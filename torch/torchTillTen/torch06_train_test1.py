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
    
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])
x_pre = np.array([12,13,14])

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
print(x_train.shape)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
x_pre = torch.FloatTensor(x_pre).unsqueeze(1).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

x_mean = torch.mean(x_train).to(DEVICE)
x_std = torch.std(x_train).to(DEVICE)

x_train = ((x_train - torch.mean(x_train))/ torch.std(x_train)).to(DEVICE)


model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 8),
    nn.Linear(8,1)
).to(DEVICE)


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)



def train(model, criterion,optimizer, x_train, y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 5000

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train,y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
        
    return loss2.item()

loss2 = evaluate(model, criterion,x_test, y_test)

print('final loss:', loss2)

x_pred = (x_pre - x_mean)/x_std

result = model(x_pred)
print('x_pred 예측값:', result.detach().cpu().numpy())

exit()
# print('10,31,211 의 예측값:', result.item())
print('10, 31, 211 의 예측값:', result.detach())
print('10, 31, 211 의 예측값:', result.detach().cpu().numpy())