import torch 
from torch.utils.data import Dataset, DataLoader, TensorDataset


#1. 커스텀 데이터셋
# class MyDataset(Dataset):
#     def __init__(self):

#         self.x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
#         self.y = [0,1,0,1,0]

#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([0,1,0,1,0])



#2 instance 생성
dataset = TensorDataset(x,y)


#3 DataLoader 에 쏙
loader = DataLoader(dataset, batch_size=2, shuffle=True)

#4 출력
for batch_idx, (xb, yb) in enumerate(loader):
    print('======================== 배치:', batch_idx, '=================================')
    print('x: 배치', batch_idx)
    print('x:', xb)
    print('y:', yb
          )