import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random
import time 

# Google Colab 환경에서 파일 다운로드를 위해 필요할 수 있습니다.
# try:
#     from google.colab import files
#     COLAB_ENV = True
# except ImportError:
#     COLAB_ENV = False
COLAB_ENV = False

##############################
# 0️⃣ 경로 설정
##############################
base_path = '/Users/jaewoo000/Desktop/IBM:RedHat/Study25/_data/quantum/'
os.makedirs(base_path, exist_ok=True)

##############################
# 1️⃣ Seed 고정
##############################
SEED = 6054
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##############################
# 2️⃣ 디바이스 선언
##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dev = qml.device("default.qubit", wires=5)

##############################
# 3️⃣ QNN 회로 정의 (동일)
##############################
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    num_qubits = 5
    layers = 3
    
    num_weights_per_layer = num_qubits * 2
    
    for l in range(layers):
        for i in range(num_qubits):
            qml.RX(inputs[i], wires=i)
        
        for i in range(num_qubits):
            qml.RY(weights[(l * num_weights_per_layer + i) % weights.shape[0]], wires=i)
            qml.RZ(weights[(l * num_weights_per_layer + i + num_qubits) % weights.shape[0]], wires=i)
        
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[num_qubits - 1, 0])

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

##############################
# 4️⃣ 데이터 준비 (동일)
##############################
transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.FashionMNIST(root=base_path, train=True, download=True, transform=transform_train)
test_dataset = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform_test)

indices = [i for i, (_, label) in enumerate(train_dataset) if label in [0, 6]]
train_dataset = Subset(train_dataset, indices)
for i in range(len(train_dataset)):
    original_idx = train_dataset.indices[i]
    if train_dataset.dataset.targets[original_idx] == 6:
        train_dataset.dataset.targets[original_idx] = 1

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

test_subset_indices = [i for i, (_, label) in enumerate(test_dataset) if label in [0, 6]]
test_subset = Subset(test_dataset, test_subset_indices)
for i in range(len(test_subset)):
    original_idx = test_subset.indices[i]
    if test_subset.dataset.targets[original_idx] == 6:
        test_subset.dataset.targets[original_idx] = 1
test_eval_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

##############################
# 5️⃣ 모델 정의 (수정)
##############################
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ 파라미터 수를 맞추기 위해 CNN 채널을 줄임
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ✅ FC 레이어와 QNN 입력 차원 조정
        self.fc1 = nn.Linear(64, 5) # GAP 후 64채널, QNN 입력 차원을 5 (큐빗 수)로 맞춤
        self.norm = nn.LayerNorm(5) # QNN 입력에 LayerNorm 적용
        
        # ✅ QNN 파라미터 개수 유지 (규격 준수)
        self.q_params = nn.Parameter(torch.rand(30)) 
        
        # ✅ FC 레이어 노드 수 조정
        self.fc2 = nn.Linear(1, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.norm(x)
        
        q_out = torch.stack([quantum_circuit(x[i], self.q_params) for i in range(x.shape[0])])
        q_out = q_out.unsqueeze(1).to(torch.float32)
        x = F.relu(self.fc2(q_out))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

##############################
# 6️⃣ 규격 검사
##############################
model_for_specs = HybridModel()
model_for_specs.eval()
dummy_q_inputs = torch.randn(5)
dummy_q_weights = model_for_specs.q_params.data
q_specs = qml.specs(quantum_circuit)(dummy_q_inputs, dummy_q_weights)
assert q_specs["num_tape_wires"] <= 8, f"❌ 큐빗 수 초과: {q_specs['num_tape_wires']} (최대 8)"
assert q_specs['resources'].depth <= 30, f"❌ 회로 깊이 초과: {q_specs['resources'].depth} (최대 30)"
assert q_specs["num_trainable_params"] <= 60, f"❌ 학습 퀀텀 파라미터 수 초과: {q_specs['num_trainable_params']} (최대 60)"
print("✅ QNN 규격 검사 통과")

total_params = sum(p.numel() for p in model_for_specs.parameters() if p.requires_grad)
assert total_params <= 50000, f"❌ 학습 전체 파라미터 수 초과: {total_params} (최대 50000)"
print(f"✅ 학습 전체 파라미터 수 검사 통과: {total_params}")
del model_for_specs

##############################
# 7️⃣ 학습 (하이퍼파라미터 조정)
##############################
model = HybridModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
criterion = nn.NLLLoss()
best_acc = 0.0
early_stopping_patience = 25
epochs_no_improve = 0
epochs = 500

for epoch in range(epochs):
    start_time = time.time()
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total * 100
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%")

    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_acc)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < old_lr:
        print(f"🔽 학습률 감소: {old_lr:.6f} → {new_lr:.6f}")

    if avg_acc > best_acc:
        best_acc = avg_acc
        epochs_no_improve = 0
        best_model_path = os.path.join(base_path, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated at epoch {epoch+1} with train acc {avg_acc:.2f}%")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"🕒 Epoch {epoch+1} 소요 시간: {epoch_time:.2f}초")

##############################
# 8️⃣ 모델 저장 (동일)
##############################
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(base_path, f'model_{now}_final_train_acc_{best_acc:.4f}.pth')
torch.save(model.state_dict(), model_path)
print(f"✅ 마지막 모델 저장 완료: {model_path}")

##############################
# 9️⃣ 추론 및 제출 생성 (동일)
##############################
model.load_state_dict(torch.load(os.path.join(base_path, 'best_model.pth')))
model.eval()
all_preds = []
correct_eval = 0
total_eval = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        output = model(images)
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())

with torch.no_grad():
    for images, labels in test_eval_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1)
        correct_eval += (pred == labels).sum().item()
        total_eval += labels.size(0)

eval_acc = (correct_eval / total_eval) * 100 if total_eval > 0 else 0
print(f"\n✅ 0/6 클래스에 대한 테스트 정확도: {eval_acc:.2f}%")

final_submission_preds = [0 if p == 0 else 6 for p in all_preds]
csv_filename = f"{base_path}y_pred_{now}.csv"
df = pd.DataFrame({"y_pred": final_submission_preds})
df.to_csv(csv_filename, index=False, header=False)
print(f"✅ 결과 저장 완료: {csv_filename}")