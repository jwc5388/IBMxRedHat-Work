import os, random, warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Module, Conv2d, BatchNorm2d, Dropout2d, Linear, AdaptiveAvgPool2d
from torch.nn.parameter import Parameter
import pennylane as qml
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

# 재현성
SEED = 6054
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 설정값
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 200
PATIENCE = 10
BEST_MODEL_PATH = "best_model.pt"
os.makedirs("submission", exist_ok=True)

# 데이터
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(28, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

full_train = torchvision.datasets.FashionMNIST("./", train=True, download=True, transform=transform_train)
test_ds = torchvision.datasets.FashionMNIST("./", train=False, download=True, transform=transform_test)

mask = (full_train.targets == 0) | (full_train.targets == 6)
full_train.targets[full_train.targets == 6] = 1
binary_ds = Subset(full_train, torch.where(mask)[0])

val_size = int(len(binary_ds) * 0.1)
train_size = len(binary_ds) - val_size
train_ds, val_ds = random_split(binary_ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# FocalLoss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# QNN 회로
class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        self.dev = qml.device("default.qubit", wires=4)
        self.params = Parameter(torch.randn(2 * 4 * 3, dtype=torch.float64), requires_grad=True)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.AngleEmbedding(x, wires=[0, 1, 2, 3])
            qml.StronglyEntanglingLayers(self.params.reshape(2, 4, 3), wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self.circuit = circuit

    def forward(self, x):
        return self.circuit(x)

# CNN + GAP + QNN + FC
class HybridCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 16, 3, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.bn2 = BatchNorm2d(32)
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.bn3 = BatchNorm2d(64)
        self.dropout = Dropout2d(0.4)
        self.gap = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(64, 128)
        self.fc2 = Linear(128, 4)
        self.qnn = QuantumCircuit()
        self.final = Linear(1, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.gap(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        q_out = torch.stack([self.qnn(vec.double()) for vec in x]).view(-1, 1).float()
        logits = self.final(q_out)
        return logits

model = HybridCNN().to(device)
loss_fn = FocalLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
best_val = float('inf')
patience_ctr = 0

# 학습 루프
for epoch in range(1, EPOCHS + 1):
    scheduler.step()
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"Train {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # 검증
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            val_loss += loss_fn(logits, target).item()
            pred = logits.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"[{epoch}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Best model saved.")
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping.")
            break

# 추론
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
all_preds = []
with torch.no_grad():
    for data, _ in tqdm(test_loader, desc="Test Inf"):
        data = data.to(device)
        logits = model(data)
        pred = logits.argmax(dim=1).cpu().numpy()
        y = np.where(pred == 1, 6, 0)
        all_preds.extend(y.tolist())

filename = f"submission/y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv"
np.savetxt(filename, all_preds, fmt='%d')
print(f"Saved submission: {filename}")

# 자원 검증
dummy_input = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
specs = qml.specs(model.qnn.circuit)(dummy_input)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert specs["num_tape_wires"] <= 8
assert specs["resources"].depth <= 30
assert specs["num_trainable_params"] <= 60
assert total_params <= 50000
print("회로 자원 조건 통과")