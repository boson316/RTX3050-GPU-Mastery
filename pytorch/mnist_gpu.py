"""
PyTorch MNIST GPU 訓練（輕量 CNN + AMP + Augmentation）
- 輕量 CNN（2 conv + 2 fc）取代 ResNet18，每 epoch 目標 1～1.5s
- AMP、CosineAnnealingLR、RandomRotation(10)
執行：python mnist_gpu.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# 加速：cudnn 自動選最快演算法（輸入尺寸固定時）
torch.backends.cudnn.benchmark = True

batch_size = 2048  # 輕量模型可開大，step 數少、每 epoch 約 1～1.5s
epochs = 10
lr = 0.001

# 訓練用：ToTensor + Augmentation（RandomRotation 10°）+ Normalize
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = torchvision.datasets.MNIST("data", train=True, download=True, transform=train_transform)
test_ds = torchvision.datasets.MNIST("data", train=False, transform=test_transform)


# 輕量 CNN（約 0.5M 參數）：28→14→7，比 ResNet18 快約 3～4 倍，MNIST 仍可 99%+
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 28→14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 14→7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


model = SmallCNN()

device = torch.device("cuda")
model = model.to(device)
# 不在此使用 torch.compile：Windows 上 inductor 需 Triton，多數環境未提供，會報錯
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler("cuda")
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

history = {"loss": [], "train_acc": [], "test_acc": []}

# DataLoader 與訓練迴圈放在 main，Windows 多進程才不會重複執行
train_loader = None
test_loader = None


def train(epoch):
    model.train()
    total_loss = 0.0
    correct_t = torch.tensor(0, device=device, dtype=torch.long)  # GPU 上累加，最後再 .item() 一次
    total = 0
    start = time.time()
    for data, target in train_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast("cuda"):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pred = output.argmax(1)
        correct_t += pred.eq(target).sum()
        total += len(target)
    scheduler.step()
    elapsed = time.time() - start
    avg_loss = total_loss / len(train_loader)
    acc = 100.0 * correct_t.item() / total
    history["loss"].append(avg_loss)
    history["train_acc"].append(acc)
    print(f"Epoch {epoch}: Loss {avg_loss:.4f}, Acc {acc:.2f}%, Time {elapsed:.1f}s")


def test():
    model.eval()
    correct_t = torch.tensor(0, device=device, dtype=torch.long)  # GPU 上累加，最後再 .item() 一次
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast("cuda"):
                output = model(data)
            pred = output.argmax(1)
            correct_t += pred.eq(target).sum()
            total += len(target)
    acc = 100.0 * correct_t.item() / total
    history["test_acc"].append(acc)
    print(f"Test Acc: {acc:.2f}%")


if __name__ == "__main__":
    _num_workers = 2  # 與 GPU 計算重疊，縮短每 epoch；若報錯可改 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=_num_workers, pin_memory=True, persistent_workers=(_num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=_num_workers, pin_memory=True, persistent_workers=(_num_workers > 0))

    print(f"Using {torch.cuda.get_device_name(0)} (SmallCNN + AMP + Aug, batch={batch_size})")

    for e in range(1, epochs + 1):
        train(e)
        test()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["loss"], "o-", color="C0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")
    ax2.plot(history["train_acc"], "o-", label="Train", color="C0")
    ax2.plot(history["test_acc"], "s-", label="Test", color="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Acc (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("mnist_acc_loss.png")
    plt.close()

    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images_gpu = images.to(device)[:8]
    with torch.no_grad():
        with autocast("cuda"):
            outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)
    plt.figure(figsize=(12, 3))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"True: {labels[i].item()}, Pred: {predicted[i].cpu().item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("mnist_pred.png")
    plt.close()
