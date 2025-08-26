from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import os

def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def fit(model, loaders: Dict[str, DataLoader], device,
        epochs: int = 3, lr: float = 1e-3, save_dir: str = "outputs/models"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        if 'train' in loaders:
            train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
            print(f"Train  | loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        if 'val' in loaders:
            val_loss, val_acc = evaluate(model, loaders['val'], criterion, device)
            print(f"Val    | loss: {val_loss:.4f}  acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"✅ Saved new best model (acc={best_val_acc:.4f})")
        else:
            # if no val set, save last
            torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    return model
