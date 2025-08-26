import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np

# Paths
data_dir  = r"C:\Users\DELL\Downloads\AI_Healthcare_Assistant\AI_Healthcare_Assistant\data"
train_dir = rf"{data_dir}\train"
val_dir   = rf"{data_dir}\val"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data Augmentation

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data   = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)


# Class Weights

class_counts = np.bincount(train_data.targets)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Model
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop with Early Stopping

def train_model(epochs=20, patience=3):
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        train_acc = correct / len(train_data)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "resnet18_best.pth"))
            print(f"✅ Best model updated at Epoch {epoch+1} (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print("🎯 Training Complete! Best model saved.")

#  Safe Guard 
if __name__ == "__main__":
    train_model(epochs=20, patience=3)
