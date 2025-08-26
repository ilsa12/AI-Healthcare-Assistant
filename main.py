from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths
data_dir = "data"

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset load test
train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_data = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

print(f"✅ Total training images: {len(train_data)}")
print(f"✅ Total validation images: {len(val_data)}")
print(f"✅ Classes: {train_data.classes}")


images, labels = next(iter(train_loader))
print(f"✅ Batch shape: {images.shape}")
print(f"✅ Labels: {labels}")
