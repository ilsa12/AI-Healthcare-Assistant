from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # many X-rays are single-channel
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224):
    data_path = Path(data_dir)
    transform = get_transforms(img_size)

    train_dir = data_path / "train"
    val_dir   = data_path / "val"
    test_dir  = data_path / "test"

    loaders = {}
    if train_dir.exists():
        train_ds = datasets.ImageFolder(train_dir.as_posix(), transform=transform)
        loaders['train'] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    if val_dir.exists():
        val_ds = datasets.ImageFolder(val_dir.as_posix(), transform=transform)
        loaders['val'] = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    if test_dir.exists():
        test_ds = datasets.ImageFolder(test_dir.as_posix(), transform=transform)
        loaders['test'] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_names = None
    for key in ['train', 'val', 'test']:
        if key in loaders:
            class_names = loaders[key].dataset.classes
            break
    return loaders, class_names
