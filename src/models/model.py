import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int = 2, pretrained: bool = True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
