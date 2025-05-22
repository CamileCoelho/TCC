from torchvision import models
import torch.nn as nn

def build_resnet18():
    model = models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def build_resnet50():
    model = models.resnet50(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def build_resnet101():
    model = models.resnet101(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
