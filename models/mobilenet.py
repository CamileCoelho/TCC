from torchvision import models
import torch.nn as nn

def build_mobilenet_v2():
    model = models.mobilenet_v2(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_mobilenet_v3_large():
    model = models.mobilenet_v3_large(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model

def build_mobilenet_v3_small():
    model = models.mobilenet_v3_small(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model
