from torchvision import models
import torch.nn as nn

def build_vgg11():
    model = models.vgg11(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model

def build_vgg16():
    model = models.vgg16(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model

def build_vgg19():
    model = models.vgg19(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model
