import torch.nn as nn
import torchvision.models as models

def build_densenet121():
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def build_densenet161():
    model = models.densenet161(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def build_densenet169():
    model = models.densenet169(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

def build_densenet201():
    model = models.densenet201(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model
