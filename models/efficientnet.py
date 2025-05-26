import torch.nn as nn
import torchvision.models as models

def build_efficientnet_b0():
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b1():
    model = models.efficientnet_b1(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b2():
    model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b3():
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b4():
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b5():
    model = models.efficientnet_b5(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b6():
    model = models.efficientnet_b6(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_efficientnet_b7():
    model = models.efficientnet_b7(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model