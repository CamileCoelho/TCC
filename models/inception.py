from torchvision import models
import torch.nn as nn

def build_inception_v3():
    model = models.inception_v3(weights='DEFAULT', aux_logits=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    return model