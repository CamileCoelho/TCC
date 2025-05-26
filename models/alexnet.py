import torch.nn as nn
import torchvision.models as models

def build_alexnet():
    model = models.alexnet(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model