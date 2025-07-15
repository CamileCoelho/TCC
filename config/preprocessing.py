import torch
from torchvision.transforms import v2
from torchvision import datasets
from models.cnn import CNN

def define_transforms(height, width):
    data_transforms = {
        'train' : v2.Compose([
                    v2.Resize((height,width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test'  : v2.Compose([
                    v2.Resize((height,width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms

def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/train/',transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/dev/',transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/test/',transform=data_transforms['test'])
    return train_data, validation_data, test_data

def pre_process_data():
    data_transforms = define_transforms(224,224)
    train_data, validation_data, test_data = read_images(data_transforms)
    num_workers = 4  # Número de workers para DataLoader / Ajuste conforme os núcleos do processador
    
    # Batch size / pin_memory otimizado baseado no dispositivo
    if torch.cuda.is_available():
        batch_size = 16  # GPU pode usar batch maior
        pin_memory = True # Melhora a performance na GPU
        print(f"📦 Batch size para GPU: {batch_size}")
    else:
        batch_size = 8   # CPU usa batch menor
        pin_memory = False # Desnecessário na CPU
        print(f"📦 Batch size para CPU: {batch_size}")

    return CNN(train_data, validation_data, test_data, batch_size, num_workers, pin_memory)