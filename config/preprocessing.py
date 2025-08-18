import torch
from torchvision.transforms import v2
from torchvision import datasets
from models.cnn import CNN
from config.data_augmentation import DataAugmentationConfig

def define_transforms(height, width, da_config=None):
    """
    Define as transformações com ou sem data augmentation
    
    Args:
        height: Altura da imagem
        width: Largura da imagem  
        da_config: Configuração de data augmentation (dict)
    """
    # Configuração de data augmentation
    da_configurator = DataAugmentationConfig(da_config)
    
    # Transformações base (sempre aplicadas)
    base_transforms = [
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Transformações de treino (com data augmentation se configurado)
    train_transforms = da_configurator.create_augmentation_transforms(height, width)
    train_transforms.extend(base_transforms)
    
    # Transformações de teste (apenas básicas)
    test_transforms = [
        v2.Resize((height, width)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    data_transforms = {
        'train': v2.Compose(train_transforms),
        'test': v2.Compose(test_transforms)
    }
    
    # Imprime resumo das técnicas de data augmentation ativas
    active_techniques = da_configurator.get_config_summary()
    if active_techniques:
        print(f"🎨 Data Augmentation ativado com: {', '.join(active_techniques)}")
    else:
        print(f"🎨 Data Augmentation desabilitado")
    
    return data_transforms

def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/train/', transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/dev/', transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/test/', transform=data_transforms['test'])
    return train_data, validation_data, test_data

def pre_process_data(da_config=None):
    """
    Pré-processa os dados com configuração de data augmentation
    
    Args:
        da_config: Configuração de data augmentation (dict)
    """
    data_transforms = define_transforms(224, 224, da_config)
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