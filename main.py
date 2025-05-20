from modelos.cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
from config.output import save_csv

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

if __name__ == '__main__':
    data_transforms = define_transforms(224,224)
    train_data, validation_data, test_data = read_images(data_transforms)
    cnn = CNN(train_data, validation_data, test_data,8)

    replications = 10
    model_names=['VGG19']
    epochs = [50]
    learning_rates = [0.0001]
    weight_decays = [0.01]

    for i in range(1, replications+1):
        print(f"Replication No: {i}")
        result = cnn.create_and_train_cnn(model_names[0],epochs[0],learning_rates[0],weight_decays[0])

        save_csv(
        replications,
        model=model_names[0],
        epochs=epochs[0],
        lr=learning_rates[0],
        wd=weight_decays[0],
        result=result)
    