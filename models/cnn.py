import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from sklearn.metrics import (
    recall_score, f1_score, fbeta_score, precision_score
)

from models.alexnet import build_alexnet
from models.mobilenet import (
    build_mobilenet_v2, build_mobilenet_v3_large, build_mobilenet_v3_small
)
from models.resnet import (
    build_resnet18, build_resnet50, build_resnet101
)
from models.vgg import (
    build_vgg11, build_vgg16, build_vgg19
)
from models.efficientnet import (
    build_efficientnet_b0, build_efficientnet_b1, build_efficientnet_b2,
    build_efficientnet_b3, build_efficientnet_b4, build_efficientnet_b5,
    build_efficientnet_b6, build_efficientnet_b7
)
from models.densenet import (
    build_densenet121, build_densenet161, build_densenet169, build_densenet201
)


class CNN:
    def __init__(self, train_data, validation_data, test_data, batch_size, num_workers, pin_memory):
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.validation_loader = data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, save_model, replication, da_name):        
        model = self.create_model(model_name)
        optimizerSGD = self.create_optimizer(model, learning_rate, weight_decay)
        criterionCEL = self.create_criterion()
        
        actual_epochs = self.train_model(model, self.train_loader, optimizerSGD, criterionCEL, num_epochs, model_name, save_model, replication, da_name) 

        metrics = self.evaluate_model(model, self.validation_loader)

        metrics['actual_epochs'] = actual_epochs

        return metrics        
    
    def create_model(self, model_name):
        if model_name == 'MobileNetV2':
            return build_mobilenet_v2()
        
        if model_name == 'MobileNetV3Large':
            return build_mobilenet_v3_large()
        
        if model_name == 'MobileNetV3Small':
            return build_mobilenet_v3_small()
        
        if model_name == 'ResNet18':
            return build_resnet18()
        
        if model_name == 'ResNet50':
            return build_resnet50()
        
        if model_name == 'ResNet101':
            return build_resnet101()
        
        if model_name == 'VGG11':
            return build_vgg11()
        
        if model_name == 'VGG16':
            return build_vgg16()
        
        if model_name == 'VGG19':
            return build_vgg19()
        
        if model_name == 'AlexNet':
            return build_alexnet()
        
        if model_name == 'EfficientNetB0':
            return build_efficientnet_b0()
        
        if model_name == 'EfficientNetB1':
            return build_efficientnet_b1()
        
        if model_name == 'EfficientNetB2':
            return build_efficientnet_b2()
        
        if model_name == 'EfficientNetB3':
            return build_efficientnet_b3()
        
        if model_name == 'EfficientNetB4':
            return build_efficientnet_b4()
        
        if model_name == 'EfficientNetB5':
            return build_efficientnet_b5()
        
        if model_name == 'EfficientNetB6':
            return build_efficientnet_b6()
        
        if model_name == 'EfficientNetB7':
            return build_efficientnet_b7()
        
        if model_name == 'DenseNet121':
            return build_densenet121()
        
        if model_name == 'DenseNet161':
            return build_densenet161()
        
        if model_name == 'DenseNet169':
            return build_densenet169()
        
        if model_name == 'DenseNet201':
            return build_densenet201()

    def create_optimizer(self, model, learning_rate, weight_decay):
        update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                update.append(param)
        optimizerSGD = optim.SGD(update, lr=learning_rate, weight_decay=weight_decay)
        return optimizerSGD

    def create_criterion(self):
        criterionCEL = nn.CrossEntropyLoss()
        return criterionCEL

    def train_model(self, model, train_loader, optimizer, criterion, num_epochs, model_name, save_model, replication, da_name): 
        from config.early_stopping import EarlyStopping
        
        model.to(self.device)

        # Early Stopping sempre ativado por padrão
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True, model_name=model_name, should_save=save_model, replication=replication, da_name=da_name)
        
        for i in range(1, num_epochs + 1):
            # Treina uma época
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Avalia no conjunto de validação
            val_metrics = self.evaluate_model(model, self.validation_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            print(f"\t[{i}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Verifica Early Stopping
            if early_stopping(val_loss, i, model):
                print(f"\t  Treinamento parado na época {i} (Early Stopping)")
                return i
            
        return num_epochs

    def train_epoch(self,model, trainLoader, optimizer, criterion):
        model.train()
        losses = []
        for X, y in trainLoader:
            X = X.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        return np.mean(losses)

    def evaluate_model(self, model, loader):
        model.to(self.device)
        model.eval()

        y_true = []
        y_pred_list = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                loss = criterion(outputs, y)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                y_true.extend(y.cpu().numpy())
                y_pred_list.extend(preds.cpu().numpy())

        # Calcula métricas
        acc = np.mean(np.array(y_true) == np.array(y_pred_list))
        precision = precision_score(y_true, y_pred_list, average='macro') 
        recall = recall_score(y_true, y_pred_list, average='macro')
        f1 = f1_score(y_true, y_pred_list, average='macro')
        fbeta = fbeta_score(y_true, y_pred_list, beta=0.5, average='macro')
        avg_loss = total_loss / len(loader)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": fbeta,
            "loss": avg_loss
        }