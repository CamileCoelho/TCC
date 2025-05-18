import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import models
from sklearn.metrics import recall_score, f1_score, fbeta_score, classification_report, precision_score

class CNN:
    def __init__(self, train_data, validation_data, test_data, batch_size):
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay):        
        model = self.create_model(model_name)
        optimizerSGD = self.create_optimizer(model,learning_rate, weight_decay)
        criterionCEL = self.create_criterion()
        self.train_model(model, self.train_loader, optimizerSGD, criterionCEL, num_epochs) 
        metrics = self.evaluate_model(model, self.validation_loader)
        return metrics

    # Método original, deixei para caso seja necessário voltar. / Talvez necessário se precisar das replicações.

    # def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replicacoes):
    #     soma = 0
    #     acc_max = 0
    #     for i in range(0,replicacoes):
    #         model = self.create_model(model_name)
    #         optimizerSGD = self.create_optimizer(model,learning_rate, weight_decay)
    #         criterionCEL = self.create_criterion()
    #         self.train_model(model, self.train_loader, optimizerSGD, criterionCEL, model_name, num_epochs, learning_rate, weight_decay, i) 
    #         acc = self.evaluate_model(model, self.validation_loader)
    #         soma = soma + acc
    #         if acc > acc_max:
    #             acc_max = acc
    #             iter_acc_max = i
    #     return soma / replicacoes, iter_acc_max
        
    
    def create_model(self, model_name):
        if model_name == 'VGG19':
            model = models.vgg19(weights='DEFAULT')
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
            return model
        
        ## Metodo pode ser utilizado para ensembles futuramente, manter estrutura.
        # elif (model_name=='VGG11'):
        #     model = models.vgg11(weights='DEFAULT')  
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     model.classifier[6] = nn.Linear(model.classifier[6].in_features,2)
        #     return model
        # elif (model_name=='Alexnet'):
        #     model = models.alexnet(weights='DEFAULT')  
        #     for param in model.parameters():
        #         param.requires_grad = False
        #     model.classifier[6] = nn.Linear(model.classifier[6].in_features,2)
        #     return model
   
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

    def train_model(self, model, train_loader, optimizer, criterion, num_epochs): 
        model.to(self.device)
        for i in (range(1,num_epochs+1)):
            print(f"\tTreinando epoca: {i}", end='\r')
            self.train_epoch(model, train_loader, optimizer, criterion)


    #Metodo de treino anterior, comentei por ter modificado e removido a parte onde ele salva o modelo. Posteriormente pode ser usado.

    # def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replicacao): 
    #     model.to(self.device)
    #     min_loss = 100
    #     e_measures = []
    #     for i in (range(1,num_epochs+1)):
    #         train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

    #         if (train_loss < min_loss):
    #             min_loss = train_loss
    #             nome_arquivo = f"./modelos/{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replicacao}.pth"
    #             torch.save(model.state_dict(), nome_arquivo)

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


