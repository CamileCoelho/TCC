import os
import torch
import numpy as np

class EarlyStopping:
    """Early stopping para parar o treinamento quando a validação para de melhorar"""

    def __init__(self, patience=7, min_delta=0.001, verbose=True, model_name="", should_save=False, replication=-1, da_name=""):
        """
        Args:
            patience (int): Quantas épocas esperar após a última melhoria
            min_delta (float): Mudança mínima para ser considerada melhoria
            verbose (bool): Se deve imprimir mensagens
            model_name (str): Nome do modelo para criar o caminho de salvamento
            should_save (bool): Se deve salvar o modelo treinado
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0
        self.should_save = should_save
        self.replication = "" if replication == -1 else replication
        self.da_name = da_name

        # Checa se deve salvar o modelo e cria o diretório caso não exista .
        if should_save:
            path = "./trained_models/" + model_name +"_Replication-"+ str(self.replication) + "_" + self.da_name + ".pth"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.path = path

    def __call__(self, val_loss, epoch, model):
        """
        Checa se deve parar o treinamento e salva se estiver configurado.
        
        Args:
            val_loss (float): Loss de validação atual
            epoch (int): Época atual
            
        Returns:
            bool: True se deve parar o treinamento
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.should_save:
                torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"\t    Early Stopping: Baseline definido com loss {val_loss:.6f}")
                
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.should_save:
                torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"\t    Early Stopping: Melhoria detectada! Novo melhor loss: {val_loss:.6f}")
                
        else:
            self.counter += 1
            if self.verbose:
                print(f"\t    Early Stopping: Sem melhoria por {self.counter}/{self.patience} épocas")
            
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\t    🛑 Early Stopping ativado na época {epoch}!")
                return True
                
        return False
