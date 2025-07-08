import torch
import numpy as np

class EarlyStopping:
    """Early stopping para parar o treinamento quando a validação para de melhorar"""
    
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): Quantas épocas esperar após a última melhoria
            min_delta (float): Mudança mínima para ser considerada melhoria
            verbose (bool): Se deve imprimir mensagens
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0
        
    def __call__(self, val_loss, epoch):
        """
        Checa se deve parar o treinamento
        
        Args:
            val_loss (float): Loss de validação atual
            epoch (int): Época atual
            
        Returns:
            bool: True se deve parar o treinamento
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"\t    Early Stopping: Baseline definido com loss {val_loss:.6f}")
                
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
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
