
import itertools
from models.cnn import CNN
from config.preprocessing import pre_process_data
from config.output import save_csv
from config.params_loader import load_parameters


if __name__ == '__main__':
    # Carrega os dados e pré-processa com tamanho 244x244
    cnn = pre_process_data()

    # Carrega os parâmetros do arquivo params.txt
    params = load_parameters('params.txt')

    # Extrai os parâmetros necessários
    replications = params['replications']
    model_names = params['model_names']
    epochs = params['epochs']
    learning_rates = params['learning_rates']
    weight_decays = params['weight_decays']

    # Gera todas as combinações possíveis de parâmetros
    combinations = list(itertools.product(model_names, epochs, learning_rates, weight_decays))

    for model, epoch, lr, wd in combinations:
            
        for i in range(1, replications+1):

            print(f"\n>>> Replication {i} | Model: {model} | Epochs: {epoch} | LR: {lr} | WD: {wd}")

            result = cnn.create_and_train_cnn(model, epoch, lr, wd)

            save_csv(replications, model=model, epoch=epoch, lr=lr, wd=wd, result=result)
        