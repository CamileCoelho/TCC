import os, csv

# Troca o "." por "," para separar corretamente os decimais.
def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}".replace('.', ',')
    return value

def save_csv(replications, model, total_epochs, actual_epochs, lr, wd, result, path='./results/training_result.csv'):

    #Separa o resultado em campos e os formata.
    
    accuracy = format_value(result['accuracy'])
    precision = format_value(result['precision'])
    recall = format_value(result['recall'])
    f1_score = format_value(result['f1'])
    f_beta = format_value(result['fbeta'])
    loss = format_value(result['loss'])
    lr = format_value(lr)
    wd = format_value(wd)
    da_config = result.get('da_config', 'none')

    row = [replications, model, total_epochs, actual_epochs, lr, wd, da_config, accuracy, precision, recall, f1_score, f_beta, loss]

    # Cria o diretório caso não exista.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Abre o arquivo para acrescentar uma linha
    with open(path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')

        # Se o arquivo estiver vazio, escreve o cabeçalho
        if file.tell() == 0:
            writer.writerow(['Replication No','Model Name', 'Total Epochs', 'Stopped at Epoch', 'Learning Rate', 'Weight Decay', 'Data Augmentation', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Fbeta', 'Loss'])

        writer.writerow(row)