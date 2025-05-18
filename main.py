from modelos.cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import csv, os

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

# Troca o "." por "," para separar corretamente os decimais.
def formatar_valor(valor):
    if isinstance(valor, float):
        return f"{valor:.6f}".replace('.', ',')
    return valor

def salvar_resultados_csv(execucoes, modelo, epocas, lr, wd, resultado, caminho='./resultados/resultados_treinamento.csv'):

    #Separa o resultado em campos e os formata.
    
    acuracia = formatar_valor(resultado['accuracy'])
    precisao = formatar_valor(resultado['precision'])
    recall = formatar_valor(resultado['recall'])
    f1_score = formatar_valor(resultado['f1'])
    f_beta = formatar_valor(resultado['fbeta'])
    loss = formatar_valor(resultado['loss'])
    lr = formatar_valor(lr)
    wd = formatar_valor(wd)

    linha = [execucoes, modelo, epocas, lr, wd, acuracia, precisao, recall, f1_score, f_beta, loss]

    # Cria o diretório caso não exista.
    os.makedirs(os.path.dirname(caminho), exist_ok=True)

    # Abre o arquivo para acrescentar uma linha
    with open(caminho, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')

        # Se o arquivo estiver vazio, escreve o cabeçalho
        if file.tell() == 0:
            writer.writerow(['Nº Execução','Modelo', 'Épocas', 'Learning Rate', 'Weight Decay', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Fbeta', 'Loss'])

        writer.writerow(linha)

    print(f"\nResultados salvos em {caminho}")

def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/train/',transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/dev/',transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/test/',transform=data_transforms['test'])
    return train_data, validation_data, test_data

if __name__ == '__main__':
    data_transforms = define_transforms(224,224)
    train_data, validation_data, test_data = read_images(data_transforms)
    cnn = CNN(train_data, validation_data, test_data,8)

    execucoes = 10 # Numero de vezes que o treino será executado.
    #replicacoes = 10 ||| Não tenho certeza quanto a esse parametro. / Verificar com o Castello sobre. / Quase certeza que nesse contexto não será usado.
    model_names=['VGG19']
    epochs = [50]
    learning_rates = [0.0001]
    weight_decays = [0.01]

    for i in range(1, execucoes+1):
        print(f"Execução Nº: {i}")
        resultado = cnn.create_and_train_cnn(model_names[0],epochs[0],learning_rates[0],weight_decays[0])

        salvar_resultados_csv(
        execucoes,
        modelo=model_names[0],
        epocas=epochs[0],
        lr=learning_rates[0],
        wd=weight_decays[0],
        resultado=resultado
)
    