
import itertools
import time
from models.cnn import CNN
from config.preprocessing import pre_process_data
from config.output import save_csv
from config.params_loader import load_parameters


if __name__ == '__main__':
    start_time = time.time()
    
    # Carrega os dados e pré-processa com tamanho 224x224
    cnn = pre_process_data()

    # Carrega os parâmetros do arquivo params.txt
    params = load_parameters('params.txt')

    # Extrai os parâmetros necessários
    replications = params['replications']
    model_names = params['model_names']
    epochs = params['epochs']
    learning_rates = params['learning_rates']
    weight_decays = params['weight_decays']

    print(f"🚀 Configuração do experimento:")
    print(f"   Replicações: {replications}")
    print(f"   Modelos: {model_names}")
    print(f"   Épocas máximas: {epochs}")
    print(f"   Learning Rates: {learning_rates}")
    print(f"   Weight Decays: {weight_decays}")
    print(f"   Early Stopping: Ativado (padrão)")

    # Gera todas as combinações possíveis de parâmetros
    combinations = list(itertools.product(model_names, epochs, learning_rates, weight_decays))
    total_experiments = len(combinations) * replications
    
    print(f"\n📋 Total de experimentos: {total_experiments}")
    print(f"📋 Total de combinações: {len(combinations)}")
    print(f"{'='*50}")
    
    experiment_count = 0

    for combo_idx, (model, epoch, lr, wd) in enumerate(combinations, 1):
        print(f"\n🎯 Combinação {combo_idx}/{len(combinations)}: {model}")
        
        for i in range(1, replications+1):
            experiment_count += 1
            
            print(f"\n>>> Replication {i}/{replications} | Progresso geral: {experiment_count}/{total_experiments} ({100*experiment_count/total_experiments:.1f}%)")
            print(f"    Model: {model} | Epochs: {epoch} | LR: {lr} | WD: {wd}")

            result = cnn.create_and_train_cnn(model, epoch, lr, wd)

            actual_epochs = result['actual_epochs']

            save_csv(i, model=model, total_epochs=epoch, actual_epochs=actual_epochs, lr=lr, wd=wd, result=result)
            
            print(f"    ✅ Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f}")
    
    # Estatísticas finais
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"🏁 Experimento concluído!")
    print(f"⏱️ Tempo total: {total_time/3600:.2f} horas")
    print(f"📊 Experimentos executados: {experiment_count}")
    print(f"💾 Resultados salvos em: ./results/training_result.csv")
        