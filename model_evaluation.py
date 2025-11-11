import os
import re
import torch
import glob
import pandas as pd
from torch.utils.data import DataLoader
from models.cnn import CNN
import numpy as np
from datetime import datetime

class ModelEvaluator:
    """Classe para carregar modelos treinados e avaliar no conjunto de validação"""
    
    def __init__(self, validation_data, batch_size=8):
        """
        Args:
            validation_data: Dataset de validação
            batch_size: Tamanho do batch para avaliação
        """
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"🔍 Avaliador de modelos inicializado")
        print(f"📱 Dispositivo: {self.device}")
        print(f"📊 Batch size: {batch_size}")

    def create_model(self, model_name):
        if model_name == 'MobileNetV2':
            from models.mobilenet import build_mobilenet_v2
            return build_mobilenet_v2()
        
        elif model_name == 'MobileNetV3Large':
            from models.mobilenet import build_mobilenet_v3_large
            return build_mobilenet_v3_large()
        
        elif model_name == 'MobileNetV3Small':
            from models.mobilenet import build_mobilenet_v3_small
            return build_mobilenet_v3_small()
        
        elif model_name == 'ResNet18':
            from models.resnet import build_resnet18
            return build_resnet18()
        
        elif model_name == 'ResNet50':
            from models.resnet import build_resnet50
            return build_resnet50()
        
        elif model_name == 'ResNet101':
            from models.resnet import build_resnet101
            return build_resnet101()
        
        elif model_name == 'VGG11':
            from models.vgg import build_vgg11
            return build_vgg11()
        
        elif model_name == 'VGG16':
            from models.vgg import build_vgg16
            return build_vgg16()
        
        elif model_name == 'VGG19':
            from models.vgg import build_vgg19
            return build_vgg19()
        
        elif model_name == 'AlexNet':
            from models.alexnet import build_alexnet
            return build_alexnet()
        
        elif model_name == 'EfficientNetB0':
            from models.efficientnet import build_efficientnet_b0
            return build_efficientnet_b0()
        
        elif model_name == 'EfficientNetB1':
            from models.efficientnet import build_efficientnet_b1
            return build_efficientnet_b1()
        
        elif model_name == 'EfficientNetB2':
            from models.efficientnet import build_efficientnet_b2
            return build_efficientnet_b2()
        
        elif model_name == 'EfficientNetB3':
            from models.efficientnet import build_efficientnet_b3
            return build_efficientnet_b3()
        
        elif model_name == 'EfficientNetB4':
            from models.efficientnet import build_efficientnet_b4
            return build_efficientnet_b4()
        
        elif model_name == 'EfficientNetB5':
            from models.efficientnet import build_efficientnet_b5
            return build_efficientnet_b5()
        
        elif model_name == 'EfficientNetB6':
            from models.efficientnet import build_efficientnet_b6
            return build_efficientnet_b6()
        
        elif model_name == 'EfficientNetB7':
            from models.efficientnet import build_efficientnet_b7
            return build_efficientnet_b7()
        
        elif model_name == 'DenseNet121':
            from models.densenet import build_densenet121
            return build_densenet121()
        
        elif model_name == 'DenseNet161':
            from models.densenet import build_densenet161
            return build_densenet161()
        
        elif model_name == 'DenseNet169':
            from models.densenet import build_densenet169
            return build_densenet169()
        
        elif model_name == 'DenseNet201':
            from models.densenet import build_densenet201
            return build_densenet201()
        
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def load_model(self, model_name):
        model_path = f"./trained_models/{model_name}.pth"
        
        if not os.path.exists(model_path):
            print(f"❌ Arquivo do modelo não encontrado: {model_path}")
            return None
        
        print(f"📥 Carregando modelo {model_name} de {model_path}")
        
        try:
            model = self.create_model(model_name)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print(f"✅ Modelo {model_name} carregado com sucesso!")
            return model
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {model_name}: {e}")
            return None

    def evaluate_single_model(self, model, model_name, threshold=0.5):
        validation_loader = DataLoader(
            self.validation_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"🔍 Avaliando modelo {model_name}...")

        model.eval()
        y_pred = []
        y_true = []
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in validation_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = model(X)
                loss = criterion(outputs, y)
                total_loss += loss.item()
               
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predicted = (probs >= threshold).astype(int)

                y_pred.extend(predicted)
                y_true.extend(y.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        fbeta = fbeta_score(y_true, y_pred, beta=0.5, pos_label=1, zero_division=0)
        avg_loss = total_loss / len(validation_loader)
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fbeta': fbeta,
            'loss': avg_loss
        }

        print(f"✅ {model_name} - Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

        return metrics

    def get_predictions_per_image(self, model, model_name, threshold=0.5):
        validation_loader = DataLoader(
            self.validation_data, 
            batch_size=1,
            shuffle=False, 
            num_workers=0
        )

        image_results = []

        print(f"📸 Analisando predições por imagem para {model_name}...")

        with torch.no_grad():
            for idx, (X, y) in enumerate(validation_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predicted = (probs >= threshold).long()

                true_label = y.cpu().numpy()[0]
                pred_label = predicted.cpu().numpy()[0]

                correct = 1 if true_label == pred_label else 0
                
                try:
                    image_path = self.validation_data.samples[idx][0]     

                    if os.path.isabs(image_path):
                        current_dir = os.getcwd()

                        if image_path.startswith(current_dir):
                            image_path = os.path.relpath(image_path, current_dir)
                    
                    image_id = image_path.replace('\\', '/')
                except (IndexError, AttributeError):
                    image_id = f"validation_image_{idx:06d}"

                image_results.append({
                    'image_id': image_id,
                    'true_label': int(true_label),
                    f'{model_name}_correct': correct
                })

        print(f"✅ Análise concluída para {model_name}: {len(image_results)} imagens")
        print(f"\n\n")

        return image_results

    def evaluate_multiple_models(self, model_names, output_file="trained_models_evaluation.csv"):
        
        print(f"🚀 Iniciando avaliação de {len(model_names)} modelos treinados")
        print(f"📊 Arquivo de saída: {output_file}")
        print("="*60)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        model_metrics = {}
        all_image_results = None

        for i, model_name in enumerate(model_names, 1):
            print(f"\n🎯 Modelo {i}/{len(model_names)}: {model_name}")

            model = self.load_model(model_name)
            if model is None:
                print(f"⚠️ Pulando modelo {model_name} (erro no carregamento)")
                continue

            metrics = self.evaluate_single_model(model, model_name)
            model_metrics[model_name] = metrics

            image_results = self.get_predictions_per_image(model, model_name)

            if all_image_results is None:
                all_image_results = pd.DataFrame(image_results)
            else:
                model_df = pd.DataFrame(image_results)
                all_image_results[f'{model_name}_correct'] = model_df[f'{model_name}_correct']

        if all_image_results is not None:
            all_image_results.to_csv(output_file, index=False, sep=';')

            print(f"\n💾 Resultados salvos em: {output_file}")

            summary_file = output_file.replace('.csv', '_summary.csv')

            self._create_summary_report(model_metrics, all_image_results, summary_file)

            self._print_final_statistics(all_image_results, model_metrics)

        print("\n🏁 Avaliação de modelos treinados concluída!")
 
    def _create_summary_report(self, model_metrics, image_results, summary_file):
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        summary_data = []

        for model_name, metrics in model_metrics.items():
            col_name = f'{model_name}_correct'

            if col_name in image_results.columns:
                total_images = len(image_results)
                correct_predictions = image_results[col_name].sum()
                accuracy_from_images = correct_predictions / total_images

                summary_data.append({
                    'Model': model_name,
                    'Total_Images': total_images,
                    'Correct_Predictions': correct_predictions,
                    'Wrong_Predictions': total_images - correct_predictions,
                    'Accuracy_from_Images': accuracy_from_images,
                    'Accuracy_from_Metrics': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1'],
                    'Fbeta_Score': metrics['fbeta'],
                    'Loss': metrics['loss']
                })

        summary_df = pd.DataFrame(summary_data)
        
        if os.path.exists(summary_file):
            existing_df = pd.read_csv(summary_file, sep=';')
            new_rows = summary_df[~summary_df['Model'].isin(existing_df['Model'])]
            final_df = pd.concat([existing_df, new_rows], ignore_index=True)
            final_df.to_csv(summary_file, index=False, sep=';')
        else:
            summary_df.to_csv(summary_file, index=False, sep=';')

        print(f"📋 Resumo salvo em: {summary_file}")

    def _print_final_statistics(self, image_results, model_metrics):
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"📸 Total de imagens analisadas: {len(image_results)}")
        print(f"🤖 Total de modelos avaliados: {len(model_metrics)}")
        
        print(f"\n🏆 RANKING POR ACCURACY:")

        ranking = []

        for model_name in model_metrics.keys():
            col_name = f'{model_name}_correct'

            if col_name in image_results.columns:
                accuracy = image_results[col_name].sum() / len(image_results)
                ranking.append((model_name, accuracy))
        
        ranking.sort(key=lambda x: x[1], reverse=True)

        for i, (model, acc) in enumerate(ranking, 1):
            print(f"   {i}º {model}: {acc:.4f} ({acc*100:.2f}%)")
        
    def find_models_by_replication(self, replication_number, models_dir="./trained_models"):
        """
        Busca todos os arquivos .pth de uma replicação específica.
        Retorna um dicionário: {nome_modelo: caminho_arquivo}
        """
        pattern = os.path.join(models_dir, f"*Replication-{replication_number}*.pth")
        files = glob.glob(pattern)
        model_dict = {}
        regex = re.compile(r"(.+)_Replication-\d+(?:_.+)?\.pth$")
        for f in files:
            match = regex.search(os.path.basename(f))
            if match:
                model_name = match.group(1)
                model_dict[model_name] = f
        return model_dict

    def load_model_from_path(self, model_name, model_path):
        print(f"📥 Carregando modelo {model_name} de {model_path}")
        try:
            model = self.create_model(model_name)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            print(f"✅ Modelo {model_name} carregado com sucesso!")
            return model
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {model_name}: {e}")
            return None

    def evaluate_all_replications(self, max_replications=10, models_dir="./trained_models", output_prefix="trained_models_evaluation"):
        """
        Avalia todos os modelos para cada replicação encontrada.
        """
        for replication in range(1, max_replications+1):
            model_paths = self.find_models_by_replication(replication, models_dir)
            if not model_paths:
                print(f"⚠️ Nenhum modelo encontrado para Replication-{replication}")
                continue

            print(f"\n🚀 Avaliando Replication-{replication} ({len(model_paths)} modelos)")
            model_metrics = {}
            all_image_results = None

            for model_name, model_path in model_paths.items():
                model = self.load_model_from_path(model_name, model_path)
                if model is None:
                    continue
                metrics = self.evaluate_single_model(model, model_name)
                model_metrics[model_name] = metrics
                image_results = self.get_predictions_per_image(model, model_name)
                if all_image_results is None:
                    all_image_results = pd.DataFrame(image_results)
                else:
                    model_df = pd.DataFrame(image_results)
                    col_name = f'{model_name}_correct'
                    if col_name not in all_image_results.columns:
                        all_image_results[col_name] = model_df[col_name]

            if all_image_results is not None:
                output_file = f"{output_prefix}_Replication-{replication}.csv"
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file, sep=';')
                    for col in all_image_results.columns:
                        if col not in existing_df.columns:
                            existing_df[col] = all_image_results[col]
                    all_image_results = existing_df
                all_image_results.to_csv(output_file, index=False, sep=';')
                print(f"\n💾 Resultados salvos em: {output_file}")
                summary_file = output_file.replace('.csv', '_summary.csv')
                self._create_summary_report(model_metrics, all_image_results, summary_file)
                self._print_final_statistics(all_image_results, model_metrics)

        print("\n🏁 Avaliação de todas as replicações concluída!")

    def evaluate_all_models_with_da(self, models_dir="./trained_models", output_prefix="./results/trained_models_evaluation", threshold=0.5):
        """
        Avalia todos os modelos .pth na pasta, considerando modelo, replicação e técnica de DA.
        Permite passar threshold para métricas binárias.
        """
        pattern = os.path.join(models_dir, "*.pth")
        files = glob.glob(pattern)
        if not files:
            print(f"Nenhum modelo encontrado em {models_dir}")
            return

        regex = re.compile(r"(.+)_Replication-(\d+)(?:_(.+))?\.pth$")
        model_info_list = []
        for f in files:
            fname = os.path.basename(f)
            match = regex.match(fname)
            if match:
                model_name = match.group(1)
                replication = match.group(2)
                da_technique = match.group(3) if match.group(3) else "None"
                model_info_list.append({
                    "model_name": model_name,
                    "replication": replication,
                    "da_technique": da_technique,
                    "path": f
                })

        if not model_info_list:
            print("Nenhum modelo válido encontrado.")
            return

        # Agrupa por (replication, da_technique)
        from collections import defaultdict
        grouped = defaultdict(list)
        for info in model_info_list:
            key = (info["replication"], info["da_technique"])
            grouped[key].append(info)

        for (replication, da_technique), models in grouped.items():
            print(f"\n🚀 Avaliando Replication-{replication} | DA: {da_technique} ({len(models)} modelos)")

            model_metrics = {}
            all_image_results = None

            for info in models:
                model = self.load_model_from_path(info["model_name"], info["path"])
                if model is None:
                    continue
                metrics = self.evaluate_single_model(model, info["model_name"], threshold=threshold)
                model_metrics[info["model_name"]] = metrics
                image_results = self.get_predictions_per_image(model, info["model_name"])
                for r in image_results:
                    r["Replication"] = replication
                    r["DA_Technique"] = da_technique
                model_df = pd.DataFrame(image_results)
                if all_image_results is None:
                    all_image_results = model_df
                else:
                    col_name = f'{info["model_name"]}_correct'
                    if col_name not in all_image_results.columns:
                        all_image_results[col_name] = model_df[col_name]

            if all_image_results is not None:
                output_file = f"{output_prefix}_Replication-{replication}_{da_technique}.csv"
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file, sep=';')
                    for col in all_image_results.columns:
                        if col not in existing_df.columns:
                            existing_df[col] = all_image_results[col]
                    all_image_results = existing_df
                all_image_results.to_csv(output_file, index=False, sep=';')
                print(f"\n💾 Resultados salvos em: {output_file}")
                summary_file = output_file.replace('.csv', '_summary.csv')
                self._create_summary_report(model_metrics, all_image_results, summary_file)
                self._print_final_statistics(all_image_results, model_metrics)

        print("\n🏁 Avaliação de todos os modelos concluída!")

    def evaluate_all_models_with_da_multiple_thresholds(
        self, models_dir="./trained_models", output_prefix="./results/trained_models_evaluation", thresholds=[0.5]
    ):
        """
        Avalia todos os modelos .pth na pasta para uma lista de thresholds.
        Salva um arquivo de resultados para cada threshold.
        """
        pattern = os.path.join(models_dir, "*.pth")
        files = glob.glob(pattern)
        if not files:
            print(f"Nenhum modelo encontrado em {models_dir}")
            return

        regex = re.compile(r"(.+)_Replication-(\d+)(?:_(.+))?\.pth$")
        model_info_list = []
        for f in files:
            fname = os.path.basename(f)
            match = regex.match(fname)
            if match:
                model_name = match.group(1)
                replication = match.group(2)
                da_technique = match.group(3) if match.group(3) else "None"
                model_info_list.append({
                    "model_name": model_name,
                    "replication": replication,
                    "da_technique": da_technique,
                    "path": f
                })

        if not model_info_list:
            print("Nenhum modelo válido encontrado.")
            return

        from collections import defaultdict
        grouped = defaultdict(list)
        for info in model_info_list:
            key = (info["replication"], info["da_technique"])
            grouped[key].append(info)

        for threshold in thresholds:
            print(f"\n=== Avaliando todos os modelos com threshold={threshold} ===")
            for (replication, da_technique), models in grouped.items():
                print(f"\n🚀 Replication-{replication} | DA: {da_technique} ({len(models)} modelos) | Threshold: {threshold}")

                model_metrics = {}
                all_image_results = None

                for info in models:
                    model = self.load_model_from_path(info["model_name"], info["path"])
                    if model is None:
                        continue
                    metrics = self.evaluate_single_model(model, info["model_name"], threshold=threshold)
                    model_metrics[info["model_name"]] = metrics
                    image_results = self.get_predictions_per_image(model, info["model_name"], threshold=threshold)
                    for r in image_results:
                        r["Replication"] = replication
                        r["DA_Technique"] = da_technique
                    model_df = pd.DataFrame(image_results)
                    if all_image_results is None:
                        all_image_results = model_df
                    else:
                        col_name = f'{info["model_name"]}_correct'
                        if col_name not in all_image_results.columns:
                            all_image_results[col_name] = model_df[col_name]

                if all_image_results is not None:
                    output_file = f"{output_prefix}_threshold_{threshold}_Replication-{replication}_{da_technique}.csv"
                    if os.path.exists(output_file):
                        existing_df = pd.read_csv(output_file, sep=';')
                        for col in all_image_results.columns:
                            if col not in existing_df.columns:
                                existing_df[col] = all_image_results[col]
                        all_image_results = existing_df
                    all_image_results.to_csv(output_file, index=False, sep=';')
                    print(f"\n💾 Resultados salvos em: {output_file}")
                    summary_file = output_file.replace('.csv', '_summary.csv')
                    self._create_summary_report(model_metrics, all_image_results, summary_file)
                    self._print_final_statistics(all_image_results, model_metrics)

        print("\n🏁 Avaliação de todos os modelos concluída para todos os thresholds!")