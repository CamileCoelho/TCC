import torch
from torchvision.transforms import v2

class DataAugmentationConfig:
    """Configurador de data augmentation com técnicas padrão"""
    
    def __init__(self, config_dict=None):
        """
        Args:
            config_dict: Dicionário com configurações de data augmentation
                        Se None, desabilita todas as técnicas
        """
        self.config = config_dict or {}
        
    def create_augmentation_transforms(self, height=224, width=224):
        """
        Cria as transformações de data augmentation baseadas na configuração
        
        Args:
            height: Altura da imagem
            width: Largura da imagem
            
        Returns:
            Lista de transformações para aplicar
        """
        transforms_list = []
        
        # Sempre redimensiona primeiro
        transforms_list.append(v2.Resize((height, width)))
        
        # Rotação aleatória
        if self.config.get('random_rotation', {}).get('enabled', False):
            degrees = self.config['random_rotation'].get('degrees', 10)
            transforms_list.append(v2.RandomRotation(degrees))
            
        # Flip horizontal aleatório
        if self.config.get('random_horizontal_flip', {}).get('enabled', False):
            prob = self.config['random_horizontal_flip'].get('probability', 0.5)
            transforms_list.append(v2.RandomHorizontalFlip(p=prob))
            
        # Flip vertical aleatório
        if self.config.get('random_vertical_flip', {}).get('enabled', False):
            prob = self.config['random_vertical_flip'].get('probability', 0.5)
            transforms_list.append(v2.RandomVerticalFlip(p=prob))
            
        # Crop aleatório e redimensionamento
        if self.config.get('random_resized_crop', {}).get('enabled', False):
            scale_min = self.config['random_resized_crop'].get('scale_min', 0.8)
            scale_max = self.config['random_resized_crop'].get('scale_max', 1.0)
            ratio_min = self.config['random_resized_crop'].get('ratio_min', 0.75)
            ratio_max = self.config['random_resized_crop'].get('ratio_max', 1.33)
            transforms_list.append(
                v2.RandomResizedCrop(
                    size=(height, width),
                    scale=(scale_min, scale_max),
                    ratio=(ratio_min, ratio_max)
                )
            )
        
        # Ajuste de brilho, contraste, saturação e matiz
        if self.config.get('color_jitter', {}).get('enabled', False):
            brightness = self.config['color_jitter'].get('brightness', 0.2)
            contrast = self.config['color_jitter'].get('contrast', 0.2)
            saturation = self.config['color_jitter'].get('saturation', 0.2)
            hue = self.config['color_jitter'].get('hue', 0.1)
            transforms_list.append(
                v2.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
            )
        
        # Escala de cinza aleatória
        if self.config.get('random_grayscale', {}).get('enabled', False):
            prob = self.config['random_grayscale'].get('probability', 0.1)
            transforms_list.append(v2.RandomGrayscale(p=prob))
        
        # Blur gaussiano aleatório
        if self.config.get('gaussian_blur', {}).get('enabled', False):
            kernel_size = self.config['gaussian_blur'].get('kernel_size', 3)
            sigma_min = self.config['gaussian_blur'].get('sigma_min', 0.1)
            sigma_max = self.config['gaussian_blur'].get('sigma_max', 2.0)
            transforms_list.append(
                v2.GaussianBlur(
                    kernel_size=kernel_size,
                    sigma=(sigma_min, sigma_max)
                )
            )
        
        # Perspectiva aleatória
        if self.config.get('random_perspective', {}).get('enabled', False):
            distortion_scale = self.config['random_perspective'].get('distortion_scale', 0.2)
            prob = self.config['random_perspective'].get('probability', 0.5)
            transforms_list.append(
                v2.RandomPerspective(
                    distortion_scale=distortion_scale,
                    p=prob
                )
            )
        
        # Affine aleatório
        if self.config.get('random_affine', {}).get('enabled', False):
            degrees = self.config['random_affine'].get('degrees', 0)
            translate = self.config['random_affine'].get('translate', None)
            scale = self.config['random_affine'].get('scale', None)
            shear = self.config['random_affine'].get('shear', None)
            transforms_list.append(
                v2.RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    shear=shear
                )
            )
        
        # Apagar aleatório (Random Erasing)
        if self.config.get('random_erasing', {}).get('enabled', False):
            prob = self.config['random_erasing'].get('probability', 0.5)
            scale_min = self.config['random_erasing'].get('scale_min', 0.02)
            scale_max = self.config['random_erasing'].get('scale_max', 0.33)
            ratio_min = self.config['random_erasing'].get('ratio_min', 0.3)
            ratio_max = self.config['random_erasing'].get('ratio_max', 3.3)
            transforms_list.append(
                v2.RandomErasing(
                    p=prob,
                    scale=(scale_min, scale_max),
                    ratio=(ratio_min, ratio_max)
                )
            )
        
        return transforms_list

    def get_config_summary(self):
        """Retorna um resumo das configurações ativas"""
        active_techniques = []
        
        for technique, settings in self.config.items():
            if isinstance(settings, dict) and settings.get('enabled', False):
                active_techniques.append(technique)
        
        return active_techniques

def parse_data_augmentation_config(config_string):
    """
    Converte uma string de configuração em um dicionário de configurações
    
    Formato esperado: "technique1:param1=value1,param2=value2;technique2:param1=value1"
    
    Args:
        config_string: String com configurações de data augmentation
        
    Returns:
        Dicionário com configurações parseadas
    """
    if not config_string or config_string.lower() == 'none':
        return {}
    
    config_dict = {}
    
    # Separa as técnicas por ';'
    techniques = config_string.split(';')
    
    for technique in techniques:
        if ':' not in technique:
            # Formato simples: apenas o nome da técnica (habilita com valores padrão)
            config_dict[technique.strip()] = {'enabled': True}
        else:
            # Formato completo: technique:param1=value1,param2=value2
            tech_name, params_str = technique.split(':', 1)
            tech_name = tech_name.strip()
            
            config_dict[tech_name] = {'enabled': True}
            
            # Processa os parâmetros
            if params_str.strip():
                params = params_str.split(',')
                for param in params:
                    if '=' in param:
                        key, value = param.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Tenta converter para tipos apropriados
                        try:
                            # Tenta float primeiro
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Se não conseguir converter, mantém como string
                            if value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                        
                        config_dict[tech_name][key] = value
    
    return config_dict