from config.data_augmentation import parse_data_augmentation_config

def load_parameters(filename):
    params = {}
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'replications':
                    params[key] = int(value)
                elif key == 'epochs':
                    params[key] = [int(v) for v in value.split(',')]
                elif key == 'model_names':
                    params[key] = [name.strip() for name in value.split(',')]
                elif key == 'learning_rates' or key == 'weight_decays':
                    params[key] = [float(lr) for lr in value.split(',')]
                elif key == 'save_model' or key == 'use_trained_model':
                    params[key] = value.lower() == 'true'
                elif key == 'data_augmentation_configs':
                    if value.lower() == 'none' or not value:
                        params[key] = [{}]
                    else:
                        config_strings = value.split('|')
                        configs = []
                        for config_str in config_strings:
                            config_dict = parse_data_augmentation_config(config_str.strip())
                            config_dict['_original_string'] = config_str.strip()
                            configs.append(config_dict)
                        params[key] = configs
                elif key == 'thresholds':
                    params[key] = [float(v) for v in value.split(',')]
                else:
                    params[key] = value
    
    if 'data_augmentation_configs' not in params:
        params['data_augmentation_configs'] = [{}]
    
    return params