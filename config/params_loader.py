def load_parameters(filename):
    params = {}
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'replications' or key == 'epochs':
                    params[key] = int(value)
                elif key == 'model_names':
                    params[key] = [name.strip() for name in value.split(',')]
                elif key == 'learning_rates' or key == 'weight_decays':
                    params[key] = [float(lr) for lr in value.split(',')]
                elif key == 'save_model' or key == 'use_trained_model':
                    params[key] = value.lower() == 'true'
                else:
                    params[key] = value
    
    return params