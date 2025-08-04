def load_parameters(filename):
    # Define o tipo esperado por parâmetro
    param_types = {
        'replications': int,
        'model_names': str,
        'epochs': int,
        'learning_rates': float,
        'weight_decays': float,
        'saveModel': lambda x: x.lower() == 'true'  # converte string para booleano
    }

    # Define quais parâmetros são listas
    list_params = {'model_names', 'epochs', 'learning_rates', 'weight_decays'}

    params = {}

    with open(filename, 'r') as f:
        for line in f:
            # ignora linhas malformadas
            if '=' not in line:
                continue 
            
            # separa o parâmetro e o valor
            key, value = line.strip().split('=', 1)

            # Verifica se o parâmetro é conhecido
            if key not in param_types:
                raise ValueError(f"Parâmetro desconhecido: {key}")

            # obtém o tipo esperado para o parâmetro
            value_type = param_types[key]

            if key in list_params:
                # separa os valores por vírgula e aplica o tipo a cada um deles
                raw_values = [v.strip() for v in value.split(',')]
                params[key] = [value_type(v) for v in raw_values]
            else:
                # converte o valor único para o tipo apropriado
                params[key] = value_type(value.strip())

    return params