# Import necessary libraries
import os
import json
import numpy as np
import torch

def set_seed(seed):
    """ Sets the seed for reproducibility """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_params_info(model):
    """
    Prints the total and trainable parameters in the model, 
    along with the percentage reduction in trainable parameters.
    
    Parameters:
    - model: The PyTorch model (could be wrapped with LoRA).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction_percent = (1 - trainable_params / total_params) * 100

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Reduction in Trainable Parameters: {reduction_percent:.2f}%")
    

def save_running_config(config, run_name): 
    base_config_run_path = config['base_config_run_path']
    os.makedirs(base_config_run_path, exist_ok=True)
    
    output_filename = f"{run_name}.json"
    path_to_config_file = os.path.join(base_config_run_path, output_filename)
    
    # Save config as JSON
    with open(path_to_config_file, 'w', encoding='utf-8') as output_file:
        json.dump(config, output_file, indent=4, ensure_ascii=False)  # Pretty-printing for readability
    
    print(f"Configuration saved to {path_to_config_file}")