# Import necessary libraries
import os
import json
import numpy as np
import torch
from scipy.stats import entropy

# Tokenization function
def tokenize_function(examples, text_column, tokenizer, max_length):
    return tokenizer(
        examples[text_column], 
        truncation=True, 
        max_length=max_length, 
        padding="max_length"
    )

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed. 
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
    """
    
    # Handle tuple logits (happens when the model is trained using LoRA)
    if isinstance(logits, tuple):
        logits = logits[1]          # logits[0] is the loss value and logits[1] are the logits used to compute loss
                                    # logits: (tensor(2.0426, device='cuda:0'), tensor([[[ 7.8750,  5.3750,  7.0938,  ..., -4.2500, -4.2500, -4.2500],
                                    #          [ 5.0938,  5.0625,  7.3750,  ..., -1.5312, -1.5312, -1.5312],
                                    #          [ 2.6562, -0.9609,  0.0728,  ..., -2.0312, -2.0312, -2.0312],
                                    #          ...,
                                    #          [ 4.1562,  1.4375, -3.6250,  ..., -2.1250, -2.1250, -2.1250],
                                    #          [ 3.7344, -1.6641, -3.8125,  ..., -1.9688, -1.9688, -1.9688],
                                    #          [ 8.1875, -1.2344, -1.6094,  ..., -3.0938, -3.0938, -3.0938]]],
                                    #        device='cuda:0'))

    # Proceed with argmax
    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids

@torch.no_grad()
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    # Cross-Entropy Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.size(-1)).float(), labels.view(-1)).item()
    
    # Perplexity
    perplexity = np.exp(loss)
    
    # Compute Accuracy
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    accuracy = correct.sum().item() / correct.numel()
    
    # Token Distribution Shift (Entropy Difference)
    token_probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
    token_entropy = entropy(token_probs)
    
    return {
        "loss": loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "token_entropy": token_entropy,     # Expected Behavior: Should remain stable or slightly decrease if the model is becoming more confident
    }

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