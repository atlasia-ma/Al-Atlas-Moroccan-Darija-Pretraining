from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import yaml
import wandb
import torch
from datasets import load_dataset
import os
from pprint import pprint


from helpers import(
    set_seed,
    save_running_config,
)

def process_function(examples):
  return tokenizer(examples["text"],truncation=True)
        
if __name__ == "__main__":
    
    # Set up logging and tracking
    wandb.login()
    
    # get training configuration
    with open('training_config.yaml') as file:
        config = yaml.safe_load(file)
    
    print('-'*50)
    print("Training configuration:")
    pprint(config)
    print('-'*50)
    
    # Training hyperparameters
    num_train_epochs = config['hyperparameters']['num_train_epochs']
    lr = config['hyperparameters']['lr']
    batch_size = config['hyperparameters']['batch_size']
    gradient_accumulation_steps = config['hyperparameters']['gradient_accumulation_steps']
    max_grad_norm = config['hyperparameters']['max_grad_norm']
    warmup_steps = config['hyperparameters']['warmup_steps']
    warmup_ratio = config['hyperparameters']['warmup_ratio']
    MAX_LEN = config['hyperparameters']['MAX_LEN']
    
    # Logging and saving
    logging_steps = config['hyperparameters']['logging_steps']
    save_steps = config['hyperparameters']['save_steps']
    eval_steps = config['hyperparameters']['eval_steps']

    # Training data path
    TRAIN_DATA_PATH = config['DATASET_PATH']
    
    # base model path
    MODEL_PATH = config['BASE_MODEL']
    
    # training precision
    FP16_TRAINING = config['FP16_TRAINING']
    
    if FP16_TRAINING:
        torch_dtype=torch.bfloat16 # bfloat16 has better precission than float16 thanks to bigger mantissa. Though not available with all GPUs architecture.
    else:
        torch_dtype=torch.float32
        
    # set seed
    SEED = config['SEED']
    set_seed(SEED)
    
    # load dataset
    train_dataset = load_dataset(TRAIN_DATA_PATH, split='train')
    test_dataset = load_dataset(TRAIN_DATA_PATH, split='test')
    
    train_dataset = train_dataset.remove_columns(["metadata","dataset_source"])

    # Load model config
    model_config = AutoConfig.from_pretrained(MODEL_PATH)

    # Enable FlashAttention 2
    model_config.use_flash_attention_2 = True  
    
    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        config=model_config
    ).to(device)
    
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Set reasonable default for models without max length
    tokenizer.model_max_length = config['hyperparameters']['MAX_LEN']

    # Set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # data collator takes care of masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True  # Set to True for Masked LM
    )
    
    # Project name for loggings and savings
    project_name = config['project_name']
    fp16 = '-FP16' if FP16_TRAINING else ''
    
    run_name = f'{MODEL_PATH.split("/")[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wp-{warmup_ratio}-gacc-{gradient_accumulation_steps}-gnm-{max_grad_norm}{fp16}-mx-{config['hyperparameters']['MAX_LEN']}-{config['version']}'
    assert '--' not in run_name, f"[WARN] Detected -- in run_name. This will cause a push_to_hub error! Found run_name={run_name} "
    assert len(run_name) < 96, f"[WARN] run_name too long, found len(run_name)={len(run_name)} > 96. This will cause a push_to_hub error! Consider squeezing it. Found run_name={run_name}"

    # Where to save the model
    MODEL_RUN_SAVE_PATH = f"BounharAbdelaziz/{run_name}"
    
    # Save the configuration to a .json file
    save_running_config(config, run_name)
    
    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=project_name,   
        # Group runs by model size
        group=MODEL_PATH,       
        # Unique run name
        name=run_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataset": TRAIN_DATA_PATH,
        }
    )
    
    # tokenize dataset
    tokenized_train = train_dataset.map(process_function, batched=True, num_proc=os.cpu_count())
    tokenized_test = test_dataset.map(process_function, batched=True, num_proc=os.cpu_count())
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_RUN_SAVE_PATH,
        eval_strategy="steps",
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
        bf16=config['FP16_TRAINING'],
        fp16_full_eval=config['FP16_TRAINING'],
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        report_to="wandb",
        push_to_hub=False,
        metric_for_best_model=config['METRIC_FOR_BEST_MODEL'],
        gradient_checkpointing=True,
        # use_cache = False, # as we set gradient_checkpointing=True
        load_best_model_at_end=True,
        optim=config['hyperparameters']['optimizer'],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config['hyperparameters']['USE_LORA'] else None,  # Avoids gradient issues in backprop when LoRA is set to True. # https://discuss.huggingface.co/t/how-to-combine-lora-and-gradient-checkpointing-in-whisper/50629
        # torch_compile=True, # Enable torch compile for faster training
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Push to Hugging Face Hub
    print("[INFO] Preparing to push to hub...")

    if config['hyperparameters']['USE_LORA']:
        print("[INFO] Merging LoRA weights before pushing...")
        from peft import merge_and_unload
        model = merge_and_unload(model)
        
    # Save the model and tokenizer locally before pushing
    trainer.save_model(MODEL_RUN_SAVE_PATH)  # This saves the model, tokenizer, and config
    tokenizer.save_pretrained(MODEL_RUN_SAVE_PATH)

    # Push to the hub
    print("[INFO] Pushing model and tokenizer to Hugging Face Hub...")
    trainer.push_to_hub()
    tokenizer.push_to_hub(MODEL_RUN_SAVE_PATH)
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f'[INFO] Results on test set: {test_results}')