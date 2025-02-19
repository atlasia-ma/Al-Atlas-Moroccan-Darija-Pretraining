import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from utils import (
    Config,
    info,
    batch_iter,
    pre_processing
)

if __name__ == "__main__":
    
    DATASET_NAME="atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset"
    
    info("Loading dataset...")
    dataset=load_dataset(DATASET_NAME)
    info("Dataset loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    configs=Config()
    info("Loading base tokenizer...")
    base_tokenizer=AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base",
        use_fast=True # Fast tokenizers are implemented in Rust and are significantly faster than the regular Python-based tokenizers.
    )

    # check if the tokenizer is not already trained
    if os.path.exists(f"{configs.base_dir}/tokenizer"):
        info("Loading the pretrained new tokenizer...")
        new_tokenizer=AutoTokenizer.from_pretrained(f"{configs.base_dir}/tokenizer",use_fast=True)
        info("New tokenizer loaded.")
        
    else:
        info("Training new Darija tokenizer")
        train_iterator=batch_iter(dataset["train"])
        new_tokenizer=base_tokenizer.train_new_from_iterator(
            text_iterator=train_iterator,
            vocab_size=configs.new_vocab_size,
            show_progress=True
        )

        info("Saving the new tokenizer...")
        new_tokenizer.save_pretrained(f"{configs.base_dir}/tokenizer")
        info("New tokenizer saved.")

        info("Load the new Darija tokenizer")
        new_tokenizer = AutoTokenizer.from_pretrained(f"{configs.base_dir}/tokenizer",use_fast=True)

    info("Tokenizing train/test datasets...")
    train_dataset=dataset["train"].map(
        lambda example: pre_processing(example,new_tokenizer,configs),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    test_dataset=dataset["test"].map(
        lambda example: pre_processing(example,new_tokenizer,configs),
        batched=True,
        remove_columns=dataset["test"].column_names
    )
    
    # info("Counting total tokens in training dataset...")
    # total_tokens = sum(len(new_tokenizer(example["text"]).input_ids) for example in dataset["train"])
    # info(f"Total tokens in training dataset: {total_tokens}")

    info("Initializing data collator...")
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=new_tokenizer,
        mlm=True,
        mlm_probability=configs.mlm_probability
    )

    info("Loading base model...")
    import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from utils import (
    Config,
    info,
    batch_iter,
    pre_processing
)

import wandb

if __name__ == "__main__":
    configs=Config()
    
    DATASET_NAME = configs.data_path
    
    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=configs.wandb_project_name,   
        # Group runs by model size
        group=configs.hub_path,       
        # Unique run name
        name=configs.run_name,
    )
    
    info("Loading dataset...")
    dataset=load_dataset(DATASET_NAME)
    info("Dataset loaded.")

    info("Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        configs.base_model_name,
        use_fast=True
    )

    if os.path.exists(f"{configs.base_dir}/tokenizer"):
        info("Loading the pretrained new tokenizer...")
        new_tokenizer=AutoTokenizer.from_pretrained(f"{configs.base_dir}/tokenizer",use_fast=True)
        info("New tokenizer loaded.")
        
    else:
        info("Training new Darija tokenizer")
        train_iterator=batch_iter(dataset["train"])
        new_tokenizer=base_tokenizer.train_new_from_iterator(
            text_iterator=train_iterator,
            vocab_size=configs.new_vocab_size,
            show_progress=True
        )

        info("Saving the new tokenizer...")
        new_tokenizer.save_pretrained(f"{configs.base_dir}/tokenizer")
        info("New tokenizer saved.")

        info("Load the new Darija tokenizer")
        new_tokenizer = AutoTokenizer.from_pretrained(f"{configs.base_dir}/tokenizer",use_fast=True)

    info("Tokenizing train/test datasets...")
    train_dataset=dataset["train"].map(
        lambda example: pre_processing(example,new_tokenizer,configs),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    test_dataset=dataset["test"].map(
        lambda example: pre_processing(example,new_tokenizer,configs),
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    # info("Counting total tokens in training dataset...")
    # total_tokens = sum(len(new_tokenizer(example["text"]).input_ids) for example in dataset["train"])
    # info(f"Total tokens in training dataset: {total_tokens}")

    info("Initializing data collator...")
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=new_tokenizer,
        mlm=True,
        mlm_probability=configs.mlm_probability
    )

    info("Load base model...")
    model_config = AutoConfig.from_pretrained(configs.base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        configs.base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        config=model_config
    ).to(device)

    info("Resizing embedding matrix...")
    model.resize_token_embeddings(configs.new_vocab_size)

    info("init training args...")
    training_args = TrainingArguments(
        output_dir=configs.output_dir,
        overwrite_output_dir=configs.overwrite_output_dir,
        num_train_epochs=configs.num_train_epochs,
        per_device_train_batch_size=configs.per_device_train_batch_size,
        per_device_eval_batch_size=configs.per_device_eval_batch_size,
        evaluation_strategy=configs.evaluation_strategy,
        eval_steps=configs.eval_steps,
        logging_steps=configs.logging_steps,
        save_steps=configs.save_steps,
        save_total_limit=configs.save_total_limit,
        learning_rate=configs.learning_rate,
        warmup_ratio=configs.warmup_ratio,
        weight_decay=configs.weight_decay,
        report_to=configs.report_to,
        run_name=configs.run_name,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
    )
    info("Done!")

    info("init trainer...")
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    info("Done!")

    info("trainer...")
    trainer.train()

    info("save result model...")
    trainer.save_model(configs.output_dir)
    new_tokenizer.save_pretrained(configs.output_dir)
    info("push result model to hub...")
    trainer.push_to_hub(configs.hub_path)
    info("Done!")