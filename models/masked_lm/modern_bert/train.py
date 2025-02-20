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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    configs=Config(
        data_path="ClusterlabAi/101_billion_arabic_words_dataset", #"wikimedia/wikipedia", #"atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset",
        hub_path="BounharAbdelaziz/Modern-BERT-Arabic-Base", #"BounharAbdelaziz/Modern-BERT-Morocco-Darija-Base",
        base_dir="./modernbert_ar_101B_base",
        max_length=1024,
        new_vocab_size=70_000,
        num_train_epochs=1,
        batch_size=16,
        gradient_accumulation_steps=16,
        max_grad_norm=1,
        lr=5e-4,
        warmup_ratio=0.05,
        version="v1",
        logging_steps=50,
        eval_steps=100,
        save_steps=100,        
    )
    ARABIC_TRAIN = True
    
    # configs=Config(
    #     data_path="atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset",
    #     hub_path="BounharAbdelaziz/Modern-BERT-Morocco-Darija-Base",
    #     base_dir="./modernbert_darija_base",
    #     num_train_epochs=2,
    #     batch_size=8,
    #     gradient_accumulation_steps=16,
    #     max_grad_norm=1,
    #     lr=5e-3,
    #     warmup_ratio=0.07,
    #     version="v1",
    #     logging_steps=50,
    #     eval_steps=100,
    #     save_steps=100,        
    # )
    # ARABIC_TRAIN = False
    
    arabic_config = "20231101.ar"
    DATASET_NAME = configs.data_path
    print(f'DATASET_NAME: {DATASET_NAME}')
    print(f'ARABIC_TRAIN: {ARABIC_TRAIN}')
    print(f'arabic_config: {arabic_config}')
    print(f'hub_path: {configs.hub_path}')
    
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
    if ARABIC_TRAIN:
        dataset=load_dataset(DATASET_NAME) #, arabic_config)
    else:
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
    if not ARABIC_TRAIN:
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
        evaluation_strategy=configs.evaluation_strategy if not ARABIC_TRAIN else "no",
        eval_steps=configs.eval_steps if not ARABIC_TRAIN else None,
        logging_steps=configs.logging_steps,
        save_steps=configs.save_steps,
        save_total_limit=configs.save_total_limit,
        learning_rate=configs.learning_rate,
        warmup_ratio=configs.warmup_ratio,
        weight_decay=configs.weight_decay,
        report_to=configs.report_to,
        run_name=configs.run_name,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        lr_scheduler_type="constant_with_warmup", # mimic the trapezoidal schedule, we do decay at the end
    )
    info("Done!")

    info("init trainer...")
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if not ARABIC_TRAIN else None,
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