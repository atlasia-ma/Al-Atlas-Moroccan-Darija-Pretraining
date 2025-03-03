import os
import torch
from datasets import (
    load_dataset,
    concatenate_datasets,
)

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

import math
from utils import (
    GradualContextLengthExtensionTrainer,
    Config,
    tokenize_function,
    extend_position_embeddings,
    info,
    batch_iter,
    pre_processing,
    preprocess_dataset,
)

import wandb

TRAINING_STAGE = 3
# Starting context length
INITIAL_MAX_LENGTH = 8192 # 1024 8192
# Target context length
FINAL_MAX_LENGTH = 8192 # 8192 16384
# language on which we trained during decay stage
DECAY_LANG = "MSA"
# DECAY_LANG = "ARY"
# DECAY_LANG = "ARZ"
# DECAY_LANG = "TUNISIAN"
# DECAY_LANG = "ALGERIAN"

# Define specific breakpoints
context_lengths_extension_breakpoints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 6 breakpoint steps
context_lengths_extension_sequence = [
    INITIAL_MAX_LENGTH,
    INITIAL_MAX_LENGTH + (FINAL_MAX_LENGTH - INITIAL_MAX_LENGTH) // 5,
    INITIAL_MAX_LENGTH + 2 * (FINAL_MAX_LENGTH - INITIAL_MAX_LENGTH) // 5,
    INITIAL_MAX_LENGTH + 3 * (FINAL_MAX_LENGTH - INITIAL_MAX_LENGTH) // 5,
    INITIAL_MAX_LENGTH + 4 * (FINAL_MAX_LENGTH - INITIAL_MAX_LENGTH) // 5,
    FINAL_MAX_LENGTH
]  
        
ARABIC_TRAIN = True

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if TRAINING_STAGE == 1:
        # train on web-crawled data
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
            version=f"v1-stage-{TRAINING_STAGE}",
            logging_steps=50,
            eval_steps=100,
            save_steps=100,    
            mlm_probability=0.15,
        )
        
        arabic_config = "20231101.ar"
        DATASET_NAME = configs.data_path
        print(f'DATASET_NAME: {DATASET_NAME}')
        print(f'ARABIC_TRAIN: {ARABIC_TRAIN}')
        print(f'arabic_config: {arabic_config}')
        print(f'hub_path: {configs.hub_path}')
        
        info("Loading dataset...")
        if ARABIC_TRAIN:
            dataset=load_dataset(DATASET_NAME) #, arabic_config)
        else:
            dataset=load_dataset(DATASET_NAME)
            
        train_dataset = dataset["train"]
        # test_dataset = dataset["test"]
        
        info("Dataset loaded.")
    
    elif TRAINING_STAGE == 2:
        # train on educational, books, and reasoning (including mathematics) data
        # + context extension 
        # + more proba to mask
        configs=Config(
            hub_path=f"ModernBERT-Arabic-base-stage-2-pre-decay-predef-bkpt-mx{FINAL_MAX_LENGTH}-r2",
            base_dir="./modernbert_ar_101B_base",
            max_length=FINAL_MAX_LENGTH,
            num_train_epochs=1,
            batch_size=4,
            gradient_accumulation_steps=64,
            max_grad_norm=1,
            lr=5e-4,
            warmup_ratio=0, # no warmup, just continue
            version=f"v1-stage-{TRAINING_STAGE}",
            logging_steps=50,
            eval_steps=100,
            save_steps=100,   
            mlm_probability=0.3, # increased difficulty and 0.3 works optimaly for english modernbert
        )
        
        # Load all datasets with their specific columns of interest
        # Books dataset
        books_dataset = load_dataset('alielfilali01/Hindawi-Books-dataset', split='train')
        books_processed = preprocess_dataset(books_dataset, "ChapterText") # ChapterText col
        info("Books dataset loaded and processed.")
        
        # Quran tafseer datasets
        quran_tafseer_1 = load_dataset('MohamedRashad/Quran-Tafseer', split='train')
        quran_tafseer_1_processed = preprocess_dataset(quran_tafseer_1, "tafsir_content") # tafsir_content col
        info("Quran tafseer 1 dataset loaded and processed.")
        
        quran_tafseer_2 = load_dataset('M-A-D/Mixed-Arabic-Datasets-Repo', name='Ara--mustapha--QuranExe', split='train')
        quran_tafseer_2_processed = preprocess_dataset(quran_tafseer_2, "text") # text col
        info("Quran tafseer 2 dataset loaded and processed.")
        
        # Wikipedia dataset
        wikipedia_dataset = load_dataset('M-A-D/Mixed-Arabic-Datasets-Repo', name='Ara--Wikipedia', split='train')
        wikipedia_processed = preprocess_dataset(wikipedia_dataset, "text") # text col
        info("Wikipedia dataset loaded and processed.")
        
        # Reasoning dataset
        reasoning_dataset = load_dataset('Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset', split='train')
        reasoning_processed = preprocess_dataset(reasoning_dataset, None, "instruction", "answer") # concat 'instruction' + 'answer' col
        info("Reasoning dataset loaded and processed.")
        
        # Combine all datasets into one
        info("Combining all datasets...")
        combined_dataset = concatenate_datasets([
            books_processed,
            quran_tafseer_1_processed,
            quran_tafseer_2_processed,
            wikipedia_processed,
            reasoning_processed
        ]) #.select(range(100))
        info(f"Combined dataset created with {len(combined_dataset)} examples.") # 2,072,271,789 tokens
        
        # Create a small validation split for tracking training progress
        combined_dataset = combined_dataset.train_test_split(test_size=0.01, seed=1998)
        train_dataset = combined_dataset["train"]
        val_dataset = combined_dataset["test"]
        info(f"train_dataset: {train_dataset}")
        info(f"val_dataset: {val_dataset}")
        
        
    elif TRAINING_STAGE == 3:
        # decay stage: finetune on a specific language
        configs=Config(
            hub_path=f"ModernBERT-Arabic-base-stage-3-decay-mx{FINAL_MAX_LENGTH}-{DECAY_LANG}",
            base_dir=f"./modernbert_ar_101B_base_decay_{DECAY_LANG}",
            max_length=FINAL_MAX_LENGTH,
            num_train_epochs=1,
            batch_size=16 // 2,
            gradient_accumulation_steps=16 * 2,
            max_grad_norm=1,
            lr=5e-4,
            warmup_ratio=0, # no warmup, decay to 0 instead
            version=f"v1-stage-{TRAINING_STAGE}",
            logging_steps=50,
            eval_steps=100,
            save_steps=100,   
            mlm_probability=0.3, # 0.3 works optimaly for english modernbert
            DECAY_LANG=DECAY_LANG,
        )
        
        # morocco news in arabic
        # https://huggingface.co/datasets/M-A-D/Mixed-Arabic-Datasets-Repo/viewer/Ara--J-Mourad--MNAD.v1?views%5B%5D=ara__j_mourad__mnadv1 
        
        if DECAY_LANG.upper() == "MSA":
        
            # Reasoning dataset
            reasoning_dataset = load_dataset('Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset', split='train')
            reasoning_processed = preprocess_dataset(reasoning_dataset, None, "instruction", "answer") # concat 'instruction' + 'answer' col
            info("Reasoning dataset loaded and processed.")
            
            # Quran tafseer dataset
            quran_tafseer_2 = load_dataset('M-A-D/Mixed-Arabic-Datasets-Repo', name='Ara--mustapha--QuranExe', split='train')
            quran_tafseer_2_processed = preprocess_dataset(quran_tafseer_2, "text") # text col
            info("Quran tafseer 2 dataset loaded and processed.")
            
            # Wikipedia dataset
            wikipedia_dataset = load_dataset('M-A-D/Mixed-Arabic-Datasets-Repo', name='Ara--Wikipedia', split='train')
            wikipedia_processed = preprocess_dataset(wikipedia_dataset, "text") # text col
            info("Wikipedia dataset loaded and processed.")
            
            # News dataset
            news_dataset = load_dataset('M-A-D/Mixed-Arabic-Datasets-Repo', name='Ara--J-Mourad--MNAD.v1', split='train')
            news_processed = preprocess_dataset(news_dataset, "Body") # text col
            info("News dataset loaded and processed.")
            
            # Combine all datasets into one
            info("Combining all datasets...")
            train_dataset = concatenate_datasets([
                reasoning_processed,
                quran_tafseer_2_processed,
                wikipedia_processed,
                news_processed,
            ])
            info(f"Combined dataset created with {len(train_dataset)} examples.") # 545,916,980 tokens
            
        elif DECAY_LANG.upper() == "ARY":
            
            # Al-Atlas dataset
            al_atlas_dataset = load_dataset('atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset', split='train')
            al_atlas_processed = preprocess_dataset(al_atlas_dataset, "text") # text col
            info("Al-Atlas dataset loaded and processed.")
            
            # Combine all datasets into one
            info("Combining all datasets...")
            train_dataset = concatenate_datasets([
                al_atlas_processed,
            ])
            
            info(f"Training dataset created with {len(train_dataset)} examples.") # 273,315,556 tokens
        
        elif DECAY_LANG.upper() == "ARZ":
        
            # Reasoning dataset
            mgb3_dataset = load_dataset('MightyStudent/Egyptian-ASR-MGB-3', split='train')
            mgb3_processed = preprocess_dataset(mgb3_dataset, "sentence") # sentence col
            info("Reasoning dataset loaded and processed.")
            
            # Wikipedia dataset
            wikipedia_dataset = load_dataset('SaiedAlshahrani/Egyptian_Arabic_Wikipedia_20230101', split='train')
            wikipedia_processed = preprocess_dataset(wikipedia_dataset, "text") # text col
            info("Wikipedia dataset loaded and processed.")
            
            # Combine all datasets into one
            info("Combining all datasets...")
            train_dataset = concatenate_datasets([
                mgb3_processed,
                wikipedia_processed,
            ])
            info(f"Combined dataset created with {len(train_dataset)} examples.") # XXX tokens
            
            
        elif DECAY_LANG.upper() == "TUNISIAN":
        
            # Reasoning dataset
            stt_dataset = load_dataset('Arbi-Houssem/Tunisian_dataset_STT-TTS15s_filtred1.0', split='train')
            stt_processed = preprocess_dataset(stt_dataset, "sentence") # sentence col
            info("Reasoning dataset loaded and processed.")
            
            # Combine all datasets into one
            info("Combining all datasets...")
            train_dataset = concatenate_datasets([
                stt_processed,
            ])
            info(f"Combined dataset created with {len(train_dataset)} examples.") # XXX tokens
            
        elif DECAY_LANG.upper() == "ALGERIAN":
        
            # Youtube comments dataset
            ytb_dataset = load_dataset('ayoubkirouane/Algerian-Darija', split='train')
            ytb_processed = preprocess_dataset(ytb_dataset, "Text") # sentence col
            info("Reasoning dataset loaded and processed.")
            
            
            # Combine all datasets into one
            info("Combining all datasets...")
            train_dataset = concatenate_datasets([
                ytb_processed,
            ])
            info(f"Combined dataset created with {len(train_dataset)} examples.") # XXX tokens
            
    # Initialize wandb
    gradual_extension = True if TRAINING_STAGE == 2 else False
    WANDB_CONFIG = {
        "initial_max_length": INITIAL_MAX_LENGTH,
        "final_max_length": FINAL_MAX_LENGTH,
        "gradual_extension": gradual_extension,
        "stage": TRAINING_STAGE
    }
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=configs.wandb_project_name,   
        # Group runs by model size
        group=configs.hub_path,       
        # Unique run name
        name=configs.run_name,
        config=WANDB_CONFIG
    )
   
    # Load tokenizer and tokenize dataset
    if TRAINING_STAGE == 1:
   
        info("Loading base tokenizer...")
        base_tokenizer = AutoTokenizer.from_pretrained(
            configs.base_model_name,
            use_fast=True,
            max_length=1024,
        )

        if os.path.exists(f"{configs.base_dir}/tokenizer"):
            info("Loading the pretrained new tokenizer...")
            tokenizer=AutoTokenizer.from_pretrained(f"{configs.base_dir}/tokenizer",use_fast=True, max_length=1024)
            info("New tokenizer loaded.")
            
        else:
            info("Training new Darija tokenizer")
            train_iterator=batch_iter(dataset["train"])
            tokenizer=base_tokenizer.train_new_from_iterator(
                text_iterator=train_iterator,
                vocab_size=configs.new_vocab_size,
                show_progress=True
            )

            info("Saving the new tokenizer...")
            tokenizer.save_pretrained(f"{configs.base_dir}/tokenizer")
            info("New tokenizer saved.")

            info("Load the new Darija tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(f"{configs.base_dir}/tokenizer",use_fast=True)
            
        info("Tokenizing train/test datasets...")
        train_dataset = train_dataset.map(
            lambda example: pre_processing(example,tokenizer,configs),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        if not ARABIC_TRAIN:
            test_dataset = test_dataset.map(
                lambda example: pre_processing(example,tokenizer,configs),
                batched=True,
                remove_columns=dataset["test"].column_names
            )
    
    elif TRAINING_STAGE == 2:
    
        info(f"[STAGE:{TRAINING_STAGE}]Loading pretrained tokenizer from /home/infres/abounhar/AtlasIA/to_my_github/Al-Atlas-Dataset/models/masked_lm/modern_bert/modernbert_arabic_base/tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(
            f"/home/infres/abounhar/AtlasIA/to_my_github/Al-Atlas-Dataset/models/masked_lm/modern_bert/modernbert_arabic_base/tokenizer",
            use_fast=True
        )
    
        # Initially tokenize with the starting context length
        info(f"Tokenizing datasets with initial max length {INITIAL_MAX_LENGTH}...")
        tokenized_train = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, INITIAL_MAX_LENGTH),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_val = val_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, INITIAL_MAX_LENGTH),
            batched=True,
            remove_columns=val_dataset.column_names
        )

    elif TRAINING_STAGE == 3:
    
        info(f"[STAGE:{TRAINING_STAGE}]Loading pretrained tokenizer from /home/infres/abounhar/AtlasIA/to_my_github/Al-Atlas-Dataset/models/masked_lm/modern_bert/modernbert_arabic_base/tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(
            f"/home/infres/abounhar/AtlasIA/to_my_github/Al-Atlas-Dataset/models/masked_lm/modern_bert/modernbert_arabic_base/tokenizer",
            use_fast=True
        )
    
        # Initially tokenize with the starting context length
        info(f"Tokenizing datasets with initial max length {INITIAL_MAX_LENGTH}...")
        tokenized_train = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, INITIAL_MAX_LENGTH),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        

    # info("Counting total tokens in training dataset...")
    # total_tokens = sum(len(tokenizer(example["text"]).input_ids) for example in train_dataset)
    # info(f"Total tokens in training dataset: {total_tokens}")
    # exit(0)

    info("Initializing data collator...")
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=configs.mlm_probability
    )

    info("Load base model...")
    if TRAINING_STAGE == 1:
        model_config = AutoConfig.from_pretrained(configs.base_model_name)
        model = AutoModelForMaskedLM.from_pretrained(
            configs.base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            config=model_config
        ).to(device)

        info("Resizing embedding matrix...")
        model.resize_token_embeddings(configs.new_vocab_size)
    
    elif TRAINING_STAGE == 2:
        model = AutoModelForMaskedLM.from_pretrained(
            "BounharAbdelaziz/ModernBERT-Arabic-base-stage-1-pre-decay",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        
    elif TRAINING_STAGE == 3:
        model = AutoModelForMaskedLM.from_pretrained(
            "BounharAbdelaziz/ModernBERT-Arabic-base-stage-2-pre-decay-ini-1024-mx-8192",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        
    # Start with the current model's position embedding size
    info(f"Initial model position embeddings: {model.config.max_position_embeddings}")


    # mimic the trapezoidal schedule, we do decay at the ending stage
    if TRAINING_STAGE == 1:
        lr_scheduler_type = "constant_with_warmup"
    elif TRAINING_STAGE == 2:
        lr_scheduler_type = "constant"
    elif TRAINING_STAGE == 3:
        lr_scheduler_type = "linear"
        
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
        lr_scheduler_type= lr_scheduler_type, # mimic the trapezoidal schedule, we do decay at the ending stage
    )
    
        
    if TRAINING_STAGE == 1:
        info("init trainer...")
        trainer=Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset if not ARABIC_TRAIN else None,
            data_collator=data_collator
        )
        info("Done!")
    elif TRAINING_STAGE == 2:
        # Calculate the number of steps for context length
        N_GPUS = torch.cuda.device_count()
        total_steps = math.ceil(
            len(tokenized_train) * configs.num_train_epochs / 
            (configs.per_device_train_batch_size * configs.gradient_accumulation_steps * N_GPUS)
        )
        
        info(f"Total training steps: {total_steps}")
        info(f"Context length will increase gradually over {context_lengths_extension_breakpoints} breakpoints from {INITIAL_MAX_LENGTH} to {FINAL_MAX_LENGTH} as follows: {context_lengths_extension_sequence}")
        
        # Create a simple namespace to store raw datasets for retokenization
        class RawDatasets:
            def __init__(self, train_dataset_raw, val_dataset_raw):
                self.train_dataset_raw = train_dataset_raw
                self.val_dataset_raw = val_dataset_raw
        
        raw_dataset_args = RawDatasets(train_dataset, val_dataset)
        
        info("Initializing custom trainer with gradual context extension...")
        trainer = GradualContextLengthExtensionTrainer(
            initial_max_length=INITIAL_MAX_LENGTH,
            final_max_length=FINAL_MAX_LENGTH,
            total_steps=total_steps,
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            raw_dataset_args=raw_dataset_args,
            context_lengths_extension_breakpoints=context_lengths_extension_breakpoints,
            context_lengths_extension_sequence=context_lengths_extension_sequence,
        )
        
    elif TRAINING_STAGE == 3:
        # Calculate the number of steps for context length
        N_GPUS = torch.cuda.device_count()
        
        info(f"Context length will increase gradually over {context_lengths_extension_breakpoints} breakpoints from {INITIAL_MAX_LENGTH} to {FINAL_MAX_LENGTH} as follows: {context_lengths_extension_sequence}")
        
        info("Initializing custom trainer with gradual context extension...")
        trainer = Trainer(
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
        )

    info("Start training...")
    
    if TRAINING_STAGE == 1:
        info(f"Start initial pretraining with context growing from {INITIAL_MAX_LENGTH} to {FINAL_MAX_LENGTH}...")
        
    elif TRAINING_STAGE == 2:
        info(f"Starting training with gradual context extension from {INITIAL_MAX_LENGTH} to {FINAL_MAX_LENGTH}...")
        
    elif TRAINING_STAGE == 3:
        info(f"Starting decay stage training with context length set to {FINAL_MAX_LENGTH}...")
        
    trainer.train()
    
    info("Saving final model...")
    # Ensure we save with the final extended position embeddings
    if model.config.max_position_embeddings < FINAL_MAX_LENGTH:
        model = extend_position_embeddings(model, FINAL_MAX_LENGTH)
    
    trainer.save_model(configs.output_dir)
    tokenizer.save_pretrained(configs.output_dir)
    
    info("Pushing model to hub...")
    trainer.push_to_hub(configs.hub_path)
    
    info("Training completed successfully with gradual context extension!")