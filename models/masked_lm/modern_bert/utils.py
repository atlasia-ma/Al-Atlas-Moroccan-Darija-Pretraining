import torch
import math
from transformers import Trainer

def extend_position_embeddings(model, new_max_length):
    """Extend the position embeddings of the model to support longer sequences."""
    current_max_length = model.config.max_position_embeddings
    
    if new_max_length <= current_max_length:
        return model
    
    # Get the current position embeddings
    old_embeddings = model.bert.embeddings.position_embeddings.weight.data
    
    # Create new position embeddings
    new_embeddings = torch.nn.Embedding(new_max_length, old_embeddings.shape[1])
    new_embeddings.to(old_embeddings.device, dtype=old_embeddings.dtype)
    
    # Initialize with the existing weights
    with torch.no_grad():
        # Copy the existing embeddings
        new_embeddings.weight[:current_max_length] = old_embeddings
        
        # Initialize new positions by interpolation or extrapolation
        for i in range(current_max_length, new_max_length):
            # Simple linear extrapolation for positions beyond the original range
            if i < 2 * current_max_length:
                scale = i / current_max_length - 1
                new_embeddings.weight[i] = old_embeddings[current_max_length-1] + scale * (old_embeddings[current_max_length-1] - old_embeddings[0])
            else:
                # For very long contexts, decay the extrapolation to prevent extreme values
                decay = math.exp(-(i - 2 * current_max_length) / current_max_length)
                scale = 1 + (1 - decay)
                new_embeddings.weight[i] = scale * old_embeddings[current_max_length-1]
    
    # Replace the old embeddings with the new ones
    model.bert.embeddings.position_embeddings = new_embeddings
    model.config.max_position_embeddings = new_max_length
    
    # Update position ids
    if hasattr(model.bert.embeddings, 'position_ids'):
        position_ids = torch.arange(new_max_length).expand((1, -1))
        model.bert.embeddings.position_ids = position_ids.to(old_embeddings.device)
    
    return model

def preprocess_dataset(dataset, text_column, instruction_column=None, answer_column=None):
    """Process dataset to extract and prepare text from the specified columns."""
    if instruction_column and answer_column:
        return dataset.map(
            lambda example: {"text": f"{str(example[instruction_column])} {str(example[answer_column])}"},
            remove_columns=dataset.column_names
        )
    else:
        return dataset.map(
            lambda example: {"text": str(example[text_column])},
            remove_columns=dataset.column_names
        )

def tokenize_function_old(examples, tokenizer, max_length):
    """Tokenize text with specified context window."""
    # Fix: Ensure text is a list of strings
    texts = examples["text"]
    if not isinstance(texts, list):
        texts = [texts]
        
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True
    )
    
def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text with specified context window."""
    texts = examples["text"]
    # Convert all elements to strings to handle any non-string entries
    texts = [str(text) for text in texts]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True
    )

def retokenize_dataset(raw_dataset, tokenizer, max_length):
    """Tokenize a raw dataset with a specific max length."""
    return raw_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
        desc=f"Retokenizing dataset with max_length={max_length}"
    )

class GradualContextLengthExtensionTrainer(Trainer):
    """Custom trainer that gradually increases sequence length during training."""
    
    def __init__(
        self,
        initial_max_length: int,
        final_max_length: int,
        total_steps: int,
        tokenizer,
        raw_dataset_args,
        context_lengths_extension_breakpoints,
        context_lengths_extension_sequence,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.initial_max_length = initial_max_length
        self.final_max_length = final_max_length
        self.total_steps = total_steps
        self.tokenizer = tokenizer
        self.current_max_length = initial_max_length
        self.raw_dataset_args = raw_dataset_args
        self.context_lengths_extension_breakpoints = context_lengths_extension_breakpoints
        self.context_lengths_extension_sequence = context_lengths_extension_sequence
    
    def get_current_max_length(self) -> int:
      """Use predefined length breakpoints."""

      progress = self.state.global_step / self.total_steps
      # print(f'lengths: {lengths}')
      # print(f'progress: {progress}')
      
      # Find the appropriate breakpoint
      current_length = self.final_max_length  # Default value
      for i in range(len(self.context_lengths_extension_breakpoints) - 1):
          if self.context_lengths_extension_breakpoints[i] <= progress < self.context_lengths_extension_breakpoints[i + 1]:
              current_length = self.context_lengths_extension_sequence[i]
              break
            
      # print(f'current_length: {current_length}')
      # print('------------------------------')
      return current_length
    
    def _get_train_dataloader(self):
        """Override to update the max length before getting the dataloader."""
        new_length = self.get_current_max_length()
        
        if new_length != self.current_max_length:
            info(f"Updating max length from {self.current_max_length} to {new_length}")
            self.current_max_length = new_length
            
            # Retokenize the dataset with the new length
            self.train_dataset = retokenize_dataset(
                self.raw_dataset_args.train_dataset_raw, 
                self.tokenizer, 
                self.current_max_length
            )
        
        return super()._get_train_dataloader()
    
    def training_step(self, model, inputs, num_items_in_batch):
      """Override to potentially update position embeddings before each step."""
      
      current_length = self.get_current_max_length()
      if current_length != self.current_max_length:
        info(f"Updating max length from {self.current_max_length} to {current_length}")
        self.current_max_length = current_length
        
        # Retokenize the dataset with the new length
        self.train_dataset = retokenize_dataset(
            self.raw_dataset_args.train_dataset_raw, 
            self.tokenizer, 
            self.current_max_length
        )
            
      # Check if we need to update the model's position embeddings
      if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
          if model.config.max_position_embeddings < current_length:
              info(f"Extending position embeddings from {model.config.max_position_embeddings} to {current_length}")
              model = extend_position_embeddings(model, current_length)
          
      # Call the parent class's training_step method
      return super().training_step(model, inputs, num_items_in_batch)
      
class Config:
  def __init__(
    self,
    data_path="atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset",
    hub_path="BounharAbdelaziz/Modern-BERT-Morocco-Darija-Base",
    base_dir="./modernbert_base",
    max_length=8192,
    new_vocab_size=70_000,
    num_train_epochs=3,
    batch_size=8,
    gradient_accumulation_steps=16,
    max_grad_norm=1,
    lr=5e-3,
    warmup_ratio=0.07,
    version="v1",
    logging_steps=50,
    eval_steps=100,
    save_steps=100,
    mlm_probability=0.15,
  ):
    self.base_model_name="answerdotai/ModernBERT-base"
    self.data_path=data_path
    self.hub_path=hub_path
    self.max_length=max_length
    self.new_vocab_size=new_vocab_size
    self.mlm_probability=mlm_probability
    
    # build and check run name    
    self.run_name = f'{self.base_model_name.split("/")[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wp-{warmup_ratio}-gacc-{gradient_accumulation_steps}-gnm-{max_grad_norm}-{version}'
    assert '--' not in self.run_name, f"[WARN] Detected -- in run_name. This will cause a push_to_hub error! Found run_name={self.run_name} "
    assert len(self.run_name) < 96, f"[WARN] run_name too long, found len(run_name)={len(self.run_name)} > 96. This will cause a push_to_hub error! Consider squeezing it. Found run_name={self.run_name}"
    
    self.base_dir=base_dir
    self.output_dir=self.base_dir+f"/{self.run_name}"
    self.num_train_epochs=num_train_epochs
    self.per_device_train_batch_size=batch_size
    self.per_device_eval_batch_size=batch_size
    self.gradient_accumulation_steps=gradient_accumulation_steps
    self.evaluation_strategy="steps"
    self.logging_steps=logging_steps
    self.eval_steps=eval_steps
    self.save_steps=save_steps
    self.save_total_limit=1
    self.learning_rate=lr
    self.warmup_ratio=warmup_ratio
    self.weight_decay=1e-5
    self.report_to="wandb"
    self.run_name="al_atlas_masked_lm"
   
    self.wandb_project_name="al_atlas_masked_lm"
    self.overwrite_output_dir = True
    
    
def info(message):
  print("="*30)
  print(f"[INFO] {message}")
  print("="*30)

# dataset to iterator
def batch_iter(ds,batch_size=1000):
    for i in range(0,len(ds),batch_size):
        yield ds[i:i+batch_size]["text"]

def pre_processing(examples,tokenizer,configs):
    return tokenizer(
      examples["text"],
      truncation=True,
      max_length=configs.max_length
    )