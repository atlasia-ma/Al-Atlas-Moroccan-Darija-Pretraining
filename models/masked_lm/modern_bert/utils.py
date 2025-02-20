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
  ):
    self.base_model_name="answerdotai/ModernBERT-base"
    self.data_path=data_path
    self.hub_path=hub_path
    self.max_length=max_length
    self.new_vocab_size=new_vocab_size
    self.mlm_probability=0.15
    
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