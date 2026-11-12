import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import subprocess


model_id = "google/gemma-4-12B"

dataset = load_dataset("json", data_files="train.jsonl", split="train")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.blofat16,
    bnb_4bit_use_double_quant=True) # FP32 -> BF16 -> NF4

lora_config = LoraConfig(r=8, lora_alpha=12, lora_dropout=0.05,
                         target_modules="all-linear", task_type="CAUSAL_LM")

args = SFTConfig(
    output_dir="./gemma_qlora",
    num_train_epochs=3,
    per_device_trian_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    max_length=1024,
    bf16=True,

    logging_steps=10,
    logging_dir='./logs/gemma',
    disable_tqdm=False,
    eval_steps=50
)

trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    args=args,
    peft_config=lora_config,
    quantization_config=quant_config
)

trainer.train()
trainer.save_model('./gemma_qlora_adapter')

base = AutoModelForCausalLM.from_pretrained(
    'google/gemma-4-12B',
    torch_dtype=torch.float16,
    device_map='cpus'
)

model = PeftModel.from_pretrained(base, './gemma_qlora_adapter')

merged = model.merge_and_unload()
merged.save_pretrained('./gemma_complete', safe_serialization=True)

# Convert to GGUF.
subprocess.run(['python', 'llama.cpp/convert_hf_to_gguf.py', './gemma_merged', '--outfile', 'gemma-f16.gguf', '--outtype', 'f16'])

# Create ModelFile.
subprocess.run(['FROM', './gemma-f16.gguf'])

# Import/quantize for Ollama.
subprocess.run(['ollama', 'create', 'gemma_atc_qlora', '-f', 'Modelfile', '--quantize', 'q4_K_M'])

subprocess.run(['ollama', 'run', 'gemma_atc_qlora'])
