import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import subprocess
from pathlib import Path


# Check the exact name on Hugging Face. Use the instruction-tuned ("-it") version: the data is in chat format.
model_id = "google/gemma-4-12B"

# The files written by sft_datagen.py.
DATA_DIR = Path(__file__).resolve().parent / "hf_qa_dataset"
dataset = load_dataset("json", data_files={
    "train": str(DATA_DIR / "train.jsonl"),
    "validation": str(DATA_DIR / "validation.jsonl"),
    "test": str(DATA_DIR / "test.jsonl"),
})

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True) # FP32 -> BF16 -> NF4

lora_config = LoraConfig(r=8, lora_alpha=12, lora_dropout=0.05,
                         target_modules="all-linear", task_type="CAUSAL_LM")

args = SFTConfig(
    output_dir="./gemma_qlora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    max_length=1024,
    bf16=True,
    model_init_kwargs={"quantization_config": quant_config},  # loads the model in 4 bits (QLoRA)

    logging_steps=10,
    disable_tqdm=False,

    # Validation: check eval_loss every 50 steps and keep the best checkpoint, not just the last one.
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=args,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model('./gemma_qlora_adapter')

# Loss on the test pages, which the model never saw during training or checkpoint picking.
print(trainer.evaluate(dataset["test"], metric_key_prefix="test"))

base = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='cpu'
)

model = PeftModel.from_pretrained(base, './gemma_qlora_adapter')

merged = model.merge_and_unload()
merged.save_pretrained('./gemma_complete', safe_serialization=True)
# The GGUF converter needs the tokenizer files next to the weights.
AutoTokenizer.from_pretrained(model_id).save_pretrained('./gemma_complete')

# Convert to GGUF.
subprocess.run(['python', 'llama.cpp/convert_hf_to_gguf.py', './gemma_complete', '--outfile', 'gemma-f16.gguf', '--outtype', 'f16'], check=True)

# Create ModelFile.
Path('Modelfile').write_text('FROM ./gemma-f16.gguf\n')

# Import/quantize for Ollama.
subprocess.run(['ollama', 'create', 'gemma_atc_qlora', '-f', 'Modelfile', '--quantize', 'q4_K_M'], check=True)

subprocess.run(['ollama', 'run', 'gemma_atc_qlora'])
