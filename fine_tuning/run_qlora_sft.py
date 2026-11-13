"""
QLoRA fine-tuning on the question-answer pairs from sft_datagen.py.

Reads:   fine_tuning/hf_qa_dataset/{train,validation,test}.jsonl
Writes:  fine_tuning/model_weights/
             checkpoints/   checkpoints saved during training (best one is reloaded at the end)
             adapter/       the trained LoRA weights only (small, needs the base model to run)
             merged/        base model + LoRA merged into one full model, with tokenizer files
             gguf/          merged model as one GGUF file + the Modelfile used for Ollama
And imports the model into Ollama (quantized to q4_K_M) as OLLAMA_NAME.

Run on Savio with:  sbatch run_qlora_sft.sh
Redo only merge + export from an already-trained adapter:  python run_qlora_sft.py --skip-train
"""

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "hf_qa_dataset"
WEIGHTS_DIR = HERE / "model_weights"
CHECKPOINT_DIR = WEIGHTS_DIR / "checkpoints"
ADAPTER_DIR = WEIGHTS_DIR / "adapter"
MERGED_DIR = WEIGHTS_DIR / "merged"
GGUF_DIR = WEIGHTS_DIR / "gguf"
GGUF_FILE = GGUF_DIR / "model-f16.gguf"
MODELFILE = GGUF_DIR / "Modelfile"

# llama.cpp checkout that has convert_hf_to_gguf.py (run_qlora_sft.sh clones it here).
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", HERE.parent / "llama.cpp"))

# The Ollama model the questions were generated with: its chat template and settings are reused.
BASE_OLLAMA_MODEL = "gemma4:12b"
OLLAMA_NAME = "gemma4_atc_qlora"

# Instruction-tuned ("-it") version: the data is in chat format. Check the exact name on Hugging Face.
MODEL_ID = "google/gemma-4-12b-it"

# The RTX 2080 Ti (Turing) has no bfloat16 support, so everything on the GPU runs in float16.
GPU_DTYPE = torch.float16


def load_data():
    files = {split: DATA_DIR / f"{split}.jsonl" for split in ("train", "validation", "test")}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Run sft_datagen.py first. Missing: " + ", ".join(missing))

    dataset = load_dataset("json", data_files={split: str(p) for split, p in files.items()})
    for split in files:
        print(f"{split}: {len(dataset[split])} pairs")
    return dataset


def train(model_id, dataset, epochs):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=GPU_DTYPE,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,  # loads the base model in 4 bits (QLoRA)
        dtype=GPU_DTYPE,
        device_map={"": 0},
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=12,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,  # trades some speed for a lot less GPU memory (2080 Ti = 11 GB)
        optim="paged_adamw_8bit",  # 8-bit optimizer state, moved to CPU when the GPU runs short
        learning_rate=3e-4,
        max_length=512,  # the QA pairs are short; shorter sequences = less activation memory
        fp16=True,  # no bf16 on Turing
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
        # Validation: check eval_loss every 50 steps and keep the best checkpoint, not just the last one.
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,  # each checkpoint holds optimizer state too; don't fill up scratch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=args,
        peft_config=lora_config,
    )

    trainer.train()

    # Saves only the LoRA weights (plus config) of the best checkpoint.
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")

    # Loss on the test pages, which the model never saw during training or checkpoint picking.
    print(trainer.evaluate(dataset["test"], metric_key_prefix="test"))

    # Free the GPU before merging.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def merge(model_id):
    # The 4-bit model can't be merged cleanly, so reload the base model in full bf16 on the CPU
    # (about 2 bytes per parameter of RAM, ~24 GB for a 12B model) and merge the adapter into it.
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    merged = model.merge_and_unload()

    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    # The GGUF converter needs the tokenizer files next to the weights.
    AutoTokenizer.from_pretrained(model_id).save_pretrained(str(MERGED_DIR))
    print(f"Saved merged model to {MERGED_DIR}")

    del base, model, merged
    gc.collect()


def to_gguf():
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"{converter} not found. Clone llama.cpp there or set LLAMA_CPP_DIR.")

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(converter), str(MERGED_DIR), "--outfile", str(GGUF_FILE), "--outtype", "f16"],
        check=True,
    )
    print(f"Saved GGUF to {GGUF_FILE}")


def write_modelfile(base_ollama_model):
    """Copies the base model's Modelfile (chat template, stop tokens, sampling settings) and
    points it at our GGUF. A bare "FROM model.gguf" would lose the chat template."""
    base = subprocess.run(
        ["ollama", "show", base_ollama_model, "--modelfile"], check=True, capture_output=True, text=True
    ).stdout

    kept = [line for line in base.splitlines() if not line.strip().upper().startswith("FROM ")]
    # Relative path: Ollama resolves it against the Modelfile's folder, so gguf/ can be moved as a whole.
    MODELFILE.write_text(f"FROM ./{GGUF_FILE.name}\n" + "\n".join(kept) + "\n")
    print(f"Wrote {MODELFILE}")


def to_ollama(base_ollama_model, ollama_name):
    write_modelfile(base_ollama_model)
    subprocess.run(
        ["ollama", "create", ollama_name, "-f", str(MODELFILE), "--quantize", "q4_K_M"], check=True
    )
    print(f"Created Ollama model {ollama_name}. Try it: ollama run {ollama_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--no-merge", action="store_true", help="only save the LoRA adapter")
    parser.add_argument("--no-export", action="store_true", help="stop after merging (no GGUF / Ollama)")
    parser.add_argument("--skip-train", action="store_true", help="reuse model_weights/adapter; only merge + export")
    parser.add_argument("--base-ollama", default=BASE_OLLAMA_MODEL)
    parser.add_argument("--ollama-name", default=OLLAMA_NAME)
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_train:
        if not (ADAPTER_DIR / "adapter_config.json").is_file():
            parser.error(f"--skip-train needs a trained adapter in {ADAPTER_DIR}")
    else:
        dataset = load_data()
        train(args.model, dataset, args.epochs)

    if args.no_merge:
        return
    merge(args.model)

    if args.no_export:
        return
    to_gguf()
    to_ollama(args.base_ollama, args.ollama_name)


if __name__ == "__main__":
    main()
