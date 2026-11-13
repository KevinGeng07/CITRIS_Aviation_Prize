#!/bin/bash
#SBATCH --job-name=qlora_sft
#SBATCH --account=fc_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=24:00:00
#SBATCH --output=qlora_sft_%j.log

SCRATCH=/global/scratch/users/$USER
export HF_HOME=$SCRATCH/CITRIS_Aviation_Prize/hf_cache   # base model download (~24 GB) stays out of home
export PYTHONUNBUFFERED=1                                # print() shows up in the log right away

module load python
VENV=$SCRATCH/CITRIS_Aviation_Prize/venvs/sft
[ -d "$VENV" ] || python3 -m venv "$VENV"
source $VENV/bin/activate
cd $SCRATCH/CITRIS_Aviation_Prize/fine_tuning
pip install --quiet -r sft_requirements.txt

# llama.cpp, only for its convert_hf_to_gguf.py script (no compiling needed).
# The converter uses its own bundled gguf-py, so its requirements file (which pins a CPU-only torch) is skipped.
export LLAMA_CPP_DIR=$SCRATCH/CITRIS_Aviation_Prize/llama.cpp
[ -d "$LLAMA_CPP_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"

# Ollama server, same setup as run_datagen.sh (the base model gemma4:12b is already in OLLAMA_MODELS).
export PATH=$SCRATCH/ollama/bin:$PATH
export OLLAMA_MODELS=$SCRATCH/ollama_models
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))
ollama serve > ollama_$SLURM_JOB_ID.log 2>&1 &
until ollama list > /dev/null 2>&1; do sleep 2; done
ollama pull gemma4:12b   # its chat template gets copied into the new Modelfile

nvidia-smi
python run_qlora_sft.py
