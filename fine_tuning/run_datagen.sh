#!/bin/bash
#SBATCH --job-name=sft_datagen
#SBATCH --account=fc_mixedav
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=12:00:00
#SBATCH --output=sft_datagen_%j.log

SCRATCH=/global/scratch/users/$USER
export PATH=$SCRATCH/ollama/bin:$PATH
export OLLAMA_MODELS=$SCRATCH/ollama_models   # keep the 8 GB model out of your home folder
export HF_HOME=$SCRATCH/CITRIS_Aviation_Prize/hf_cache              # same for the embedding model
export OLLAMA_NUM_PARALLEL=3                  # this is where the server setting goes
# Use a port of your own: other people's Ollama servers may be running on the same node.
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))

ollama serve > ollama_$SLURM_JOB_ID.log 2>&1 &
until ollama list > /dev/null 2>&1; do sleep 2; done   # wait until the server is up

ollama pull gemma4:12b   # instant after the first time

module load python   # so python3/venv are available on the compute node
VENV=$SCRATCH/CITRIS_Aviation_Prize/venvs/sft
[ -d "$VENV" ] || python3 -m venv "$VENV"     # create the venv on the first run
source $VENV/bin/activate
cd $SCRATCH/CITRIS_Aviation_Prize/fine_tuning
pip install --quiet --upgrade pip
pip install --quiet -r sft_requirements.txt  # no-op once everything is installed
python sft_datagen.py --pdf data/7110.65BB_Bsc_w_Chg_1_2_and_3_dtd_7-9-26_Final.pdf
