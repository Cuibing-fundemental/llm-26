#!/usr/bin/env bash
set -e
cd /mnt/data/baojian/backup-git/llm-26/lecture-08-sft-rm-ppo

export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

LOG="run_full_$(date +%Y%m%d_%H%M%S).log"
echo "$(date)  Starting full-data SFT/RM/PPO run on GPU 1" | tee "$LOG"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a "$LOG"

.venv/bin/jupyter nbconvert \
    --to notebook \
    --execute lecture-08-sft-rm-ppo.ipynb \
    --output lecture-08-sft-rm-ppo.executed.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    2>&1 | tee -a "$LOG"

echo "$(date)  Run complete." | tee -a "$LOG"
