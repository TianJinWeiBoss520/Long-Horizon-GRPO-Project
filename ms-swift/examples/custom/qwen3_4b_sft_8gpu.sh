#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash examples/custom/qwen3_4b_sft_8gpu.sh
#
# Notes:
# - This script uses DDP on 8 GPUs (no deepspeed dependency required).
# - Validation file uses "answer" key, so we map it to "messages" via --columns.

source .venv/bin/activate

NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$NPROC_PER_NODE \
PYTHONPATH=. \
python swift/cli/sft.py \
  --model /home/linbiyuan/models/Qwen3-4B \
  --tuner_type lora \
  --dataset /home/wangyin/travel_agent/data/data/train_sft_fix_v2.jsonl \
  --val_dataset /home/wangyin/travel_agent/data/data/test.jsonl \
  --columns '{"answer":"messages"}' \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --load_best_model_at_end false \
  --early_stop_interval 6 \
  --logging_steps 5 \
  --max_length 8192 \
  --truncation_strategy left \
  --dataset_num_proc 8 \
  --dataloader_num_workers 0 \
  --output_dir output/qwen3_4b_lora_sft_8gpu
