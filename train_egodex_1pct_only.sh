#!/bin/bash

# EgoDex 1% Only Mid-Training Script
# Uses 100% EgoDex batch balance but only 1% of available EgoDex data for quick testing

cd /home/jose-barreiros/molmoact
source molmoact_env/bin/activate

# Set environment variables
export EGODEX_DATA_DIR="/home/jose-barreiros/egodex/organized"
export EGODEX_DATA_PERCENTAGE="0.01"  # Use only 1% of available EgoDex data
export PYTHONPATH="/home/jose-barreiros/molmoact:${PYTHONPATH:-}"

# CUDA debugging and memory management
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create timestamp for unique experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 Starting EgoDex 1% only mid-training (100% EgoDex batch balance, 1% of data)"
echo "📁 EgoDex data directory: $EGODEX_DATA_DIR"
echo "📊 EgoDex data percentage: $EGODEX_DATA_PERCENTAGE (1% of available data)"
echo "⏰ Timestamp: $TIMESTAMP"

# Start with single GPU to isolate the issue
torchrun --nnodes=1 --nproc-per-node=4 \
    launch_scripts/train_multitask_model.py \
    egodex-pose-only allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=egodex_1pct_only_${TIMESTAMP} \
    --wandb.entity=jose-barreiros-879-jb \
    --wandb.project=molmoact_egodex \
    --save_folder=checkpoints/egodex_1pct_only_${TIMESTAMP} \
    --save_overwrite \
    --duration 50000 \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 128 \
    --lr_connector 5e-6 \
    --lr_vit 5e-6 \
    --lr_llm 1e-5 \
    --save_interval 1000 \
    --save_num_checkpoints_to_keep 3 \
    --save_final_unsharded_checkpoint \
    --max_images 2 