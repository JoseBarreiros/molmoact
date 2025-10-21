#!/bin/bash

# EgoDex 100% Only Mid-Training Script
# Uses 100% EgoDex data with no other datasets - pure timestep sampling

cd /home/jose-barreiros/jose/molmoact
source molmoact_env/bin/activate

# Set environment variables
export EGODEX_DATA_DIR="/home/jose-barreiros/jose/molmoact/data/egodex/organized"
export PYTHONPATH="/home/jose-barreiros/jose/molmoact:${PYTHONPATH:-}"

# Create timestamp for unique experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 Starting EgoDex 100% only mid-training (pure EgoDex timestep sampling)"
echo "📁 EgoDex data directory: $EGODEX_DATA_DIR"
echo "⏰ Timestamp: $TIMESTAMP"

torchrun --nnodes=1 --nproc-per-node=1 \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-100pct-only allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=egodex_100pct_only_${TIMESTAMP} \
    --wandb.entity=your_wandb_entity \
    --wandb.project=molmoact_egodex \
    --save_folder=checkpoints/egodex_100pct_only_${TIMESTAMP} \
    --save_overwrite \
    --duration 50000 \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 256 \
    --lr_connector 5e-6 \
    --lr_vit 5e-6 \
    --lr_llm 1e-5 \
    --save_interval 10000 \
    --save_num_checkpoints_to_keep 5 \
    --save_final_unsharded_checkpoint \
    --max_images 2
