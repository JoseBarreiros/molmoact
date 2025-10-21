#!/bin/bash

# EgoDex Scaling Laws Training: 25% EgoDex + 10% MolmoAct
# For scaling laws analysis

cd /home/jose-barreiros/jose/molmoact
source molmoact_env/bin/activate

# Set environment variables
export EGODEX_DATA_DIR="/home/jose-barreiros/jose/molmoact/data/egodex/organized"
export EGODEX_DATA_PERCENTAGE=0.25  # Use 25% of EgoDex data
export PYTHONPATH="/home/jose-barreiros/jose/molmoact:${PYTHONPATH:-}"  # Fix module import issue

# Create timestamp for unique experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)_25pct

echo "🚀 Starting EgoDex Scaling Laws Training (25% EgoDex)"
echo "📊 EgoDex percentage: 25% of available data"
echo "📊 MolmoAct percentage: 10% (fixed)"
echo "🆔 Run ID: scaling_25pct_${TIMESTAMP}"
echo "📁 EgoDex data directory: $EGODEX_DATA_DIR"

torchrun --nnodes=1 --nproc-per-node=1 \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-scaling-25pct allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=egodex_scaling_25pct_${TIMESTAMP} \
    --wandb.entity=your_wandb_entity \
    --wandb.project=molmoact_egodex_scaling_laws \
    --save_folder=checkpoints/egodex_scaling_25pct_${TIMESTAMP} \
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
