#!/bin/bash

# Test EgoDex Scaling Laws Training: 25% EgoDex + 10% MolmoAct
# Short test run to verify configuration

cd /home/jose-barreiros/jose/molmoact
source molmoact_env/bin/activate

# Set environment variables
export EGODEX_DATA_DIR="/home/jose-barreiros/jose/molmoact/data/egodex/organized"
export EGODEX_DATA_PERCENTAGE=0.25  # Use 25% of EgoDex data
export PYTHONPATH="/home/jose-barreiros/jose/molmoact:${PYTHONPATH:-}"  # Fix module import issue

# Create timestamp for unique experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)_test25pct

echo "🚀 Starting EgoDex Scaling Laws TEST (25% EgoDex)"
echo "📊 EgoDex percentage: 25% of available data"
echo "📊 MolmoAct percentage: 10% (fixed)"
echo "🆔 Run ID: test_scaling_25pct_${TIMESTAMP}"
echo "📁 EgoDex data directory: $EGODEX_DATA_DIR"
echo "🔧 Test duration: 100 steps (short test)"
echo ""

# Test the configuration first
echo "=== Testing Configuration ==="
python -c "
import os
print(f'EGODEX_DATA_DIR: {os.environ.get(\"EGODEX_DATA_DIR\", \"NOT SET\")}')
print(f'EGODEX_DATA_PERCENTAGE: {os.environ.get(\"EGODEX_DATA_PERCENTAGE\", \"NOT SET\")}')

# Test dataset loading
try:
    from olmo.data.get_dataset import get_dataset_by_name
    print('✅ Dataset loading module imported successfully')
except ImportError as e:
    print(f'❌ Failed to import dataset module: {e}')
"

echo ""
echo "=== Starting Training Test ==="

torchrun --nnodes=1 --nproc-per-node=1 \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-scaling-25pct allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=test_egodex_scaling_25pct_${TIMESTAMP} \
    --wandb=null \
    --save_folder=checkpoints/test_egodex_scaling_25pct_${TIMESTAMP} \
    --save_overwrite \
    --duration 100 \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size 8 \
    --lr_connector 5e-6 \
    --lr_vit 5e-6 \
    --lr_llm 1e-5 \
    --save_interval 50 \
    --save_num_checkpoints_to_keep 2 \
    --save_final_unsharded_checkpoint \
    --max_images 2 \
    --turn_off_inference

echo ""
echo "✅ Test completed! Check the logs above for any issues."
echo "📁 Checkpoint saved to: checkpoints/test_egodex_scaling_25pct_${TIMESTAMP}"
