#!/bin/bash

# EgoDex Action Chunk Visualization Script
# Creates visualizations showing hand trajectories for action chunks

cd /home/jose-barreiros/jose/molmoact
source molmoact_env/bin/activate

# Set environment variables
export EGODEX_DATA_DIR="/home/jose-barreiros/jose/molmoact/data/egodex/organized"
export PYTHONPATH="/home/jose-barreiros/jose/molmoact:${PYTHONPATH:-}"

echo "🎨 EgoDex Action Chunk Visualization"
echo "=================================="

# Default parameters
if [ "$1" = "episode" ]; then
    # Episode mode: visualize all timesteps of an episode
    EPISODE_ID=${2:-"35"}
    OUTPUT_DIR=${3:-"egodex_episode_${EPISODE_ID}_visualizations"}
    ACTION_SEQUENCE_LENGTH=${4:-8}
    TASK_NAME=${5:-""}
    
    echo "📊 Episode Mode Parameters:"
    echo "  Episode ID: $EPISODE_ID"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Action chunk size: $ACTION_SEQUENCE_LENGTH"
    if [ -n "$TASK_NAME" ]; then
        echo "  Task name: $TASK_NAME"
    fi
    echo ""
    
    # Build command arguments
    CMD_ARGS=(
        --data_dir "$EGODEX_DATA_DIR"
        --output_dir "$OUTPUT_DIR"
        --episode_id "$EPISODE_ID"
        --action_sequence_length "$ACTION_SEQUENCE_LENGTH"
        --split test
        --data_percentage 0.01
    )
    
    # Add task_name if specified
    if [ -n "$TASK_NAME" ]; then
        CMD_ARGS+=(--task_name "$TASK_NAME")
    fi
    
    # Run the visualization script for the episode
    python scripts/visualize_egodex_action_chunks.py "${CMD_ARGS[@]}"
else
    # Default random sampling mode
    NUM_SAMPLES=${1:-5}
    OUTPUT_DIR=${2:-"egodex_action_chunk_visualizations"}
    ACTION_SEQUENCE_LENGTH=${3:-8}
    
    echo "📊 Random Sampling Parameters:"
    echo "  Number of samples: $NUM_SAMPLES"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Action chunk size: $ACTION_SEQUENCE_LENGTH"
    echo ""
    
    # Run the visualization script
    python scripts/visualize_egodex_action_chunks.py \
        --data_dir "$EGODEX_DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --action_sequence_length "$ACTION_SEQUENCE_LENGTH" \
        --split test \
        --data_percentage 0.01
fi

echo ""
echo "✅ Visualization complete!"
echo "📁 Check the generated images in: $(pwd)/$OUTPUT_DIR"
