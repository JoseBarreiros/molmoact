#!/bin/bash

# EgoDex Episode Action Chunk Visualization Script
# Shows all action chunks from an episode overlaid on one plot

cd /home/jose-barreiros/jose/molmoact
source molmoact_env/bin/activate

# Set environment variables
export EGODEX_DATA_DIR="/home/jose-barreiros/jose/molmoact/data/egodex/organized"
export PYTHONPATH="/home/jose-barreiros/jose/molmoact:${PYTHONPATH:-}"

echo "🎨 EgoDex Episode Action Chunk Visualization"
echo "============================================="

# Default parameters
EPISODE_ID=${1:-"0"}
TASK_NAME=${2:-""}
OUTPUT_DIR=${3:-"episode_action_chunks"}
MAX_TIMESTEPS=${4:-50}
ACTION_SEQUENCE_LENGTH=${5:-8}

echo "📊 Parameters:"
echo "  Episode ID: $EPISODE_ID"
if [ -n "$TASK_NAME" ]; then
    echo "  Task name: $TASK_NAME"
fi
echo "  Output directory: $OUTPUT_DIR"
echo "  Max timesteps: $MAX_TIMESTEPS"
echo "  Action chunk size: $ACTION_SEQUENCE_LENGTH"
echo ""

# Build command arguments
CMD_ARGS=(
    --data_dir "$EGODEX_DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --episode_id "$EPISODE_ID"
    --max_timesteps "$MAX_TIMESTEPS"
    --action_sequence_length "$ACTION_SEQUENCE_LENGTH"
    --split test
    --data_percentage 0.01
)

# Add task_name if specified
if [ -n "$TASK_NAME" ]; then
    CMD_ARGS+=(--task_name "$TASK_NAME")
fi

# Run the visualization script
python scripts/visualize_episode_action_chunks.py "${CMD_ARGS[@]}"

echo ""
echo "✅ Episode visualization complete!"
echo "📁 Check the generated plot in: $(pwd)/$OUTPUT_DIR"
echo ""
echo "💡 Usage examples:"
echo "  ./visualize_episode_chunks.sh 0                    # Episode 0, all tasks"
echo "  ./visualize_episode_chunks.sh 0 add_remove_lid     # Episode 0, specific task"
echo "  ./visualize_episode_chunks.sh 1 basic_fold 50      # Episode 1, basic_fold task, 50 timesteps"
