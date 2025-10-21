#!/bin/bash
# EgoDex testing script
# Runs all EgoDex integration tests

set -e  # Exit on any error

echo "🚀 EgoDex Testing Suite"
echo "======================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "💡 Run: source molmoact_env/bin/activate"
    exit 1
fi

# Check if EgoDex data directory is set
if [[ "$EGODEX_DATA_DIR" == "" ]]; then
    echo "⚠️  EGODEX_DATA_DIR not set"
    echo "💡 Run: source egodex_test_data/organized/setup_egodex_test_env.sh"
    exit 1
fi

echo "✅ Environment ready"
echo "📁 Data directory: $EGODEX_DATA_DIR"
echo ""

# Run tests
echo "🧪 Running quick test..."
python scripts/test_egodex_quick.py

echo ""
echo "🧪 Running complete test..."
python scripts/test_egodex_complete.py

echo ""
echo "🧪 Running ablation study..."
python scripts/test_egodex_ablation.py

echo ""
echo "🎉 All tests completed!"
echo ""
echo "Next steps:"
echo "1. Run conversation example: python scripts/egodex_conversation_example.py --example-only"
echo "2. Compute statistics: python scripts/compute_egodex_stats.py"
echo "3. Start training: python launch_scripts/train_multitask_model.py --mixture molmoact-midtrain-egodex"


