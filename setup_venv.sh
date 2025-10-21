#!/bin/bash
# MolmoAct venv setup script

set -e  # Exit on any error

echo "🚀 Setting up MolmoAct with venv..."

# Check if Python 3.11 is available
if command -v python3.11 &> /dev/null; then
    echo "✅ Python 3.11 found"
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    echo "⚠️  Python 3.11 not found, using Python 3.13 (not officially supported)"
    PYTHON_CMD="python3"
else
    echo "❌ No Python 3 found. Please install Python 3.11 or 3.13"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv molmoact_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source molmoact_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch>=2.3.1 torchvision>=0.18.1 --index-url https://download.pytorch.org/whl/cu124

# Install MolmoAct
echo "🤖 Installing MolmoAct..."
pip install -e .[all]

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install h5py opencv-python pillow

# Test installation
echo "🧪 Testing installation..."
python -c "import olmo; print('✅ MolmoAct imported successfully')"
python -c "from olmo.data.egodex_dataset import EgoDexPoseActions; print('✅ EgoDex integration working')"
python -c "import torch; print(f'✅ PyTorch {torch.__version__} imported'); print(f'✅ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "🎉 Setup complete! To activate the environment, run:"
echo "   source molmoact_env/bin/activate"
echo ""
echo "To test EgoDex integration, run:"
echo "   python scripts/egodex_conversation_example.py --example-only"


