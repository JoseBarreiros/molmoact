#!/bin/bash
# GCP Instance Setup Script for MolmoAct + EgoDex

set -e

echo "🚀 Setting up GCP instance for MolmoAct + EgoDex"

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected. Please ensure you're using a GPU-enabled GCP instance."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1-2)
echo "📊 Detected CUDA Version: $CUDA_VERSION"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    tmux \
    vim \
    python3.11 \
    python3.11-venv \
    python3.11-pip \
    python3.11-dev

# Verify Python
echo "🐍 Verifying Python installation..."
python3.11 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv molmoact_env
source molmoact_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.4
echo "🔥 Installing PyTorch with CUDA 12.4..."
pip install torch>=2.3.1 torchvision>=0.18.1 --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
echo "🧪 Testing PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Install MolmoAct dependencies
echo "🤖 Installing MolmoAct..."
pip install -e .[all]

# Install additional EgoDex dependencies
echo "📚 Installing EgoDex dependencies..."
pip install h5py opencv-python pillow psutil

# Test EgoDex integration
echo "🧪 Testing EgoDex integration..."
python -c "
try:
    from olmo.data.egodex_dataset import EgoDexPoseActions
    print('✅ EgoDex integration imported successfully')
except ImportError as e:
    print(f'❌ EgoDex integration failed: {e}')
"

# Create useful aliases
echo "⚙️  Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# MolmoAct aliases
alias activate_molmoact='source molmoact_env/bin/activate'
alias test_egodex='python scripts/test_egodex_quick.py'
alias stats_egodex='python scripts/compute_egodex_stats.py'

EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: source molmoact_env/bin/activate"
echo "2. Set EgoDex data directory: export EGODEX_DATA_DIR=/path/to/egodex/data"
echo "3. Run quick test: python scripts/test_egodex_quick.py"
echo ""
echo "💡 Useful commands:"
echo "- nvidia-smi                    # Check GPU status"
echo "- activate_molmoact            # Activate environment (after sourcing ~/.bashrc)"
echo "- test_egodex                  # Run EgoDex quick test"
echo "- stats_egodex                 # Compute EgoDex statistics"
echo ""
echo "🐛 Troubleshooting:"
echo "- If CUDA issues: Restart instance after setup"
echo "- If import errors: Check virtual environment is activated"
echo "- If GPU not detected: Verify GCP instance has GPU attached"
