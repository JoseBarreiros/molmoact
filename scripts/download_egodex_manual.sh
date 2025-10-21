#!/bin/bash
# Manual EgoDex download script
# Run this step by step to download EgoDex test data

set -e  # Exit on any error

echo "🚀 Manual EgoDex Download Script"
echo "================================="

# Check if wget is available
if ! command -v wget &> /dev/null; then
    echo "❌ wget not found. Please install wget first:"
    echo "   sudo apt install wget"
    exit 1
fi

# Set variables
OUTPUT_DIR="./egodex_test_data"
MAX_EPISODES=10
TEST_URL="https://ml-site.cdn-apple.com/datasets/egodex/test.zip"

echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Max episodes per task: $MAX_EPISODES"
echo "🌐 Test URL: $TEST_URL"
echo ""

# Create output directory
echo "📁 Creating output directory..."
mkdir -p "$OUTPUT_DIR"

# Download test set
echo "📥 Downloading test set..."
echo "⚠️  This is a large file (~16GB), it may take a while..."
echo "💡 You can press Ctrl+C to cancel if needed"
echo ""

# Ask user if they want to continue
read -p "🤔 Do you want to continue with the download? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Download cancelled by user"
    exit 1
fi

# Download with wget
echo "🚀 Starting download..."
wget --continue \
     --tries 3 \
     --timeout 30 \
     --waitretry 5 \
     --progress dot \
     --output-document "$OUTPUT_DIR/test.zip" \
     "$TEST_URL"

if [ $? -eq 0 ]; then
    echo "✅ Download completed successfully!"
else
    echo "❌ Download failed"
    exit 1
fi

# Extract test set
echo ""
echo "📦 Extracting test set..."
cd "$OUTPUT_DIR"
unzip -o test.zip -d test/
echo "✅ Test set extracted successfully"

# Remove zip file to save space
rm test.zip
echo "🗑️  Test zip removed"

# Go back to project root
cd ..

# Organize data
echo ""
echo "📁 Organizing data..."
mkdir -p "$OUTPUT_DIR/organized/test"

# Move test data to organized structure
if [ -d "$OUTPUT_DIR/test" ]; then
    for task_dir in "$OUTPUT_DIR/test"/*; do
        if [ -d "$task_dir" ]; then
            task_name=$(basename "$task_dir")
            dest="$OUTPUT_DIR/organized/test/$task_name"
            if [ ! -d "$dest" ]; then
                mv "$task_dir" "$dest"
                echo "  ✅ Moved $task_name"
            fi
        fi
    done
fi

# Limit episodes per task
echo "📊 Limiting to $MAX_EPISODES episodes per task..."
for task_dir in "$OUTPUT_DIR/organized/test"/*; do
    if [ -d "$task_dir" ]; then
        task_name=$(basename "$task_dir")
        echo "  📁 Processing $task_name..."
        
        # Count episodes
        episode_count=0
        for episode_dir in "$task_dir"/episode_*; do
            if [ -d "$episode_dir" ]; then
                episode_count=$((episode_count + 1))
            fi
        done
        
        echo "    📊 Found $episode_count episodes"
        
        # Remove excess episodes
        if [ $episode_count -gt $MAX_EPISODES ]; then
            echo "    🗑️  Removing excess episodes..."
            # Sort episodes and remove excess
            ls -1 "$task_dir"/episode_* | tail -n +$((MAX_EPISODES + 1)) | xargs rm -rf
            echo "    ✅ Limited to $MAX_EPISODES episodes"
        fi
    fi
done

# Create environment setup script
echo "📝 Creating environment setup script..."
cat > "$OUTPUT_DIR/organized/setup_egodex_test_env.sh" << 'EOF'
#!/bin/bash
# EgoDex test environment setup for MolmoAct

# Set the EgoDex data directory
export EGODEX_DATA_DIR="$(pwd)"

echo "🧪 EgoDex test data directory set to: $EGODEX_DATA_DIR"
echo "📊 This is a small sample for testing MolmoAct integration"
echo ""
echo "Available test scripts:"
echo "  python scripts/test_egodex_quick.py"
echo "  python scripts/test_egodex_complete.py"
echo "  python scripts/test_egodex_ablation.py"
echo "  python scripts/egodex_conversation_example.py --example-only"
echo "  python scripts/compute_egodex_stats.py"
echo ""
echo "To make this permanent, add to your ~/.bashrc:"
echo "  echo 'export EGODEX_DATA_DIR=\"$(pwd)\"' >> ~/.bashrc"
EOF

chmod +x "$OUTPUT_DIR/organized/setup_egodex_test_env.sh"

echo ""
echo "🎉 EgoDex test data download complete!"
echo "📁 Data location: $OUTPUT_DIR/organized"
echo "📊 Sample size: $MAX_EPISODES episodes per task"
echo ""
echo "Next steps:"
echo "1. Set environment variable: source $OUTPUT_DIR/organized/setup_egodex_test_env.sh"
echo "2. Run test scripts:"
echo "   python scripts/test_egodex_quick.py"
echo "   python scripts/test_egodex_complete.py"
echo ""
echo "✅ Setup complete!"


