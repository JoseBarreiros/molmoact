#!/usr/bin/env python3
"""
EgoDex download script using wget with resume capability and better error handling.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import shutil


def download_with_wget(url: str, output_file: Path, max_retries: int = 3):
    """
    Download a file using wget with retry logic and resume capability.
    
    Args:
        url: URL to download
        output_file: Path to save the file
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        print(f"📥 Download attempt {attempt + 1}/{max_retries}")
        print(f"   URL: {url}")
        print(f"   Output: {output_file}")
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build wget command with resume capability and better error handling
        cmd = [
            'wget',
            '--continue',  # Resume partial downloads
            '--tries', '3',  # Number of retries
            '--timeout', '30',  # Timeout for each attempt
            '--waitretry', '5',  # Wait between retries
            '--progress', 'bar',  # Show progress bar
            '--output-document', str(output_file),
            url
        ]
        
        try:
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Download successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Download failed (attempt {attempt + 1}): {e}")
            if e.stderr:
                print(f"   Error details: {e.stderr}")
            
            # If this isn't the last attempt, wait before retrying
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                print(f"   ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ All {max_retries} attempts failed")
                return False
    
    return False


def download_egodex_test_wget(output_dir: str, max_episodes: int = 10):
    """
    Download EgoDex test data using wget.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Downloading EgoDex test data using wget...")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Max episodes per task: {max_episodes}")
    
    # Download test set
    print("\n📥 Downloading test set...")
    test_url = 'https://ml-site.cdn-apple.com/datasets/egodex/test.zip'
    test_zip = output_path / "test.zip"
    
    success = download_with_wget(test_url, test_zip, max_retries=3)
    
    if not success:
        print("❌ Failed to download test set after all retries")
        return False
    
    # Extract test set
    print("\n📦 Extracting test set...")
    test_dir = output_path / "test"
    test_dir.mkdir(exist_ok=True)
    
    try:
        subprocess.run(['unzip', '-o', str(test_zip), '-d', str(test_dir)], check=True)
        print("✅ Test set extracted successfully")
        
        # Remove zip file to save space
        test_zip.unlink()
        print("🗑️  Test zip removed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to extract test set: {e}")
        return False
    
    # Organize data
    print("\n📁 Organizing data...")
    organized_dir = organize_test_data(output_path, max_episodes)
    
    # Create environment setup
    create_test_setup(organized_dir)
    
    print("\n" + "="*60)
    print("🎉 EgoDex test data download complete!")
    print(f"📁 Data location: {organized_dir}")
    print(f"📊 Sample size: {max_episodes} episodes per task")
    print("\nNext steps:")
    print("1. Set environment variable: export EGODEX_DATA_DIR='{organized_dir}'")
    print("2. Run test scripts:")
    print("   python scripts/test_egodex_quick.py")
    print("   python scripts/test_egodex_complete.py")
    print("="*60)
    
    return True


def organize_test_data(data_dir: Path, max_episodes: int):
    """Organize the test data into train/test structure."""
    organized_dir = data_dir / "organized"
    test_dir = organized_dir / "test"
    
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Process test data
    test_source = data_dir / "test"
    if test_source.exists():
        print("📁 Organizing test data...")
        for task_dir in test_source.iterdir():
            if task_dir.is_dir():
                dest = test_dir / task_dir.name
                if not dest.exists():
                    shutil.move(str(task_dir), str(dest))
                    print(f"  ✅ Moved {task_dir.name}")
    
    # Limit episodes per task
    print(f"📊 Limiting to {max_episodes} episodes per task...")
    for task_dir in test_dir.iterdir():
        if task_dir.is_dir():
            limit_episodes_in_task(task_dir, max_episodes)
    
    return str(organized_dir)


def limit_episodes_in_task(task_dir: Path, max_episodes: int):
    """Limit the number of episodes in a task directory."""
    episodes = []
    
    # Find all episode directories
    for item in task_dir.iterdir():
        if item.is_dir() and item.name.startswith('episode_'):
            episodes.append(item)
    
    if len(episodes) <= max_episodes:
        return
    
    # Sort episodes and keep only the first max_episodes
    episodes.sort(key=lambda x: x.name)
    episodes_to_remove = episodes[max_episodes:]
    
    for episode_dir in episodes_to_remove:
        shutil.rmtree(episode_dir)
        print(f"    🗑️  Removed {episode_dir.name}")


def create_test_setup(data_dir: str):
    """Create environment setup for test data."""
    setup_script = Path(data_dir) / "setup_egodex_test_env.sh"
    
    content = f"""#!/bin/bash
# EgoDex test environment setup for MolmoAct

# Set the EgoDex data directory
export EGODEX_DATA_DIR="{data_dir}"

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
echo "  echo 'export EGODEX_DATA_DIR=\"{data_dir}\"' >> ~/.bashrc"
"""
    
    with open(setup_script, 'w') as f:
        f.write(content)
    
    setup_script.chmod(0o755)
    print(f"📝 Created test setup script: {setup_script}")


def main():
    parser = argparse.ArgumentParser(description="Download EgoDex test data using wget")
    parser.add_argument("--output-dir", default="./egodex_test_data", 
                       help="Directory to save EgoDex test data")
    parser.add_argument("--max-episodes", type=int, default=10,
                       help="Maximum number of episodes per task")
    
    args = parser.parse_args()
    
    success = download_egodex_test_wget(args.output_dir, args.max_episodes)
    
    if success:
        print("\n✅ Download complete! Run 'source egodex_test_data/organized/setup_egodex_test_env.sh' to get started.")
    else:
        print("\n❌ Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()


