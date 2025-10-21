#!/usr/bin/env python3
"""
Download a small portion of EgoDex data for testing MolmoAct integration.
This script downloads only the test set and a small sample from part1 for quick testing.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import json


def download_small_egodex_sample(output_dir: str, max_episodes: int = 10):
    """
    Download a small sample of EgoDex data for testing.
    
    Args:
        output_dir: Directory to save the data
        max_episodes: Maximum number of episodes to keep from each task
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Downloading small EgoDex sample for testing...")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Max episodes per task: {max_episodes}")
    
    # Download test set (smaller)
    print("\n📥 Downloading test set...")
    test_url = 'https://ml-site.cdn-apple.com/datasets/egodex/test.zip'
    test_zip = output_path / "test.zip"
    
    try:
        subprocess.run(['curl', test_url, '-o', str(test_zip)], check=True)
        print("✅ Test set downloaded")
        
        # Extract test set
        test_dir = output_path / "test"
        test_dir.mkdir(exist_ok=True)
        subprocess.run(['unzip', str(test_zip), '-d', str(test_dir)], check=True)
        print("✅ Test set extracted")
        
        # Remove zip file
        test_zip.unlink()
        print("🗑️  Test zip removed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download test set: {e}")
        return False
    
    # Download a small part of part1 for more data
    print("\n📥 Downloading part1 (first 100MB)...")
    part1_url = 'https://ml-site.cdn-apple.com/datasets/egodex/part1.zip'
    part1_zip = output_path / "part1.zip"
    
    try:
        # Download with curl and limit to first 100MB
        subprocess.run(['curl', part1_url, '-o', str(part1_zip), '--max-filesize', '104857600'], check=True)
        print("✅ Part1 sample downloaded")
        
        # Extract part1
        part1_dir = output_path / "part1"
        part1_dir.mkdir(exist_ok=True)
        subprocess.run(['unzip', str(part1_zip), '-d', str(part1_dir)], check=True)
        print("✅ Part1 sample extracted")
        
        # Remove zip file
        part1_zip.unlink()
        print("🗑️  Part1 zip removed")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to download part1 sample: {e}")
        print("Continuing with test set only...")
    
    # Organize data
    print("\n📁 Organizing data...")
    organized_dir = organize_small_sample(output_path, max_episodes)
    
    # Create environment setup
    create_test_setup(organized_dir)
    
    print("\n" + "="*60)
    print("🎉 EgoDex test data preparation complete!")
    print(f"📁 Data location: {organized_dir}")
    print(f"📊 Sample size: {max_episodes} episodes per task")
    print("\nNext steps:")
    print("1. Set environment variable: export EGODEX_DATA_DIR='{organized_dir}'")
    print("2. Run test scripts:")
    print("   python scripts/test_egodex_integration.py")
    print("   python scripts/test_egodex_pose_actions.py")
    print("   python scripts/egodex_conversation_example.py --example-only")
    print("="*60)
    
    return True


def organize_small_sample(data_dir: Path, max_episodes: int):
    """Organize the small sample into train/test structure."""
    organized_dir = data_dir / "organized"
    train_dir = organized_dir / "train"
    test_dir = organized_dir / "test"
    
    for split_dir in [train_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Process part1 data (training)
    part1_source = data_dir / "part1"
    if part1_source.exists():
        print("📁 Organizing training data...")
        for task_dir in part1_source.iterdir():
            if task_dir.is_dir():
                dest = train_dir / task_dir.name
                if not dest.exists():
                    shutil.move(str(task_dir), str(dest))
                    print(f"  ✅ Moved {task_dir.name}")
    
    # Limit episodes per task
    print(f"📊 Limiting to {max_episodes} episodes per task...")
    for split_dir in [train_dir, test_dir]:
        for task_dir in split_dir.iterdir():
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
echo "  python scripts/test_egodex_integration.py"
echo "  python scripts/test_egodex_pose_actions.py"
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
    parser = argparse.ArgumentParser(description="Download small EgoDex sample for testing")
    parser.add_argument("--output-dir", default="./egodex_test_data", 
                       help="Directory to save EgoDex test data")
    parser.add_argument("--max-episodes", type=int, default=10,
                       help="Maximum number of episodes per task")
    
    args = parser.parse_args()
    
    success = download_small_egodex_sample(args.output_dir, args.max_episodes)
    
    if success:
        print("\n✅ Setup complete! Run 'source egodex_test_data/organized/setup_egodex_test_env.sh' to get started.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()


