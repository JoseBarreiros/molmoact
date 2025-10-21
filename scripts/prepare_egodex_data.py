#!/usr/bin/env python3
"""
Script to prepare EgoDex data for MolmoAct training.

This script helps download, extract, and organize EgoDex data for use with MolmoAct.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil


def download_egodex_data(output_dir: str, parts: list = None):
    """
    Download EgoDex dataset parts.
    
    Args:
        output_dir: Directory to save the data
        parts: List of parts to download (1-5 for training, 'test', 'additional')
    """
    if parts is None:
        parts = ['test']  # Start with test set for initial exploration
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # EgoDex download URLs
    urls = {
        'part1': 'https://ml-site.cdn-apple.com/datasets/egodex/part1.zip',
        'part2': 'https://ml-site.cdn-apple.com/datasets/egodex/part2.zip', 
        'part3': 'https://ml-site.cdn-apple.com/datasets/egodex/part3.zip',
        'part4': 'https://ml-site.cdn-apple.com/datasets/egodex/part4.zip',
        'part5': 'https://ml-site.cdn-apple.com/datasets/egodex/part5.zip',
        'test': 'https://ml-site.cdn-apple.com/datasets/egodex/test.zip',
        'additional': 'https://ml-site.cdn-apple.com/datasets/egodex/additional.zip'
    }
    
    for part in parts:
        if part not in urls:
            print(f"Unknown part: {part}. Available parts: {list(urls.keys())}")
            continue
            
        url = urls[part]
        zip_file = output_path / f"{part}.zip"
        
        print(f"Downloading {part}...")
        print(f"URL: {url}")
        print(f"Output: {zip_file}")
        
        # Download using curl
        cmd = ['curl', url, '-o', str(zip_file)]
        try:
            subprocess.run(cmd, check=True)
            print(f"Downloaded {part} successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {part}: {e}")
            continue
        
        # Extract the zip file
        print(f"Extracting {part}...")
        extract_dir = output_path / part
        extract_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run(['unzip', str(zip_file), '-d', str(extract_dir)], check=True)
            print(f"Extracted {part} successfully")
            
            # Remove zip file to save space
            zip_file.unlink()
            print(f"Removed {zip_file} to save space")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract {part}: {e}")
            continue


def organize_egodex_data(data_dir: str):
    """
    Organize EgoDex data into train/test/additional splits.
    
    Args:
        data_dir: Directory containing extracted EgoDex data
    """
    data_path = Path(data_dir)
    
    # Create organized directory structure
    organized_dir = data_path / "organized"
    train_dir = organized_dir / "train"
    test_dir = organized_dir / "test"
    additional_dir = organized_dir / "additional"
    
    for split_dir in [train_dir, test_dir, additional_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize data by split
    for part_dir in data_path.iterdir():
        if not part_dir.is_dir() or part_dir.name.startswith('.'):
            continue
            
        print(f"Organizing {part_dir.name}...")
        
        if part_dir.name == 'test':
            # Move test data
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    dest = test_dir / task_dir.name
                    if not dest.exists():
                        shutil.move(str(task_dir), str(dest))
                    else:
                        print(f"Warning: {dest} already exists, skipping")
                        
        elif part_dir.name == 'additional':
            # Move additional data
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    dest = additional_dir / task_dir.name
                    if not dest.exists():
                        shutil.move(str(task_dir), str(dest))
                    else:
                        print(f"Warning: {dest} already exists, skipping")
                        
        elif part_dir.name.startswith('part'):
            # Move training data
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    dest = train_dir / task_dir.name
                    if not dest.exists():
                        shutil.move(str(task_dir), str(dest))
                    else:
                        print(f"Warning: {dest} already exists, skipping")
    
    print(f"Data organized into: {organized_dir}")
    return str(organized_dir)


def create_environment_setup(data_dir: str):
    """
    Create environment setup script for EgoDex data.
    
    Args:
        data_dir: Path to organized EgoDex data
    """
    setup_script = Path(data_dir) / "setup_egodex_env.sh"
    
    content = f"""#!/bin/bash
# EgoDex environment setup for MolmoAct

# Set the EgoDex data directory
export EGODEX_DATA_DIR="{data_dir}"

# Add to your ~/.bashrc or ~/.zshrc to make permanent:
# echo 'export EGODEX_DATA_DIR="{data_dir}"' >> ~/.bashrc

echo "EgoDex data directory set to: $EGODEX_DATA_DIR"
echo "You can now run MolmoAct training with EgoDex data"
"""
    
    with open(setup_script, 'w') as f:
        f.write(content)
    
    # Make executable
    setup_script.chmod(0o755)
    
    print(f"Created environment setup script: {setup_script}")
    print("Run: source setup_egodex_env.sh")


def main():
    parser = argparse.ArgumentParser(description="Prepare EgoDex data for MolmoAct training")
    parser.add_argument("--output-dir", required=True, help="Directory to save EgoDex data")
    parser.add_argument("--parts", nargs="+", 
                       choices=['part1', 'part2', 'part3', 'part4', 'part5', 'test', 'additional'],
                       default=['test'],
                       help="EgoDex data parts to download")
    parser.add_argument("--organize-only", action="store_true",
                       help="Only organize existing data, don't download")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, only organize")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    
    if not args.organize_only and not args.skip_download:
        print("Downloading EgoDex data...")
        download_egodex_data(str(output_dir), args.parts)
    
    if not args.skip_download:
        print("Organizing EgoDex data...")
        organized_dir = organize_egodex_data(str(output_dir))
        
        print("Creating environment setup...")
        create_environment_setup(organized_dir)
        
        print("\n" + "="*60)
        print("EgoDex data preparation complete!")
        print(f"Data location: {organized_dir}")
        print("\nNext steps:")
        print("1. Set environment variable: export EGODEX_DATA_DIR='{organized_dir}'")
        print("2. Run MolmoAct training with EgoDex:")
        print("   python launch_scripts/train_multitask_model.py --mixture molmoact-midtrain-egodex")
        print("="*60)


if __name__ == "__main__":
    main()



