#!/usr/bin/env python3
"""
Extract first frames from EgoDex MP4 videos for data inspection.

This script goes through each task directory in ~/egodex/organized/train/,
finds all MP4 files, and extracts the first frame as a PNG image.
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video and save it as PNG."""
    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        # Read the first frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Could not read first frame from: {video_path}")
            cap.release()
            return False
        
        # Save the first frame as PNG
        success = cv2.imwrite(str(output_path), frame)
        cap.release()
        
        if success:
            return True
        else:
            print(f"❌ Could not save frame: {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return False


def process_task_directory(task_dir, base_dir):
    """Process all MP4 files in a task directory."""
    task_name = task_dir.name
    print(f"\n📁 Processing task: {task_name}")
    
    # Find all MP4 files in the task directory
    mp4_files = list(task_dir.glob("*.mp4"))
    
    if not mp4_files:
        print(f"   ⚠️  No MP4 files found in {task_dir}")
        return 0, 0
    
    print(f"   📹 Found {len(mp4_files)} MP4 files")
    
    success_count = 0
    total_count = len(mp4_files)
    
    # Process each MP4 file
    for mp4_file in tqdm(mp4_files, desc=f"   Extracting frames", leave=False):
        # Create output filename (replace .mp4 with _first_frame.png)
        output_filename = mp4_file.stem + "_first_frame.png"
        output_path = task_dir / output_filename
        
        # Skip if output already exists
        if output_path.exists():
            print(f"   ⏭️  Skipping {mp4_file.name} (frame already exists)")
            success_count += 1
            continue
        
        # Extract first frame
        if extract_first_frame(mp4_file, output_path):
            success_count += 1
    
    print(f"   ✅ Successfully processed {success_count}/{total_count} videos")
    return success_count, total_count


def main():
    parser = argparse.ArgumentParser(description="Extract first frames from EgoDex MP4 videos")
    parser.add_argument(
        "--data-dir", 
        default="/home/jose-barreiros/egodex/organized/train",
        help="Path to EgoDex train directory (default: /home/jose-barreiros/egodex/organized/train)"
    )
    parser.add_argument(
        "--task-filter",
        help="Process only specific task (e.g., 'fry_egg') or tasks matching pattern"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        return 1
    
    print(f"🔍 Scanning EgoDex data directory: {data_dir}")
    
    # Find all task directories
    task_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('part')]
    
    if not task_dirs:
        print(f"❌ No task directories found in {data_dir}")
        return 1
    
    # Apply task filter if specified
    if args.task_filter:
        task_dirs = [d for d in task_dirs if args.task_filter in d.name]
        if not task_dirs:
            print(f"❌ No tasks found matching filter: {args.task_filter}")
            return 1
    
    # Apply max tasks limit if specified
    if args.max_tasks:
        task_dirs = task_dirs[:args.max_tasks]
    
    print(f"📊 Found {len(task_dirs)} task directories to process")
    
    # Process each task directory
    total_success = 0
    total_videos = 0
    
    for task_dir in tqdm(task_dirs, desc="Processing tasks"):
        success, videos = process_task_directory(task_dir, data_dir)
        total_success += success
        total_videos += videos
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Total videos processed: {total_success}/{total_videos}")
    print(f"📁 Task directories processed: {len(task_dirs)}")
    
    if total_success > 0:
        print(f"\n💡 First frames saved as: *_first_frame.png")
        print(f"💡 You can now inspect the data by viewing these PNG files")
    
    return 0


if __name__ == "__main__":
    exit(main())
