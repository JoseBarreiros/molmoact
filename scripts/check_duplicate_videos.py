#!/usr/bin/env python3
"""
Check for duplicate MP4 files in EgoDex task directories.

This script compares MP4 files by their content (using MD5 hashes) to detect
duplicates even if they have different filenames.
"""

import os
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def calculate_file_hash(file_path, chunk_size=8192):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None


def get_video_info(file_path):
    """Get basic information about a video file."""
    try:
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime
        }
    except Exception as e:
        print(f"❌ Error getting info for {file_path}: {e}")
        return None


def check_duplicates_in_directory(task_dir, task_name):
    """Check for duplicates within a single task directory."""
    print(f"\n📁 Checking task: {task_name}")
    
    # Find all MP4 files
    mp4_files = list(task_dir.glob("*.mp4"))
    
    if len(mp4_files) < 2:
        print(f"   ⚠️  Only {len(mp4_files)} MP4 files found, no duplicates possible")
        return [], {}
    
    print(f"   📹 Found {len(mp4_files)} MP4 files")
    
    # Group files by size first (quick pre-filter)
    size_groups = defaultdict(list)
    for mp4_file in mp4_files:
        info = get_video_info(mp4_file)
        if info:
            size_groups[info['size']].append(mp4_file)
    
    print(f"   📊 Files grouped by size: {len(size_groups)} groups")
    
    # Check for exact duplicates within each size group
    hash_groups = defaultdict(list)
    duplicates = []
    
    for size, files in size_groups.items():
        if len(files) == 1:
            continue  # No potential duplicates for this size
            
        print(f"   🔍 Checking {len(files)} files of size {size:,} bytes")
        
        for mp4_file in tqdm(files, desc=f"   Computing hashes", leave=False):
            file_hash = calculate_file_hash(mp4_file)
            if file_hash:
                hash_groups[file_hash].append(mp4_file)
    
    # Find actual duplicates
    for file_hash, files in hash_groups.items():
        if len(files) > 1:
            duplicates.append({
                'hash': file_hash,
                'files': files,
                'count': len(files)
            })
    
    return duplicates, hash_groups


def analyze_duplicates(duplicates, task_name):
    """Analyze and report duplicate findings."""
    if not duplicates:
        print(f"   ✅ No duplicates found in {task_name}")
        return
    
    print(f"   🚨 Found {len(duplicates)} duplicate groups in {task_name}:")
    
    for i, dup_group in enumerate(duplicates, 1):
        print(f"\n   📋 Duplicate Group {i}:")
        print(f"      Hash: {dup_group['hash']}")
        print(f"      Count: {dup_group['count']} files")
        print(f"      Files:")
        
        for j, file_path in enumerate(dup_group['files']):
            info = get_video_info(file_path)
            size_str = f"{info['size']:,} bytes" if info else "unknown size"
            print(f"         {j+1}. {file_path.name} ({size_str})")


def main():
    parser = argparse.ArgumentParser(description="Check for duplicate MP4 files in EgoDex data")
    parser.add_argument(
        "--data-dir", 
        default="/home/jose-barreiros/egodex/organized/train",
        help="Path to EgoDex train directory"
    )
    parser.add_argument(
        "--task-filter",
        default="add_remove_lid",
        help="Task name to check (default: add_remove_lid)"
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Check all tasks for duplicates"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        return 1
    
    print(f"🔍 Checking for duplicate MP4 files in: {data_dir}")
    
    if args.all_tasks:
        # Check all task directories
        task_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('part')]
        print(f"📊 Found {len(task_dirs)} task directories to check")
        
        total_duplicates = 0
        tasks_with_duplicates = 0
        
        for task_dir in tqdm(task_dirs, desc="Checking tasks"):
            duplicates, _ = check_duplicates_in_directory(task_dir, task_dir.name)
            analyze_duplicates(duplicates, task_dir.name)
            
            if duplicates:
                total_duplicates += len(duplicates)
                tasks_with_duplicates += 1
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Tasks with duplicates: {tasks_with_duplicates}/{len(task_dirs)}")
        print(f"📊 Total duplicate groups found: {total_duplicates}")
        
    else:
        # Check specific task
        task_dir = data_dir / args.task_filter
        
        if not task_dir.exists():
            print(f"❌ Task directory does not exist: {task_dir}")
            return 1
        
        duplicates, hash_groups = check_duplicates_in_directory(task_dir, args.task_filter)
        analyze_duplicates(duplicates, args.task_filter)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Duplicate groups found: {len(duplicates)}")
        print(f"📊 Unique files: {len(hash_groups)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
