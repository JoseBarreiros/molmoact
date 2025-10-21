#!/usr/bin/env python3
"""
Compute EgoDex action statistics for normalization.

This script computes the actual statistics (q01, q99, mean, std) from EgoDex data
for proper action normalization, similar to how LIBERO statistics are computed.
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions


def compute_egodex_statistics(
    data_dir: str,
    split: str = "train",
    sample_size: int = 1000,
    data_percentage: float = 1.0,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Compute action statistics from EgoDex dataset.
    
    Args:
        data_dir: Path to EgoDex data directory
        split: Dataset split to use ("train", "test", "additional")
        sample_size: Number of episodes to sample for statistics
        data_percentage: Percentage of data to use (0.0 to 1.0) for ablation studies
        output_file: Optional file to save statistics to
        
    Returns:
        Dictionary containing computed statistics
    """
    print(f"Computing EgoDex action statistics from {split} split...")
    print(f"Data directory: {data_dir}")
    print(f"Sample size: {sample_size}")
    print(f"Data percentage: {data_percentage:.1%}")
    
    # Create dataset instance
    dataset = EgoDexPoseActions(
        data_dir=data_dir,
        split=split,
        normalize_actions=False,  # Don't normalize during computation
        discretize_actions=False,
        action_sequence_length=8,
        data_percentage=data_percentage
    )
    
    print(f"Dataset loaded with {len(dataset)} episodes")
    
    # Collect actions from sample of episodes
    all_actions = []
    actual_sample_size = min(sample_size, len(dataset))
    
    print(f"Processing {actual_sample_size} episodes...")
    
    for i in range(actual_sample_size):
        if i % 100 == 0:
            print(f"  Processed {i}/{actual_sample_size} episodes...")
            
        try:
            episode = dataset.episodes[i]
            pose_data = dataset._load_pose_data(episode['pose_path'])
            
            # Extract hand poses and finger tips
            transforms = pose_data.get('transforms', {})
            left_hand_poses = transforms.get('leftHand', [])
            right_hand_poses = transforms.get('rightHand', [])
            
            # Convert to lists if they are numpy arrays
            if isinstance(left_hand_poses, np.ndarray):
                left_hand_poses = [left_hand_poses[i] for i in range(len(left_hand_poses))]
            if isinstance(right_hand_poses, np.ndarray):
                right_hand_poses = [right_hand_poses[i] for i in range(len(right_hand_poses))]
            
            # Use primary hand (right if available, otherwise left)
            primary_poses = right_hand_poses if len(right_hand_poses) > 0 else left_hand_poses
            hand_name = "right" if len(right_hand_poses) > 0 else "left"
            
            if len(primary_poses) == 0:
                continue
            
            # Sample poses from this episode
            n_poses = len(primary_poses)
            if n_poses <= 8:
                sampled_poses = primary_poses
                indices = list(range(n_poses))
            else:
                indices = np.linspace(0, n_poses - 1, 8, dtype=int)
                sampled_poses = [primary_poses[i] for i in indices]
            
            # Get finger tip positions
            finger_tips = dataset._extract_finger_tip_positions(transforms, hand_name, indices)
            
            # Convert to action format
            for j, pose in enumerate(sampled_poses):
                # Extract hand position (x, y, z)
                hand_position = pose[:3, 3]
                
                # Extract hand rotation (convert rotation matrix to euler angles)
                rotation_matrix = pose[:3, :3]
                rotation = dataset._rotation_matrix_to_euler(rotation_matrix)
                
                # Combine hand pose: [x, y, z, rx, ry, rz]
                hand_action = np.concatenate([hand_position, rotation])
                
                # Add finger tip positions
                finger_action = finger_tips[j] if j < len(finger_tips) else np.zeros(15)
                
                # Combine hand pose + finger tips
                action = np.concatenate([hand_action, finger_action])
                all_actions.append(action)
                
        except Exception as e:
            print(f"Warning: Failed to process episode {i}: {e}")
            continue
    
    if not all_actions:
        raise ValueError("No valid actions found in dataset")
    
    # Convert to numpy array
    all_actions = np.array(all_actions)
    print(f"Computed statistics from {len(all_actions)} actions")
    
    # Compute statistics
    q01 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)
    mean = np.mean(all_actions, axis=0)
    std = np.std(all_actions, axis=0)
    
    # Check for zero-range dimensions
    range_diff = q99 - q01
    zero_range_dims = np.where(np.abs(range_diff) < 1e-10)[0]
    if len(zero_range_dims) > 0:
        print(f"Warning: {len(zero_range_dims)} dimensions have zero range: {zero_range_dims.tolist()}")
        print("These dimensions will be handled with epsilon in normalization.")
    
    # Create mask (all dimensions normalized)
    mask = np.ones(21, dtype=bool)
    
    stats = {
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mask": mask.tolist(),
        "n_samples": len(all_actions),
        "dataset_split": split,
        "sample_size": actual_sample_size,
        "data_percentage": data_percentage
    }
    
    print(f"\nAction statistics computed:")
    print(f"  - Hand position range: [{q01[0]:.3f}, {q99[0]:.3f}] x [{q01[1]:.3f}, {q99[1]:.3f}] x [{q01[2]:.3f}, {q99[2]:.3f}]")
    print(f"  - Hand rotation range: [{q01[3]:.3f}, {q99[3]:.3f}] x [{q01[4]:.3f}, {q99[4]:.3f}] x [{q01[5]:.3f}, {q99[5]:.3f}]")
    print(f"  - Finger tip range: [{q01[6]:.3f}, {q99[6]:.3f}] to [{q01[20]:.3f}, {q99[20]:.3f}]")
    
    # Save to file if specified
    if output_file and output_file.strip():
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute EgoDex action statistics for normalization")
    parser.add_argument("--data-dir", required=True, help="Path to EgoDex data directory")
    parser.add_argument("--split", default="train", choices=["train", "test", "additional"], 
                       help="Dataset split to use")
    parser.add_argument("--sample-size", type=int, default=1000, 
                       help="Number of episodes to sample for statistics")
    parser.add_argument("--data-percentage", type=float, default=1.0,
                       help="Percentage of data to use (0.0 to 1.0) for ablation studies")
    parser.add_argument("--output-file", help="File to save statistics to (JSON format)")
    
    args = parser.parse_args()
    
    try:
        stats = compute_egodex_statistics(
            data_dir=args.data_dir,
            split=args.split,
            sample_size=args.sample_size,
            data_percentage=args.data_percentage,
            output_file=args.output_file
        )
        
        print("\nStatistics computation completed successfully!")
        
    except Exception as e:
        print(f"Error computing statistics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
