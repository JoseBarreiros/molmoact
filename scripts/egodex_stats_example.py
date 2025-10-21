#!/usr/bin/env python3
"""
Example script showing how to compute and use EgoDex action statistics.

This demonstrates the complete workflow:
1. Compute statistics from EgoDex data
2. Save statistics to file
3. Load statistics and use them for normalization
4. Compare normalized vs unnormalized actions
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions
from scripts.compute_egodex_stats import compute_egodex_statistics


def main():
    # Configuration
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/path/to/egodex/organized')
    stats_file = 'egodex_action_stats.json'
    
    print("EgoDex Action Statistics Example")
    print("=" * 50)
    
    # Step 1: Compute statistics from training data
    print("\n1. Computing action statistics from training data...")
    try:
        stats = compute_egodex_statistics(
            data_dir=data_dir,
            split="train",
            sample_size=500,  # Use smaller sample for demo
            output_file=stats_file
        )
        print(f"✓ Statistics computed and saved to {stats_file}")
    except Exception as e:
        print(f"✗ Error computing statistics: {e}")
        return
    
    # Step 2: Load dataset with precomputed statistics
    print("\n2. Loading dataset with precomputed statistics...")
    try:
        dataset_with_stats = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            max_frames=2,
            action_sequence_length=4,
            normalize_actions=True,
            discretize_actions=True,
            action_stats_file=stats_file
        )
        print("✓ Dataset loaded with precomputed statistics")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Step 3: Load dataset without normalization for comparison
    print("\n3. Loading dataset without normalization for comparison...")
    try:
        dataset_without_stats = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            max_frames=2,
            action_sequence_length=4,
            normalize_actions=False,
            discretize_actions=False
        )
        print("✓ Dataset loaded without normalization")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Step 4: Compare actions
    print("\n4. Comparing normalized vs unnormalized actions...")
    try:
        rng = np.random.RandomState(42)
        
        # Get sample from both datasets
        sample_with_stats = dataset_with_stats.get(0, rng)
        sample_without_stats = dataset_without_stats.get(0, rng)
        
        print("\nAction processing comparison:")
        print(f"  - With stats: {sample_with_stats['annotation']['action_processing']}")
        print(f"  - Without stats: {sample_without_stats['annotation']['action_processing']}")
        
        # Show action statistics
        if 'action_stats' in sample_with_stats['annotation']:
            stats = sample_with_stats['annotation']['action_stats']
            print(f"\nAction statistics used:")
            print(f"  - Samples: {stats['n_samples']}")
            print(f"  - Hand position range: [{stats['q01'][0]:.3f}, {stats['q99'][0]:.3f}]")
            print(f"  - Hand rotation range: [{stats['q01'][3]:.3f}, {stats['q99'][3]:.3f}]")
            print(f"  - Finger tip range: [{stats['q01'][6]:.3f}, {stats['q99'][6]:.3f}]")
        
        # Show action format differences
        answer_with_stats = sample_with_stats['answers']
        answer_without_stats = sample_without_stats['answers']
        
        print(f"\nAction format comparison:")
        print(f"  - With stats (discretized): {'discretized' in answer_with_stats}")
        print(f"  - Without stats (continuous): {'discretized' not in answer_without_stats}")
        
        # Extract action values for comparison
        import re
        
        # Find action lines in answers
        lines_with_stats = [line for line in answer_with_stats.split('\n') if '[' in line and ']' in line and '#' in line]
        lines_without_stats = [line for line in answer_without_stats.split('\n') if '[' in line and ']' in line and '#' in line]
        
        if lines_with_stats and lines_without_stats:
            # Extract numbers from first action line
            numbers_with_stats = re.findall(r'\d+', lines_with_stats[0])
            numbers_without_stats = re.findall(r'-?\d+\.\d+', lines_without_stats[0])
            
            print(f"\nAction value comparison (first action):")
            print(f"  - With stats: {len(numbers_with_stats)} discrete values")
            print(f"  - Without stats: {len(numbers_without_stats)} continuous values")
            
            if len(numbers_with_stats) > 0:
                print(f"  - Sample discrete values: {numbers_with_stats[:5]}...")
            if len(numbers_without_stats) > 0:
                print(f"  - Sample continuous values: {[float(x) for x in numbers_without_stats[:5]]}...")
        
        print("\n✓ Action comparison completed successfully!")
        
    except Exception as e:
        print(f"✗ Error comparing actions: {e}")
        return
    
    # Step 5: Show how to use statistics in training
    print("\n5. Usage in training:")
    print("   To use precomputed statistics in training:")
    print(f"   dataset = EgoDexPoseActions(")
    print(f"       data_dir='{data_dir}',")
    print(f"       action_stats_file='{stats_file}',")
    print(f"       normalize_actions=True,")
    print(f"       discretize_actions=True")
    print(f"   )")
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()


