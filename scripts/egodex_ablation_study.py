#!/usr/bin/env python3
"""
EgoDex Ablation Study Script

This script demonstrates how to train models with different percentages of EgoDex data
for ablation studies. It shows how to:
1. Compute statistics for different data percentages
2. Train models with varying amounts of data
3. Compare performance across different data sizes
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions
from scripts.compute_egodex_stats import compute_egodex_statistics


def run_ablation_study(
    data_dir: str,
    percentages: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    split: str = "train",
    output_dir: str = "ablation_results"
):
    """
    Run ablation study with different data percentages.
    
    Args:
        data_dir: Path to EgoDex data directory
        percentages: List of data percentages to test
        split: Dataset split to use
        output_dir: Directory to save results
    """
    print("EgoDex Ablation Study")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for percentage in percentages:
        print(f"\n{'='*20} Testing {percentage:.1%} of data {'='*20}")
        
        # 1. Compute statistics for this percentage
        stats_file = os.path.join(output_dir, f"egodex_stats_{percentage:.2f}.json")
        
        print(f"Computing statistics for {percentage:.1%} of data...")
        try:
            stats = compute_egodex_statistics(
                data_dir=data_dir,
                split=split,
                sample_size=1000,  # Use all available data
                data_percentage=percentage,
                output_file=stats_file
            )
            print(f"✓ Statistics computed: {stats['n_samples']} samples")
        except Exception as e:
            print(f"✗ Error computing statistics: {e}")
            continue
        
        # 2. Load dataset with this percentage
        print(f"Loading dataset with {percentage:.1%} of data...")
        try:
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split=split,
                data_percentage=percentage,
                action_stats_file=stats_file,
                normalize_actions=True,
                discretize_actions=True,
                action_sequence_length=8
            )
            print(f"✓ Dataset loaded: {len(dataset)} episodes")
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            continue
        
        # 3. Analyze dataset characteristics
        print("Analyzing dataset characteristics...")
        try:
            # Get a sample to analyze
            rng = None
            sample = dataset.get(0, rng)
            
            # Extract action processing info
            action_processing = sample['annotation'].get('action_processing', {})
            action_stats = sample['annotation'].get('action_stats', {})
            
            # Count episodes by task
            task_counts = {}
            for episode in dataset.episodes:
                task_name = episode['task_name']
                task_counts[task_name] = task_counts.get(task_name, 0) + 1
            
            analysis = {
                'total_episodes': len(dataset),
                'data_percentage': percentage,
                'action_processing': action_processing,
                'action_stats_samples': action_stats.get('n_samples', 0),
                'task_distribution': task_counts,
                'num_tasks': len(task_counts)
            }
            
            results[percentage] = analysis
            
            print(f"✓ Analysis completed:")
            print(f"  - Total episodes: {analysis['total_episodes']}")
            print(f"  - Number of tasks: {analysis['num_tasks']}")
            print(f"  - Action processing: {action_processing}")
            print(f"  - Statistics samples: {action_stats.get('n_samples', 0)}")
            
        except Exception as e:
            print(f"✗ Error analyzing dataset: {e}")
            continue
    
    # 4. Save results
    results_file = os.path.join(output_dir, "ablation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Ablation Study Summary")
    print(f"{'='*50}")
    
    for percentage in sorted(results.keys()):
        analysis = results[percentage]
        print(f"\n{percentage:.1%} of data:")
        print(f"  - Episodes: {analysis['total_episodes']}")
        print(f"  - Tasks: {analysis['num_tasks']}")
        print(f"  - Statistics samples: {analysis['action_stats_samples']}")
    
    print(f"\nResults saved to: {results_file}")
    print("\nTo use in training:")
    print("  python launch_scripts/train_multitask_model.py \\")
    print("    --mixture egodex-pose-only \\")
    print("    --data-dir /path/to/egodex \\")
    print("    --egodex-data-percentage 0.5  # Use 50% of data")


def main():
    parser = argparse.ArgumentParser(description="Run EgoDex ablation study with different data percentages")
    parser.add_argument("--data-dir", required=True, help="Path to EgoDex data directory")
    parser.add_argument("--percentages", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0],
                       help="Data percentages to test (e.g., 0.1 0.25 0.5 1.0)")
    parser.add_argument("--split", default="train", choices=["train", "test", "additional"],
                       help="Dataset split to use")
    parser.add_argument("--output-dir", default="ablation_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Validate percentages
    for p in args.percentages:
        if not 0.0 < p <= 1.0:
            print(f"Error: Percentage {p} must be between 0.0 and 1.0")
            sys.exit(1)
    
    try:
        run_ablation_study(
            data_dir=args.data_dir,
            percentages=args.percentages,
            split=args.split,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error running ablation study: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


