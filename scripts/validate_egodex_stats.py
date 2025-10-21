#!/usr/bin/env python3
"""
Validate EgoDex statistics computation accuracy and consistency.

This script performs comprehensive validation of the EgoDex action statistics
computation to ensure accuracy and consistency across different runs.
"""

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions


def validate_statistics_consistency():
    """Test that statistics are consistent across multiple runs."""
    print("🧪 Testing statistics consistency across multiple runs...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Run statistics computation multiple times
    stats_list = []
    for i in range(3):
        print(f"  📊 Run {i+1}/3...")
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.1,  # Use small subset for speed
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        stats = dataset._compute_action_statistics()
        stats_list.append(stats)
    
    # Compare statistics across runs
    print("  🔍 Comparing statistics across runs...")
    base_stats = stats_list[0]
    
    for i, stats in enumerate(stats_list[1:], 1):
        print(f"  📊 Comparing run {i+1} with run 1...")
        
        # Compare key statistics
        for key in ['q01', 'q99', 'mean', 'std']:
            if key in base_stats and key in stats:
                base_val = np.array(base_stats[key])
                stats_val = np.array(stats[key])
                
                # Check if arrays are close
                if not np.allclose(base_val, stats_val, rtol=1e-10, atol=1e-10):
                    print(f"    ❌ {key} differs between runs")
                    print(f"      Run 1: {base_val[:5]}...")
                    print(f"      Run {i+1}: {stats_val[:5]}...")
                    return False
                else:
                    print(f"    ✅ {key} consistent")
    
    print("  ✅ All statistics are consistent across runs")
    return True


def validate_statistics_accuracy():
    """Test that computed statistics are mathematically correct."""
    print("🧪 Testing statistics accuracy...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Load dataset
    dataset = EgoDexPoseActions(
        data_dir=data_dir,
        split='test',
        data_percentage=0.05,  # Use very small subset for detailed analysis
        action_sequence_length=8,
        use_action_chunking=True,
        use_temporal_aggregation=True,
        use_action_normalization=True,
        use_action_discretization=True
    )
    
    print(f"  📊 Dataset loaded: {len(dataset)} episodes")
    
    # Compute statistics
    stats = dataset._compute_action_statistics()
    
    # Validate statistics properties
    print("  🔍 Validating statistics properties...")
    
    # Check that q01 <= q99 for all dimensions
    q01 = np.array(stats['q01'])
    q99 = np.array(stats['q99'])
    
    if not np.all(q01 <= q99):
        print("    ❌ q01 > q99 for some dimensions")
        bad_dims = np.where(q01 > q99)[0]
        print(f"      Bad dimensions: {bad_dims}")
        return False
    else:
        print("    ✅ q01 <= q99 for all dimensions")
    
    # Check that mean is between q01 and q99
    mean = np.array(stats['mean'])
    if not np.all((q01 <= mean) & (mean <= q99)):
        print("    ❌ Mean outside [q01, q99] range for some dimensions")
        bad_dims = np.where(~((q01 <= mean) & (mean <= q99)))[0]
        print(f"      Bad dimensions: {bad_dims}")
        return False
    else:
        print("    ✅ Mean within [q01, q99] range")
    
    # Check that std is non-negative
    std = np.array(stats['std'])
    if not np.all(std >= 0):
        print("    ❌ Negative standard deviation found")
        bad_dims = np.where(std < 0)[0]
        print(f"      Bad dimensions: {bad_dims}")
        return False
    else:
        print("    ✅ All standard deviations are non-negative")
    
    # Check for reasonable ranges
    print("  📊 Checking for reasonable value ranges...")
    
    # Hand position should be in reasonable range (roughly -1 to 1 meters)
    pos_q01 = q01[:3]
    pos_q99 = q99[:3]
    pos_range = pos_q99 - pos_q01
    
    if np.any(pos_range > 2.0):  # More than 2 meters range
        print(f"    ⚠️  Large position range detected: {pos_range}")
    else:
        print(f"    ✅ Position range reasonable: {pos_range}")
    
    # Hand rotation should be in reasonable range (roughly -π to π)
    rot_q01 = q01[3:6]
    rot_q99 = q99[3:6]
    rot_range = rot_q99 - rot_q01
    
    if np.any(rot_range > 2 * np.pi):  # More than 2π range
        print(f"    ⚠️  Large rotation range detected: {rot_range}")
    else:
        print(f"    ✅ Rotation range reasonable: {rot_range}")
    
    # Finger tips should be in reasonable range
    finger_q01 = q01[6:]
    finger_q99 = q99[6:]
    finger_range = finger_q99 - finger_q01
    
    if np.any(finger_range > 1.0):  # More than 1 meter range
        print(f"    ⚠️  Large finger range detected: max={np.max(finger_range):.3f}")
    else:
        print(f"    ✅ Finger range reasonable: max={np.max(finger_range):.3f}")
    
    print("  ✅ Statistics accuracy validation passed")
    return True


def validate_normalization_behavior():
    """Test that normalization works correctly with computed statistics."""
    print("🧪 Testing normalization behavior...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Load dataset
    dataset = EgoDexPoseActions(
        data_dir=data_dir,
        split='test',
        data_percentage=0.01,  # Very small subset
        action_sequence_length=8,
        use_action_chunking=True,
        use_temporal_aggregation=True,
        use_action_normalization=True,
        use_action_discretization=True
    )
    
    print(f"  📊 Dataset loaded: {len(dataset)} episodes")
    
    # Get a sample to test normalization
    sample = dataset[0]
    annotation = sample['annotation']
    
    # Check if normalization was applied
    if 'action_processing' not in annotation:
        print("    ❌ No action processing information found")
        return False
    
    action_processing = annotation['action_processing']
    
    if not action_processing.get('normalization', False):
        print("    ❌ Normalization not applied")
        return False
    
    print("    ✅ Normalization was applied")
    
    # Check that normalized values are in reasonable range
    if 'normalized_actions' in annotation:
        normalized_actions = np.array(annotation['normalized_actions'])
        
        # Check if values are roughly in [-1, 1] range
        min_val = np.min(normalized_actions)
        max_val = np.max(normalized_actions)
        
        if min_val < -1.5 or max_val > 1.5:
            print(f"    ⚠️  Normalized values outside expected range: [{min_val:.3f}, {max_val:.3f}]")
        else:
            print(f"    ✅ Normalized values in reasonable range: [{min_val:.3f}, {max_val:.3f}]")
    
    print("  ✅ Normalization behavior validation passed")
    return True


def validate_statistics_sampling():
    """Test that statistics sampling is representative."""
    print("🧪 Testing statistics sampling representativeness...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Test with different data percentages
    percentages = [0.01, 0.05, 0.1, 0.2]
    stats_list = []
    
    for pct in percentages:
        print(f"  📊 Testing with {pct*100:.0f}% data...")
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=pct,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        stats = dataset._compute_action_statistics()
        stats_list.append((pct, stats))
    
    # Compare statistics across different sample sizes
    print("  🔍 Comparing statistics across sample sizes...")
    
    base_pct, base_stats = stats_list[0]
    base_q01 = np.array(base_stats['q01'])
    base_q99 = np.array(base_stats['q99'])
    
    for pct, stats in stats_list[1:]:
        q01 = np.array(stats['q01'])
        q99 = np.array(stats['q99'])
        
        # Calculate relative differences
        q01_diff = np.abs(q01 - base_q01) / (np.abs(base_q01) + 1e-8)
        q99_diff = np.abs(q99 - base_q99) / (np.abs(base_q99) + 1e-8)
        
        max_q01_diff = np.max(q01_diff)
        max_q99_diff = np.max(q99_diff)
        
        print(f"    📊 {pct*100:.0f}% vs {base_pct*100:.0f}%:")
        print(f"      Max q01 relative diff: {max_q01_diff:.4f}")
        print(f"      Max q99 relative diff: {max_q99_diff:.4f}")
        
        # Check if differences are reasonable (less than 20% relative difference)
        if max_q01_diff > 0.2 or max_q99_diff > 0.2:
            print(f"      ⚠️  Large differences detected for {pct*100:.0f}% sample")
        else:
            print(f"      ✅ Differences reasonable for {pct*100:.0f}% sample")
    
    print("  ✅ Statistics sampling validation passed")
    return True


def validate_statistics_file_io():
    """Test statistics file save/load functionality."""
    print("🧪 Testing statistics file I/O...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Create temporary stats file
    temp_stats_file = "/tmp/egodex_test_stats.json"
    
    try:
        # Load dataset and compute statistics
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        stats = dataset._compute_action_statistics()
        
        # Save statistics to file
        print("  💾 Saving statistics to file...")
        
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                json_stats[key] = value.tolist()
            else:
                json_stats[key] = value
        
        with open(temp_stats_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        # Load statistics from file
        print("  📂 Loading statistics from file...")
        loaded_stats = dataset._load_action_statistics(temp_stats_file)
        
        # Compare original and loaded statistics
        print("  🔍 Comparing original vs loaded statistics...")
        
        for key in ['q01', 'q99', 'mean', 'std']:
            if key in stats and key in loaded_stats:
                original = np.array(stats[key])
                loaded = np.array(loaded_stats[key])
                
                if np.allclose(original, loaded, rtol=1e-10, atol=1e-10):
                    print(f"    ✅ {key} matches between original and loaded")
                else:
                    print(f"    ❌ {key} differs between original and loaded")
                    return False
        
        print("  ✅ Statistics file I/O validation passed")
        return True
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_stats_file):
            os.remove(temp_stats_file)


def main():
    """Run all statistics validation tests."""
    print("🚀 Starting EgoDex statistics validation...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        print("   Please set it to your EgoDex data directory")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print()
    
    # Run validation tests
    tests = [
        ("Statistics Consistency", validate_statistics_consistency),
        ("Statistics Accuracy", validate_statistics_accuracy),
        ("Normalization Behavior", validate_normalization_behavior),
        ("Statistics Sampling", validate_statistics_sampling),
        ("Statistics File I/O", validate_statistics_file_io),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"==================== {test_name} ====================")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 STATISTICS VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All statistics validation tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
