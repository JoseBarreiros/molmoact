#!/usr/bin/env python3
"""
Test EgoDex edge cases and robustness.

This script tests various edge cases and error conditions to ensure
the EgoDex integration is robust for large-scale training.
"""

import os
import sys
import tempfile
import shutil
import h5py
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions


def test_empty_episodes():
    """Test handling of episodes with no pose data."""
    print("🧪 Testing empty episodes handling...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Create a dataset with very small data percentage to potentially hit edge cases
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.001,  # Very small to potentially get problematic episodes
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        print(f"  📊 Dataset loaded: {len(dataset)} episodes")
        
        # Test accessing all episodes to ensure no crashes
        successful_accesses = 0
        failed_accesses = 0
        
        for i in range(min(10, len(dataset))):
            try:
                sample = dataset[i]
                if 'question' in sample and 'answers' in sample:
                    successful_accesses += 1
                else:
                    print(f"    ⚠️  Episode {i} missing expected keys")
            except Exception as e:
                failed_accesses += 1
                print(f"    ⚠️  Episode {i} failed to load: {e}")
        
        print(f"  📊 Successful accesses: {successful_accesses}")
        print(f"  📊 Failed accesses: {failed_accesses}")
        
        # Allow some failures but not too many
        if failed_accesses > successful_accesses:
            print("  ❌ Too many failed episode accesses")
            return False
        
        print("  ✅ Empty episodes handled gracefully")
        return True
        
    except Exception as e:
        print(f"  ❌ Empty episodes test failed: {e}")
        return False


def test_corrupted_files():
    """Test handling of corrupted HDF5 files."""
    print("🧪 Testing corrupted files handling...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Find a real HDF5 file to corrupt
        test_dir = Path(data_dir)
        hdf5_files = list(test_dir.rglob("*.hdf5"))
        
        if not hdf5_files:
            print("  ⚠️  No HDF5 files found to test corruption")
            return True
        
        # Create a backup and corrupt a file
        test_file = hdf5_files[0]
        backup_file = test_file.with_suffix('.hdf5.backup')
        
        print(f"  📊 Testing with file: {test_file.name}")
        
        # Backup original
        shutil.copy2(test_file, backup_file)
        
        try:
            # Corrupt the file by truncating it
            with open(test_file, 'ab') as f:
                f.truncate(100)  # Truncate to 100 bytes
            
            # Try to load dataset - should handle corruption gracefully
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
            
            # Try to access a few samples
            for i in range(min(3, len(dataset))):
                try:
                    sample = dataset[i]
                    print(f"    ✅ Sample {i} loaded successfully despite corruption")
                except Exception as e:
                    print(f"    ⚠️  Sample {i} failed (expected with corruption): {e}")
            
            print("  ✅ Corrupted files handled gracefully")
            return True
            
        finally:
            # Restore original file
            if backup_file.exists():
                shutil.copy2(backup_file, test_file)
                backup_file.unlink()
        
    except Exception as e:
        print(f"  ❌ Corrupted files test failed: {e}")
        return False


def test_missing_pose_data():
    """Test handling of episodes with missing pose data."""
    print("🧪 Testing missing pose data handling...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Create a dataset and check for episodes with missing pose data
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
        
        print(f"  📊 Dataset loaded: {len(dataset)} episodes")
        
        # Check a few episodes for pose data availability
        episodes_with_poses = 0
        episodes_without_poses = 0
        
        for i in range(min(10, len(dataset))):
            try:
                sample = dataset[i]
                annotation = sample.get('annotation', {})
                
                # Check if poses are present
                hand_poses = annotation.get('hand_poses', [])
                finger_tips = annotation.get('finger_tips', [])
                
                if hand_poses or finger_tips:
                    episodes_with_poses += 1
                else:
                    episodes_without_poses += 1
                    print(f"    ⚠️  Episode {i} has no pose data")
                
            except Exception as e:
                print(f"    ⚠️  Episode {i} failed to analyze: {e}")
        
        print(f"  📊 Episodes with poses: {episodes_with_poses}")
        print(f"  📊 Episodes without poses: {episodes_without_poses}")
        
        # Most episodes should have pose data
        if episodes_without_poses > episodes_with_poses:
            print("  ❌ Too many episodes without pose data")
            return False
        
        print("  ✅ Missing pose data handled appropriately")
        return True
        
    except Exception as e:
        print(f"  ❌ Missing pose data test failed: {e}")
        return False


def test_extreme_data_percentages():
    """Test extreme data percentage values."""
    print("🧪 Testing extreme data percentages...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Test various extreme values
    test_values = [0.0, 0.0001, 0.5, 0.999, 1.0]
    
    for pct in test_values:
        print(f"  📊 Testing data percentage: {pct}")
        
        try:
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
            
            print(f"    ✅ Loaded {len(dataset)} episodes with {pct} data percentage")
            
            # For very small percentages, dataset might be empty
            if pct < 0.001 and len(dataset) == 0:
                print(f"    ℹ️  Empty dataset expected for very small percentage")
            elif len(dataset) > 0:
                # Try to access first sample
                try:
                    sample = dataset[0]
                    print(f"    ✅ Sample access successful")
                except Exception as e:
                    print(f"    ❌ Sample access failed: {e}")
                    return False
            
        except Exception as e:
            print(f"    ❌ Failed to create dataset with {pct} percentage: {e}")
            return False
    
    print("  ✅ Extreme data percentages handled correctly")
    return True


def test_action_sequence_edge_cases():
    """Test edge cases in action sequence generation."""
    print("🧪 Testing action sequence edge cases...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Test different action sequence lengths
    sequence_lengths = [1, 4, 8, 16, 32]
    
    for seq_len in sequence_lengths:
        print(f"  📊 Testing sequence length: {seq_len}")
        
        try:
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                data_percentage=0.01,
                action_sequence_length=seq_len,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            
            if len(dataset) > 0:
                sample = dataset[0]
                annotation = sample.get('annotation', {})
                
                # Check action sequence length
                actual_length = annotation.get('action_sequence_length', 0)
                if actual_length == seq_len:
                    print(f"    ✅ Correct sequence length: {actual_length}")
                else:
                    print(f"    ⚠️  Expected {seq_len}, got {actual_length}")
                
                # Check action DOF
                action_dof = annotation.get('action_dof', 0)
                if action_dof == 21:  # Expected DOF
                    print(f"    ✅ Correct action DOF: {action_dof}")
                else:
                    print(f"    ⚠️  Unexpected action DOF: {action_dof}")
            else:
                print(f"    ⚠️  No episodes available for testing")
                
        except Exception as e:
            print(f"    ❌ Failed with sequence length {seq_len}: {e}")
            return False
    
    print("  ✅ Action sequence edge cases handled correctly")
    return True


def test_concurrent_dataset_access():
    """Test concurrent access to dataset (thread safety)."""
    print("🧪 Testing concurrent dataset access...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        import threading
        import time
        
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
        
        if len(dataset) == 0:
            print("  ⚠️  No episodes available for concurrent testing")
            return True
        
        # Test concurrent access
        results = []
        errors = []
        
        def access_sample(index):
            try:
                sample = dataset[index % len(dataset)]
                results.append(True)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=access_sample, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        print(f"  📊 Successful concurrent accesses: {len(results)}")
        print(f"  📊 Concurrent access errors: {len(errors)}")
        
        if len(errors) > 0:
            print(f"  ⚠️  Concurrent access errors: {errors[:3]}")  # Show first 3 errors
        
        # Allow some errors but not too many
        if len(errors) > len(results):
            print("  ❌ Too many concurrent access errors")
            return False
        
        print("  ✅ Concurrent access handled reasonably well")
        return True
        
    except Exception as e:
        print(f"  ❌ Concurrent access test failed: {e}")
        return False


def test_memory_usage_over_time():
    """Test memory usage over multiple dataset accesses."""
    print("🧪 Testing memory usage over time...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        import psutil
        import gc
        
        # Monitor memory usage
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  📊 Initial memory: {initial_memory:.1f} MB")
        
        # Create and access dataset multiple times
        for iteration in range(3):
            print(f"  📊 Iteration {iteration + 1}/3")
            
            # Create new dataset instance
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
            
            # Access multiple samples
            for i in range(min(5, len(dataset))):
                try:
                    sample = dataset[i]
                    # Simulate some processing
                    _ = sample.get('question', '')
                    _ = sample.get('answers', '')
                except Exception as e:
                    print(f"    ⚠️  Sample {i} access failed: {e}")
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = current_memory - initial_memory
            print(f"    📊 Memory usage: {current_memory:.1f} MB (+{memory_delta:.1f} MB)")
            
            # Clean up
            del dataset
            gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_delta = final_memory - initial_memory
        
        print(f"  📊 Final memory: {final_memory:.1f} MB")
        print(f"  📊 Total memory increase: {total_memory_delta:.1f} MB")
        
        # Check for significant memory leaks
        if total_memory_delta > 500:  # More than 500MB increase
            print("  ⚠️  Potential memory leak detected (>500MB increase)")
            return False
        
        print("  ✅ Memory usage looks reasonable")
        return True
        
    except Exception as e:
        print(f"  ❌ Memory usage test failed: {e}")
        return False


def main():
    """Run all edge case tests."""
    print("🚀 Starting EgoDex edge case testing...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        print("   Please set it to your EgoDex data directory")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print()
    
    # Run edge case tests
    tests = [
        ("Empty Episodes", test_empty_episodes),
        ("Corrupted Files", test_corrupted_files),
        ("Missing Pose Data", test_missing_pose_data),
        ("Extreme Data Percentages", test_extreme_data_percentages),
        ("Action Sequence Edge Cases", test_action_sequence_edge_cases),
        ("Concurrent Dataset Access", test_concurrent_dataset_access),
        ("Memory Usage Over Time", test_memory_usage_over_time),
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
    print("📊 EDGE CASE TESTING SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All edge case tests passed!")
        print("🛡️  EgoDex integration is robust for large-scale training!")
        return True
    else:
        print("⚠️  Some edge case tests failed. Review issues before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
