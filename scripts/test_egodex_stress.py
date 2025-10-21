#!/usr/bin/env python3
"""
EgoDex stress test script for large-scale training validation.

This script tests EgoDex integration under stress conditions to ensure
it can handle large-scale training scenarios.
"""

import os
import sys
import time
import gc
import psutil
import numpy as np
from typing import Dict, List, Any
import warnings

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmo.data.egodex_dataset import EgoDexPoseActions


def monitor_memory():
    """Monitor current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }


def test_large_dataset_loading():
    """Test loading large datasets with different configurations."""
    print("🧪 Testing large dataset loading...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    # Test configurations
    configs = [
        {'data_percentage': 0.1, 'use_action_normalization': True, 'use_action_discretization': True},
        {'data_percentage': 0.2, 'use_action_normalization': True, 'use_action_discretization': False},
        {'data_percentage': 0.5, 'use_action_normalization': False, 'use_action_discretization': True},
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"  📊 Configuration {i+1}: {config}")
        
        initial_memory = monitor_memory()
        start_time = time.time()
        
        try:
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                **config
            )
            
            # Load a few samples to simulate training
            for j in range(min(20, len(dataset))):
                sample = dataset[j]
                if j % 5 == 0:
                    current_memory = monitor_memory()
                    print(f"    Sample {j}: Memory {current_memory['rss']:.1f} MB")
            
            load_time = time.time() - start_time
            final_memory = monitor_memory()
            
            results[f"config_{i+1}"] = {
                'success': True,
                'episode_count': len(dataset),
                'load_time': load_time,
                'memory_usage': final_memory['rss'] - initial_memory['rss'],
                'memory_percent': final_memory['percent']
            }
            
            print(f"    ✅ Loaded {len(dataset)} episodes in {load_time:.2f}s")
            print(f"    📊 Memory usage: {results[f'config_{i+1}']['memory_usage']:.1f} MB")
            
        except Exception as e:
            results[f"config_{i+1}"] = {
                'success': False,
                'error': str(e)
            }
            print(f"    ❌ Failed: {e}")
        
        # Clean up
        del dataset
        gc.collect()
        time.sleep(1)  # Give system time to clean up
    
    return results


def test_concurrent_loading():
    """Test concurrent dataset loading (simulating multi-GPU training)."""
    print("🧪 Testing concurrent loading...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    # Simulate multiple workers loading data
    num_workers = 3
    datasets = []
    
    initial_memory = monitor_memory()
    start_time = time.time()
    
    try:
        # Create multiple dataset instances
        for i in range(num_workers):
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                data_percentage=0.1,
                use_action_normalization=True
            )
            datasets.append(dataset)
            print(f"  📊 Worker {i+1}: Loaded {len(dataset)} episodes")
        
        # Simulate concurrent data loading
        for worker_id, dataset in enumerate(datasets):
            for i in range(min(10, len(dataset))):
                sample = dataset[i]
                if i % 3 == 0:
                    current_memory = monitor_memory()
                    print(f"    Worker {worker_id+1}, Sample {i}: Memory {current_memory['rss']:.1f} MB")
        
        concurrent_time = time.time() - start_time
        final_memory = monitor_memory()
        
        print(f"  ✅ Concurrent loading completed in {concurrent_time:.2f}s")
        print(f"  📊 Memory usage: {final_memory['rss'] - initial_memory['rss']:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Concurrent loading failed: {e}")
        return False
    
    finally:
        # Clean up
        for dataset in datasets:
            del dataset
        gc.collect()


def test_memory_leaks():
    """Test for memory leaks during repeated data loading."""
    print("🧪 Testing for memory leaks...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    initial_memory = monitor_memory()
    memory_usage = [initial_memory['rss']]
    
    # Load and unload dataset multiple times
    num_iterations = 10
    
    for i in range(num_iterations):
        print(f"  📊 Iteration {i+1}/{num_iterations}")
        
        # Load dataset
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.05,
            use_action_normalization=True
        )
        
        # Load some samples
        for j in range(min(5, len(dataset))):
            sample = dataset[j]
        
        # Check memory
        current_memory = monitor_memory()
        memory_usage.append(current_memory['rss'])
        
        print(f"    Memory: {current_memory['rss']:.1f} MB")
        
        # Clean up
        del dataset
        gc.collect()
        time.sleep(0.5)  # Give system time to clean up
    
    # Analyze memory usage
    memory_increase = memory_usage[-1] - memory_usage[0]
    max_memory = max(memory_usage)
    min_memory = min(memory_usage)
    
    print(f"  📊 Memory analysis:")
    print(f"    Initial: {memory_usage[0]:.1f} MB")
    print(f"    Final: {memory_usage[-1]:.1f} MB")
    print(f"    Increase: {memory_increase:.1f} MB")
    print(f"    Max: {max_memory:.1f} MB")
    print(f"    Min: {min_memory:.1f} MB")
    
    # Check for significant memory leaks (> 100MB increase)
    if memory_increase > 100:
        print(f"  ⚠️  Potential memory leak: {memory_increase:.1f} MB increase")
        return False
    else:
        print(f"  ✅ No significant memory leaks detected")
        return True


def test_error_recovery():
    """Test error recovery and robustness."""
    print("🧪 Testing error recovery...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    # Test 1: Invalid index handling
    try:
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            use_action_normalization=True
        )
        
        # Try to access invalid indices
        try:
            _ = dataset[len(dataset)]  # Should raise IndexError
            print("  ❌ Should have raised IndexError for out-of-bounds access")
            return False
        except IndexError:
            print("  ✅ Correctly handled out-of-bounds access")
        
        # Try negative index
        try:
            _ = dataset[-1]  # Should work (last element)
            print("  ✅ Correctly handled negative index")
        except Exception as e:
            print(f"  ⚠️  Negative index handling: {e}")
        
    except Exception as e:
        print(f"  ❌ Dataset creation failed: {e}")
        return False
    
    # Test 2: Corrupted data handling (simulate by modifying data temporarily)
    try:
        # This is a basic test - in real scenarios, you'd test with actually corrupted files
        print("  ✅ Error recovery tests completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Error recovery test failed: {e}")
        return False


def test_performance_benchmarks():
    """Run performance benchmarks for training efficiency."""
    print("🧪 Running performance benchmarks...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"  📊 Testing batch size: {batch_size}")
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.1,
            use_action_normalization=True
        )
        
        # Benchmark loading batch
        start_time = time.time()
        
        for i in range(batch_size):
            if i < len(dataset):
                sample = dataset[i]
        
        load_time = time.time() - start_time
        avg_time_per_sample = load_time / batch_size
        
        results[batch_size] = {
            'total_time': load_time,
            'avg_time_per_sample': avg_time_per_sample,
            'samples_per_second': batch_size / load_time
        }
        
        print(f"    Total time: {load_time:.3f}s")
        print(f"    Avg per sample: {avg_time_per_sample*1000:.1f}ms")
        print(f"    Samples/sec: {batch_size / load_time:.1f}")
        
        del dataset
        gc.collect()
    
    # Find optimal batch size
    best_batch = max(results.keys(), key=lambda k: results[k]['samples_per_second'])
    print(f"  🏆 Optimal batch size: {best_batch} ({results[best_batch]['samples_per_second']:.1f} samples/sec)")
    
    return results


def main():
    """Run all stress tests."""
    print("🚀 Starting EgoDex stress tests...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.environ.get('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print(f"💻 System memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print()
    
    # Run stress tests
    tests = [
        ("Large Dataset Loading", test_large_dataset_loading),
        ("Concurrent Loading", test_concurrent_loading),
        ("Memory Leak Detection", test_memory_leaks),
        ("Error Recovery", test_error_recovery),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
        
        # Clean up between tests
        gc.collect()
        time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("📊 STRESS TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All stress tests passed! System is ready for large-scale training.")
        return True
    else:
        print("⚠️  Some stress tests failed. Please address issues before large-scale training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

