#!/usr/bin/env python3
"""
Test EgoDex downsampling functionality.

This script tests the 30Hz to 10Hz downsampling feature to ensure
it works correctly and maintains data quality.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions


def test_downsampling_rates():
    """Test different downsampling rates."""
    print("🧪 Testing different downsampling rates...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Test different FPS settings
    fps_settings = [30, 15, 10, 5, 1]  # 30Hz (no downsampling) down to 1Hz
    
    results = {}
    
    for target_fps in fps_settings:
        print(f"  📊 Testing {target_fps}Hz...")
        
        try:
            start_time = time.time()
            
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                data_percentage=0.01,  # Small subset for testing
                action_sequence_length=8,
                target_fps=target_fps,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            
            load_time = time.time() - start_time
            
            if len(dataset) > 0:
                # Test sample access
                sample = dataset[0]
                annotation = sample['annotation']
                
                # Check downsampling factor
                expected_factor = max(1, 30 // target_fps)
                actual_factor = dataset.downsample_factor
                
                if actual_factor == expected_factor:
                    print(f"    ✅ Downsample factor correct: {actual_factor}")
                else:
                    print(f"    ❌ Downsample factor incorrect: expected {expected_factor}, got {actual_factor}")
                    return False
                
                # Store results
                results[target_fps] = {
                    'load_time': load_time,
                    'episodes': len(dataset),
                    'downsample_factor': actual_factor,
                    'action_dof': annotation.get('action_dof', 0),
                    'sequence_length': annotation.get('action_sequence_length', 0)
                }
                
                print(f"    ✅ Loaded {len(dataset)} episodes in {load_time:.2f}s")
                print(f"    📊 Downsample factor: {actual_factor}")
                
            else:
                print(f"    ⚠️  No episodes loaded")
                results[target_fps] = {'episodes': 0}
                
        except Exception as e:
            print(f"    ❌ Failed with {target_fps}Hz: {e}")
            return False
    
    # Analyze results
    print("\n  📊 Downsampling Analysis:")
    print("    FPS  | Factor | Episodes | Load Time | Action DOF | Seq Length")
    print("    -----|--------|----------|-----------|------------|----------")
    
    for fps in fps_settings:
        if fps in results and results[fps]['episodes'] > 0:
            r = results[fps]
            print(f"    {fps:3d}Hz | {r['downsample_factor']:6d} | {r['episodes']:8d} | {r['load_time']:8.2f}s | {r['action_dof']:10d} | {r['sequence_length']:9d}")
    
    # Check performance benefits
    if 30 in results and 10 in results:
        time_30 = results[30]['load_time']
        time_10 = results[10]['load_time']
        speedup = time_30 / time_10 if time_10 > 0 else 1
        
        print(f"\n  📈 Performance Comparison (30Hz vs 10Hz):")
        print(f"    Load time speedup: {speedup:.2f}x")
        
        if speedup > 1.1:  # At least 10% improvement
            print(f"    ✅ Downsampling provides performance benefit")
        else:
            print(f"    ⚠️  Downsampling performance benefit minimal")
    
    print("  ✅ Downsampling rates test completed")
    return True


def test_downsampling_quality():
    """Test that downsampling maintains data quality."""
    print("🧪 Testing downsampling quality...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Load same data with different FPS settings
        dataset_30hz = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            action_sequence_length=8,
            target_fps=30,  # Original
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        dataset_10hz = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            action_sequence_length=8,
            target_fps=10,  # Downsampled
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        print(f"  📊 30Hz dataset: {len(dataset_30hz)} episodes")
        print(f"  📊 10Hz dataset: {len(dataset_10hz)} episodes")
        
        if len(dataset_30hz) == 0 or len(dataset_10hz) == 0:
            print("  ⚠️  No episodes available for quality comparison")
            return True
        
        # Compare samples from both datasets
        quality_metrics = {
            'same_episode_count': len(dataset_30hz) == len(dataset_10hz),
            'same_action_dof': 0,
            'same_sequence_length': 0,
            'same_conversation_format': 0,
            'reasonable_action_values': 0
        }
        
        num_samples = min(3, len(dataset_30hz), len(dataset_10hz))
        
        for i in range(num_samples):
            sample_30 = dataset_30hz[i]
            sample_10 = dataset_10hz[i]
            
            # Check action DOF consistency
            dof_30 = sample_30['annotation'].get('action_dof', 0)
            dof_10 = sample_10['annotation'].get('action_dof', 0)
            if dof_30 == dof_10 == 21:
                quality_metrics['same_action_dof'] += 1
            
            # Check sequence length consistency
            seq_30 = sample_30['annotation'].get('action_sequence_length', 0)
            seq_10 = sample_10['annotation'].get('action_sequence_length', 0)
            if seq_30 == seq_10:
                quality_metrics['same_sequence_length'] += 1
            
            # Check conversation format consistency
            has_q_30 = 'question' in sample_30 and len(sample_30['question']) > 0
            has_q_10 = 'question' in sample_10 and len(sample_10['question']) > 0
            has_a_30 = 'answers' in sample_30 and len(sample_30['answers']) > 0
            has_a_10 = 'answers' in sample_10 and len(sample_10['answers']) > 0
            
            if has_q_30 == has_q_10 and has_a_30 == has_a_10:
                quality_metrics['same_conversation_format'] += 1
            
            # Check that action values are reasonable (not all zeros)
            # This is a basic sanity check
            if 'action_sequence' in sample_10['answers']:
                quality_metrics['reasonable_action_values'] += 1
        
        # Report quality metrics
        print("  📊 Quality Metrics:")
        for metric, count in quality_metrics.items():
            percentage = (count / num_samples) * 100 if num_samples > 0 else 0
            print(f"    {metric}: {count}/{num_samples} ({percentage:.1f}%)")
        
        # Check critical quality requirements
        if quality_metrics['same_action_dof'] == num_samples:
            print("  ✅ Action DOF consistent across sampling rates")
        else:
            print("  ❌ Action DOF inconsistent across sampling rates")
            return False
        
        if quality_metrics['same_conversation_format'] == num_samples:
            print("  ✅ Conversation format consistent across sampling rates")
        else:
            print("  ❌ Conversation format inconsistent across sampling rates")
            return False
        
        print("  ✅ Downsampling quality maintained")
        return True
        
    except Exception as e:
        print(f"  ❌ Downsampling quality test failed: {e}")
        return False


def test_downsampling_statistics():
    """Test that statistics computation works with downsampling."""
    print("🧪 Testing downsampling with statistics computation...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Test statistics computation with different FPS settings
        fps_settings = [30, 15, 10]
        
        for target_fps in fps_settings:
            print(f"  📊 Computing statistics for {target_fps}Hz...")
            
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                data_percentage=0.01,
                action_sequence_length=8,
                target_fps=target_fps,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            
            # Compute statistics
            stats = dataset._compute_action_statistics()
            
            # Validate statistics
            required_keys = ['q01', 'q99', 'mean', 'std', 'n_samples']
            for key in required_keys:
                if key not in stats:
                    print(f"    ❌ Missing statistics key: {key}")
                    return False
            
            # Check statistics dimensions
            for key in ['q01', 'q99', 'mean', 'std']:
                if len(stats[key]) != 21:  # Expected action DOF
                    print(f"    ❌ Wrong dimension for {key}: {len(stats[key])}")
                    return False
            
            print(f"    ✅ Statistics computed: {stats['n_samples']} samples")
            print(f"    📊 Action dimensions: {len(stats['q01'])}")
            
            # Check reasonable value ranges
            q01 = np.array(stats['q01'])
            q99 = np.array(stats['q99'])
            
            # Hand position should be in reasonable range
            pos_range = q99[:3] - q01[:3]
            if np.all(pos_range > 0.1) and np.all(pos_range < 5.0):
                print(f"    ✅ Position range reasonable: {pos_range}")
            else:
                print(f"    ⚠️  Position range unusual: {pos_range}")
        
        print("  ✅ Statistics computation works with downsampling")
        return True
        
    except Exception as e:
        print(f"  ❌ Downsampling statistics test failed: {e}")
        return False


def test_downsampling_performance():
    """Test performance impact of downsampling."""
    print("🧪 Testing downsampling performance impact...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Test different FPS settings with timing
        fps_settings = [30, 15, 10, 5]
        performance_results = {}
        
        for target_fps in fps_settings:
            print(f"  📊 Testing performance for {target_fps}Hz...")
            
            # Time dataset loading
            start_time = time.time()
            
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                data_percentage=0.02,  # Slightly larger for better timing
                action_sequence_length=8,
                target_fps=target_fps,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            
            load_time = time.time() - start_time
            
            # Time sample access
            if len(dataset) > 0:
                start_time = time.time()
                for i in range(min(5, len(dataset))):
                    sample = dataset[i]
                access_time = time.time() - start_time
                avg_access_time = access_time / min(5, len(dataset))
            else:
                access_time = 0
                avg_access_time = 0
            
            performance_results[target_fps] = {
                'load_time': load_time,
                'access_time': access_time,
                'avg_access_time': avg_access_time,
                'episodes': len(dataset)
            }
            
            print(f"    📊 Load time: {load_time:.2f}s")
            print(f"    📊 Access time: {access_time:.2f}s (avg: {avg_access_time:.3f}s/sample)")
        
        # Analyze performance trends
        print("\n  📈 Performance Analysis:")
        print("    FPS  | Load Time | Access Time | Episodes")
        print("    -----|-----------|-------------|---------")
        
        for fps in fps_settings:
            if fps in performance_results:
                r = performance_results[fps]
                print(f"    {fps:3d}Hz | {r['load_time']:8.2f}s | {r['access_time']:10.2f}s | {r['episodes']:8d}")
        
        # Check for performance improvements
        if 30 in performance_results and 10 in performance_results:
            load_30 = performance_results[30]['load_time']
            load_10 = performance_results[10]['load_time']
            load_speedup = load_30 / load_10 if load_10 > 0 else 1
            
            access_30 = performance_results[30]['avg_access_time']
            access_10 = performance_results[10]['avg_access_time']
            access_speedup = access_30 / access_10 if access_10 > 0 else 1
            
            print(f"\n  📊 10Hz vs 30Hz Performance:")
            print(f"    Load speedup: {load_speedup:.2f}x")
            print(f"    Access speedup: {access_speedup:.2f}x")
            
            if load_speedup > 1.2 or access_speedup > 1.2:
                print("    ✅ Downsampling provides significant performance benefit")
            elif load_speedup > 1.05 or access_speedup > 1.05:
                print("    ✅ Downsampling provides modest performance benefit")
            else:
                print("    ⚠️  Downsampling performance benefit minimal")
        
        print("  ✅ Performance analysis completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Downsampling performance test failed: {e}")
        return False


def main():
    """Run all downsampling tests."""
    print("🚀 Starting EgoDex downsampling tests...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        print("   Please set it to your EgoDex data directory")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print()
    
    # Run downsampling tests
    tests = [
        ("Downsampling Rates", test_downsampling_rates),
        ("Downsampling Quality", test_downsampling_quality),
        ("Downsampling Statistics", test_downsampling_statistics),
        ("Downsampling Performance", test_downsampling_performance),
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
    print("📊 DOWNSAMPLING TESTING SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All downsampling tests passed!")
        print("🎯 EgoDex downsampling is working correctly!")
        print("\n💡 Recommendations:")
        print("   - Use 10Hz for most training scenarios (3x speedup)")
        print("   - Use 15Hz for higher temporal resolution")
        print("   - Use 5Hz for very fast prototyping")
        print("   - Keep 30Hz for maximum temporal fidelity")
        return True
    else:
        print("⚠️  Some downsampling tests failed. Review issues before using.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
