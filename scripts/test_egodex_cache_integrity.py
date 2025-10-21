#!/usr/bin/env python3
"""
Test script to verify EgoDex caching works correctly and data integrity is maintained.
"""

import os
import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from olmo.data.egodex_dataset import EgoDexPoseActions


def test_cache_integrity():
    """Test that caching works correctly and data is not mixed up."""
    print("🧪 EgoDex Cache Integrity Test")
    print("=" * 50)
    
    # Initialize dataset
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/jose/molmoact/data/egodex/organized')
    
    dataset = EgoDexPoseActions(
        data_dir=data_dir,
        split='test',
        data_percentage=0.01,  # Small subset for testing
        normalize_actions=False,
        discretize_actions=False
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} timesteps")
    
    # Test 1: Verify same timestep returns identical data
    print("\n🔍 Test 1: Same timestep consistency")
    rng = np.random.RandomState(42)
    test_idx = 100
    
    # Get sample twice
    sample1 = dataset.get(test_idx, rng)
    sample2 = dataset.get(test_idx, rng)
    
    # Check key fields
    timestep_info = dataset.timesteps[test_idx]
    episode = timestep_info['episode']
    timestep = timestep_info['timestep']
    
    print(f"  Testing timestep {test_idx}: episode {episode['episode_id']}, timestep {timestep}")
    
    # Verify consistency
    checks = [
        ("Question", sample1['question'] == sample2['question']),
        ("Answer length", len(sample1['answers']) == len(sample2['answers'])),
        ("Image size", sample1['image'][0].size == sample2['image'][0].size),
        ("Annotation task", sample1['annotation']['task_name'] == sample2['annotation']['task_name']),
        ("Annotation episode", sample1['annotation']['episode_id'] == sample2['annotation']['episode_id']),
        ("Annotation timestep", sample1['annotation']['timestep'] == sample2['annotation']['timestep']),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"    {status} {check_name}: {passed}")
        if not passed:
            all_passed = False
    
    print(f"  Result: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    
    # Test 2: Verify different timesteps return different data
    print("\n🔍 Test 2: Different timestep uniqueness")
    test_indices = [100, 200, 300, 400, 500]
    samples = []
    
    for idx in test_indices:
        sample = dataset.get(idx, rng)
        timestep_info = dataset.timesteps[idx]
        samples.append({
            'idx': idx,
            'episode_id': timestep_info['episode']['episode_id'],
            'timestep': timestep_info['timestep'],
            'task_name': timestep_info['episode']['task_name'],
            'question': sample['question'],
            'answer_length': len(sample['answers'])
        })
    
    # Check uniqueness
    unique_questions = len(set(s['question'] for s in samples))
    unique_episodes = len(set(s['episode_id'] for s in samples))
    unique_timesteps = len(set((s['episode_id'], s['timestep']) for s in samples))
    
    print(f"  Unique questions: {unique_questions}/{len(samples)}")
    print(f"  Unique episodes: {unique_episodes}")
    print(f"  Unique (episode, timestep) pairs: {unique_timesteps}/{len(samples)}")
    
    uniqueness_passed = unique_timesteps == len(samples)
    print(f"  Result: {'✅ PASSED' if uniqueness_passed else '❌ FAILED'}")
    
    # Test 3: Cache effectiveness test
    print("\n🔍 Test 3: Cache effectiveness")
    
    # Clear caches
    dataset._pose_data_cache.clear()
    dataset._frame_cache.clear()
    dataset._video_handles.clear()
    
    # Test repeated access to same episode
    episode_timesteps = []
    for i, timestep_info in enumerate(dataset.timesteps[:1000]):  # Check first 1000
        if timestep_info['episode']['episode_id'] == '0':
            episode_timesteps.append(i)
            if len(episode_timesteps) >= 5:  # Get 5 timesteps from episode 0
                break
    
    if len(episode_timesteps) >= 2:
        print(f"  Testing cache with {len(episode_timesteps)} timesteps from episode 0")
        
        # First pass (cache miss)
        start_time = time.time()
        for idx in episode_timesteps:
            dataset.get(idx, rng)
        first_pass_time = time.time() - start_time
        
        # Second pass (cache hit)
        start_time = time.time()
        for idx in episode_timesteps:
            dataset.get(idx, rng)
        second_pass_time = time.time() - start_time
        
        speedup = first_pass_time / second_pass_time if second_pass_time > 0 else float('inf')
        
        print(f"    First pass (cache miss): {first_pass_time:.3f}s")
        print(f"    Second pass (cache hit): {second_pass_time:.3f}s")
        print(f"    Speedup: {speedup:.1f}x")
        print(f"    Cache size - poses: {len(dataset._pose_data_cache)}, frames: {len(dataset._frame_cache)}")
        
        cache_passed = speedup > 2.0  # Expect at least 2x speedup
        print(f"  Result: {'✅ PASSED' if cache_passed else '❌ FAILED'}")
    else:
        print("  ⚠️  Not enough timesteps from episode 0 for cache test")
        cache_passed = True
    
    # Test 4: Data consistency across different random states
    print("\n🔍 Test 4: Random state independence")
    
    # Same timestep with different random states should return same data
    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(456)
    
    sample_rng1 = dataset.get(test_idx, rng1)
    sample_rng2 = dataset.get(test_idx, rng2)
    
    # Check that deterministic parts are the same
    deterministic_checks = [
        ("Question", sample_rng1['question'] == sample_rng2['question']),
        ("Image size", sample_rng1['image'][0].size == sample_rng2['image'][0].size),
        ("Annotation task", sample_rng1['annotation']['task_name'] == sample_rng2['annotation']['task_name']),
        ("Annotation episode", sample_rng1['annotation']['episode_id'] == sample_rng2['annotation']['episode_id']),
        ("Annotation timestep", sample_rng1['annotation']['timestep'] == sample_rng2['annotation']['timestep']),
    ]
    
    rng_independence_passed = all(passed for _, passed in deterministic_checks)
    for check_name, passed in deterministic_checks:
        status = "✅" if passed else "❌"
        print(f"    {status} {check_name}: {passed}")
    
    print(f"  Result: {'✅ PASSED' if rng_independence_passed else '❌ FAILED'}")
    
    # Test 5: Memory usage and cleanup
    print("\n🔍 Test 5: Memory management")
    
    initial_pose_cache = len(dataset._pose_data_cache)
    initial_frame_cache = len(dataset._frame_cache)
    initial_video_handles = len(dataset._video_handles)
    
    print(f"  Initial cache sizes:")
    print(f"    Pose data: {initial_pose_cache}")
    print(f"    Frames: {initial_frame_cache}")
    print(f"    Video handles: {initial_video_handles}")
    
    # Test cleanup
    dataset.__del__()
    
    final_pose_cache = len(dataset._pose_data_cache)
    final_frame_cache = len(dataset._frame_cache)
    final_video_handles = len(dataset._video_handles)
    
    print(f"  After cleanup:")
    print(f"    Pose data: {final_pose_cache}")
    print(f"    Frames: {final_frame_cache}")
    print(f"    Video handles: {final_video_handles}")
    
    cleanup_passed = (final_pose_cache == 0 and 
                     final_frame_cache == 0 and 
                     final_video_handles == 0)
    print(f"  Result: {'✅ PASSED' if cleanup_passed else '❌ FAILED'}")
    
    # Overall results
    print("\n" + "=" * 50)
    print("🎯 OVERALL RESULTS:")
    
    all_tests = [
        ("Same timestep consistency", all_passed),
        ("Different timestep uniqueness", uniqueness_passed),
        ("Cache effectiveness", cache_passed),
        ("Random state independence", rng_independence_passed),
        ("Memory cleanup", cleanup_passed)
    ]
    
    passed_tests = sum(1 for _, passed in all_tests if passed)
    total_tests = len(all_tests)
    
    for test_name, passed in all_tests:
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Caching works correctly and data integrity is maintained.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed_tests == total_tests


if __name__ == '__main__':
    success = test_cache_integrity()
    sys.exit(0 if success else 1)
