#!/usr/bin/env python3
"""
Test EgoDex performance improvements with pose data caching.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_egodex_performance():
    """Test EgoDex loading performance with and without caching."""
    from olmo.data.egodex_dataset import EgoDexPoseActions
    import numpy as np
    
    print("🚀 Testing EgoDex Performance Improvements")
    print("=" * 50)
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/jose/molmoact/data/egodex/organized')
    
    try:
        # Initialize dataset with very small data percentage for testing
        print("📊 Initializing EgoDex dataset...")
        start_time = time.time()
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.001,  # Very small for quick test
            normalize_actions=False,
            discretize_actions=False
        )
        init_time = time.time() - start_time
        print(f"   ⏱️  Initialization: {init_time:.2f}s")
        print(f"   📈 Episodes: {len(dataset.episodes)}")
        print(f"   📈 Timesteps: {len(dataset.timesteps)}")
        
        if len(dataset) == 0:
            print("   ⚠️  No data available for testing")
            return
        
        # Test repeated access to same episode (should hit cache)
        print(f"\n🔄 Testing pose data caching performance...")
        rng = np.random.RandomState(42)
        
        # Get a few sample indices from the same episode
        sample_indices = []
        episode_ids = set()
        for i in range(min(10, len(dataset))):
            timestep_info = dataset.timesteps[i]
            episode_id = timestep_info['episode']['episode_id']
            episode_ids.add(episode_id)
            if len(sample_indices) < 5:  # Get up to 5 samples
                sample_indices.append(i)
        
        print(f"   📊 Testing with {len(sample_indices)} samples from {len(episode_ids)} episodes")
        
        # First pass - should populate cache
        print("   🔥 First pass (cache miss):")
        first_pass_times = []
        for i, item_idx in enumerate(sample_indices):
            start_time = time.time()
            sample = dataset.get(item_idx, rng)
            get_time = time.time() - start_time
            first_pass_times.append(get_time)
            
            timestep_info = dataset.timesteps[item_idx]
            print(f"      Sample {i+1}: {get_time:.3f}s (ep{timestep_info['episode']['episode_id']})")
        
        # Second pass - should hit cache
        print("   ⚡ Second pass (cache hit):")
        second_pass_times = []
        for i, item_idx in enumerate(sample_indices):
            start_time = time.time()
            sample = dataset.get(item_idx, rng)
            get_time = time.time() - start_time
            second_pass_times.append(get_time)
            
            timestep_info = dataset.timesteps[item_idx]
            print(f"      Sample {i+1}: {get_time:.3f}s (ep{timestep_info['episode']['episode_id']})")
        
        # Calculate improvement
        avg_first = sum(first_pass_times) / len(first_pass_times)
        avg_second = sum(second_pass_times) / len(second_pass_times)
        speedup = avg_first / avg_second if avg_second > 0 else float('inf')
        
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   🔥 First pass (cache miss): {avg_first:.3f}s avg")
        print(f"   ⚡ Second pass (cache hit): {avg_second:.3f}s avg")
        print(f"   🚀 Speedup: {speedup:.1f}x faster with caching")
        
        # Show cache statistics
        cache_size = len(dataset._pose_data_cache)
        print(f"   💾 Cache size: {cache_size} pose data files cached")
        
        if speedup > 2:
            print(f"   ✅ Caching is working effectively!")
        elif speedup > 1.5:
            print(f"   ⚠️  Caching provides some improvement")
        else:
            print(f"   ❌ Caching may not be working as expected")
        
        print(f"\n💡 The pose data caching should provide significant speedup")
        print(f"   when accessing multiple timesteps from the same episodes,")
        print(f"   which is common during training.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_egodex_performance()
