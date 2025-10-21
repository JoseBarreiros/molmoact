#!/usr/bin/env python3
"""
Test script to analyze EgoDex sampling logic and understand episode indexing.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from olmo.data.egodex_dataset import EgoDexPoseActions


def test_episode_indexing(dataset):
    """Test episode indexing and timestep organization."""
    print("🔍 Testing Episode Indexing")
    print("=" * 50)
    
    # Analyze the first 500 timesteps
    sample_size = min(500, len(dataset.timesteps))
    
    episode_stats = {}
    episode_ranges = {}
    
    for i in range(sample_size):
        timestep_info = dataset.timesteps[i]
        episode_id = timestep_info['episode']['episode_id']
        timestep = timestep_info['timestep']
        
        if episode_id not in episode_stats:
            episode_stats[episode_id] = {
                'count': 0,
                'timesteps': [],
                'task_name': timestep_info['episode']['task_name'],
                'video_path': timestep_info['episode']['video_path']
            }
            episode_ranges[episode_id] = {'start': i, 'end': i}
        
        episode_stats[episode_id]['count'] += 1
        episode_stats[episode_id]['timesteps'].append(timestep)
        episode_ranges[episode_id]['end'] = i
    
    print(f"📊 Episode Analysis (first {sample_size} timesteps):")
    for episode_id in sorted(episode_stats.keys()):
        stats = episode_stats[episode_id]
        ranges = episode_ranges[episode_id]
        timesteps = sorted(stats['timesteps'])
        
        print(f"\n  Episode {episode_id}:")
        print(f"    Task: {stats['task_name']}")
        print(f"    Dataset range: {ranges['start']} to {ranges['end']}")
        print(f"    Timestep count: {stats['count']}")
        print(f"    Timestep range: {min(timesteps)} to {max(timesteps)}")
        print(f"    Video: {Path(stats['video_path']).name}")
        
        # Check for gaps or issues
        expected_timesteps = list(range(min(timesteps), max(timesteps) + 1))
        missing = set(expected_timesteps) - set(timesteps)
        if missing:
            print(f"    ⚠️  Missing timesteps: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
        
        # Check for duplicates
        if len(timesteps) != len(set(timesteps)):
            duplicates = len(timesteps) - len(set(timesteps))
            print(f"    ⚠️  Duplicate timesteps: {duplicates}")
    
    return episode_stats


def test_episode_finding_logic(dataset, target_episode_id="0"):
    """Test the episode finding logic specifically."""
    print(f"\n🔍 Testing Episode Finding Logic for Episode '{target_episode_id}'")
    print("=" * 60)
    
    # Simulate the find_episode_timesteps function
    episode_timesteps = []
    
    for idx, timestep_info in enumerate(dataset.timesteps):
        if timestep_info['episode']['episode_id'] == target_episode_id:
            episode_timesteps.append(idx)
            
            # Show first few and last few for debugging
            if len(episode_timesteps) <= 10:
                print(f"  Found timestep {len(episode_timesteps)}: dataset_idx={idx}, "
                      f"actual_timestep={timestep_info['timestep']}")
    
    print(f"\n📊 Results for episode '{target_episode_id}':")
    print(f"  Total timesteps found: {len(episode_timesteps)}")
    
    if len(episode_timesteps) > 10:
        print(f"  First 10 dataset indices: {episode_timesteps[:10]}")
        print(f"  Last 10 dataset indices: {episode_timesteps[-10:]}")
    
    # Analyze the actual timestep values
    actual_timesteps = []
    for idx in episode_timesteps[:100]:  # Check first 100 to avoid too much output
        timestep_info = dataset.timesteps[idx]
        actual_timesteps.append(timestep_info['timestep'])
    
    print(f"  Actual timestep values (first 100): {actual_timesteps[:20]}{'...' if len(actual_timesteps) > 20 else ''}")
    
    if len(episode_timesteps) > 100:
        last_100_indices = episode_timesteps[-100:]
        last_timesteps = []
        for idx in last_100_indices:
            timestep_info = dataset.timesteps[idx]
            last_timesteps.append(timestep_info['timestep'])
        print(f"  Actual timestep values (last 100): {last_timesteps[-20:]}{'...' if len(last_timesteps) > 20 else ''}")
    
    # Check for consistency
    min_timestep = min(actual_timesteps)
    max_timestep = max(actual_timesteps)
    expected_count = max_timestep - min_timestep + 1
    
    print(f"  Timestep range: {min_timestep} to {max_timestep}")
    print(f"  Expected count: {expected_count}")
    print(f"  Actual count: {len(episode_timesteps)}")
    
    if len(episode_timesteps) != expected_count:
        print(f"  ⚠️  Mismatch! Expected {expected_count}, got {len(episode_timesteps)}")
    
    return episode_timesteps


def test_dataset_consistency(dataset):
    """Test overall dataset consistency."""
    print(f"\n🔍 Testing Dataset Consistency")
    print("=" * 50)
    
    print(f"Total timesteps in dataset: {len(dataset.timesteps)}")
    print(f"Total episodes in dataset: {len(dataset.episodes)}")
    
    # Check for episode ID type consistency
    episode_id_types = set()
    for timestep_info in dataset.timesteps[:100]:
        ep_id = timestep_info['episode']['episode_id']
        episode_id_types.add(type(ep_id).__name__)
    
    print(f"Episode ID types found: {episode_id_types}")
    
    # Check if episode IDs match dataset episodes
    dataset_episode_ids = set()
    for episode in dataset.episodes:
        dataset_episode_ids.add(episode['episode_id'])
    
    timestep_episode_ids = set()
    for timestep_info in dataset.timesteps[:1000]:  # Check first 1000
        timestep_episode_ids.add(timestep_info['episode']['episode_id'])
    
    print(f"Dataset episode IDs (first 10): {sorted(list(dataset_episode_ids))[:10]}")
    print(f"Timestep episode IDs (first 1000): {sorted(list(timestep_episode_ids))}")
    
    # Check if there are timesteps that don't correspond to dataset episodes
    orphan_timesteps = timestep_episode_ids - dataset_episode_ids
    if orphan_timesteps:
        print(f"⚠️  Orphan timesteps with episode IDs not in dataset: {orphan_timesteps}")


def main():
    print("🧪 EgoDex Sampling Logic Test")
    print("=" * 60)
    
    # Initialize dataset with small percentage for testing
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/jose/molmoact/data/egodex/organized')
    
    try:
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.001,  # Very small for testing
            normalize_actions=False,
            discretize_actions=False
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total timesteps: {len(dataset.timesteps)}")
        print(f"📊 Total episodes: {len(dataset.episodes)}")
        
        # Run tests
        test_dataset_consistency(dataset)
        episode_stats = test_episode_indexing(dataset)
        test_episode_finding_logic(dataset, "0")
        
        # Test a different episode if available
        available_episodes = list(episode_stats.keys())[:5]
        if len(available_episodes) > 1:
            other_episode = available_episodes[1]
            test_episode_finding_logic(dataset, other_episode)
        
        print(f"\n🎉 Tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
