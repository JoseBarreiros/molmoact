#!/usr/bin/env python3
"""
Quick EgoDex profiling script focusing on main bottlenecks.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from olmo.data.egodex_dataset import EgoDexPoseActions


def quick_profile():
    """Quick profile focusing on main bottlenecks."""
    print("🚀 Quick EgoDex Loading Profile")
    print("=" * 40)
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/jose/molmoact/data/egodex/organized')
    
    # Test 1: Dataset initialization time
    print("1️⃣  Testing dataset initialization...")
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
    print(f"   📊 Episodes: {len(dataset.episodes)}, Timesteps: {len(dataset.timesteps)}")
    
    if len(dataset) == 0:
        print("   ⚠️  No data available for profiling")
        return
    
    # Test 2: Individual get() calls timing
    print("\n2️⃣  Testing get() method performance...")
    rng = np.random.RandomState(42)
    
    # Test first few samples
    num_tests = min(5, len(dataset))
    get_times = []
    
    for i in range(num_tests):
        start_time = time.time()
        sample = dataset.get(i, rng)
        get_time = time.time() - start_time
        get_times.append(get_time)
        
        timestep_info = dataset.timesteps[i]
        print(f"   Sample {i}: {get_time:.3f}s (ep{timestep_info['episode']['episode_id']}_t{timestep_info['timestep']})")
    
    avg_get_time = np.mean(get_times)
    print(f"   ⏱️  Average get() time: {avg_get_time:.3f}s")
    
    # Test 3: Breakdown of a single get() call
    print(f"\n3️⃣  Detailed breakdown of sample 0...")
    timestep_info = dataset.timesteps[0]
    episode = timestep_info['episode']
    timestep = timestep_info['timestep']
    
    # Frame extraction
    start_time = time.time()
    frame = dataset._get_single_frame(episode['video_path'], timestep)
    frame_time = time.time() - start_time
    print(f"   🎬 Frame extraction: {frame_time:.3f}s")
    
    # Pose loading
    start_time = time.time()
    pose_data = dataset._load_pose_data(episode['pose_path'])
    pose_time = time.time() - start_time
    print(f"   📁 Pose loading: {pose_time:.3f}s")
    
    # Action chunk generation
    start_time = time.time()
    description = pose_data['metadata'].get('llm_description', f"Perform the {episode['task_name']} task")
    answer = dataset._get_timestep_action_chunk(pose_data, timestep, description)
    chunk_time = time.time() - start_time
    print(f"   ⚡ Action chunk generation: {chunk_time:.3f}s")
    
    total_breakdown = frame_time + pose_time + chunk_time
    print(f"   📊 Breakdown total: {total_breakdown:.3f}s")
    
    # Test 4: Video file access speed
    print(f"\n4️⃣  Testing video file access...")
    cv2_times = []
    for i in range(min(3, len(dataset.episodes))):
        episode = dataset.episodes[i]
        start_time = time.time()
        cap = cv2.VideoCapture(episode['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()
        cv_time = time.time() - start_time
        cv2_times.append(cv_time)
        print(f"   Episode {episode['episode_id']}: {cv_time:.3f}s")
    
    avg_cv_time = np.mean(cv2_times)
    print(f"   ⏱️  Average OpenCV time: {avg_cv_time:.3f}s")
    
    # Summary and recommendations
    print(f"\n📊 SUMMARY & RECOMMENDATIONS")
    print("=" * 40)
    
    if init_time > 5:
        print(f"🐌 Dataset initialization is slow ({init_time:.1f}s)")
        print("   💡 Consider caching episode metadata")
    
    if avg_get_time > 0.1:
        print(f"🐌 get() method is slow ({avg_get_time:.3f}s per sample)")
        
        if pose_time > frame_time and pose_time > chunk_time:
            print("   🎯 Main bottleneck: Pose data loading from HDF5")
            print("   💡 Consider caching pose data in memory")
        elif frame_time > pose_time and frame_time > chunk_time:
            print("   🎯 Main bottleneck: Video frame extraction")
            print("   💡 Consider frame caching or pre-extraction")
        elif chunk_time > 0.05:
            print("   🎯 Action chunk generation is slow")
            print("   💡 Consider optimizing pose processing")
    
    if avg_cv_time > 0.05:
        print(f"🐌 OpenCV video access is slow ({avg_cv_time:.3f}s)")
        print("   💡 Consider using faster video library or frame caching")


if __name__ == '__main__':
    quick_profile()
