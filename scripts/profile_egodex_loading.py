#!/usr/bin/env python3
"""
Profile EgoDex data loading to identify bottlenecks.
Measures different components of the loading process.
"""

import os
import sys
import time
import cProfile
import pstats
from pathlib import Path
import numpy as np
import h5py
import cv2
from PIL import Image
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from olmo.data.egodex_dataset import EgoDexPoseActions


class EgoDexProfiler:
    """Profiler for EgoDex data loading performance."""
    
    def __init__(self, dataset_path, split='test', data_percentage=0.01):
        self.dataset_path = dataset_path
        self.split = split
        self.data_percentage = data_percentage
        self.timings = {}
        
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def profile_dataset_initialization(self):
        """Profile the dataset initialization process."""
        print("🔍 Profiling dataset initialization...")
        
        # Time dataset creation
        start_time = time.time()
        dataset = EgoDexPoseActions(
            data_dir=self.dataset_path,
            split=self.split,
            data_percentage=self.data_percentage,
            normalize_actions=False,
            discretize_actions=False
        )
        init_time = time.time() - start_time
        
        self.timings['dataset_init'] = init_time
        print(f"  ⏱️  Dataset initialization: {init_time:.2f}s")
        print(f"  📊 Total episodes loaded: {len(dataset.episodes)}")
        print(f"  📊 Total timesteps indexed: {len(dataset.timesteps)}")
        
        return dataset
    
    def profile_pose_data_loading(self, dataset, num_samples=10):
        """Profile pose data loading from HDF5 files."""
        print(f"🔍 Profiling pose data loading ({num_samples} samples)...")
        
        pose_load_times = []
        
        for i in range(min(num_samples, len(dataset.episodes))):
            episode = dataset.episodes[i]
            
            # Time HDF5 loading
            start_time = time.time()
            pose_data = dataset._load_pose_data(episode['pose_path'])
            load_time = time.time() - start_time
            pose_load_times.append(load_time)
            
            # Get size info
            transforms = pose_data.get('transforms', {})
            left_poses = len(transforms.get('leftHand', []))
            right_poses = len(transforms.get('rightHand', []))
            
            if i < 3:  # Show details for first few
                print(f"  Episode {episode['episode_id']}: {load_time:.3f}s ({left_poses}L, {right_poses}R poses)")
        
        avg_pose_time = np.mean(pose_load_times)
        self.timings['pose_data_loading'] = {
            'avg_time': avg_pose_time,
            'total_samples': len(pose_load_times),
            'all_times': pose_load_times
        }
        print(f"  ⏱️  Average pose loading time: {avg_pose_time:.3f}s")
        
        return pose_load_times
    
    def profile_video_frame_extraction(self, dataset, num_samples=10):
        """Profile video frame extraction."""
        print(f"🔍 Profiling video frame extraction ({num_samples} samples)...")
        
        frame_times = []
        
        for i in range(min(num_samples, len(dataset.episodes))):
            episode = dataset.episodes[i]
            
            # Time single frame extraction
            start_time = time.time()
            cap = cv2.VideoCapture(episode['video_path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()
            extract_time = time.time() - start_time
            
            if ret:
                frame_times.append(extract_time)
                if i < 3:  # Show details for first few
                    print(f"  Episode {episode['episode_id']}: {extract_time:.3f}s")
            else:
                print(f"  ⚠️  Failed to read frame from {episode['episode_id']}")
        
        avg_frame_time = np.mean(frame_times) if frame_times else 0
        self.timings['frame_extraction'] = {
            'avg_time': avg_frame_time,
            'total_samples': len(frame_times),
            'all_times': frame_times
        }
        print(f"  ⏱️  Average frame extraction time: {avg_frame_time:.3f}s")
        
        return frame_times
    
    def profile_get_method(self, dataset, num_samples=10):
        """Profile the complete get() method."""
        print(f"🔍 Profiling get() method ({num_samples} samples)...")
        
        get_times = []
        rng = np.random.RandomState(42)
        
        # Sample random indices
        if len(dataset) > num_samples:
            sample_indices = rng.choice(len(dataset), size=num_samples, replace=False)
        else:
            sample_indices = list(range(len(dataset)))
        
        for i, item_idx in enumerate(sample_indices):
            start_time = time.time()
            
            try:
                sample = dataset.get(item_idx, rng)
                get_time = time.time() - start_time
                get_times.append(get_time)
                
                if i < 3:  # Show details for first few
                    timestep_info = dataset.timesteps[item_idx]
                    print(f"  Sample {i+1} (idx {item_idx}): {get_time:.3f}s - ep{timestep_info['episode']['episode_id']}_t{timestep_info['timestep']}")
                    
            except Exception as e:
                print(f"  ❌ Error in sample {item_idx}: {e}")
                get_times.append(0)
        
        avg_get_time = np.mean(get_times)
        self.timings['get_method'] = {
            'avg_time': avg_get_time,
            'total_samples': len(get_times),
            'all_times': get_times
        }
        print(f"  ⏱️  Average get() method time: {avg_get_time:.3f}s")
        
        return get_times
    
    def profile_get_method_components(self, dataset, num_samples=5):
        """Profile individual components of the get() method."""
        print(f"🔍 Profiling get() method components ({num_samples} samples)...")
        
        component_times = {
            'frame_extraction': [],
            'pose_loading': [],
            'action_chunk_generation': [],
            'question_generation': []
        }
        
        rng = np.random.RandomState(42)
        
        if len(dataset) > num_samples:
            sample_indices = rng.choice(len(dataset), size=num_samples, replace=False)
        else:
            sample_indices = list(range(len(dataset)))
        
        for i, item_idx in enumerate(sample_indices):
            timestep_info = dataset.timesteps[item_idx]
            episode = timestep_info['episode']
            timestep = timestep_info['timestep']
            
            print(f"\n  Sample {i+1} breakdown:")
            
            # 1. Frame extraction
            start_time = time.time()
            frame = dataset._get_single_frame(episode['video_path'], timestep)
            frame_time = time.time() - start_time
            component_times['frame_extraction'].append(frame_time)
            print(f"    Frame extraction: {frame_time:.3f}s")
            
            # 2. Pose loading
            start_time = time.time()
            pose_data = dataset._load_pose_data(episode['pose_path'])
            pose_time = time.time() - start_time
            component_times['pose_loading'].append(pose_time)
            print(f"    Pose loading: {pose_time:.3f}s")
            
            # 3. Action chunk generation
            start_time = time.time()
            description = pose_data['metadata'].get('llm_description', f"Perform the {episode['task_name']} task")
            answer = dataset._get_timestep_action_chunk(pose_data, timestep, description)
            chunk_time = time.time() - start_time
            component_times['action_chunk_generation'].append(chunk_time)
            print(f"    Action chunk generation: {chunk_time:.3f}s")
            
            # 4. Question generation (just timing the string creation)
            start_time = time.time()
            question = (
                f"The task is {description}. "
                "What is the action that the robot should take. "
                f"To figure out the action that the robot should take to {description}, "
                "let's think through it step by step. "
                "First, how far are the objects from the hand? "
                "Second, how does the hand move during the task? "
                "Third, how do the fingers move for precise manipulation? "
                "Based on the distance, hand movement, and finger control, "
                "what actions should the robot take?"
            )
            question_time = time.time() - start_time
            component_times['question_generation'].append(question_time)
            print(f"    Question generation: {question_time:.3f}s")
        
        # Calculate averages
        for component, times in component_times.items():
            if times:
                avg_time = np.mean(times)
                self.timings[f'component_{component}'] = {
                    'avg_time': avg_time,
                    'times': times
                }
                print(f"\n  ⏱️  Average {component}: {avg_time:.3f}s")
    
    def profile_hdf5_access_patterns(self, dataset, num_samples=5):
        """Profile HDF5 file access patterns."""
        print(f"🔍 Profiling HDF5 access patterns ({num_samples} samples)...")
        
        hdf5_times = []
        
        for i in range(min(num_samples, len(dataset.episodes))):
            episode = dataset.episodes[i]
            hdf5_path = episode['pose_path']
            
            print(f"  Analyzing {hdf5_path}...")
            
            # Time opening file
            start_time = time.time()
            with h5py.File(hdf5_path, 'r') as f:
                open_time = time.time() - start_time
                
                # Time reading different sections
                start_time = time.time()
                transforms = f['transforms'][:]
                transforms_time = time.time() - start_time
                
                start_time = time.time()
                confidences = f['confidences'][:]
                confidences_time = time.time() - start_time
                
                start_time = time.time()
                metadata = dict(f['metadata'].attrs)
                metadata_time = time.time() - start_time
            
            total_time = open_time + transforms_time + confidences_time + metadata_time
            hdf5_times.append(total_time)
            
            print(f"    Open: {open_time:.3f}s, Transforms: {transforms_time:.3f}s, Confidences: {confidences_time:.3f}s, Metadata: {metadata_time:.3f}s")
        
        avg_hdf5_time = np.mean(hdf5_times)
        self.timings['hdf5_access'] = {
            'avg_time': avg_hdf5_time,
            'times': hdf5_times
        }
        print(f"  ⏱️  Average HDF5 access time: {avg_hdf5_time:.3f}s")
    
    def print_summary(self):
        """Print a summary of all timing results."""
        print("\n" + "="*60)
        print("📊 EGODEX LOADING PERFORMANCE SUMMARY")
        print("="*60)
        
        # Dataset initialization
        if 'dataset_init' in self.timings:
            print(f"🗂️  Dataset Initialization: {self.timings['dataset_init']:.2f}s")
        
        # Pose data loading
        if 'pose_data_loading' in self.timings:
            pose_data = self.timings['pose_data_loading']
            print(f"📁 Pose Data Loading: {pose_data['avg_time']:.3f}s per file (avg)")
        
        # Frame extraction
        if 'frame_extraction' in self.timings:
            frame_data = self.timings['frame_extraction']
            print(f"🎬 Frame Extraction: {frame_data['avg_time']:.3f}s per frame (avg)")
        
        # Get method
        if 'get_method' in self.timings:
            get_data = self.timings['get_method']
            print(f"🔄 Get Method: {get_data['avg_time']:.3f}s per sample (avg)")
        
        # Component breakdown
        print(f"\n🔧 Component Breakdown:")
        for key, value in self.timings.items():
            if key.startswith('component_'):
                component = key.replace('component_', '')
                if isinstance(value, dict) and 'avg_time' in value:
                    print(f"  • {component}: {value['avg_time']:.3f}s")
        
        # HDF5 access
        if 'hdf5_access' in self.timings:
            hdf5_data = self.timings['hdf5_access']
            print(f"💾 HDF5 Access: {hdf5_data['avg_time']:.3f}s per file (avg)")
        
        # Recommendations
        print(f"\n💡 Performance Recommendations:")
        
        if 'get_method' in self.timings and 'pose_data_loading' in self.timings:
            get_time = self.timings['get_method']['avg_time']
            pose_time = self.timings['pose_data_loading']['avg_time']
            pose_ratio = pose_time / get_time * 100
            
            if pose_ratio > 50:
                print(f"  🐌 Pose data loading is {pose_ratio:.1f}% of get() time - consider caching")
        
        if 'frame_extraction' in self.timings:
            frame_time = self.timings['frame_extraction']['avg_time']
            if frame_time > 0.1:
                print(f"  🐌 Frame extraction is slow ({frame_time:.3f}s) - consider frame caching")
        
        if 'dataset_init' in self.timings and self.timings['dataset_init'] > 10:
            print(f"  🐌 Dataset initialization is slow - consider optimizing timestep indexing")


def main():
    parser = argparse.ArgumentParser(description='Profile EgoDex data loading performance')
    parser.add_argument('--data_dir', type=str, 
                       default=os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/jose/molmoact/data/egodex/organized'),
                       help='EgoDex data directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'additional'],
                       help='Dataset split to profile')
    parser.add_argument('--data_percentage', type=float, default=0.01,
                       help='Percentage of data to use for profiling')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to profile')
    parser.add_argument('--profile_methods', action='store_true',
                       help='Use cProfile for detailed method profiling')
    
    args = parser.parse_args()
    
    print("🚀 Starting EgoDex Loading Performance Profile")
    print("=" * 50)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Data percentage: {args.data_percentage}")
    print(f"🎯 Number of samples: {args.num_samples}")
    print()
    
    # Initialize profiler
    profiler = EgoDexProfiler(args.data_dir, args.split, args.data_percentage)
    
    # Run profiling
    try:
        # 1. Dataset initialization
        dataset = profiler.profile_dataset_initialization()
        print()
        
        # 2. Pose data loading
        profiler.profile_pose_data_loading(dataset, args.num_samples)
        print()
        
        # 3. Video frame extraction
        profiler.profile_video_frame_extraction(dataset, args.num_samples)
        print()
        
        # 4. HDF5 access patterns
        profiler.profile_hdf5_access_patterns(dataset, min(args.num_samples, 5))
        print()
        
        # 5. Get method profiling
        profiler.profile_get_method(dataset, args.num_samples)
        print()
        
        # 6. Component breakdown
        profiler.profile_get_method_components(dataset, min(args.num_samples, 5))
        
        # Print summary
        profiler.print_summary()
        
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
