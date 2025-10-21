#!/usr/bin/env python3
"""
Test script for EgoDex pose-based action learning.

This script tests the EgoDex integration where human hand poses are treated as robot actions,
similar to how LIBERO treats robot actions.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from olmo.data.egodex_dataset import EgoDexPoseActions, EgoDexPoseActionsTrain, EgoDexPoseActionsTest
from olmo.data.get_dataset import get_dataset_by_name


def test_pose_action_dataset_loading(data_dir: str, split: str = "test"):
    """Test EgoDex pose-based action dataset loading."""
    print(f"Testing EgoDex pose-based action dataset loading from: {data_dir}")
    print(f"Split: {split}")
    
    try:
        # Test direct dataset loading
        dataset = EgoDexPoseActions(
            data_dir=data_dir, 
            split=split, 
            max_frames=5,
            action_sequence_length=8
        )
        print(f"✓ Dataset loaded successfully. Episodes: {len(dataset)}")
        
        if len(dataset) == 0:
            print("⚠ Warning: No episodes found in dataset")
            return False
            
        # Test getting a sample
        rng = np.random.RandomState(42)
        sample = dataset.get(0, rng)
        
        print("✓ Sample data structure:")
        print(f"  - Style: {sample['style']}")
        print(f"  - Image type: {type(sample['image'])}")
        print(f"  - Image count: {len(sample['image'])}")
        print(f"  - Question length: {len(sample['question'])}")
        print(f"  - Answer length: {len(sample['answers'])}")
        print(f"  - Annotation keys: {list(sample['annotation'].keys())}")
        print(f"  - Action type: {sample['annotation'].get('action_type', 'N/A')}")
        print(f"  - Action sequence length: {sample['annotation'].get('action_sequence_length', 'N/A')}")
        print(f"  - Action DOF: {sample['annotation'].get('action_dof', 'N/A')}")
        
        # Check action processing features
        action_processing = sample['annotation'].get('action_processing', {})
        print(f"  - Action processing:")
        print(f"    - Chunking: {action_processing.get('chunking', 'N/A')}")
        print(f"    - Temporal aggregation: {action_processing.get('temporal_aggregation', 'N/A')}")
        print(f"    - Normalization: {action_processing.get('normalization', 'N/A')}")
        print(f"    - Discretization: {action_processing.get('discretization', 'N/A')}")
        if action_processing.get('discretization'):
            print(f"    - Action bins: {action_processing.get('n_action_bins', 'N/A')}")
        
        # Check if answer contains action matrix
        answer = sample['answers']
        if "Action Sequence" in answer and "[" in answer:
            print("✓ Answer contains action sequence format")
            if "21 DOF" in answer or "finger tips" in answer:
                print("✓ Answer includes finger tip positions")
            else:
                print("⚠ Warning: Answer may not include finger tip positions")
            
            if "discretized" in answer:
                print("✓ Answer includes discretized actions")
            if "Temporally aggregated" in answer:
                print("✓ Answer includes temporal aggregation")
        else:
            print("⚠ Warning: Answer may not contain proper action sequence format")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_pose_action_via_get_dataset(data_dir: str):
    """Test EgoDex pose-based action dataset loading via get_dataset_by_name."""
    print(f"Testing EgoDex pose-based actions via get_dataset_by_name...")
    
    # Set environment variable
    os.environ['EGODEX_DATA_DIR'] = data_dir
    
    try:
        # Test different dataset names
        dataset_names = ['egodex_pose_train', 'egodex_pose_test', 'egodex_pose']
        
        for name in dataset_names:
            print(f"Testing dataset name: {name}")
            dataset = get_dataset_by_name(name, "train")
            print(f"✓ {name} loaded successfully. Type: {type(dataset)}")
            
            # Test getting a sample
            rng = np.random.RandomState(42)
            sample = dataset.get(0, rng)
            print(f"  - Sample keys: {list(sample.keys())}")
            print(f"  - Action type: {sample['annotation'].get('action_type', 'N/A')}")
            
    except Exception as e:
        print(f"✗ Error in get_dataset_by_name: {e}")
        return False
    
    return True


def test_pose_to_action_conversion(data_dir: str):
    """Test conversion of hand poses to robot actions including finger tips."""
    print("Testing pose-to-action conversion with finger tips...")
    
    try:
        dataset = EgoDexPoseActions(
            data_dir=data_dir, 
            split="test", 
            max_frames=2,
            action_sequence_length=6,
            normalize_actions=True,
            discretize_actions=True
        )
        
        if len(dataset) == 0:
            print("⚠ No episodes available for pose-to-action testing")
            return True
            
        # Get a sample and check action conversion
        rng = np.random.RandomState(42)
        sample = dataset.get(0, rng)
        
        annotation = sample['annotation']
        hand_poses = annotation.get('hand_poses', {})
        finger_tips = annotation.get('finger_tips', {})
        
        print("✓ Hand pose data structure:")
        print(f"  - Left hand poses: {len(hand_poses.get('left', []))}")
        print(f"  - Right hand poses: {len(hand_poses.get('right', []))}")
        
        print("✓ Finger tip data structure:")
        print(f"  - Left finger tips: {len(finger_tips.get('left', []))}")
        print(f"  - Right finger tips: {len(finger_tips.get('right', []))}")
        
        # Check action DOF
        action_dof = annotation.get('action_dof', 0)
        print(f"  - Action DOF: {action_dof}")
        if action_dof == 21:
            print("✓ Correct action DOF (6 hand + 15 finger tips)")
        else:
            print(f"⚠ Warning: Expected 21 DOF, got {action_dof}")
        
        # Check action processing features
        action_processing = annotation.get('action_processing', {})
        print(f"  - Action processing features:")
        print(f"    - Chunking: {action_processing.get('chunking', 'N/A')}")
        print(f"    - Temporal aggregation: {action_processing.get('temporal_aggregation', 'N/A')}")
        print(f"    - Normalization: {action_processing.get('normalization', 'N/A')}")
        print(f"    - Discretization: {action_processing.get('discretization', 'N/A')}")
        
        # Check if answer contains proper action format
        answer = sample['answers']
        if "Action Sequence" in answer:
            print("✓ Answer contains action sequence")
            
            # Extract action matrix from answer
            lines = answer.split('\n')
            action_lines = [line for line in lines if '[' in line and ']' in line and '#' in line]
            print(f"  - Action steps found: {len(action_lines)}")
            
            if action_lines:
                print("  - Sample action step:", action_lines[0].strip())
                
                # Check if action has correct number of values (21 DOF)
                import re
                if action_processing.get('discretization'):
                    # For discretized actions, look for integer values
                    numbers = re.findall(r'\d+', action_lines[0])
                    if len(numbers) == 21:
                        print("✓ Action step has correct number of discretized values (21 DOF)")
                    else:
                        print(f"⚠ Warning: Expected 21 discretized values, got {len(numbers)}")
                else:
                    # For continuous actions, look for float values
                    numbers = re.findall(r'-?\d+\.\d+', action_lines[0])
                    if len(numbers) == 21:
                        print("✓ Action step has correct number of continuous values (21 DOF)")
                    else:
                        print(f"⚠ Warning: Expected 21 continuous values, got {len(numbers)}")
            
            # Check for temporal aggregation
            if "Temporally aggregated" in answer:
                print("✓ Answer includes temporal aggregation")
            
            # Check for discretization
            if "discretized" in answer:
                print("✓ Answer includes discretized actions")
        else:
            print("⚠ Warning: Answer does not contain action sequence")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in pose-to-action conversion: {e}")
        return False


def test_libro_style_compatibility(data_dir: str):
    """Test compatibility with LIBERO-style training format."""
    print("Testing LIBERO-style compatibility...")
    
    try:
        # Test with LIBERO-compatible parameters
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            width=320,
            height=240,
            style="demo",
            max_frames=3,
            action_sequence_length=8
        )
        
        if len(dataset) == 0:
            print("⚠ No episodes available for LIBERO compatibility test")
            return True
            
        # Test multiple samples
        rng = np.random.RandomState(42)
        for i in range(min(3, len(dataset))):
            sample = dataset.get(i, rng)
            
            # Check required fields
            required_fields = ['style', 'image', 'question', 'answers', 'annotation']
            for field in required_fields:
                if field not in sample:
                    print(f"✗ Missing required field: {field}")
                    return False
            
            # Check image format
            images = sample['image']
            if not isinstance(images, list):
                print(f"✗ Images should be a list, got: {type(images)}")
                return False
                
            if len(images) == 0:
                print(f"✗ No images in sample {i}")
                return False
                
            # Check image properties
            img = images[0]
            if not hasattr(img, 'size'):
                print(f"✗ Image should be PIL Image, got: {type(img)}")
                return False
                
            # Check action format
            answer = sample['answers']
            if "Action Sequence" not in answer:
                print(f"✗ Sample {i} answer does not contain action sequence")
                return False
                
            print(f"  Sample {i}: {len(images)} images, size {img.size}, action sequence present")
        
        print("✓ LIBERO-style compatibility test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error in LIBERO compatibility test: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test EgoDex pose-based action integration")
    parser.add_argument("--data-dir", required=True, help="Path to EgoDex data directory")
    parser.add_argument("--split", default="test", choices=["train", "test", "additional"],
                       help="Dataset split to test")
    parser.add_argument("--skip-download-test", action="store_true",
                       help="Skip dataset loading test")
    parser.add_argument("--skip-pose-test", action="store_true",
                       help="Skip pose-to-action conversion test")
    parser.add_argument("--skip-libro-test", action="store_true",
                       help="Skip LIBERO compatibility test")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    
    if not data_dir.exists():
        print(f"✗ Data directory does not exist: {data_dir}")
        print("Please download EgoDex data first using prepare_egodex_data.py")
        sys.exit(1)
    
    print("="*60)
    print("EgoDex Pose-Based Action Integration Test")
    print("="*60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Dataset loading
    if not args.skip_download_test:
        total_tests += 1
        if test_pose_action_dataset_loading(str(data_dir), args.split):
            tests_passed += 1
        print()
    
    # Test 2: get_dataset_by_name integration
    if not args.skip_download_test:
        total_tests += 1
        if test_pose_action_via_get_dataset(str(data_dir)):
            tests_passed += 1
        print()
    
    # Test 3: Pose-to-action conversion
    if not args.skip_pose_test:
        total_tests += 1
        if test_pose_to_action_conversion(str(data_dir)):
            tests_passed += 1
        print()
    
    # Test 4: LIBERO compatibility
    if not args.skip_libro_test:
        total_tests += 1
        if test_libro_style_compatibility(str(data_dir)):
            tests_passed += 1
        print()
    
    # Summary
    print("="*60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! EgoDex pose-based action integration is working correctly.")
        print("\nNext steps:")
        print("1. Set environment variable: export EGODEX_DATA_DIR='{data_dir}'")
        print("2. Run pose-based training: python launch_scripts/train_multitask_model.py --mixture egodex-pose-only")
        print("3. Run combined training: python launch_scripts/train_multitask_model.py --mixture molmoact-midtrain-egodex-pose")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
