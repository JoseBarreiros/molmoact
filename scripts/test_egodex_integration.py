#!/usr/bin/env python3
"""
Test script to verify EgoDex integration with MolmoAct.

This script tests the EgoDex dataset loading and data processing pipeline.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from olmo.data.egodex_dataset import EgoDexDataset, EgoDexTrain, EgoDexTest
from olmo.data.get_dataset import get_dataset_by_name


def test_egodex_dataset_loading(data_dir: str, split: str = "test"):
    """Test basic EgoDex dataset loading."""
    print(f"Testing EgoDex dataset loading from: {data_dir}")
    print(f"Split: {split}")
    
    try:
        # Test direct dataset loading
        dataset = EgoDexDataset(data_dir=data_dir, split=split, max_frames=5)
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
        print(f"  - Question: {sample['question'][:100]}...")
        print(f"  - Answer: {sample['answers'][:100]}...")
        print(f"  - Annotation keys: {list(sample['annotation'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_egodex_via_get_dataset(data_dir: str):
    """Test EgoDex dataset loading via get_dataset_by_name."""
    print(f"Testing EgoDex via get_dataset_by_name...")
    
    # Set environment variable
    os.environ['EGODEX_DATA_DIR'] = data_dir
    
    try:
        # Test different dataset names
        dataset_names = ['egodex_train', 'egodex_test', 'egodex']
        
        for name in dataset_names:
            print(f"Testing dataset name: {name}")
            dataset = get_dataset_by_name(name, "train")
            print(f"✓ {name} loaded successfully. Type: {type(dataset)}")
            
            # Test getting a sample
            rng = np.random.RandomState(42)
            sample = dataset.get(0, rng)
            print(f"  - Sample keys: {list(sample.keys())}")
            
    except Exception as e:
        print(f"✗ Error in get_dataset_by_name: {e}")
        return False
    
    return True


def test_pose_data_processing(data_dir: str):
    """Test pose data processing from HDF5 files."""
    print("Testing pose data processing...")
    
    try:
        dataset = EgoDexDataset(data_dir=data_dir, split="test", max_frames=2)
        
        if len(dataset) == 0:
            print("⚠ No episodes available for pose testing")
            return True
            
        # Get a sample and check pose data
        rng = np.random.RandomState(42)
        sample = dataset.get(0, rng)
        
        annotation = sample['annotation']
        pose_data = annotation.get('pose_data', {})
        
        print("✓ Pose data structure:")
        print(f"  - Transforms keys: {list(pose_data.get('transforms', {}).keys())}")
        print(f"  - Confidences keys: {list(pose_data.get('confidences', {}).keys())}")
        print(f"  - Metadata keys: {list(pose_data.get('metadata', {}).keys())}")
        print(f"  - Pose description: {annotation.get('pose_description', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in pose data processing: {e}")
        return False


def test_training_compatibility(data_dir: str):
    """Test compatibility with MolmoAct training format."""
    print("Testing training compatibility...")
    
    try:
        # Test with MolmoAct-compatible parameters
        dataset = EgoDexDataset(
            data_dir=data_dir,
            split="test",
            width=320,
            height=240,
            style="demo",
            max_frames=3
        )
        
        if len(dataset) == 0:
            print("⚠ No episodes available for training compatibility test")
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
                
            print(f"  Sample {i}: {len(images)} images, size {img.size}")
        
        print("✓ Training compatibility test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error in training compatibility test: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test EgoDex integration with MolmoAct")
    parser.add_argument("--data-dir", required=True, help="Path to EgoDex data directory")
    parser.add_argument("--split", default="test", choices=["train", "test", "additional"],
                       help="Dataset split to test")
    parser.add_argument("--skip-download-test", action="store_true",
                       help="Skip dataset loading test")
    parser.add_argument("--skip-pose-test", action="store_true",
                       help="Skip pose data processing test")
    parser.add_argument("--skip-training-test", action="store_true",
                       help="Skip training compatibility test")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    
    if not data_dir.exists():
        print(f"✗ Data directory does not exist: {data_dir}")
        print("Please download EgoDex data first using prepare_egodex_data.py")
        sys.exit(1)
    
    print("="*60)
    print("EgoDex Integration Test")
    print("="*60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Dataset loading
    if not args.skip_download_test:
        total_tests += 1
        if test_egodex_dataset_loading(str(data_dir), args.split):
            tests_passed += 1
        print()
    
    # Test 2: get_dataset_by_name integration
    if not args.skip_download_test:
        total_tests += 1
        if test_egodex_via_get_dataset(str(data_dir)):
            tests_passed += 1
        print()
    
    # Test 3: Pose data processing
    if not args.skip_pose_test:
        total_tests += 1
        if test_pose_data_processing(str(data_dir)):
            tests_passed += 1
        print()
    
    # Test 4: Training compatibility
    if not args.skip_training_test:
        total_tests += 1
        if test_training_compatibility(str(data_dir)):
            tests_passed += 1
        print()
    
    # Summary
    print("="*60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! EgoDex integration is working correctly.")
        print("\nNext steps:")
        print("1. Set environment variable: export EGODEX_DATA_DIR='{data_dir}'")
        print("2. Run training: python launch_scripts/train_multitask_model.py --mixture molmoact-midtrain-egodex")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()



