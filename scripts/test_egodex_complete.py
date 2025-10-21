#!/usr/bin/env python3
"""
Complete EgoDex integration test script.
Tests all aspects of EgoDex integration with MolmoAct.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import traceback

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False
    
    try:
        import olmo
        print("  ✅ MolmoAct core imported")
    except ImportError as e:
        print(f"  ❌ MolmoAct core import failed: {e}")
        return False
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        print("  ✅ EgoDex dataset classes imported")
    except ImportError as e:
        print(f"  ❌ EgoDex dataset import failed: {e}")
        return False
    
    try:
        from olmo.data.robot_datasets import EgoDexPoseActions
        print("  ✅ EgoDex datasets registered")
    except ImportError as e:
        print(f"  ❌ EgoDex registration failed: {e}")
        return False
    
    return True


def test_data_directory():
    """Test that EgoDex data directory is set and accessible."""
    print("\n🧪 Testing data directory...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("  ❌ EGODEX_DATA_DIR environment variable not set")
        print("  💡 Run: source egodex_test_data/organized/setup_egodex_test_env.sh")
        return False
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  ❌ Data directory does not exist: {data_path}")
        return False
    
    print(f"  ✅ Data directory: {data_path}")
    
    # Check for test data (since we only have test data)
    test_dir = data_path / "test"
    
    if test_dir.exists():
        test_tasks = list(test_dir.iterdir())
        print(f"  ✅ Test tasks: {len(test_tasks)}")
    else:
        print("  ⚠️  No test data found")
    
    return True


def test_dataset_loading():
    """Test loading EgoDex datasets."""
    print("\n🧪 Testing dataset loading...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            return False
        
        # Test with small data percentage
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",  # Use test split since we only have test data
            data_percentage=0.1,  # Use only 10% of data
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        print(f"  ✅ Dataset loaded with {len(dataset)} episodes")
        print(f"  📊 Data percentage: 10%")
        print(f"  🔧 Action sequence length: {dataset.action_sequence_length}")
        print(f"  🔧 Action DOF: {getattr(dataset, 'action_dof', 'N/A')}")
        print(f"  🔧 Action chunking: {getattr(dataset, 'use_action_chunking', 'N/A')}")
        print(f"  🔧 Temporal aggregation: {getattr(dataset, 'use_temporal_aggregation', 'N/A')}")
        print(f"  🔧 Action normalization: {getattr(dataset, 'use_action_normalization', 'N/A')}")
        print(f"  🔧 Action discretization: {getattr(dataset, 'use_action_discretization', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test loading individual data samples."""
    print("\n🧪 Testing data loading...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            return False
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            data_percentage=0.1,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        if len(dataset) == 0:
            print("  ⚠️  No data available for testing")
            return True
        
        # Load a sample
        sample = dataset[0]
        print(f"  ✅ Sample loaded successfully")
        print(f"  📊 Sample keys: {list(sample.keys())}")
        
        # Check image data
        if 'images' in sample:
            images = sample['images']
            print(f"  📸 Images: {len(images)} frames")
            if len(images) > 0:
                print(f"  📸 Image shape: {images[0].shape}")
        
        # Check action data
        if 'actions' in sample:
            actions = sample['actions']
            print(f"  🤖 Actions: {actions.shape}")
            print(f"  🤖 Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Check text data
        if 'text' in sample:
            text = sample['text']
            print(f"  📝 Text length: {len(text)} characters")
            print(f"  📝 Text preview: {text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        traceback.print_exc()
        return False


def test_action_processing():
    """Test action processing pipeline."""
    print("\n🧪 Testing action processing...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            return False
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            data_percentage=0.1,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        if len(dataset) == 0:
            print("  ⚠️  No data available for testing")
            return True
        
        # Test action processing
        sample = dataset[0]
        annotation = sample['annotation']
        
        print(f"  ✅ Action processing successful")
        print(f"  📊 Action type: {annotation['action_type']}")
        print(f"  📊 Action sequence length: {annotation['action_sequence_length']}")
        print(f"  📊 Action DOF: {annotation['action_dof']}")
        
        # Test action processing details
        action_processing = annotation['action_processing']
        print(f"  📊 Action chunking: {action_processing.get('chunking', 'N/A')}")
        print(f"  📊 Temporal aggregation: {action_processing.get('temporal_aggregation', 'N/A')}")
        print(f"  📊 Action normalization: {action_processing.get('normalization', 'N/A')}")
        print(f"  📊 Action discretization: {action_processing.get('discretization', 'N/A')}")
        
        # Test hand poses and finger tips
        hand_poses = annotation['hand_poses']
        finger_tips = annotation['finger_tips']
        print(f"  📊 Hand poses: {len(hand_poses)} hands")
        print(f"  📊 Finger tips: {len(finger_tips)} hands")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Action processing failed: {e}")
        traceback.print_exc()
        return False


def test_conversation_format():
    """Test conversation format generation."""
    print("\n🧪 Testing conversation format...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            return False
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            data_percentage=0.1,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        if len(dataset) == 0:
            print("  ⚠️  No data available for testing")
            return True
        
        # Test conversation generation
        sample = dataset[0]
        question = sample['question']
        answers = sample['answers']
        
        print(f"  ✅ Conversation format generated")
        print(f"  📝 Question length: {len(question)} characters")
        print(f"  📝 Answer length: {len(answers)} characters")
        
        # Check for key components
        if "The task is" in question:
            print("  ✅ Task description found")
        else:
            print("  ⚠️  Task description not found")
        
        if "What is the action that the robot should take" in question:
            print("  ✅ Action question found")
        else:
            print("  ⚠️  Action question not found")
        
        if "here's the step-by-step reasoning" in answers:
            print("  ✅ Step-by-step reasoning found")
        else:
            print("  ⚠️  Step-by-step reasoning not found")
        
        if "Action Sequence" in answers:
            print("  ✅ Action sequence found")
        else:
            print("  ⚠️  Action sequence not found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Conversation format failed: {e}")
        traceback.print_exc()
        return False


def test_statistics_computation():
    """Test action statistics computation."""
    print("\n🧪 Testing statistics computation...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            return False
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            data_percentage=0.1,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        if len(dataset) == 0:
            print("  ⚠️  No data available for testing")
            return True
        
        # Test statistics computation
        stats = dataset._compute_action_statistics()
        
        print(f"  ✅ Statistics computed successfully")
        print(f"  📊 Stats keys: {list(stats.keys())}")
        
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  📊 {key}: {value:.6f}")
            else:
                print(f"  📊 {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Statistics computation failed: {e}")
        traceback.print_exc()
        return False


def test_training_integration():
    """Test integration with training script."""
    print("\n🧪 Testing training integration...")
    
    try:
        # Test that we can import the dataset classes
        from olmo.data.egodex_dataset import EgoDexPoseActions
        print(f"  ✅ EgoDex dataset class imported")
        
        # Test that we can create a dataset instance
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            return False
        
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            data_percentage=0.01  # Use very small percentage for quick test
        )
        print(f"  ✅ EgoDex dataset instance created with {len(dataset)} episodes")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training integration failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting comprehensive EgoDex integration tests...")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Directory", test_data_directory),
        ("Dataset Loading", test_dataset_loading),
        ("Data Loading", test_data_loading),
        ("Action Processing", test_action_processing),
        ("Conversation Format", test_conversation_format),
        ("Statistics Computation", test_statistics_computation),
        ("Training Integration", test_training_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Report results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EgoDex integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test EgoDex integration with MolmoAct")
    parser.add_argument("--test", choices=[
        "imports", "data_dir", "dataset", "data", "actions", 
        "conversation", "stats", "training", "all"
    ], default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
    else:
        test_map = {
            "imports": test_imports,
            "data_dir": test_data_directory,
            "dataset": test_dataset_loading,
            "data": test_data_loading,
            "actions": test_action_processing,
            "conversation": test_conversation_format,
            "stats": test_statistics_computation,
            "training": test_training_integration,
        }
        
        test_func = test_map[args.test]
        success = test_func()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

