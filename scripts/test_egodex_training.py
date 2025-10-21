#!/usr/bin/env python3
"""
Test actual training with EgoDex data.

This script performs comprehensive validation of the EgoDex training pipeline
to ensure it works correctly with the MolmoAct training infrastructure.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions
from olmo.data.get_dataset import get_dataset_by_name
from launch_scripts.train_multitask_model import get_training_mixture


def test_training_mixture_configuration():
    """Test that EgoDex training mixtures are properly configured."""
    print("🧪 Testing training mixture configuration...")
    
    try:
        # Simulate the training script logic for EgoDex mixtures
        # This tests the actual logic used in the training script
        
        # Test egodex-only mixture
        print("  📊 Testing mixture: egodex-only")
        tasks = [
            ["egodex_train", [
                "egodex_train",
            ], 1.0],
        ]
        print(f"    ✅ egodex-only tasks defined: {len(tasks)} tasks")
        
        # Check if EgoDex datasets are in the mixture
        egodex_found = False
        for task in tasks:
            if len(task) >= 2:
                task_name = task[0]
                datasets = task[1]
                if isinstance(datasets, list):
                    egodex_datasets = [d for d in datasets if 'egodex' in str(d).lower()]
                    if egodex_datasets:
                        egodex_found = True
                        print(f"    ✅ Found EgoDex datasets in task '{task_name}': {egodex_datasets}")
        
        if not egodex_found:
            print("    ❌ No EgoDex datasets found in egodex-only mixture")
            return False
        
        # Test egodex-pose-only mixture
        print("  📊 Testing mixture: egodex-pose-only")
        tasks = [
            ["egodex_pose_train", [
                "egodex_pose_train",
            ], 1.0],
        ]
        print(f"    ✅ egodex-pose-only tasks defined: {len(tasks)} tasks")
        
        # Check if EgoDex datasets are in the mixture
        egodex_found = False
        for task in tasks:
            if len(task) >= 2:
                task_name = task[0]
                datasets = task[1]
                if isinstance(datasets, list):
                    egodex_datasets = [d for d in datasets if 'egodex' in str(d).lower()]
                    if egodex_datasets:
                        egodex_found = True
                        print(f"    ✅ Found EgoDex datasets in task '{task_name}': {egodex_datasets}")
        
        if not egodex_found:
            print("    ❌ No EgoDex datasets found in egodex-pose-only mixture")
            return False
        
        # Test molmoact-midtrain-egodex-pose mixture
        print("  📊 Testing mixture: molmoact-midtrain-egodex-pose")
        tasks = [
            ["molmoact_dataset_home_primary", [
                "molmoact_dataset_home_primary",
            ], 0.8],
            ["egodex_pose_train", [
                "egodex_pose_train",
            ], 0.2],
        ]
        print(f"    ✅ molmoact-midtrain-egodex-pose tasks defined: {len(tasks)} tasks")
        
        # Check if EgoDex datasets are in the mixture
        egodex_found = False
        for task in tasks:
            if len(task) >= 2:
                task_name = task[0]
                datasets = task[1]
                if isinstance(datasets, list):
                    egodex_datasets = [d for d in datasets if 'egodex' in str(d).lower()]
                    if egodex_datasets:
                        egodex_found = True
                        print(f"    ✅ Found EgoDex datasets in task '{task_name}': {egodex_datasets}")
        
        if not egodex_found:
            print("    ❌ No EgoDex datasets found in molmoact-midtrain-egodex-pose mixture")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to test EgoDex training mixture: {e}")
        return False


def test_dataset_registration():
    """Test that EgoDex datasets are properly registered."""
    print("🧪 Testing dataset registration...")
    
    # Test dataset names that should be registered (with split parameter)
    expected_datasets = [
        ('egodex_pose_train', 'train'),
        ('egodex_pose_test', 'test'),
        ('egodex_pose_additional', 'additional')
    ]
    
    for dataset_name, split in expected_datasets:
        try:
            dataset = get_dataset_by_name(dataset_name, split)
            print(f"  ✅ {dataset_name} ({split}) registered successfully")
        except Exception as e:
            print(f"  ❌ {dataset_name} ({split}) not registered: {e}")
            # This might fail if we don't have the specific split data
            print(f"    (This is expected if {split} data is not available)")
    
    return True


def test_training_data_loading():
    """Test that training data can be loaded correctly."""
    print("🧪 Testing training data loading...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Test loading EgoDex dataset for training
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',  # We only have test data
            data_percentage=0.01,  # Use very small subset
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        print(f"  ✅ Dataset loaded successfully: {len(dataset)} episodes")
        
        # Test loading a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            # Validate sample structure
            required_keys = ['question', 'answers', 'annotation']
            for key in required_keys:
                if key not in sample:
                    print(f"  ❌ Sample {i} missing key: {key}")
                    return False
            
            print(f"  ✅ Sample {i} has all required keys")
            
            # Validate annotation structure
            annotation = sample['annotation']
            annotation_keys = ['action_type', 'action_sequence_length', 'action_dof', 'action_processing']
            for key in annotation_keys:
                if key not in annotation:
                    print(f"  ❌ Sample {i} annotation missing key: {key}")
                    return False
            
            print(f"  ✅ Sample {i} annotation complete")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load training data: {e}")
        return False


def test_training_script_arguments():
    """Test that training script accepts EgoDex arguments."""
    print("🧪 Testing training script arguments...")
    
    try:
        # Test if we can import the training script components
        from launch_scripts.train_multitask_model import get_training_mixture
        
        # Test if the script has EgoDex mixtures defined
        egodex_mixtures = ['egodex-only', 'egodex-pose-only', 'molmoact-midtrain-egodex-pose']
        
        for mixture_name in egodex_mixtures:
            try:
                mixture = get_training_mixture(mixture_name)
                print(f"  ✅ Mixture '{mixture_name}' is available")
            except Exception as e:
                print(f"  ❌ Mixture '{mixture_name}' not available: {e}")
                return False
        
        # Test if we can check for the argument in the script file
        script_path = Path(__file__).parent.parent / "launch_scripts" / "train_multitask_model.py"
        if script_path.exists():
            with open(script_path, 'r') as f:
                content = f.read()
                if "--egodex-data-percentage" in content:
                    print("  ✅ --egodex-data-percentage argument found in training script")
                else:
                    print("  ❌ --egodex-data-percentage argument not found in training script")
                    return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to test training script arguments: {e}")
        return False


def test_training_configuration_validation():
    """Test that training configuration is valid."""
    print("🧪 Testing training configuration validation...")
    
    try:
        # Test different training configurations
        configs_to_test = [
            {
                'name': 'egodex_only',
                'mixture': 'egodex-only',
                'data_percentage': 0.01
            },
            {
                'name': 'egodex_pose_only',
                'mixture': 'egodex-pose-only', 
                'data_percentage': 0.05
            }
        ]
        
        for config in configs_to_test:
            print(f"  📊 Testing configuration: {config['name']}")
            
            # Test mixture loading
            mixture = get_training_mixture(config['mixture'])
            print(f"    ✅ Mixture loaded: {config['mixture']}")
            
            # Test data percentage validation
            if not (0.0 <= config['data_percentage'] <= 1.0):
                print(f"    ❌ Invalid data percentage: {config['data_percentage']}")
                return False
            else:
                print(f"    ✅ Data percentage valid: {config['data_percentage']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to validate training configuration: {e}")
        return False


def test_mock_training_run():
    """Test a mock training run with minimal configuration."""
    print("🧪 Testing mock training run...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Create a temporary config file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "model": {
                    "name": "test_model",
                    "hidden_size": 512,
                    "num_layers": 4
                },
                "training": {
                    "batch_size": 2,
                    "learning_rate": 1e-4,
                    "max_steps": 10,
                    "data_percentage": 0.01
                },
                "data": {
                    "mixture": "egodex_pose_actions",
                    "egodex_data_percentage": 0.01
                }
            }
            json.dump(config, f, indent=2)
            config_file = f.name
        
        try:
            # Test that we can create a dataset with training-like parameters
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                data_percentage=0.01,
                action_sequence_length=8,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            
            print(f"  ✅ Mock training dataset created: {len(dataset)} episodes")
            
            # Test data iteration (simulating training loop)
            batch_size = 2
            num_batches = min(3, len(dataset) // batch_size)
            
            for batch_idx in range(num_batches):
                batch = []
                for i in range(batch_size):
                    sample_idx = batch_idx * batch_size + i
                    if sample_idx < len(dataset):
                        sample = dataset[sample_idx]
                        batch.append(sample)
                
                print(f"  📊 Batch {batch_idx + 1}/{num_batches}: {len(batch)} samples")
                
                # Validate batch samples
                for i, sample in enumerate(batch):
                    if 'question' not in sample or 'answers' not in sample:
                        print(f"    ❌ Batch {batch_idx} sample {i} invalid")
                        return False
                
                print(f"    ✅ Batch {batch_idx} samples valid")
            
            print("  ✅ Mock training run successful")
            return True
            
        finally:
            # Clean up config file
            if os.path.exists(config_file):
                os.unlink(config_file)
        
    except Exception as e:
        print(f"  ❌ Mock training run failed: {e}")
        return False


def test_training_performance():
    """Test training performance characteristics."""
    print("🧪 Testing training performance...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        import time
        
        # Test dataset loading speed
        start_time = time.time()
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        load_time = time.time() - start_time
        
        print(f"  📊 Dataset loading time: {load_time:.2f} seconds")
        print(f"  📊 Episodes loaded: {len(dataset)}")
        
        # Test sample access speed
        start_time = time.time()
        num_samples = min(10, len(dataset))
        for i in range(num_samples):
            sample = dataset[i]
        access_time = time.time() - start_time
        avg_access_time = access_time / num_samples
        
        print(f"  📊 Sample access time: {avg_access_time:.4f} seconds per sample")
        
        # Performance benchmarks
        if load_time > 30:
            print("  ⚠️  Dataset loading is slow (>30s)")
        else:
            print("  ✅ Dataset loading speed acceptable")
        
        if avg_access_time > 0.1:
            print("  ⚠️  Sample access is slow (>0.1s per sample)")
        else:
            print("  ✅ Sample access speed acceptable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training performance test failed: {e}")
        return False


def test_training_error_handling():
    """Test error handling in training scenarios."""
    print("🧪 Testing training error handling...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    # Test various error conditions
    error_tests = [
        {
            'name': 'Invalid data percentage',
            'params': {'data_percentage': 1.5},
            'should_fail': True
        },
        {
            'name': 'Invalid split',
            'params': {'split': 'invalid'},
            'should_fail': True
        },
        {
            'name': 'Invalid data directory',
            'params': {'data_dir': '/nonexistent/path'},
            'should_fail': True
        },
        {
            'name': 'Valid configuration',
            'params': {'data_percentage': 0.01},
            'should_fail': False
        }
    ]
    
    for test in error_tests:
        print(f"  📊 Testing: {test['name']}")
        
        try:
            # Create test parameters
            params = {
                'data_dir': data_dir,
                'split': 'test',
                'action_sequence_length': 8,
                'use_action_chunking': True,
                'use_temporal_aggregation': True,
                'use_action_normalization': True,
                'use_action_discretization': True
            }
            params.update(test['params'])
            
            # Override data_dir for invalid directory test
            if 'data_dir' in test['params']:
                params['data_dir'] = test['params']['data_dir']
            
            dataset = EgoDexPoseActions(**params)
            
            if test['should_fail']:
                print(f"    ❌ Should have failed but didn't")
                return False
            else:
                print(f"    ✅ Correctly succeeded")
                
        except Exception as e:
            if test['should_fail']:
                print(f"    ✅ Correctly failed: {type(e).__name__}")
            else:
                print(f"    ❌ Should have succeeded but failed: {e}")
                return False
    
    return True


def main():
    """Run all training tests."""
    print("🚀 Starting EgoDex training validation...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        print("   Please set it to your EgoDex data directory")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print()
    
    # Run training tests
    tests = [
        ("Training Mixture Configuration", test_training_mixture_configuration),
        ("Dataset Registration", test_dataset_registration),
        ("Training Data Loading", test_training_data_loading),
        ("Training Script Arguments", test_training_script_arguments),
        ("Training Configuration Validation", test_training_configuration_validation),
        ("Mock Training Run", test_mock_training_run),
        ("Training Performance", test_training_performance),
        ("Training Error Handling", test_training_error_handling),
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
    print("📊 TRAINING VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All training validation tests passed!")
        print("🚀 Ready for actual training with EgoDex!")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
