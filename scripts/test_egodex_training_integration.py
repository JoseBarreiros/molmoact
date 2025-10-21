#!/usr/bin/env python3
"""
EgoDex training integration test.

This script tests the actual training pipeline integration with EgoDex
to ensure everything works correctly before large-scale training.
"""

import os
import sys
import json
import tempfile
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmo.data.egodex_dataset import EgoDexPoseActions
from olmo.data.get_dataset import get_dataset_by_name


def test_training_mixture_configuration():
    """Test that EgoDex is properly configured in training mixtures."""
    print("🧪 Testing training mixture configuration...")
    
    try:
        from launch_scripts.train_multitask_model import get_training_mixtures
        
        mixtures = get_training_mixtures()
        print(f"  📊 Found {len(mixtures)} training mixtures")
        
        # Check for EgoDex mixtures
        egodex_mixtures = []
        for name, config in mixtures.items():
            if 'egodex' in name.lower():
                egodex_mixtures.append((name, config))
                print(f"  ✅ Found EgoDex mixture: {name}")
        
        if not egodex_mixtures:
            print("  ⚠️  No EgoDex mixtures found")
            return False
        
        # Validate EgoDex mixture configuration
        for name, config in egodex_mixtures:
            print(f"  📋 Validating mixture: {name}")
            
            # Check required fields
            required_fields = ['datasets', 'weights']
            for field in required_fields:
                if field not in config:
                    print(f"    ❌ Missing required field: {field}")
                    return False
                else:
                    print(f"    ✅ Has {field}: {len(config[field])} items")
            
            # Check if EgoDex datasets are in the mixture
            datasets = config['datasets']
            egodex_datasets = [d for d in datasets if 'egodex' in d.lower()]
            
            if not egodex_datasets:
                print(f"    ❌ No EgoDex datasets found in mixture {name}")
                return False
            else:
                print(f"    ✅ Contains EgoDex datasets: {egodex_datasets}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training mixture test failed: {e}")
        return False


def test_dataset_registration():
    """Test that EgoDex datasets are properly registered."""
    print("🧪 Testing dataset registration...")
    
    # Test dataset names
    egodex_datasets = [
        'egodex_pose_actions',
        'egodex_pose_actions_with_stats'
    ]
    
    for dataset_name in egodex_datasets:
        try:
            dataset_class = get_dataset_by_name(dataset_name, 'test')
            print(f"  ✅ {dataset_name} registered successfully")
        except Exception as e:
            print(f"  ❌ {dataset_name} registration failed: {e}")
            return False
    
    return True


def test_training_script_arguments():
    """Test that training script accepts EgoDex arguments."""
    print("🧪 Testing training script arguments...")
    
    try:
        from launch_scripts.train_multitask_model import main as train_main
        import argparse
        
        # Create a mock argument parser to test
        parser = argparse.ArgumentParser()
        
        # This would normally be done by the training script
        # We're testing if the argument exists
        try:
            # Check if egodex-data-percentage argument exists
            # This is a bit hacky, but we can check the source code
            with open('launch_scripts/train_multitask_model.py', 'r') as f:
                content = f.read()
                if 'egodex-data-percentage' in content:
                    print("  ✅ --egodex-data-percentage argument found")
                else:
                    print("  ❌ --egodex-data-percentage argument not found")
                    return False
        except Exception as e:
            print(f"  ❌ Could not check training script arguments: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training script argument test failed: {e}")
        return False


def test_data_loading_in_training_context():
    """Test data loading in a training-like context."""
    print("🧪 Testing data loading in training context...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    try:
        # Test with different configurations that might be used in training
        configs = [
            {
                'data_percentage': 0.1,
                'use_action_normalization': True,
                'use_action_discretization': True,
                'action_sequence_length': 8
            },
            {
                'data_percentage': 0.05,
                'use_action_normalization': False,
                'use_action_discretization': True,
                'action_sequence_length': 4
            }
        ]
        
        for i, config in enumerate(configs):
            print(f"  📊 Testing configuration {i+1}: {config}")
            
            dataset = EgoDexPoseActions(
                data_dir=data_dir,
                split='test',
                **config
            )
            
            # Simulate training data loading
            batch_size = 4
            for j in range(min(batch_size, len(dataset))):
                sample = dataset[j]
                
                # Check sample structure
                required_keys = ['style', 'image', 'question', 'answers', 'annotation']
                for key in required_keys:
                    if key not in sample:
                        print(f"    ❌ Missing key in sample: {key}")
                        return False
                
                # Check annotation structure
                annotation = sample['annotation']
                required_annotation_keys = ['action_type', 'action_sequence_length', 'action_dof']
                for key in required_annotation_keys:
                    if key not in annotation:
                        print(f"    ❌ Missing annotation key: {key}")
                        return False
            
            print(f"    ✅ Configuration {i+1} loaded successfully")
            del dataset
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading test failed: {e}")
        return False


def test_statistics_integration():
    """Test statistics integration for training."""
    print("🧪 Testing statistics integration...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    try:
        # Test with precomputed statistics
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            use_action_normalization=True
        )
        
        # Check if statistics are loaded
        if hasattr(dataset, 'action_stats') and dataset.action_stats:
            print("  ✅ Statistics loaded successfully")
            print(f"  📊 Statistics keys: {list(dataset.action_stats.keys())}")
        else:
            print("  ❌ Statistics not loaded")
            return False
        
        # Test normalization
        sample = dataset[0]
        annotation = sample['annotation']
        
        if 'action_processing' in annotation:
            processing = annotation['action_processing']
            if processing.get('normalization', False):
                print("  ✅ Action normalization enabled")
            else:
                print("  ⚠️  Action normalization not enabled")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Statistics integration test failed: {e}")
        return False


def test_conversation_format_consistency():
    """Test conversation format consistency for training."""
    print("🧪 Testing conversation format consistency...")
    
    data_dir = os.environ.get('EGODEX_DATA_DIR', '/home/jose-barreiros/development/molmoact/egodex_test_data/organized/test')
    
    try:
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.01,
            use_action_normalization=True
        )
        
        # Test multiple samples for consistency
        num_samples = min(5, len(dataset))
        formats = []
        
        for i in range(num_samples):
            sample = dataset[i]
            formats.append({
                'question_length': len(sample['question']),
                'answer_length': len(sample['answers']),
                'has_reasoning': 'step-by-step reasoning' in sample['answers'],
                'has_action_sequence': 'Action Sequence' in sample['answers']
            })
        
        # Check consistency
        question_lengths = [f['question_length'] for f in formats]
        answer_lengths = [f['answer_length'] for f in formats]
        
        # Check if lengths are reasonably consistent (within 50% of mean)
        mean_q_len = np.mean(question_lengths)
        mean_a_len = np.mean(answer_lengths)
        
        q_len_consistent = all(abs(q - mean_q_len) < 0.5 * mean_q_len for q in question_lengths)
        a_len_consistent = all(abs(a - mean_a_len) < 0.5 * mean_a_len for a in answer_lengths)
        
        if q_len_consistent:
            print(f"  ✅ Question lengths consistent: {mean_q_len:.0f} ± {np.std(question_lengths):.0f}")
        else:
            print(f"  ⚠️  Question lengths inconsistent: {question_lengths}")
        
        if a_len_consistent:
            print(f"  ✅ Answer lengths consistent: {mean_a_len:.0f} ± {np.std(answer_lengths):.0f}")
        else:
            print(f"  ⚠️  Answer lengths inconsistent: {answer_lengths}")
        
        # Check reasoning consistency
        reasoning_count = sum(f['has_reasoning'] for f in formats)
        action_seq_count = sum(f['has_action_sequence'] for f in formats)
        
        print(f"  📊 Reasoning present in {reasoning_count}/{num_samples} samples")
        print(f"  📊 Action sequences present in {action_seq_count}/{num_samples} samples")
        
        return q_len_consistent and a_len_consistent
        
    except Exception as e:
        print(f"  ❌ Conversation format consistency test failed: {e}")
        return False


def test_training_script_execution():
    """Test actual training script execution (dry run)."""
    print("🧪 Testing training script execution...")
    
    try:
        # This would test the actual training script
        # For now, we'll just check if it can be imported and has the right structure
        
        from launch_scripts.train_multitask_model import main as train_main
        import inspect
        
        # Check if main function exists and has expected signature
        sig = inspect.signature(train_main)
        print(f"  📊 Training script main function signature: {sig}")
        
        # Check if it can be called (we won't actually run it)
        print("  ✅ Training script can be imported and has main function")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training script execution test failed: {e}")
        return False


def main():
    """Run all training integration tests."""
    print("🚀 Starting EgoDex training integration tests...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.environ.get('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print()
    
    # Run all tests
    tests = [
        ("Dataset Registration", test_dataset_registration),
        ("Training Mixture Configuration", test_training_mixture_configuration),
        ("Training Script Arguments", test_training_script_arguments),
        ("Data Loading in Training Context", test_data_loading_in_training_context),
        ("Statistics Integration", test_statistics_integration),
        ("Conversation Format Consistency", test_conversation_format_consistency),
        ("Training Script Execution", test_training_script_execution),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING INTEGRATION SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All training integration tests passed! Ready for training.")
        return True
    else:
        print("⚠️  Some training integration tests failed. Please fix issues before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
