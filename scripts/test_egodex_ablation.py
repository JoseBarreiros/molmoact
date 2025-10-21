#!/usr/bin/env python3
"""
EgoDex ablation study test script.
Tests different data percentages and configurations.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_data_percentages():
    """Test different data percentages."""
    print("🧪 Testing data percentages...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        percentages = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        for pct in percentages:
            print(f"\n📊 Testing {pct*100:.1f}% of data...")
            
            start_time = time.time()
            dataset = EgoDexPoseActions(
                data_percentage=pct,
                action_sequence_length=8,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            load_time = time.time() - start_time
            
            print(f"  ✅ Loaded {len(dataset)} episodes in {load_time:.2f}s")
            
            if len(dataset) > 0:
                # Test loading a sample
                start_time = time.time()
                sample = dataset[0]
                sample_time = time.time() - start_time
                
                print(f"  ✅ Sample loaded in {sample_time:.3f}s")
                
                # Check action processing
                if 'actions' in sample:
                    actions = sample['actions']
                    print(f"  🤖 Actions: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
                
                # Check text generation
                if 'text' in sample:
                    text = sample['text']
                    print(f"  📝 Text: {len(text)} characters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data percentage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_configurations():
    """Test different action processing configurations."""
    print("\n🧪 Testing action configurations...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        configs = [
            {"name": "Basic", "chunking": False, "aggregation": False, "normalization": False, "discretization": False},
            {"name": "Chunking Only", "chunking": True, "aggregation": False, "normalization": False, "discretization": False},
            {"name": "Chunking + Aggregation", "chunking": True, "aggregation": True, "normalization": False, "discretization": False},
            {"name": "Chunking + Aggregation + Normalization", "chunking": True, "aggregation": True, "normalization": True, "discretization": False},
            {"name": "Full Pipeline", "chunking": True, "aggregation": True, "normalization": True, "discretization": True},
        ]
        
        for config in configs:
            print(f"\n🔧 Testing {config['name']}...")
            
            dataset = EgoDexPoseActions(
                data_percentage=0.1,
                action_sequence_length=8,
                use_action_chunking=config['chunking'],
                use_temporal_aggregation=config['aggregation'],
                use_action_normalization=config['normalization'],
                use_action_discretization=config['discretization']
            )
            
            print(f"  ✅ Dataset loaded with {len(dataset)} episodes")
            
            if len(dataset) > 0:
                sample = dataset[0]
                
                if 'actions' in sample:
                    actions = sample['actions']
                    print(f"  🤖 Actions: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
                    
                    # Check if actions are properly processed
                    if config['normalization']:
                        if actions.min() >= -1.1 and actions.max() <= 1.1:
                            print("  ✅ Actions appear normalized")
                        else:
                            print("  ⚠️  Actions may not be normalized")
                    
                    if config['discretization']:
                        unique_values = len(torch.unique(actions))
                        total_values = actions.numel()
                        if unique_values < total_values * 0.1:
                            print("  ✅ Actions appear discretized")
                        else:
                            print("  ⚠️  Actions may not be discretized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Action configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test performance with different configurations."""
    print("\n🧪 Testing performance...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        import time
        
        # Test loading time
        print("⏱️  Testing loading time...")
        
        configs = [
            {"name": "Small (1%)", "pct": 0.01},
            {"name": "Medium (10%)", "pct": 0.1},
            {"name": "Large (50%)", "pct": 0.5},
        ]
        
        for config in configs:
            print(f"\n📊 Testing {config['name']}...")
            
            start_time = time.time()
            dataset = EgoDexPoseActions(
                data_percentage=config['pct'],
                action_sequence_length=8,
                use_action_chunking=True,
                use_temporal_aggregation=True,
                use_action_normalization=True,
                use_action_discretization=True
            )
            load_time = time.time() - start_time
            
            print(f"  ✅ Loaded {len(dataset)} episodes in {load_time:.2f}s")
            
            if len(dataset) > 0:
                # Test sample loading time
                start_time = time.time()
                for i in range(min(10, len(dataset))):
                    sample = dataset[i]
                sample_time = time.time() - start_time
                
                print(f"  ✅ Loaded 10 samples in {sample_time:.3f}s")
                print(f"  📈 Average per sample: {sample_time/10:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_quality():
    """Test conversation format quality."""
    print("\n🧪 Testing conversation quality...")
    
    try:
        from olmo.data.egodex_dataset import EgoDexPoseActions
        
        dataset = EgoDexPoseActions(
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
        
        print(f"  📊 Testing {min(5, len(dataset))} samples...")
        
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            text = sample['text']
            
            print(f"\n  📝 Sample {i+1}:")
            print(f"    Length: {len(text)} characters")
            
            # Check for key components
            components = [
                "The task is",
                "What is the action that the robot should take",
                "let's think through it step by step",
                "How far are the objects from the hand",
                "How does the hand move during the task",
                "How do the fingers move for precise manipulation",
                "Action Sequence",
                "hand pose and finger tip actions"
            ]
            
            found_components = []
            for component in components:
                if component in text:
                    found_components.append(component)
            
            print(f"    Components found: {len(found_components)}/{len(components)}")
            
            if len(found_components) >= len(components) * 0.8:
                print("    ✅ High quality conversation")
            elif len(found_components) >= len(components) * 0.6:
                print("    ⚠️  Medium quality conversation")
            else:
                print("    ❌ Low quality conversation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Conversation quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_ablation_study():
    """Run comprehensive ablation study."""
    print("🚀 Starting EgoDex ablation study...")
    print("="*60)
    
    tests = [
        ("Data Percentages", test_data_percentages),
        ("Action Configurations", test_action_configurations),
        ("Performance", test_performance),
        ("Conversation Quality", test_conversation_quality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Report results
    print("\n" + "="*60)
    print("📊 ABLATION STUDY RESULTS")
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
        print("🎉 All ablation tests passed!")
        return True
    else:
        print("⚠️  Some ablation tests failed.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run EgoDex ablation study tests")
    parser.add_argument("--test", choices=[
        "data_percentages", "action_configs", "performance", "conversation", "all"
    ], default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_ablation_study()
    else:
        test_map = {
            "data_percentages": test_data_percentages,
            "action_configs": test_action_configurations,
            "performance": test_performance,
            "conversation": test_conversation_quality,
        }
        
        test_func = test_map[args.test]
        success = test_func()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


