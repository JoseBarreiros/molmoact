#!/usr/bin/env python3
"""
Quick EgoDex integration test script.
Fast test to verify EgoDex is working with MolmoAct.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def quick_test():
    """Run a quick test of EgoDex integration."""
    print("🚀 Quick EgoDex integration test...")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        import torch
        from olmo.data.egodex_dataset import EgoDexPoseActions
        print("  ✅ All imports successful")
        
        # Test data directory
        print("📁 Testing data directory...")
        data_dir = os.getenv('EGODEX_DATA_DIR')
        if not data_dir:
            print("  ❌ EGODEX_DATA_DIR not set")
            print("  💡 Run: source egodex_test_data/organized/setup_egodex_test_env.sh")
            return False
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"  ❌ Data directory not found: {data_path}")
            return False
        
        print(f"  ✅ Data directory: {data_path}")
        
        # Test dataset loading
        print("🤖 Testing dataset loading...")
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
        
        if len(dataset) == 0:
            print("  ⚠️  No data available")
            return True
        
        # Test data loading
        print("📊 Testing data loading...")
        sample = dataset[0]
        print(f"  ✅ Sample loaded successfully")
        print(f"  📊 Sample keys: {list(sample.keys())}")
        
        # Test action processing
        if 'actions' in sample:
            actions = sample['actions']
            print(f"  🤖 Actions: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Test conversation format
        if 'text' in sample:
            text = sample['text']
            print(f"  📝 Text: {len(text)} characters")
            print(f"  📝 Preview: {text[:100]}...")
        
        print("\n🎉 Quick test passed! EgoDex integration is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

