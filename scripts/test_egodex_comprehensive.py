#!/usr/bin/env python3
"""
Comprehensive EgoDex testing suite.

This script runs all EgoDex tests and provides a final readiness assessment
for large-scale training and evaluation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_test_script(script_name, description):
    """Run a test script and return the result."""
    print(f"🧪 Running {description}...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Return result
        success = result.returncode == 0
        
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📊 Result: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return success, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Failed to run {script_name}: {e}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        return False, duration


def main():
    """Run comprehensive EgoDex testing suite."""
    print("🚀 Starting Comprehensive EgoDex Testing Suite")
    print("=" * 80)
    
    # Check environment
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        print("   Please set it to your EgoDex data directory")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print(f"🐍 Python executable: {sys.executable}")
    print()
    
    # Define all test scripts
    test_suite = [
        {
            'script': 'test_egodex_quick.py',
            'description': 'Quick Integration Test',
            'category': 'Basic Functionality'
        },
        {
            'script': 'test_egodex_complete.py',
            'description': 'Complete Integration Test',
            'category': 'Basic Functionality'
        },
        {
            'script': 'test_egodex_pre_training.py',
            'description': 'Pre-training Validation',
            'category': 'Training Readiness'
        },
        {
            'script': 'validate_egodex_stats.py',
            'description': 'Statistics Validation',
            'category': 'Data Quality'
        },
        {
            'script': 'test_egodex_training.py',
            'description': 'Training Pipeline Test',
            'category': 'Training Readiness'
        },
        {
            'script': 'test_egodex_edge_cases.py',
            'description': 'Edge Cases and Robustness',
            'category': 'Robustness'
        },
        {
            'script': 'test_egodex_validation.py',
            'description': 'Validation and Evaluation',
            'category': 'Evaluation Readiness'
        }
    ]
    
    # Run all tests
    results = []
    total_duration = 0
    
    for test in test_suite:
        script_path = Path(__file__).parent / test['script']
        
        if not script_path.exists():
            print(f"⚠️  Test script not found: {test['script']}")
            results.append({
                'name': test['description'],
                'category': test['category'],
                'success': False,
                'duration': 0,
                'error': 'Script not found'
            })
            continue
        
        success, duration = run_test_script(str(script_path), test['description'])
        total_duration += duration
        
        results.append({
            'name': test['description'],
            'category': test['category'],
            'success': success,
            'duration': duration,
            'error': None
        })
        
        print()
    
    # Generate comprehensive report
    print("=" * 80)
    print("📊 COMPREHENSIVE EGOdex TESTING REPORT")
    print("=" * 80)
    
    # Group results by category
    categories = {}
    for result in results:
        category = result['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    # Report by category
    for category, category_results in categories.items():
        print(f"\n📋 {category}")
        print("-" * 40)
        
        passed = sum(1 for r in category_results if r['success'])
        total = len(category_results)
        category_duration = sum(r['duration'] for r in category_results)
        
        for result in category_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration_str = f"{result['duration']:.1f}s"
            print(f"  {status} {result['name']} ({duration_str})")
            if not result['success'] and result['error']:
                print(f"      Error: {result['error']}")
        
        print(f"  📊 Category Summary: {passed}/{total} passed ({category_duration:.1f}s total)")
    
    # Overall summary
    total_passed = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"\n🎯 OVERALL SUMMARY")
    print("-" * 40)
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_tests - total_passed}")
    print(f"⏱️  Total Duration: {total_duration:.1f} seconds")
    print(f"📈 Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    # Readiness assessment
    print(f"\n🎯 READINESS ASSESSMENT")
    print("-" * 40)
    
    # Critical categories that must pass
    critical_categories = ['Basic Functionality', 'Training Readiness']
    critical_passed = True
    
    for category in critical_categories:
        if category in categories:
            category_results = categories[category]
            category_passed = sum(1 for r in category_results if r['success'])
            category_total = len(category_results)
            
            if category_passed == category_total:
                print(f"✅ {category}: READY")
            else:
                print(f"❌ {category}: NOT READY ({category_passed}/{category_total} passed)")
                critical_passed = False
    
    # Optional categories
    optional_categories = ['Data Quality', 'Robustness', 'Evaluation Readiness']
    for category in optional_categories:
        if category in categories:
            category_results = categories[category]
            category_passed = sum(1 for r in category_results if r['success'])
            category_total = len(category_results)
            
            if category_passed == category_total:
                print(f"✅ {category}: READY")
            else:
                print(f"⚠️  {category}: PARTIAL ({category_passed}/{category_total} passed)")
    
    # Final recommendation
    print(f"\n🚀 FINAL RECOMMENDATION")
    print("-" * 40)
    
    if critical_passed and total_passed >= total_tests * 0.8:  # 80% pass rate
        print("🎉 READY FOR LARGE-SCALE TRAINING!")
        print("   All critical tests passed. EgoDex integration is robust.")
        
        if total_passed == total_tests:
            print("🌟 EXCELLENT: All tests passed!")
        else:
            print(f"📊 Good: {total_passed}/{total_tests} tests passed")
            
        print("\n📋 Next Steps:")
        print("   1. Start with small-scale training to validate setup")
        print("   2. Monitor training metrics and convergence")
        print("   3. Scale up to full dataset once confident")
        
        return True
        
    elif critical_passed:
        print("⚠️  CONDITIONALLY READY")
        print("   Critical tests passed, but some issues detected.")
        print("   Review failed tests before proceeding.")
        
        print("\n📋 Next Steps:")
        print("   1. Fix issues in failed tests")
        print("   2. Re-run comprehensive test suite")
        print("   3. Proceed with caution when training")
        
        return False
        
    else:
        print("❌ NOT READY FOR TRAINING")
        print("   Critical tests failed. Must fix issues before training.")
        
        print("\n📋 Next Steps:")
        print("   1. Fix critical issues first")
        print("   2. Re-run individual test scripts")
        print("   3. Ensure all basic functionality works")
        print("   4. Only proceed when critical tests pass")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
