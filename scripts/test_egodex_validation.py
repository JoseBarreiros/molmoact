#!/usr/bin/env python3
"""
Test EgoDex validation and evaluation.

This script tests validation split handling, evaluation metrics,
and model inference capabilities with EgoDex data.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions
from olmo.data.get_dataset import get_dataset_by_name


def test_validation_split_handling():
    """Test proper handling of validation/test splits."""
    print("🧪 Testing validation split handling...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Test different splits
        splits_to_test = ['test', 'additional']  # We don't have train data
        
        for split in splits_to_test:
            print(f"  📊 Testing split: {split}")
            
            try:
                dataset = EgoDexPoseActions(
                    data_dir=data_dir,
                    split=split,
                    data_percentage=0.1,
                    action_sequence_length=8,
                    use_action_chunking=True,
                    use_temporal_aggregation=True,
                    use_action_normalization=True,
                    use_action_discretization=True
                )
                
                print(f"    ✅ {split} split loaded: {len(dataset)} episodes")
                
                # Validate split-specific behavior
                if len(dataset) > 0:
                    sample = dataset[0]
                    
                    # Check that samples have the expected structure for evaluation
                    required_keys = ['question', 'answers', 'annotation']
                    for key in required_keys:
                        if key not in sample:
                            print(f"    ❌ Missing key '{key}' in {split} split")
                            return False
                    
                    # Check annotation has evaluation-relevant information
                    annotation = sample['annotation']
                    eval_keys = ['action_type', 'action_sequence_length', 'action_dof']
                    for key in eval_keys:
                        if key not in annotation:
                            print(f"    ❌ Missing evaluation key '{key}' in {split} split")
                            return False
                    
                    print(f"    ✅ {split} split has proper evaluation structure")
                
            except Exception as e:
                print(f"    ❌ Failed to load {split} split: {e}")
                return False
        
        print("  ✅ Validation split handling works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation split test failed: {e}")
        return False


def test_evaluation_metrics_structure():
    """Test that data structure supports evaluation metrics."""
    print("🧪 Testing evaluation metrics structure...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.05,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        print(f"  📊 Dataset loaded: {len(dataset)} episodes")
        
        if len(dataset) == 0:
            print("  ⚠️  No episodes available for evaluation testing")
            return True
        
        # Test multiple samples for evaluation structure
        eval_metrics = {
            'has_questions': 0,
            'has_answers': 0,
            'has_reasoning': 0,
            'has_action_sequences': 0,
            'has_pose_data': 0,
            'has_finger_data': 0,
            'valid_action_dof': 0,
            'valid_sequence_length': 0
        }
        
        num_samples = min(10, len(dataset))
        
        for i in range(num_samples):
            try:
                sample = dataset[i]
                
                # Check question presence
                if 'question' in sample and sample['question']:
                    eval_metrics['has_questions'] += 1
                
                # Check answer presence
                if 'answers' in sample and sample['answers']:
                    eval_metrics['has_answers'] += 1
                
                # Check reasoning presence
                if 'answers' in sample and 'step-by-step reasoning' in sample['answers'].lower():
                    eval_metrics['has_reasoning'] += 1
                
                # Check action sequence presence
                if 'answers' in sample and 'action sequence' in sample['answers'].lower():
                    eval_metrics['has_action_sequences'] += 1
                
                # Check annotation structure
                annotation = sample.get('annotation', {})
                
                # Check pose data
                hand_poses = annotation.get('hand_poses', [])
                if hand_poses:
                    eval_metrics['has_pose_data'] += 1
                
                # Check finger data
                finger_tips = annotation.get('finger_tips', [])
                if finger_tips:
                    eval_metrics['has_finger_data'] += 1
                
                # Check action DOF
                action_dof = annotation.get('action_dof', 0)
                if action_dof == 21:  # Expected DOF
                    eval_metrics['valid_action_dof'] += 1
                
                # Check sequence length
                seq_length = annotation.get('action_sequence_length', 0)
                if seq_length > 0:
                    eval_metrics['valid_sequence_length'] += 1
                
            except Exception as e:
                print(f"    ⚠️  Sample {i} evaluation failed: {e}")
        
        # Report evaluation metrics
        print("  📊 Evaluation metrics structure:")
        for metric, count in eval_metrics.items():
            percentage = (count / num_samples) * 100
            print(f"    {metric}: {count}/{num_samples} ({percentage:.1f}%)")
        
        # Check critical metrics
        critical_metrics = ['has_questions', 'has_answers', 'has_pose_data', 'valid_action_dof']
        for metric in critical_metrics:
            if eval_metrics[metric] < num_samples * 0.8:  # Less than 80%
                print(f"    ❌ {metric} below threshold: {eval_metrics[metric]}/{num_samples}")
                return False
        
        print("  ✅ Evaluation metrics structure is adequate")
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation metrics test failed: {e}")
        return False


def test_conversation_quality_metrics():
    """Test conversation quality for evaluation."""
    print("🧪 Testing conversation quality metrics...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split='test',
            data_percentage=0.02,
            action_sequence_length=8,
            use_action_chunking=True,
            use_temporal_aggregation=True,
            use_action_normalization=True,
            use_action_discretization=True
        )
        
        print(f"  📊 Dataset loaded: {len(dataset)} episodes")
        
        if len(dataset) == 0:
            print("  ⚠️  No episodes available for conversation quality testing")
            return True
        
        # Quality metrics
        quality_metrics = {
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'questions_with_task': 0,
            'answers_with_reasoning': 0,
            'answers_with_actions': 0,
            'reasonable_lengths': 0
        }
        
        total_questions = 0
        total_answers = 0
        num_samples = min(5, len(dataset))
        
        for i in range(num_samples):
            try:
                sample = dataset[i]
                
                # Analyze question
                question = sample.get('question', '')
                if question:
                    quality_metrics['avg_question_length'] += len(question)
                    total_questions += 1
                    
                    if 'task is' in question.lower():
                        quality_metrics['questions_with_task'] += 1
                
                # Analyze answer
                answers = sample.get('answers', '')
                if answers:
                    quality_metrics['avg_answer_length'] += len(answers)
                    total_answers += 1
                    
                    if 'step-by-step reasoning' in answers.lower():
                        quality_metrics['answers_with_reasoning'] += 1
                    
                    if 'action sequence' in answers.lower():
                        quality_metrics['answers_with_actions'] += 1
                
                # Check reasonable lengths
                if len(question) > 50 and len(answers) > 100:
                    quality_metrics['reasonable_lengths'] += 1
                
            except Exception as e:
                print(f"    ⚠️  Sample {i} quality analysis failed: {e}")
        
        # Calculate averages
        if total_questions > 0:
            quality_metrics['avg_question_length'] /= total_questions
        if total_answers > 0:
            quality_metrics['avg_answer_length'] /= total_answers
        
        # Report quality metrics
        print("  📊 Conversation quality metrics:")
        print(f"    Average question length: {quality_metrics['avg_question_length']:.1f} chars")
        print(f"    Average answer length: {quality_metrics['avg_answer_length']:.1f} chars")
        print(f"    Questions with task context: {quality_metrics['questions_with_task']}/{num_samples}")
        print(f"    Answers with reasoning: {quality_metrics['answers_with_reasoning']}/{num_samples}")
        print(f"    Answers with actions: {quality_metrics['answers_with_actions']}/{num_samples}")
        print(f"    Reasonable conversation lengths: {quality_metrics['reasonable_lengths']}/{num_samples}")
        
        # Quality checks
        if quality_metrics['avg_question_length'] < 50:
            print("    ⚠️  Questions seem too short")
        if quality_metrics['avg_answer_length'] < 100:
            print("    ⚠️  Answers seem too short")
        if quality_metrics['answers_with_reasoning'] < num_samples * 0.5:
            print("    ⚠️  Too few answers have reasoning")
        if quality_metrics['answers_with_actions'] < num_samples * 0.5:
            print("    ⚠️  Too few answers have action sequences")
        
        print("  ✅ Conversation quality metrics calculated")
        return True
        
    except Exception as e:
        print(f"  ❌ Conversation quality test failed: {e}")
        return False


def test_inference_compatibility():
    """Test that data is compatible with model inference."""
    print("🧪 Testing inference compatibility...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
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
        
        print(f"  📊 Dataset loaded: {len(dataset)} episodes")
        
        if len(dataset) == 0:
            print("  ⚠️  No episodes available for inference testing")
            return True
        
        # Test inference compatibility
        inference_checks = {
            'has_image_input': 0,
            'has_text_input': 0,
            'has_action_output': 0,
            'consistent_format': 0,
            'valid_tokens': 0
        }
        
        num_samples = min(3, len(dataset))
        
        for i in range(num_samples):
            try:
                sample = dataset[i]
                
                # Check for image input (should be in the conversation format)
                question = sample.get('question', '')
                if question and '[image]' in question:
                    inference_checks['has_image_input'] += 1
                
                # Check for text input
                if question and len(question) > 10:
                    inference_checks['has_text_input'] += 1
                
                # Check for action output
                answers = sample.get('answers', '')
                if answers and 'action sequence' in answers.lower():
                    inference_checks['has_action_output'] += 1
                
                # Check consistent format
                if 'question' in sample and 'answers' in sample and 'annotation' in sample:
                    inference_checks['consistent_format'] += 1
                
                # Check for reasonable token lengths (basic check)
                if len(question) < 2000 and len(answers) < 2000:
                    inference_checks['valid_tokens'] += 1
                
            except Exception as e:
                print(f"    ⚠️  Sample {i} inference check failed: {e}")
        
        # Report inference compatibility
        print("  📊 Inference compatibility checks:")
        for check, count in inference_checks.items():
            percentage = (count / num_samples) * 100
            print(f"    {check}: {count}/{num_samples} ({percentage:.1f}%)")
        
        # Critical checks
        critical_checks = ['has_text_input', 'consistent_format', 'valid_tokens']
        for check in critical_checks:
            if inference_checks[check] < num_samples:
                print(f"    ❌ {check} not satisfied for all samples")
                return False
        
        print("  ✅ Inference compatibility looks good")
        return True
        
    except Exception as e:
        print(f"  ❌ Inference compatibility test failed: {e}")
        return False


def test_evaluation_dataset_creation():
    """Test creation of evaluation-specific datasets."""
    print("🧪 Testing evaluation dataset creation...")
    
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        return False
    
    try:
        # Test different evaluation configurations
        eval_configs = [
            {
                'name': 'small_eval',
                'data_percentage': 0.01,
                'action_sequence_length': 8
            },
            {
                'name': 'medium_eval',
                'data_percentage': 0.05,
                'action_sequence_length': 4
            }
        ]
        
        for config in eval_configs:
            print(f"  📊 Testing evaluation config: {config['name']}")
            
            try:
                dataset = EgoDexPoseActions(
                    data_dir=data_dir,
                    split='test',
                    data_percentage=config['data_percentage'],
                    action_sequence_length=config['action_sequence_length'],
                    use_action_chunking=True,
                    use_temporal_aggregation=True,
                    use_action_normalization=True,
                    use_action_discretization=True
                )
                
                print(f"    ✅ {config['name']} dataset created: {len(dataset)} episodes")
                
                # Validate dataset properties
                if len(dataset) > 0:
                    sample = dataset[0]
                    annotation = sample.get('annotation', {})
                    
                    # Check sequence length matches config
                    actual_length = annotation.get('action_sequence_length', 0)
                    if actual_length == config['action_sequence_length']:
                        print(f"    ✅ Correct sequence length: {actual_length}")
                    else:
                        print(f"    ⚠️  Expected {config['action_sequence_length']}, got {actual_length}")
                
            except Exception as e:
                print(f"    ❌ Failed to create {config['name']} dataset: {e}")
                return False
        
        print("  ✅ Evaluation dataset creation works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation dataset creation test failed: {e}")
        return False


def main():
    """Run all validation and evaluation tests."""
    print("🚀 Starting EgoDex validation and evaluation testing...")
    print("=" * 60)
    
    # Check environment
    data_dir = os.getenv('EGODEX_DATA_DIR')
    if not data_dir:
        print("❌ EGODEX_DATA_DIR environment variable not set")
        print("   Please set it to your EgoDex data directory")
        return False
    
    print(f"📁 Using data directory: {data_dir}")
    print()
    
    # Run validation tests
    tests = [
        ("Validation Split Handling", test_validation_split_handling),
        ("Evaluation Metrics Structure", test_evaluation_metrics_structure),
        ("Conversation Quality Metrics", test_conversation_quality_metrics),
        ("Inference Compatibility", test_inference_compatibility),
        ("Evaluation Dataset Creation", test_evaluation_dataset_creation),
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
    print("📊 VALIDATION AND EVALUATION TESTING SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation and evaluation tests passed!")
        print("📊 EgoDex data is ready for model evaluation!")
        return True
    else:
        print("⚠️  Some validation tests failed. Review issues before evaluation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
