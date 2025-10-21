#!/usr/bin/env python3
"""
EgoDex Conversation Example

This script demonstrates the complete EgoDex conversation format with the new
natural language questions and shows what a realistic answer looks like.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions


def demonstrate_egodex_conversation(data_dir: str):
    """Demonstrate the complete EgoDex conversation format."""
    print("EgoDex Conversation Example")
    print("=" * 60)
    
    try:
        # Load dataset
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            max_frames=2,
            action_sequence_length=8,
            normalize_actions=True,
            discretize_actions=True
        )
        
        if len(dataset) == 0:
            print("No episodes available for conversation demonstration")
            return
        
        # Get a sample
        rng = None
        sample = dataset.get(0, rng)
        
        print(f"\n📋 EPISODE INFORMATION")
        print("-" * 30)
        annotation = sample['annotation']
        print(f"Task: {annotation['task_name']}")
        print(f"Episode ID: {annotation['episode_id']}")
        print(f"Action Type: {annotation['action_type']}")
        print(f"Action DOF: {annotation['action_dof']}")
        print(f"Action Sequence Length: {annotation['action_sequence_length']}")
        
        print(f"\n❓ QUESTION (User Input)")
        print("-" * 30)
        question = sample['question']
        print(question)
        
        print(f"\n🤖 ANSWER (Model Response)")
        print("-" * 30)
        answer = sample['answers']
        print(answer)
        
        print(f"\n🔍 CONVERSATION ANALYSIS")
        print("-" * 30)
        
        # Analyze the conversation structure
        lines = answer.split('\n')
        
        print("Answer Structure:")
        print("1. Reasoning Header")
        print("2. Three Step-by-Step Analyses:")
        print("   - Distance analysis")
        print("   - Hand movement analysis") 
        print("   - Finger movement analysis")
        print("3. Action Sequence (8x21 matrix)")
        print("4. Temporally Aggregated Action")
        print("5. Execution Instructions")
        
        print(f"\n📊 ACTION PROCESSING DETAILS")
        print("-" * 30)
        action_processing = annotation['action_processing']
        for key, value in action_processing.items():
            print(f"  {key}: {value}")
        
        if 'action_stats' in annotation and annotation['action_stats']:
            stats = annotation['action_stats']
            print(f"\n📈 ACTION STATISTICS")
            print("-" * 30)
            print(f"  Samples used: {stats.get('n_samples', 'N/A')}")
            print(f"  Hand position range: [{stats['q01'][0]:.3f}, {stats['q99'][0]:.3f}]")
            print(f"  Hand rotation range: [{stats['q01'][3]:.3f}, {stats['q99'][3]:.3f}]")
            print(f"  Finger tip range: [{stats['q01'][6]:.3f}, {stats['q99'][6]:.3f}]")
        
        print(f"\n🎯 KEY FEATURES DEMONSTRATED")
        print("-" * 30)
        print("✅ Natural language questions (no technical jargon)")
        print("✅ Clear step-by-step reasoning")
        print("✅ 3D spatial analysis (distance)")
        print("✅ Hand trajectory analysis (movement)")
        print("✅ Finger manipulation analysis (fine control)")
        print("✅ 21-DOF action sequence (6 hand + 15 finger tips)")
        print("✅ Action processing (chunking, aggregation, normalization)")
        print("✅ LIBERO-compatible format")
        
        print(f"\n💡 CONVERSATION FLOW")
        print("-" * 30)
        print("1. Task Description → Sets context")
        print("2. Chain-of-Thought Questions → Guides reasoning")
        print("3. Step-by-Step Analysis → Breaks down the problem")
        print("4. Action Generation → Produces executable actions")
        print("5. Execution Instructions → Tells robot what to do")
        
    except Exception as e:
        print(f"Error demonstrating conversation: {e}")


def show_example_conversation():
    """Show a realistic example conversation without loading data."""
    print("EgoDex Conversation Example (Realistic)")
    print("=" * 60)
    
    print(f"\n❓ QUESTION (User Input)")
    print("-" * 30)
    question = """The task is pick up the red cup and place it on the table. 
What is the action that the robot should take. 
To figure out the action that the robot should take to pick up the red cup and place it on the table, 
let's think through it step by step. 
First, how far are the objects from the hand? 
Second, how does the hand move during the task? 
Third, how do the fingers move for precise manipulation? 
Based on the distance, hand movement, and finger control, 
what actions should the robot take?"""
    print(question)
    
    print(f"\n🤖 ANSWER (Model Response)")
    print("-" * 30)
    answer = """Based on the egocentric visual analysis, here's the step-by-step reasoning:

1. How far are the objects from the hand: right hand starts at position [0.123, -0.456, 0.789], indicating objects at approximately 0.79m depth.

2. How does the hand move during the task: right hand moves from position [0.123, -0.456, 0.789] to [0.145, -0.432, 0.801], covering a distance of 0.025m. The movement vector is [0.022, 0.024, 0.012], indicating a fine manipulation action.

3. How do the fingers move for precise manipulation: Index finger moved 0.015m; Middle finger stable; Ring finger moved 0.008m; Pinky finger stable; Thumb finger moved 0.012m.

Action Sequence (8 steps, 21 DOF: hand pose + finger tips):
[
  [0.123, -0.456, 0.789, 0.012, -0.023, 0.045, 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888, 0.999, 1.111, 1.222, 1.333, 1.444, 1.555, 1.666],  # Step 1
  [0.125, -0.454, 0.790, 0.013, -0.022, 0.046, 0.112, 0.223, 0.334, 0.445, 0.556, 0.667, 0.778, 0.889, 1.000, 1.112, 1.223, 1.334, 1.445, 1.556, 1.667],  # Step 2
  [0.127, -0.452, 0.791, 0.014, -0.021, 0.047, 0.113, 0.224, 0.335, 0.446, 0.557, 0.668, 0.779, 0.890, 1.001, 1.113, 1.224, 1.335, 1.446, 1.557, 1.668],  # Step 3
  [0.129, -0.450, 0.792, 0.015, -0.020, 0.048, 0.114, 0.225, 0.336, 0.447, 0.558, 0.669, 0.780, 0.891, 1.002, 1.114, 1.225, 1.336, 1.447, 1.558, 1.669],  # Step 4
  [0.131, -0.448, 0.793, 0.016, -0.019, 0.049, 0.115, 0.226, 0.337, 0.448, 0.559, 0.670, 0.781, 0.892, 1.003, 1.115, 1.226, 1.337, 1.448, 1.559, 1.670],  # Step 5
  [0.133, -0.446, 0.794, 0.017, -0.018, 0.050, 0.116, 0.227, 0.338, 0.449, 0.560, 0.671, 0.782, 0.893, 1.004, 1.116, 1.227, 1.338, 1.449, 1.560, 1.671],  # Step 6
  [0.135, -0.444, 0.795, 0.018, -0.017, 0.051, 0.117, 0.228, 0.339, 0.450, 0.561, 0.672, 0.783, 0.894, 1.005, 1.117, 1.228, 1.339, 1.450, 1.561, 1.672],  # Step 7
  [0.137, -0.442, 0.796, 0.019, -0.016, 0.052, 0.118, 0.229, 0.340, 0.451, 0.562, 0.673, 0.784, 0.895, 1.006, 1.118, 1.229, 1.340, 1.451, 1.562, 1.673]   # Step 8
]

Temporally aggregated action: [0.130, -0.449, 0.792, 0.016, -0.020, 0.049, 0.115, 0.226, 0.337, 0.448, 0.559, 0.670, 0.781, 0.892, 1.003, 1.115, 1.226, 1.337, 1.448, 1.559, 1.670]

To perform pick up the red cup and place it on the table, the robot should execute these hand pose and finger tip actions in sequence, following the demonstrated human trajectory."""
    print(answer)
    
    print(f"\n🔍 CONVERSATION BREAKDOWN")
    print("-" * 30)
    print("Question Components:")
    print("  • Task description: 'pick up the red cup and place it on the table'")
    print("  • Chain-of-thought prompt: 'let's think through it step by step'")
    print("  • Three natural questions:")
    print("    1. 'How far are the objects from the hand?'")
    print("    2. 'How does the hand move during the task?'")
    print("    3. 'How do the fingers move for precise manipulation?'")
    print("  • Action request: 'what actions should the robot take?'")
    
    print(f"\nAnswer Components:")
    print("  • Reasoning header: 'Based on the egocentric visual analysis...'")
    print("  • Three analyses matching the questions:")
    print("    1. Distance analysis: 3D hand position → object depth")
    print("    2. Movement analysis: hand trajectory → movement type")
    print("    3. Finger analysis: individual finger movements")
    print("  • Action sequence: 8x21 matrix (8 timesteps × 21 DOF)")
    print("  • Aggregated action: single 21-DOF action")
    print("  • Execution instructions: tells robot what to do")
    
    print(f"\n📊 ACTION FORMAT EXPLANATION")
    print("-" * 30)
    print("Each action step contains 21 values:")
    print("  • Positions 0-2: Hand position (x, y, z)")
    print("  • Positions 3-5: Hand rotation (rx, ry, rz)")
    print("  • Positions 6-8: Index finger tip (x, y, z)")
    print("  • Positions 9-11: Middle finger tip (x, y, z)")
    print("  • Positions 12-14: Ring finger tip (x, y, z)")
    print("  • Positions 15-17: Pinky finger tip (x, y, z)")
    print("  • Positions 18-20: Thumb finger tip (x, y, z)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate EgoDex conversation format")
    parser.add_argument("--data-dir", help="Path to EgoDex data directory (optional)")
    parser.add_argument("--example-only", action="store_true", help="Show example without loading data")
    
    args = parser.parse_args()
    
    if args.example_only:
        show_example_conversation()
    elif args.data_dir:
        demonstrate_egodex_conversation(args.data_dir)
    else:
        print("Usage:")
        print("  python scripts/egodex_conversation_example.py --example-only")
        print("  python scripts/egodex_conversation_example.py --data-dir /path/to/egodex")


if __name__ == "__main__":
    main()


