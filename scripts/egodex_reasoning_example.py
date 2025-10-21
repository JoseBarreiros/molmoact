#!/usr/bin/env python3
"""
EgoDex Reasoning Example

This script demonstrates how reasoning is handled in EgoDex data processing.
It shows the different types of reasoning generated and how they're structured.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.data.egodex_dataset import EgoDexPoseActions


def demonstrate_egodex_reasoning(data_dir: str):
    """Demonstrate EgoDex reasoning components."""
    print("EgoDex Reasoning Components")
    print("=" * 50)
    
    try:
        # Load dataset
        dataset = EgoDexPoseActions(
            data_dir=data_dir,
            split="test",
            max_frames=2,
            action_sequence_length=4,
            normalize_actions=True,
            discretize_actions=True
        )
        
        if len(dataset) == 0:
            print("No episodes available for reasoning demonstration")
            return
        
        # Get a sample
        rng = None
        sample = dataset.get(0, rng)
        
        print(f"\n1. CONVERSATION STRUCTURE")
        print("-" * 30)
        
        # Show the question structure
        question = sample['questions'][0]
        print("Question (LIBERO-style chain-of-thought prompting):")
        print(question)
        
        print(f"\n2. REASONING COMPONENTS")
        print("-" * 30)
        
        # Show the answer structure
        answer = sample['answers']
        print("Answer (structured reasoning + actions):")
        print(answer)
        
        print(f"\n3. REASONING BREAKDOWN")
        print("-" * 30)
        
        # Extract reasoning components
        lines = answer.split('\n')
        
        print("A. Depth Analysis:")
        for line in lines:
            if "Depth analysis:" in line:
                print(f"  {line.strip()}")
                break
        
        print("\nB. Trajectory Analysis:")
        for line in lines:
            if "Trajectory analysis:" in line:
                print(f"  {line.strip()}")
                break
        
        print("\nC. Finger Movement Analysis:")
        for line in lines:
            if "Finger tip analysis:" in line:
                print(f"  {line.strip()}")
                break
        
        print("\nD. Action Sequence:")
        action_started = False
        for line in lines:
            if "Action Sequence" in line:
                action_started = True
                print(f"  {line.strip()}")
            elif action_started and line.strip().startswith('['):
                print(f"  {line.strip()}")
            elif action_started and line.strip() and not line.strip().startswith('['):
                break
        
        print(f"\n4. ANNOTATION DETAILS")
        print("-" * 30)
        
        annotation = sample['annotation']
        print(f"Task: {annotation['task_name']}")
        print(f"Action Type: {annotation['action_type']}")
        print(f"Action DOF: {annotation['action_dof']}")
        print(f"Action Sequence Length: {annotation['action_sequence_length']}")
        
        if 'action_processing' in annotation:
            processing = annotation['action_processing']
            print(f"Action Processing:")
            for key, value in processing.items():
                print(f"  - {key}: {value}")
        
        print(f"\n5. REASONING FEATURES")
        print("-" * 30)
        
        print("✓ Chain-of-thought prompting (LIBERO-style)")
        print("✓ Depth perception reasoning")
        print("✓ Trajectory analysis")
        print("✓ Finger movement analysis")
        print("✓ Step-by-step action generation")
        print("✓ Multi-modal reasoning (vision + pose)")
        print("✓ Task-specific descriptions")
        
        print(f"\n6. REASONING FLOW")
        print("-" * 30)
        
        print("1. Visual Analysis → Extract hand poses and finger tips")
        print("2. Depth Reasoning → Analyze 3D positions and object depth")
        print("3. Trajectory Reasoning → Analyze movement patterns and direction")
        print("4. Finger Analysis → Analyze fine manipulation details")
        print("5. Action Generation → Create 21-DOF action sequence")
        print("6. Integration → Combine reasoning with executable actions")
        
    except Exception as e:
        print(f"Error demonstrating reasoning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate EgoDex reasoning components")
    parser.add_argument("--data-dir", required=True, help="Path to EgoDex data directory")
    
    args = parser.parse_args()
    
    demonstrate_egodex_reasoning(args.data_dir)


if __name__ == "__main__":
    import argparse
    main()


