# EgoDex Reasoning System

## Overview

EgoDex implements a comprehensive reasoning system that generates structured, step-by-step explanations for human manipulation actions. The reasoning system follows the LIBERO pattern but is adapted for human hand poses and finger tip movements.

## Reasoning Architecture

### 1. Chain-of-Thought Prompting

EgoDex uses LIBERO-style chain-of-thought prompting to encourage the model to reason through manipulation tasks:

```
The task is {description}. 
What is the action that the robot should take. 
To figure out the action that the robot should take to {description}, 
let's think through it step by step. 
First, what is the depth map for the first image? 
Second, what is the trajectory of the end effector in the first image? 
Based on the depth map of the first image and the trajectory of the end effector in the first image, 
along with other images from different camera views as additional information, 
what is the action that the robot should take?
```

### 2. Multi-Component Reasoning

The reasoning system generates four main components:

#### A. Depth Analysis
- **Purpose**: Analyze 3D hand positions to infer object depth
- **Method**: Extract hand position (x, y, z) from pose data
- **Output**: `"Depth analysis: {hand} hand starts at position {pos}, indicating objects at approximately {z:.2f}m depth."`

#### B. Trajectory Analysis
- **Purpose**: Analyze hand movement patterns and direction
- **Method**: Calculate movement vector from start to end positions
- **Output**: `"Trajectory analysis: {hand} hand moves from {start} to {end}, covering {distance:.3f}m. The movement vector is {movement}, indicating a {'grasping' if distance > 0.1 else 'fine manipulation'} action."`

#### C. Finger Movement Analysis
- **Purpose**: Analyze individual finger tip movements for fine manipulation
- **Method**: Track 5 finger tips (Index, Middle, Ring, Pinky, Thumb) across time
- **Output**: `"Finger tip analysis: {finger} finger moved {distance:.3f}m"` or `"{finger} finger stable"`

#### D. Action Sequence Generation
- **Purpose**: Generate executable 21-DOF action sequence
- **Method**: Combine hand pose (6 DOF) + finger tips (15 DOF) with processing
- **Output**: Structured action matrix with chunking, temporal aggregation, normalization, and discretization

## Reasoning Flow

```
1. Visual Analysis
   ↓
2. Depth Reasoning (3D position analysis)
   ↓
3. Trajectory Reasoning (movement pattern analysis)
   ↓
4. Finger Analysis (fine manipulation analysis)
   ↓
5. Action Generation (21-DOF sequence creation)
   ↓
6. Integration (reasoning + executable actions)
```

## Implementation Details

### Depth Reasoning (`_generate_depth_reasoning`)

```python
def _generate_depth_reasoning(self, transforms: Dict[str, Any]) -> str:
    """Generate depth perception reasoning from pose data."""
    left_hand = transforms.get('leftHand', [])
    right_hand = transforms.get('rightHand', [])
    
    if left_hand and len(left_hand) > 0:
        left_pos = left_hand[0][:3, 3]
        depth_info = f"Depth analysis: Left hand at position {left_pos}, indicating objects at approximately {left_pos[2]:.2f}m depth."
    elif right_hand and len(right_hand) > 0:
        right_pos = right_hand[0][:3, 3]
        depth_info = f"Depth analysis: Right hand at position {right_pos}, indicating objects at approximately {right_pos[2]:.2f}m depth."
    else:
        depth_info = "Depth analysis: Unable to determine precise depth from hand poses."
    
    return depth_info
```

### Trajectory Reasoning (`_generate_trajectory_reasoning`)

```python
def _generate_trajectory_reasoning(self, left_poses: List[np.ndarray], right_poses: List[np.ndarray]) -> str:
    """Generate trajectory reasoning from hand pose sequences."""
    # Analyze trajectory of the primary hand
    primary_poses = right_poses if right_poses else left_poses
    hand_name = "right" if right_poses else "left"
    
    # Extract positions and calculate movement
    positions = [pose[:3, 3] for pose in primary_poses]
    start_pos = positions[0]
    end_pos = positions[-1]
    movement = end_pos - start_pos
    distance = np.linalg.norm(movement)
    
    trajectory_info = f"""Trajectory analysis: {hand_name} hand moves from position {start_pos} to {end_pos}, 
covering a distance of {distance:.3f}m. The movement vector is {movement}, 
indicating a {'grasping' if distance > 0.1 else 'fine manipulation'} action."""
    
    return trajectory_info
```

### Finger Movement Analysis (`_analyze_finger_tip_movement`)

```python
def _analyze_finger_tip_movement(self, finger_tips: List[np.ndarray]) -> str:
    """Analyze finger tip movement patterns."""
    finger_names = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
    finger_analysis = []
    
    for i, finger_name in enumerate(finger_names):
        start_pos = finger_tips[0][i*3:(i+1)*3]
        end_pos = finger_tips[-1][i*3:(i+1)*3]
        movement = end_pos - start_pos
        distance = np.linalg.norm(movement)
        
        if distance > 0.01:  # Significant movement threshold
            finger_analysis.append(f"{finger_name} finger moved {distance:.3f}m")
        else:
            finger_analysis.append(f"{finger_name} finger stable")
    
    return f"Finger tip analysis: {'; '.join(finger_analysis)}."
```

## Reasoning Output Format

### Complete Reasoning Example

```
Based on the visual analysis, here's the step-by-step reasoning:

Depth analysis: right hand starts at position [0.123, -0.456, 0.789], indicating objects at approximately 0.79m depth.

Trajectory analysis: right hand moves from position [0.123, -0.456, 0.789] to [0.145, -0.432, 0.801], covering a distance of 0.025m. The movement vector is [0.022, 0.024, 0.012], indicating a fine manipulation action.

Finger tip analysis: Index finger moved 0.015m; Middle finger stable; Ring finger moved 0.008m; Pinky finger stable; Thumb finger moved 0.012m.

Action Sequence (8 steps, 21 DOF: hand pose + finger tips):
[
  [0.123, -0.456, 0.789, 0.012, -0.023, 0.045, 0.111, 0.222, 0.333, ...],  # Step 1
  [0.125, -0.454, 0.790, 0.013, -0.022, 0.046, 0.112, 0.223, 0.334, ...],  # Step 2
  ...
]

Temporally aggregated action: [0.124, -0.455, 0.790, 0.013, -0.023, 0.046, 0.112, 0.223, 0.334, ...]

To perform {task_description}, the robot should execute these hand pose and finger tip actions in sequence, following the demonstrated human trajectory.
```

## Key Features

### 1. **Multi-Modal Reasoning**
- Combines visual information (egocentric images) with pose data
- Integrates 3D spatial reasoning with temporal analysis

### 2. **Hierarchical Analysis**
- High-level trajectory reasoning (gross movements)
- Fine-grained finger analysis (precise manipulation)
- Depth perception for spatial understanding

### 3. **Actionable Output**
- Generates executable 21-DOF action sequences
- Includes processing steps (chunking, aggregation, normalization)
- Provides both discrete and continuous action formats

### 4. **Task-Specific Adaptation**
- Uses task descriptions to contextualize reasoning
- Adapts reasoning style based on manipulation type
- Handles reversible tasks with appropriate descriptions

### 5. **LIBERO Compatibility**
- Follows LIBERO's chain-of-thought prompting pattern
- Maintains consistent reasoning structure
- Enables direct comparison with robot manipulation data

## Usage Examples

### Basic Usage

```python
# Load dataset with reasoning
dataset = EgoDexPoseActions(
    data_dir='/path/to/egodex/organized',
    normalize_actions=True,
    discretize_actions=True
)

# Get sample with reasoning
sample = dataset.get(0)
question = sample['questions'][0]  # Chain-of-thought prompt
answer = sample['answers']         # Structured reasoning + actions
```

### Reasoning Analysis

```python
# Analyze reasoning components
lines = answer.split('\n')

# Extract depth reasoning
depth_line = [line for line in lines if "Depth analysis:" in line][0]

# Extract trajectory reasoning  
trajectory_line = [line for line in lines if "Trajectory analysis:" in line][0]

# Extract finger analysis
finger_line = [line for line in lines if "Finger tip analysis:" in line][0]
```

## Benefits

1. **Interpretable Actions**: Clear reasoning for each action decision
2. **Multi-Scale Analysis**: From gross movements to fine finger control
3. **Spatial Understanding**: 3D depth and trajectory reasoning
4. **Task Context**: Reasoning adapted to specific manipulation tasks
5. **LIBERO Compatibility**: Consistent with robot manipulation reasoning
6. **Actionable Output**: Directly executable action sequences

The EgoDex reasoning system provides comprehensive, structured explanations that bridge the gap between human manipulation demonstrations and robot action execution, enabling effective transfer learning from human to robot manipulation.


