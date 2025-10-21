# EgoDex Integration with MolmoAct

This document describes how to integrate and use the EgoDex dataset with MolmoAct for mid-training, treating human hand poses as robot actions similar to LIBERO.

## Overview

EgoDex is a large-scale dataset of egocentric dexterous manipulation collected with ARKit on Apple Vision Pro. It provides:
- 829 hours of 30 Hz 1080p egocentric video
- 3D pose annotations for head, upper body, and hands
- Natural language task descriptions
- 194 diverse tabletop manipulation tasks

## Key Innovation: Enhanced Pose-Based Actions with Finger Tips

Unlike traditional approaches, this integration treats **human hand poses AND finger tip positions as robot actions**, following the LIBERO pattern with advanced processing:
- **Observations**: Egocentric camera images (like LIBERO's camera views)
- **Actions**: Human hand poses + finger tip positions converted to 21-DOF robot actions (6 hand + 15 finger tips)
- **Action Processing**: Chunking, temporal aggregation, normalization, and discretization (like LIBERO)
- **Learning**: Direct imitation of human manipulation trajectories with detailed finger control

## Quick Start

### 1. Download EgoDex Data

```bash
# Download test set (16 GB) for initial exploration
python scripts/prepare_egodex_data.py --output-dir /path/to/egodex --parts test

# Download training data (1.5 TB total, split into 5 parts)
python scripts/prepare_egodex_data.py --output-dir /path/to/egodex --parts part1 part2 part3 part4 part5

# Download all data
python scripts/prepare_egodex_data.py --output-dir /path/to/egodex --parts part1 part2 part3 part4 part5 test additional
```

### 2. Set Environment Variables

```bash
export EGODEX_DATA_DIR="/path/to/egodex/organized"
export EGODEX_DATA_PERCENTAGE=1.0  # Use full dataset (0.25, 0.5, 1.0 for scaling laws)
export PYTHONPATH="/path/to/molmoact:${PYTHONPATH:-}"  # Required for module imports
```

**Environment Variables:**
- `EGODEX_DATA_DIR`: Path to organized EgoDex data
- `EGODEX_DATA_PERCENTAGE`: Percentage of EgoDex data to use (0.0-1.0)
- `PYTHONPATH`: Must include MolmoAct project root for proper module imports

### 3. Compute Action Statistics

```bash
# Compute statistics from training data
python scripts/compute_egodex_stats.py \
    --data-dir /path/to/egodex/organized \
    --split train \
    --sample-size 1000 \
    --output-file egodex_action_stats.json

# Use statistics in training
python scripts/egodex_stats_example.py
```

### 4. Test Integration

```bash
# Test basic EgoDex integration
python scripts/test_egodex_integration.py --data-dir /path/to/egodex/organized

# Test pose-based action integration with statistics
python scripts/test_egodex_pose_actions.py --data-dir /path/to/egodex/organized
```

### 5. Run Training

```bash
# EgoDex pose-based actions only (100% of data)
python launch_scripts/train_multitask_model.py --mixture egodex-pose-only

# EgoDex with 50% of data for ablation study
python launch_scripts/train_multitask_model.py \
    --mixture egodex-pose-only \
    --egodex-data-percentage 0.5

# EgoDex pose-based + MolmoAct combined
python launch_scripts/train_multitask_model.py --mixture molmoact-midtrain-egodex-pose

# Combined training with 25% EgoDex data
python launch_scripts/train_multitask_model.py \
    --mixture molmoact-midtrain-egodex-pose \
    --egodex-data-percentage 0.25

# Original EgoDex approach (for comparison)
python launch_scripts/train_multitask_model.py --mixture egodex-only
```

### 6. Ablation Studies

```bash
# Run comprehensive ablation study
python scripts/egodex_ablation_study.py \
    --data-dir /path/to/egodex/organized \
    --percentages 0.1 0.25 0.5 0.75 1.0 \
    --output-dir ablation_results

# Compute statistics for specific percentage
python scripts/compute_egodex_stats.py \
    --data-dir /path/to/egodex/organized \
    --data-percentage 0.5 \
    --output-file egodex_stats_50pct.json
```

## Dataset Structure

After running the preparation script, your data will be organized as:

```
/path/to/egodex/organized/
├── train/           # Training data (725 hours)
│   ├── task1/
│   │   ├── 0.hdf5   # Pose annotations
│   │   ├── 0.mp4    # Video
│   │   ├── 1.hdf5
│   │   ├── 1.mp4
│   │   └── ...
│   └── task2/
│       └── ...
├── test/            # Test data (7 hours)
│   └── ...
└── additional/      # Additional data (97 hours)
    └── ...
```

## Data Format

### HDF5 Pose Files

Each HDF5 file contains:
- `camera/intrinsic`: 3x3 camera intrinsics matrix
- `transforms/`: 4x4 transformation matrices for all joints (N frames)
  - `camera`, `leftHand`, `rightHand`, `leftIndexFingerTip`, etc.
- `confidences/`: Joint confidence scores (0-1, optional)
- Attributes: `llm_description`, `llm_description2`, `which_llm_description`

### Video Files

- 30 Hz, 1080p MP4 videos
- Corresponding to pose data frame-by-frame
- N frames in video = N pose matrices in HDF5

## Usage Examples

### Pose-Based Action Learning (Recommended)

```python
from olmo.data.egodex_dataset import EgoDexPoseActions

# Load test data with pose-based actions
dataset = EgoDexPoseActions(
    data_dir="/path/to/egodex/organized",
    split="test",
    width=320,
    height=240,
    action_sequence_length=8  # Generate 8 pose actions per episode
)

# Get a sample
import numpy as np
rng = np.random.RandomState(42)
sample = dataset.get(0, rng)

print(f"Task: {sample['annotation']['task_name']}")
print(f"Action Type: {sample['annotation']['action_type']}")
print(f"Question: {sample['question'][:100]}...")
print(f"Answer contains action sequence: {'Action Sequence' in sample['answers']}")
print(f"Hand poses available: {len(sample['annotation']['hand_poses']['right'])}")
```

### Basic Dataset Loading (Original Approach)

```python
from olmo.data.egodex_dataset import EgoDexDataset

# Load test data
dataset = EgoDexDataset(
    data_dir="/path/to/egodex/organized",
    split="test",
    width=320,
    height=240,
    max_frames=10
)

# Get a sample
import numpy as np
rng = np.random.RandomState(42)
sample = dataset.get(0, rng)

print(f"Task: {sample['annotation']['task_name']}")
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answers']}")
print(f"Images: {len(sample['image'])} frames")
```

### Multi-Frame Dataset

```python
from olmo.data.egodex_dataset import EgoDexDatasetMultiFrame

# Load with multiple frames per episode
dataset = EgoDexDatasetMultiFrame(
    data_dir="/path/to/egodex/organized",
    split="train",
    frame_interval=5,  # Every 5th frame
    max_frames=20
)
```

### Via Dataset Registry

```python
from olmo.data.get_dataset import get_dataset_by_name

# Set environment variable
import os
os.environ['EGODEX_DATA_DIR'] = "/path/to/egodex/organized"

# Load via registry
dataset = get_dataset_by_name("egodex_train", "train")
```

## How Pose-Based Actions Work

### Data Flow

```
EgoDex Episode → [Egocentric Image + Task Description] → MolmoAct Model → Hand Pose Actions
```

### Input Format

**1. Visual Input:**
- **Egocentric camera image**: First-person view from Apple Vision Pro
- **Task description**: Natural language instruction (e.g., "pick up the black bowl")

**2. LIBERO-Style Prompting:**
```
"The task is {description}. What is the action that the robot should take. 
To figure out the action that the robot should take to {description}, 
let's think through it step by step. 
First, what is the depth map for the first image? 
Second, what is the trajectory of the end effector in the first image? 
Based on the depth map of the first image and the trajectory of the end effector in the first image, 
along with other images from different camera views as additional information, 
what is the action that the robot should take?"
```

### Output Format

**Target Response:**
```
Based on the visual analysis, here's the step-by-step reasoning:

Depth analysis: right hand starts at position [x, y, z], indicating objects at approximately z m depth.

Trajectory analysis: right hand moves from position [x1, y1, z1] to [x2, y2, z2], 
covering a distance of d m. The movement vector is [dx, dy, dz], 
indicating a grasping action.

Action Sequence (8 steps):
[
  [x, y, z, rx, ry, rz, gripper],  # right hand pose 1
  [x, y, z, rx, ry, rz, gripper],  # right hand pose 2
  ...
]

To perform {description}, the robot should execute these hand pose actions in sequence, 
following the demonstrated human trajectory.
```

### Pose-to-Action Conversion

**21-DOF Action Format:**
- **Hand Position**: `[x, y, z]` - 3D hand position from ARKit
- **Hand Rotation**: `[rx, ry, rz]` - Euler angles from rotation matrix
- **Finger Tips**: `[tip1_x, tip1_y, tip1_z, tip2_x, tip2_y, tip2_z, ...]` - 5 finger tip positions (15 values)
  - Index finger tip: `[tip1_x, tip1_y, tip1_z]`
  - Middle finger tip: `[tip2_x, tip2_y, tip2_z]`
  - Ring finger tip: `[tip3_x, tip3_y, tip3_z]`
  - Pinky finger tip: `[tip4_x, tip4_y, tip4_z]`
  - Thumb tip: `[tip5_x, tip5_y, tip5_z]`

**Enhanced Action Processing (like LIBERO):**
- **Action Chunking**: Generate 8×21 action matrices (8 timesteps × 21 DOF)
- **Temporal Aggregation**: Average finger tips and hand poses across timesteps
- **Normalization**: Per-dimension statistics (q01/q99) for consistent scaling
- **Discretization**: Convert to discrete tokens for language model integration

**Note**: No gripper state is included since human hands don't have a single gripper mechanism - finger tip positions provide the detailed manipulation information.

## Action Statistics Computation

### Why Statistics Are Needed

EgoDex actions need proper normalization for stable training, similar to LIBERO. The statistics computation process:

1. **Samples episodes** from the training split (default: 1000 episodes)
2. **Extracts actions** from hand poses and finger tip positions
3. **Computes percentiles** (q01, q99) for robust normalization
4. **Saves statistics** to JSON file for reuse

### Computing Statistics

```bash
# Basic computation
python scripts/compute_egodex_stats.py \
    --data-dir /path/to/egodex/organized \
    --split train \
    --sample-size 1000 \
    --output-file egodex_action_stats.json

# Quick computation (smaller sample)
python scripts/compute_egodex_stats.py \
    --data-dir /path/to/egodex/organized \
    --split train \
    --sample-size 100 \
    --output-file egodex_action_stats_small.json
```

### Using Precomputed Statistics

```python
# Load dataset with precomputed statistics
dataset = EgoDexPoseActions(
    data_dir='/path/to/egodex/organized',
    action_stats_file='egodex_action_stats.json',
    normalize_actions=True,
    discretize_actions=True
)

# Statistics are automatically loaded and used for normalization
```

### Statistics Format

```json
{
  "q01": [x_min, y_min, z_min, rx_min, ry_min, rz_min, tip1_x_min, ...],
  "q99": [x_max, y_max, z_max, rx_max, ry_max, rz_max, tip1_x_max, ...],
  "mean": [x_mean, y_mean, z_mean, rx_mean, ry_mean, rz_mean, tip1_x_mean, ...],
  "std": [x_std, y_std, z_std, rx_std, ry_std, rz_std, tip1_x_std, ...],
  "mask": [true, true, true, true, true, true, true, ...],
  "n_samples": 8000,
  "dataset_split": "train",
  "sample_size": 1000
}
```

## Data Percentage for Ablation Studies

### Overview

EgoDex supports training with a percentage of the total data for ablation studies. This allows you to:

- **Study data efficiency**: How much EgoDex data is needed for good performance?
- **Compare approaches**: Test different data percentages systematically
- **Resource optimization**: Train with less data when resources are limited
- **Reproducible experiments**: Deterministic sampling ensures consistent results

### Usage

```python
# Load 50% of EgoDex data
dataset = EgoDexPoseActions(
    data_dir='/path/to/egodex/organized',
    data_percentage=0.5,  # Use 50% of available data
    normalize_actions=True,
    discretize_actions=True
)

# Load 10% for quick experiments
dataset = EgoDexPoseActions(
    data_dir='/path/to/egodex/organized',
    data_percentage=0.1,  # Use 10% of available data
    normalize_actions=True,
    discretize_actions=True
)
```

**Using Environment Variable:**
```bash
# Set data percentage via environment variable (used by training scripts)
export EGODEX_DATA_PERCENTAGE=0.25  # Use 25% of data for scaling laws

# Then run training - the dataset will automatically use this percentage
python launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-scaling-25pct \
    allenai/MolmoAct-7B-D-Pretrain-0812
```

### Key Features

- **Deterministic Sampling**: Uses fixed seed (42) for reproducible results
- **Statistics Consistency**: Statistics computed from the sampled data
- **Task Distribution**: Maintains proportional task distribution
- **Memory Efficient**: Only loads the specified percentage of data

### Ablation Study Workflow

1. **Compute statistics** for each data percentage
2. **Train models** with different percentages
3. **Compare performance** across data sizes
4. **Analyze results** to determine optimal data usage

## Training Configurations

### Available Mixtures

1. **`egodex-pose-only`**: Train only on EgoDex pose-based actions with finger tips (21-DOF)
2. **`molmoact-midtrain-egodex-pose`**: Combine EgoDex pose-based actions with finger tips with MolmoAct mid-training data
3. **`egodex-only`**: Original EgoDex approach (for comparison)
4. **`molmoact-midtrain-egodex`**: Original combined approach
5. **`molmoact-midtrain-egodex-scaling-25pct`**: Scaling laws with 25% EgoDex data (90% EgoDex batches, 10% MolmoAct batches)
6. **`molmoact-midtrain-egodex-scaling-50pct`**: Scaling laws with 50% EgoDex data (90% EgoDex batches, 10% MolmoAct batches)
7. **`molmoact-midtrain-egodex-scaling-100pct`**: Scaling laws with 100% EgoDex data (90% EgoDex batches, 10% MolmoAct batches)

### Training Command Examples

```bash
# EgoDex only training (with finger tips)
python launch_scripts/train_multitask_model.py \
    --mixture egodex-pose-only \
    --model-config configs/molmoact_7b.yaml \
    --data-config configs/data_config.yaml

# Combined training (with finger tips)
python launch_scripts/train_multitask_model.py \
    --mixture molmoact-midtrain-egodex-pose \
    --model-config configs/molmoact_7b.yaml \
    --data-config configs/data_config.yaml
```

### Scaling Laws Experiments

For studying how performance scales with EgoDex data size, use the scaling configurations:

```bash
# Training with different EgoDex data percentages
# All use 90% EgoDex batches + 10% MolmoAct batches

# 25% EgoDex data
torchrun --nnodes=1 --nproc-per-node=1 \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-scaling-25pct allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=egodex_scaling_25pct \
    --save_folder=checkpoints/egodex_scaling_25pct \
    --duration 50000

# 50% EgoDex data  
torchrun --nnodes=1 --nproc-per-node=1 \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-scaling-50pct allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=egodex_scaling_50pct \
    --save_folder=checkpoints/egodex_scaling_50pct \
    --duration 50000

# 100% EgoDex data
torchrun --nnodes=1 --nproc-per-node=1 \
    launch_scripts/train_multitask_model.py \
    molmoact-midtrain-egodex-scaling-100pct allenai/MolmoAct-7B-D-Pretrain-0812 \
    --wandb.name=egodex_scaling_100pct \
    --save_folder=checkpoints/egodex_scaling_100pct \
    --duration 50000
```

**Key Features for Scaling Laws:**
- **Fixed batch ratio**: 90% EgoDex, 10% MolmoAct across all experiments
- **Variable data size**: 25%, 50%, 100% of total EgoDex data
- **Controlled via environment**: `EGODEX_DATA_PERCENTAGE=0.25|0.5|1.0`
- **Reproducible sampling**: Deterministic data subset selection

## Customization

### Task Filtering

```python
# Load only specific tasks
dataset = EgoDexDataset(
    data_dir="/path/to/egodex/organized",
    split="train",
    task_filter=["basic_pick_place", "stack_objects"]
)
```

### Frame Selection

```python
# Custom frame selection
dataset = EgoDexDataset(
    data_dir="/path/to/egodex/organized",
    split="train",
    max_frames=5,  # Limit frames per video
    width=640,     # Custom resolution
    height=480
)
```

### Pose Processing

The dataset automatically processes 3D poses into text descriptions. You can access raw pose data via:

```python
sample = dataset.get(0, rng)
pose_data = sample['annotation']['pose_data']

# Access raw transforms
transforms = pose_data['transforms']
left_hand_poses = transforms['leftHand']  # Shape: (N, 4, 4)

# Access confidences
confidences = pose_data['confidences']
left_hand_conf = confidences['leftHand']  # Shape: (N,)
```

## Troubleshooting

### Common Issues

1. **"Data directory not found"**
   - Ensure you've downloaded and organized the data using `prepare_egodex_data.py`
   - Check that `EGODEX_DATA_DIR` environment variable is set correctly

2. **"No episodes found"**
   - Verify the data directory structure matches the expected format
   - Check that HDF5 and MP4 files are paired correctly

3. **Memory issues with large videos**
   - Use `max_frames` parameter to limit frames per video
   - Consider using `EgoDexDatasetMultiFrame` with `frame_interval` > 1

4. **Pose data processing errors**
   - Ensure HDF5 files are not corrupted
   - Check that pose data contains expected joint names

5. **Module import errors (`ModuleNotFoundError: No module named 'launch_scripts'`)**
   - Add MolmoAct project root to PYTHONPATH: `export PYTHONPATH="/path/to/molmoact:${PYTHONPATH:-}"`
   - Ensure you're running from the correct directory
   - Verify the virtual environment is activated

### Performance Tips

1. **Use smaller image sizes** for faster training:
   ```python
   dataset = EgoDexDataset(width=224, height=224)
   ```

2. **Limit frames per video** for memory efficiency:
   ```python
   dataset = EgoDexDataset(max_frames=5)
   ```

3. **Use frame intervals** for trajectory learning:
   ```python
   dataset = EgoDexDatasetMultiFrame(frame_interval=10)
   ```

## Citation

If you use EgoDex data in your research, please cite:

```bibtex
@misc{egodex,
      title={EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video}, 
      author={Ryan Hoque and Peide Huang and David J. Yoon and Mouli Sivapurapu and Jian Zhang},
      year={2025},
      eprint={2505.11709},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11709}, 
}
```

## Visualization Tools

MolmoAct includes several visualization tools to help understand and verify the EgoDex integration:

### 1. Individual Action Chunk Visualization

Visualize individual action chunks with hand trajectories and task descriptions:

```bash
# Visualize random action chunks
./visualize_egodex_chunks.sh 5 output_dir 8

# Visualize all timesteps from a specific episode
./visualize_egodex_chunks.sh episode 0 output_dir 8

# Visualize specific episode and task
./visualize_egodex_chunks.sh episode 0 output_dir 8 "add_remove_lid"
```

**Features:**
- Shows single action chunk with hand trajectory
- Displays task description overlay
- 3D and 2D projection plots
- Configurable action chunk size

### 2. Episode Action Chunk Overview

Visualize all action chunks from an episode overlaid on one plot to verify overlap:

```bash
# Basic episode overview
./visualize_episode_chunks.sh 0

# Specific task with more timesteps
./visualize_episode_chunks.sh 0 add_remove_lid episode_chunks_test 50 8

# Different episode and task
./visualize_episode_chunks.sh 1 basic_pick_place episode_chunks_test 30 8
```

**Features:**
- **6-panel visualization** with multiple perspectives
- **3D trajectory plot** with all chunks overlaid
- **Timeline plot** showing chunk overlap patterns
- **Statistics panel** with overlap analysis
- **Color-coded chunks** for easy identification

**What to Look For:**
- Consecutive chunks should overlap by `(action_sequence_length - 1)` timesteps
- Smooth hand trajectories without sudden jumps
- Task-specific movement patterns
- Overlap percentage should be close to expected value

### 3. Performance Profiling

Test data loading performance and caching effectiveness:

```bash
# Quick performance test
python scripts/test_egodex_performance.py

# Comprehensive cache integrity test
python scripts/test_egodex_cache_integrity.py

# Detailed bottleneck analysis
python scripts/analyze_egodex_bottlenecks.py
```

### 4. Integration Testing

Verify EgoDex integration works correctly:

```bash
# Basic integration test
python scripts/test_egodex_integration.py

# Pose-based action test
python scripts/test_egodex_pose_actions.py

# End-to-end conversation test
python scripts/egodex_conversation_example.py
```

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Fix: Set PYTHONPATH
export PYTHONPATH="/path/to/molmoact:${PYTHONPATH:-}"
```

**2. Episode ID Collisions**
- EgoDex reuses episode IDs across different tasks
- Use `--task_name` parameter to filter specific tasks
- Example: `./visualize_episode_chunks.sh 0 add_remove_lid`

**3. Memory Issues**
- Use smaller `data_percentage` for testing
- Clear caches between tests
- Monitor memory usage with profiling tools

**4. Performance Issues**
- Enable pose data and frame caching (automatic)
- Use video handle reuse (automatic)
- Consider pre-computing action chunks for very large datasets

### Performance Optimization

The EgoDex integration includes several performance optimizations:

- **Pose data caching**: 111x speedup for repeated episodes
- **Frame caching**: Eliminates repeated video frame extraction
- **Video handle reuse**: Keeps handles open for efficiency
- **Fast initialization**: 10x faster dataset loading
- **Memory management**: Proper cleanup prevents leaks

Expected performance:
- **Initialization**: <10 seconds for 3,243 episodes
- **Sampling**: ~0.05 seconds per sample with caching
- **Cache speedup**: 100x+ for repeated episodes

## References

- [EgoDex Paper](https://arxiv.org/abs/2505.11709)
- [EgoDex GitHub](https://github.com/apple/ml-egodex)
- [MolmoAct Paper](https://arxiv.org/abs/2406.12324)
