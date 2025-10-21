"""
EgoDex dataset loader for MolmoAct mid-training.

This module provides dataset classes for loading and processing EgoDex data,
which consists of egocentric videos with 3D pose annotations from Apple Vision Pro.
"""

import os
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

from olmo.data.dataset import Dataset as BaseDataset


class EgoDexDataset(BaseDataset):
    """
    Dataset loader for EgoDex egocentric manipulation data.
    
    EgoDex provides:
    - 30 Hz 1080p egocentric videos (MP4)
    - 3D pose annotations for head, upper body, and hands (HDF5)
    - Natural language task descriptions
    - 194 diverse tabletop manipulation tasks
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        width: Optional[int] = 320,
        height: Optional[int] = 240,
        style: str = "demo",
        max_frames: Optional[int] = None,
        task_filter: Optional[List[str]] = None,
        data_percentage: float = 1.0,
        **kwargs
    ):
        """
        Initialize EgoDex dataset.
        
        Args:
            data_dir: Path to EgoDex dataset directory
            split: Dataset split ('train', 'test', 'additional')
            width: Target image width for resizing
            height: Target image height for resizing
            style: Conversation style for MolmoAct
            max_frames: Maximum number of frames per video (for memory efficiency)
            task_filter: List of task names to include (None for all tasks)
            data_percentage: Percentage of data to use (0.0 to 1.0) for ablation studies
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.width = width
        self.height = height
        self.style = style
        self.max_frames = max_frames
        self.task_filter = task_filter
        self.data_percentage = data_percentage
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"EgoDex data directory not found: {data_dir}")
        
        # Validate split
        valid_splits = ['train', 'test', 'additional']
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of: {valid_splits}")
        
        # Load dataset index
        self._load_dataset_index()
        
    def _load_dataset_index(self):
        """Load the dataset index by scanning for task directories and video files."""
        self.episodes = []
        
        # Look for split directory (train, test, additional)
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            # If no split directory, assume all data is in the root
            split_dir = self.data_dir
            
        # Scan for task directories
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            
            # Apply task filter if specified
            if self.task_filter is not None and task_name not in self.task_filter:
                continue
                
            # Find all video files in this task
            for video_file in sorted(task_dir.glob("*.mp4")):
                hdf5_file = video_file.with_suffix('.hdf5')
                if hdf5_file.exists():
                    self.episodes.append({
                        'task_name': task_name,
                        'video_path': str(video_file),
                        'pose_path': str(hdf5_file),
                        'episode_id': video_file.stem
                    })
        
        # Apply data percentage filtering for ablation studies
        if self.data_percentage < 1.0:
            original_count = len(self.episodes)
            target_count = int(original_count * self.data_percentage)
            
            # Use deterministic sampling for reproducibility
            import random
            random.seed(42)  # Fixed seed for reproducibility
            self.episodes = random.sample(self.episodes, target_count)
            
            print(f"Applied data percentage filtering: {original_count} -> {len(self.episodes)} episodes ({self.data_percentage:.1%})")
        
        print(f"Loaded {len(self.episodes)} EgoDex episodes from {self.split} split")
        
    def __len__(self):
        return len(self.episodes)
    
    def _load_pose_data(self, pose_path: str) -> Dict[str, Any]:
        """Load pose data from HDF5 file."""
        with h5py.File(pose_path, 'r') as f:
            pose_data = {}
            
            # Load camera intrinsics
            if 'camera/intrinsic' in f:
                pose_data['camera_intrinsic'] = f['camera/intrinsic'][:]
            
            # Load transforms for all joints
            transforms = {}
            if 'transforms' in f:
                for joint_name in f['transforms'].keys():
                    transforms[joint_name] = f[f'transforms/{joint_name}'][:]
            pose_data['transforms'] = transforms
            
            # Load confidences if available
            confidences = {}
            if 'confidences' in f:
                for joint_name in f['confidences'].keys():
                    confidences[joint_name] = f[f'confidences/{joint_name}'][:]
            pose_data['confidences'] = confidences
            
            # Load language metadata
            metadata = {}
            for key in f.attrs.keys():
                metadata[key] = f.attrs[key]
            pose_data['metadata'] = metadata
            
        return pose_data
    
    def _extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize if specified
            if self.width is not None and self.height is not None:
                pil_image = pil_image.resize((self.width, self.height), Image.BILINEAR)
            
            frames.append(pil_image)
            frame_count += 1
            
            # Limit frames if specified
            if max_frames is not None and frame_count >= max_frames:
                break
        
        cap.release()
        return frames
    
    def _create_conversation(self, task_name: str, metadata: Dict[str, Any], pose_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create conversation format for MolmoAct training with pose-based actions."""
        # Get task description from metadata
        description = metadata.get('llm_description', f"Perform the {task_name} task")
        
        # Handle reversible tasks
        if 'which_llm_description' in metadata:
            which_desc = metadata['which_llm_description']
            if which_desc == 2 and 'llm_description2' in metadata:
                description = metadata['llm_description2']
        
        # Create LIBERO-style conversation format
        question = (
            f"The task is {description}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {description}, "
            "let's think through it step by step. "
            "First, what is the depth map for the first image? "
            "Second, what is the trajectory of the end effector in the first image? "
            "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
            "along with other images from different camera views as additional information, "
            "what is the action that the robot should take?"
        )
        
        # Generate pose-based action description
        answer = self._generate_pose_based_answer(pose_data, description)
        
        return {
            "value": [question, answer]
        }
    
    def _generate_pose_based_answer(self, pose_data: Dict[str, Any], description: str) -> str:
        """Generate LIBERO-style answer with pose-based actions."""
        transforms = pose_data.get('transforms', {})
        confidences = pose_data.get('confidences', {})
        
        # Extract hand poses as actions
        left_hand_poses = transforms.get('leftHand', [])
        right_hand_poses = transforms.get('rightHand', [])
        
        # Generate depth perception reasoning
        depth_reasoning = self._generate_depth_reasoning(transforms)
        
        # Generate trajectory reasoning
        trajectory_reasoning = self._generate_trajectory_reasoning(left_hand_poses, right_hand_poses)
        
        # Generate action matrix
        action_matrix = self._generate_action_matrix(left_hand_poses, right_hand_poses, confidences, transforms)
        
        # Combine all components
        answer = f"""Based on the visual analysis, here's the step-by-step reasoning:

{depth_reasoning}

{trajectory_reasoning}

Action Matrix (8x7 format representing hand poses over time):
{action_matrix}

To perform {description}, the robot should execute these hand pose actions in sequence, following the demonstrated human trajectory."""
        
        return answer
    
    def _generate_depth_reasoning(self, transforms: Dict[str, Any]) -> str:
        """Generate depth perception reasoning from pose data."""
        # Use hand positions to infer depth
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
    
    def _generate_trajectory_reasoning(self, left_poses: List[np.ndarray], right_poses: List[np.ndarray]) -> str:
        """Generate trajectory reasoning from hand pose sequences."""
        if len(left_poses) == 0 and len(right_poses) == 0:
            return "Trajectory analysis: No hand pose data available for trajectory analysis."
        
        # Analyze trajectory of the primary hand (right hand if available, otherwise left)
        primary_poses = right_poses if len(right_poses) > 0 else left_poses
        hand_name = "right" if len(right_poses) > 0 else "left"
        
        if len(primary_poses) < 2:
            return f"Trajectory analysis: {hand_name} hand pose data insufficient for trajectory analysis."
        
        # Extract positions
        positions = [pose[:3, 3] for pose in primary_poses]
        
        # Calculate movement direction and magnitude
        start_pos = positions[0]
        end_pos = positions[-1]
        movement = end_pos - start_pos
        distance = np.linalg.norm(movement)
        
        trajectory_info = f"""Trajectory analysis: {hand_name} hand moves from position {start_pos} to {end_pos}, 
covering a distance of {distance:.3f}m. The movement vector is {movement}, 
indicating a {'grasping' if distance > 0.1 else 'fine manipulation'} action."""
        
        return trajectory_info
    
    def _generate_action_matrix(self, left_poses: List[np.ndarray], right_poses: List[np.ndarray], confidences: Dict[str, List[float]], transforms: Dict[str, Any]) -> str:
        """Generate action matrix from hand poses and finger tip positions with chunking and temporal aggregation."""
        # Use the primary hand (right if available, otherwise left)
        primary_poses = right_poses if len(right_poses) > 0 else left_poses
        hand_name = "right" if len(right_poses) > 0 else "left"
        
        if len(primary_poses) == 0:
            return "No hand pose data available for action generation."
        
        # Sample 8 poses evenly across the sequence (action chunking)
        n_poses = len(primary_poses)
        if n_poses <= 8:
            sampled_poses = primary_poses
            indices = list(range(n_poses))
        else:
            indices = np.linspace(0, n_poses - 1, 8, dtype=int)
            sampled_poses = [primary_poses[i] for i in indices]
        
        # Get finger tip positions for the primary hand
        finger_tips = self._extract_finger_tip_positions(transforms, hand_name, indices)
        
        # Convert poses to action format (6 DOF hand + 5 finger tips * 3 = 21 DOF total)
        action_matrix = []
        for i, pose in enumerate(sampled_poses):
            # Extract hand position (x, y, z)
            hand_position = pose[:3, 3]
            
            # Extract hand rotation (convert rotation matrix to euler angles)
            rotation_matrix = pose[:3, :3]
            rotation = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Combine hand pose: [x, y, z, rx, ry, rz] (no gripper state)
            hand_action = np.concatenate([hand_position, rotation])
            
            # Add finger tip positions: [tip1_x, tip1_y, tip1_z, tip2_x, tip2_y, tip2_z, ...]
            finger_action = finger_tips[i] if i < len(finger_tips) else np.zeros(15)  # 5 fingers * 3 coords
            
            # Combine hand pose + finger tips
            action = np.concatenate([hand_action, finger_action])
            action_matrix.append(action)
        
        # Apply temporal aggregation (like LIBERO)
        aggregated_action = self._temporal_aggregate_actions(action_matrix)
        
        # Apply normalization and discretization if enabled
        if self.normalize_actions:
            normalized_actions = [self._normalize_action(action) for action in action_matrix]
            normalized_aggregated = self._normalize_action(aggregated_action)
        else:
            normalized_actions = action_matrix
            normalized_aggregated = aggregated_action
        
        if self.discretize_actions:
            discretized_actions = [self._discretize_action(action) for action in normalized_actions]
            discretized_aggregated = self._discretize_action(normalized_aggregated)
        else:
            discretized_actions = normalized_actions
            discretized_aggregated = normalized_aggregated
        
        # Format as string
        action_str = "[\n"
        for i, action in enumerate(action_matrix):
            if self.discretize_actions:
                action_str += f"  [{', '.join([str(x) for x in discretized_actions[i]])}],  # Step {i+1} (discretized)\n"
            else:
                action_str += f"  [{', '.join([f'{x:.3f}' for x in action])}],  # Step {i+1} (hand + finger tips)\n"
        action_str += f"]\n\nTemporally aggregated action: "
        
        if self.discretize_actions:
            action_str += f"[{', '.join([str(x) for x in discretized_aggregated])}] (discretized)"
        else:
            action_str += f"[{', '.join([f'{x:.3f}' for x in aggregated_action])}]"
        
        return action_str
    
    def _temporal_aggregate_actions(self, action_matrix: List[np.ndarray]) -> np.ndarray:
        """Apply temporal aggregation to action matrix (like LIBERO)."""
        if not action_matrix:
            return np.zeros(21)
        
        action_matrix = np.array(action_matrix)
        
        # For hand pose (first 6 DOF): average across timesteps
        hand_pose_avg = np.mean(action_matrix[:, :6], axis=0)
        
        # For finger tips (last 15 DOF): average across timesteps
        finger_tips_avg = np.mean(action_matrix[:, 6:], axis=0)
        
        # Combine averaged components
        aggregated_action = np.concatenate([hand_pose_avg, finger_tips_avg])
        
        return aggregated_action
    
    def _extract_finger_tip_positions(self, transforms: Dict[str, Any], hand_name: str, indices: List[int]) -> List[np.ndarray]:
        """Extract finger tip positions for the specified hand."""
        # EgoDex finger tip joint names (based on ARKit hand tracking)
        finger_tip_joints = [
            f"{hand_name}IndexFingerTip",
            f"{hand_name}MiddleFingerTip", 
            f"{hand_name}RingFingerTip",
            f"{hand_name}PinkyFingerTip",
            f"{hand_name}ThumbTip"
        ]
        
        finger_tips = []
        for idx in indices:
            tip_positions = []
            for joint_name in finger_tip_joints:
                if joint_name in transforms and len(transforms[joint_name]) > idx:
                    # Extract position from transform matrix
                    tip_transform = transforms[joint_name][idx]
                    tip_position = tip_transform[:3, 3]  # x, y, z
                    tip_positions.extend(tip_position)
                else:
                    # Fill with zeros if joint not available
                    tip_positions.extend([0.0, 0.0, 0.0])
            
            finger_tips.append(np.array(tip_positions))
        
        return finger_tips

    def _process_pose_for_visualization(self, pose_data: Dict[str, Any]) -> str:
        """Process pose data into a text description for the model."""
        transforms = pose_data.get('transforms', {})
        confidences = pose_data.get('confidences', {})
        
        # Extract key hand poses
        hand_info = []
        for hand in ['leftHand', 'rightHand']:
            if hand in transforms:
                # Get hand position (translation part of transform)
                hand_transform = transforms[hand]
                if len(hand_transform) > 0:
                    hand_pos = hand_transform[0][:3, 3]  # First frame position
                    hand_conf = confidences.get(hand, [1.0])[0] if hand in confidences else 1.0
                    hand_info.append(f"{hand}: position {hand_pos}, confidence {hand_conf:.2f}")
        
        pose_description = "Hand poses: " + "; ".join(hand_info) if hand_info else "No hand pose data available"
        return pose_description
    
    def get(self, item: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """Get a single episode from the dataset."""
        episode = self.episodes[item]
        
        # Load video frames
        frames = self._extract_frames(episode['video_path'], self.max_frames)
        
        # Load pose data
        pose_data = self._load_pose_data(episode['pose_path'])
        
        # Create conversation with pose-based actions
        conversation = self._create_conversation(episode['task_name'], pose_data['metadata'], pose_data)
        
        # Process pose data for model input
        pose_description = self._process_pose_for_visualization(pose_data)
        
        # Use egocentric frame as primary observation (like LIBERO camera view)
        primary_image = frames[0] if frames else Image.new('RGB', (self.width or 320, self.height or 240))
        
        # Create annotation with pose information
        annotation = {
            'task_name': episode['task_name'],
            'episode_id': episode['episode_id'],
            'pose_description': pose_description,
            'num_frames': len(frames),
            'pose_data': pose_data,  # Include full pose data for advanced processing
            'action_type': 'pose_based',  # Indicate this uses pose-based actions
            'hand_poses': {
                'left': pose_data.get('transforms', {}).get('leftHand', []),
                'right': pose_data.get('transforms', {}).get('rightHand', [])
            }
        }
        
        return {
            'style': self.style,
            'image': [primary_image],  # Egocentric observation
            'question': conversation['value'][0],
            'answers': conversation['value'][1],
            'annotation': annotation
        }


class EgoDexDatasetMultiFrame(EgoDexDataset):
    """
    Extended EgoDex dataset that uses multiple frames from each video.
    This is more suitable for trajectory learning tasks.
    """
    
    def __init__(self, frame_interval: int = 5, **kwargs):
        """
        Initialize multi-frame EgoDex dataset.
        
        Args:
            frame_interval: Interval between selected frames (e.g., 5 = every 5th frame)
            **kwargs: Arguments passed to parent EgoDexDataset
        """
        super().__init__(**kwargs)
        self.frame_interval = frame_interval
    
    def get(self, item: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """Get a single episode with multiple frames."""
        episode = self.episodes[item]
        
        # Load video frames
        frames = self._extract_frames(episode['video_path'], self.max_frames)
        
        # Select frames at specified interval
        selected_frames = frames[::self.frame_interval]
        
        # Load pose data
        pose_data = self._load_pose_data(episode['pose_path'])
        
        # Create conversation with pose-based actions
        conversation = self._create_conversation(episode['task_name'], pose_data['metadata'], pose_data)
        
        # Process pose data
        pose_description = self._process_pose_for_visualization(pose_data)
        
        # Create annotation
        annotation = {
            'task_name': episode['task_name'],
            'episode_id': episode['episode_id'],
            'pose_description': pose_description,
            'num_frames': len(selected_frames),
            'frame_interval': self.frame_interval,
            'pose_data': pose_data,
            'action_type': 'pose_based',
            'hand_poses': {
                'left': pose_data.get('transforms', {}).get('leftHand', []),
                'right': pose_data.get('transforms', {}).get('rightHand', [])
            }
        }
        
        return {
            'style': self.style,
            'image': selected_frames,  # Multiple frames
            'question': conversation['value'][0],
            'answers': conversation['value'][1],
            'annotation': annotation
        }


class EgoDexPoseActions(EgoDexDataset):
    """
    EgoDex dataset that treats human hand poses as robot actions.
    This follows the LIBERO pattern where poses are the "actions" and images are "observations".
    Includes action chunking, temporal aggregation, normalization, and discretization.
    """
    
    def __init__(self, action_sequence_length: int = 8, normalize_actions: bool = True, discretize_actions: bool = True, 
                 action_stats_file: str = None, data_percentage: float = 1.0, target_fps: int = 10, **kwargs):
        """
        Initialize EgoDex dataset with pose-based actions.
        
        Args:
            action_sequence_length: Number of pose actions to generate per episode
            normalize_actions: Whether to normalize actions using dataset statistics
            discretize_actions: Whether to discretize actions for language model integration
            action_stats_file: Path to precomputed action statistics file (JSON)
            data_percentage: Percentage of data to use (0.0 to 1.0) for ablation studies
            target_fps: Target FPS for downsampling (default: 10Hz from original 30Hz)
            **kwargs: Arguments passed to parent EgoDexDataset
        """
        # Validate data_percentage
        if not (0.0 <= data_percentage <= 1.0):
            raise ValueError(f"data_percentage must be between 0.0 and 1.0, got {data_percentage}")
        
        super().__init__(**kwargs)
        self.action_sequence_length = action_sequence_length
        self.normalize_actions = normalize_actions
        self.discretize_actions = discretize_actions
        self.target_fps = target_fps
        self.source_fps = 30  # EgoDex is recorded at 30Hz
        
        # Calculate downsampling factor
        self.downsample_factor = max(1, self.source_fps // self.target_fps)
        
        # Initialize action statistics for normalization
        if normalize_actions:
            if action_stats_file and os.path.exists(action_stats_file):
                self.action_stats = self._load_action_statistics(action_stats_file)
            else:
                self.action_stats = self._compute_action_statistics()
        else:
            self.action_stats = None
        
        # Initialize discretization parameters
        self.n_action_bins = 256
        self.bin_centers = self._compute_bin_centers() if discretize_actions else None
    
    def _create_pose_action_conversation(self, task_name: str, metadata: Dict[str, Any], pose_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create conversation format specifically for pose-based action learning."""
        description = metadata.get('llm_description', f"Perform the {task_name} task")
        
        # Handle reversible tasks
        if 'which_llm_description' in metadata:
            which_desc = metadata['which_llm_description']
            if which_desc == 2 and 'llm_description2' in metadata:
                description = metadata['llm_description2']
        
        # EgoDex-specific question format (natural language, clearer and less ambiguous)
        question = (
            f"The task is {description}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {description}, "
            "let's think through it step by step. "
            "First, how far are the objects from the hand? "
            "Second, how does the hand move during the task? "
            "Third, how do the fingers move for precise manipulation? "
            "Based on the distance, hand movement, and finger control, "
            "what actions should the robot take?"
        )
        
        # Generate pose-based action sequence
        answer = self._generate_pose_action_sequence(pose_data, description)
        
        return {
            "value": [question, answer]
        }
    
    def _generate_pose_action_sequence(self, pose_data: Dict[str, Any], description: str) -> str:
        """Generate a sequence of pose-based actions from hand pose data including finger tips."""
        transforms = pose_data.get('transforms', {})
        confidences = pose_data.get('confidences', {})
        
        # Extract hand poses
        left_hand_poses = transforms.get('leftHand', [])
        right_hand_poses = transforms.get('rightHand', [])
        
        # Use primary hand (right if available, otherwise left)
        primary_poses = right_hand_poses if len(right_hand_poses) > 0 else left_hand_poses
        hand_name = "right" if len(right_hand_poses) > 0 else "left"
        
        if len(primary_poses) == 0:
            return f"Error: No hand pose data available for {description}"
        
        # Apply downsampling first (30Hz -> target_fps)
        n_poses = len(primary_poses)
        if self.downsample_factor > 1:
            # Downsample by taking every downsample_factor-th frame
            downsampled_indices = list(range(0, n_poses, self.downsample_factor))
            primary_poses = [primary_poses[i] for i in downsampled_indices]
            n_poses = len(primary_poses)
        
        # Sample poses for action sequence from downsampled data
        if n_poses <= self.action_sequence_length:
            sampled_poses = primary_poses
            indices = list(range(n_poses))
        else:
            indices = np.linspace(0, n_poses - 1, self.action_sequence_length, dtype=int)
            sampled_poses = [primary_poses[i] for i in indices]
        
        # Get finger tip positions for the primary hand
        finger_tips = self._extract_finger_tip_positions(transforms, hand_name, indices)
        
        # Convert poses to action format (6 DOF hand + 5 finger tips * 3 = 21 DOF total)
        actions = []
        for i, pose in enumerate(sampled_poses):
            # Extract hand position (x, y, z)
            hand_position = pose[:3, 3]
            
            # Extract hand rotation (convert rotation matrix to euler angles)
            rotation_matrix = pose[:3, :3]
            rotation = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Combine hand pose: [x, y, z, rx, ry, rz] (no gripper state)
            hand_action = np.concatenate([hand_position, rotation])
            
            # Add finger tip positions: [tip1_x, tip1_y, tip1_z, tip2_x, tip2_y, tip2_z, ...]
            finger_action = finger_tips[i] if i < len(finger_tips) else np.zeros(15)  # 5 fingers * 3 coords
            
            # Combine hand pose + finger tips
            action = np.concatenate([hand_action, finger_action])
            actions.append(action)
        
        # Apply temporal aggregation (like LIBERO)
        aggregated_action = self._temporal_aggregate_actions(actions)
        
        # Apply normalization and discretization if enabled
        if self.normalize_actions:
            normalized_actions = [self._normalize_action(action) for action in actions]
            normalized_aggregated = self._normalize_action(aggregated_action)
        else:
            normalized_actions = actions
            normalized_aggregated = aggregated_action
        
        if self.discretize_actions:
            discretized_actions = [self._discretize_action(action) for action in normalized_actions]
            discretized_aggregated = self._discretize_action(normalized_aggregated)
        else:
            discretized_actions = normalized_actions
            discretized_aggregated = normalized_aggregated
        
        # Format as action sequence
        action_sequence = "[\n"
        for i, action in enumerate(actions):
            if self.discretize_actions:
                action_sequence += f"  [{', '.join([str(x) for x in discretized_actions[i]])}],  # {hand_name} hand + finger tips {i+1} (discretized)\n"
            else:
                action_sequence += f"  [{', '.join([f'{x:.3f}' for x in action])}],  # {hand_name} hand + finger tips {i+1}\n"
        action_sequence += f"]\n\nTemporally aggregated action: "
        
        if self.discretize_actions:
            action_sequence += f"[{', '.join([str(x) for x in discretized_aggregated])}] (discretized)"
        else:
            action_sequence += f"[{', '.join([f'{x:.3f}' for x in aggregated_action])}]"
        
        # Generate reasoning
        start_pos = actions[0][:3]
        end_pos = actions[-1][:3]
        movement = end_pos - start_pos
        distance = np.linalg.norm(movement)
        
        # Analyze finger tip movement
        finger_movement_analysis = self._analyze_finger_tip_movement(finger_tips)
        
        reasoning = f"""Based on the egocentric visual analysis, here's the step-by-step reasoning:

1. How far are the objects from the hand: {hand_name} hand starts at position {start_pos}, indicating objects at approximately {start_pos[2]:.2f}m depth.

2. How does the hand move during the task: {hand_name} hand moves from position {start_pos} to {end_pos}, covering a distance of {distance:.3f}m. The movement vector is {movement}, indicating a {'grasping' if distance > 0.1 else 'fine manipulation'} action.

3. How do the fingers move for precise manipulation: {finger_movement_analysis}

Action Sequence ({self.action_sequence_length} steps, 21 DOF: hand pose + finger tips):
{action_sequence}

To perform {description}, the robot should execute these hand pose and finger tip actions in sequence, following the demonstrated human trajectory."""
        
        return reasoning
    
    def _extract_finger_tip_positions(self, transforms: Dict[str, Any], hand_name: str, indices: List[int]) -> List[np.ndarray]:
        """Extract finger tip positions for the specified hand."""
        # EgoDex finger tip joint names (based on ARKit hand tracking)
        finger_tip_joints = [
            f"{hand_name}IndexFingerTip",
            f"{hand_name}MiddleFingerTip", 
            f"{hand_name}RingFingerTip",
            f"{hand_name}PinkyFingerTip",
            f"{hand_name}ThumbTip"
        ]
        
        finger_tips = []
        for idx in indices:
            tip_positions = []
            for joint_name in finger_tip_joints:
                if joint_name in transforms and len(transforms[joint_name]) > idx:
                    # Extract position from transform matrix
                    tip_transform = transforms[joint_name][idx]
                    tip_position = tip_transform[:3, 3]  # x, y, z
                    tip_positions.extend(tip_position)
                else:
                    # Fill with zeros if joint not available
                    tip_positions.extend([0.0, 0.0, 0.0])
            
            finger_tips.append(np.array(tip_positions))
        
        return finger_tips
    
    def _analyze_finger_tip_movement(self, finger_tips: List[np.ndarray]) -> str:
        """Analyze finger tip movement patterns."""
        if len(finger_tips) == 0 or len(finger_tips) < 2:
            return "Finger tip analysis: Insufficient data for finger movement analysis."
        
        # Analyze movement of each finger tip
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
    
    def _compute_action_statistics(self) -> Dict[str, Any]:
        """Compute action statistics for normalization from actual EgoDex data."""
        print("Computing EgoDex action statistics for normalization...")
        
        # Collect actions from a sample of episodes
        all_actions = []
        # Use all available episodes (already filtered by data_percentage)
        sample_size = len(self)
        
        for i in range(sample_size):
            try:
                episode = self.episodes[i]
                pose_data = self._load_pose_data(episode['pose_path'])
                
                # Extract hand poses and finger tips
                transforms = pose_data.get('transforms', {})
                left_hand_poses = transforms.get('leftHand', [])
                right_hand_poses = transforms.get('rightHand', [])
                
                # Convert to lists if they are numpy arrays
                if isinstance(left_hand_poses, np.ndarray):
                    left_hand_poses = [left_hand_poses[i] for i in range(len(left_hand_poses))]
                if isinstance(right_hand_poses, np.ndarray):
                    right_hand_poses = [right_hand_poses[i] for i in range(len(right_hand_poses))]
                
                # Use primary hand (right if available, otherwise left)
                primary_poses = right_hand_poses if len(right_hand_poses) > 0 else left_hand_poses
                hand_name = "right" if len(right_hand_poses) > 0 else "left"
                
                if len(primary_poses) == 0:
                    continue
                
                # Apply downsampling first (30Hz -> target_fps)
                n_poses = len(primary_poses)
                if self.downsample_factor > 1:
                    # Downsample by taking every downsample_factor-th frame
                    downsampled_indices = list(range(0, n_poses, self.downsample_factor))
                    primary_poses = [primary_poses[i] for i in downsampled_indices]
                    n_poses = len(primary_poses)
                
                # Sample poses from this episode (from downsampled data)
                if n_poses <= 8:
                    sampled_poses = primary_poses
                    indices = list(range(n_poses))
                else:
                    indices = np.linspace(0, n_poses - 1, 8, dtype=int)
                    sampled_poses = [primary_poses[i] for i in indices]
                
                # Get finger tip positions
                finger_tips = self._extract_finger_tip_positions(transforms, hand_name, indices)
                
                # Convert to action format
                for i, pose in enumerate(sampled_poses):
                    # Extract hand position (x, y, z)
                    hand_position = pose[:3, 3]
                    
                    # Extract hand rotation (convert rotation matrix to euler angles)
                    rotation_matrix = pose[:3, :3]
                    rotation = self._rotation_matrix_to_euler(rotation_matrix)
                    
                    # Combine hand pose: [x, y, z, rx, ry, rz]
                    hand_action = np.concatenate([hand_position, rotation])
                    
                    # Add finger tip positions
                    finger_action = finger_tips[i] if i < len(finger_tips) else np.zeros(15)
                    
                    # Combine hand pose + finger tips
                    action = np.concatenate([hand_action, finger_action])
                    all_actions.append(action)
                    
            except Exception as e:
                print(f"Warning: Failed to process episode {i}: {e}")
                continue
        
        if not all_actions:
            print("Warning: No valid actions found, using default statistics")
            return self._get_default_action_statistics()
        
        # Convert to numpy array
        all_actions = np.array(all_actions)
        print(f"Computed statistics from {len(all_actions)} actions")
        
        # Compute statistics
        q01 = np.percentile(all_actions, 1, axis=0)
        q99 = np.percentile(all_actions, 99, axis=0)
        mean = np.mean(all_actions, axis=0)
        std = np.std(all_actions, axis=0)
        
        # Create mask (all dimensions normalized)
        mask = np.ones(21, dtype=bool)
        
        stats = {
            "q01": q01,
            "q99": q99,
            "mean": mean,
            "std": std,
            "mask": mask,
            "n_samples": len(all_actions)
        }
        
        print(f"Action statistics computed:")
        print(f"  - Hand position range: [{q01[0]:.3f}, {q99[0]:.3f}] x [{q01[1]:.3f}, {q99[1]:.3f}] x [{q01[2]:.3f}, {q99[2]:.3f}]")
        print(f"  - Hand rotation range: [{q01[3]:.3f}, {q99[3]:.3f}] x [{q01[4]:.3f}, {q99[4]:.3f}] x [{q01[5]:.3f}, {q99[5]:.3f}]")
        print(f"  - Finger tip range: [{q01[6]:.3f}, {q99[6]:.3f}] to [{q01[20]:.3f}, {q99[20]:.3f}]")
        
        return stats
    
    def _load_action_statistics(self, stats_file: str) -> Dict[str, Any]:
        """Load precomputed action statistics from JSON file."""
        import json
        
        print(f"Loading precomputed action statistics from: {stats_file}")
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Convert lists back to numpy arrays
        for key in ['q01', 'q99', 'mean', 'std']:
            stats[key] = np.array(stats[key])
        stats['mask'] = np.array(stats['mask'], dtype=bool)
        
        print(f"Loaded statistics from {stats['n_samples']} samples")
        print(f"  - Hand position range: [{stats['q01'][0]:.3f}, {stats['q99'][0]:.3f}] x [{stats['q01'][1]:.3f}, {stats['q99'][1]:.3f}] x [{stats['q01'][2]:.3f}, {stats['q99'][2]:.3f}]")
        print(f"  - Hand rotation range: [{stats['q01'][3]:.3f}, {stats['q99'][3]:.3f}] x [{stats['q01'][4]:.3f}, {stats['q99'][4]:.3f}] x [{stats['q01'][5]:.3f}, {stats['q99'][5]:.3f}]")
        print(f"  - Finger tip range: [{stats['q01'][6]:.3f}, {stats['q99'][6]:.3f}] to [{stats['q01'][20]:.3f}, {stats['q99'][20]:.3f}]")
        
        return stats
    
    def _get_default_action_statistics(self) -> Dict[str, Any]:
        """Get default action statistics when computation fails."""
        return {
            "q01": np.array([-0.5, -0.5, 0.0, -3.14, -3.14, -3.14] + [-0.1] * 15),
            "q99": np.array([0.5, 0.5, 1.0, 3.14, 3.14, 3.14] + [0.1] * 15),
            "mean": np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0] * 15),
            "std": np.array([0.2, 0.2, 0.2, 1.0, 1.0, 1.0] + [0.05] * 15),
            "mask": np.ones(21, dtype=bool),
            "n_samples": 0
        }
    
    def _compute_bin_centers(self) -> np.ndarray:
        """Compute bin centers for action discretization."""
        # Create 256 bins from -1 to 1
        return np.linspace(-1.0, 1.0, self.n_action_bins)
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action using dataset statistics."""
        if not self.normalize_actions or self.action_stats is None:
            return action
        
        q01 = self.action_stats["q01"]
        q99 = self.action_stats["q99"]
        mask = self.action_stats["mask"]
        
        # Normalize to [-1, 1] range
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        range_diff = q99 - q01
        range_diff = np.where(np.abs(range_diff) < epsilon, epsilon, range_diff)
        normalized = 2.0 * (action - q01) / range_diff - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # Apply mask (some dimensions might not be normalized)
        result = np.where(mask, normalized, action)
        
        return result
    
    def _discretize_action(self, action: np.ndarray) -> List[int]:
        """Discretize normalized action to bin indices."""
        if not self.discretize_actions or self.bin_centers is None:
            return action.tolist()
        
        # Find closest bin for each dimension
        bin_indices = []
        for i, value in enumerate(action):
            bin_idx = np.argmin(np.abs(self.bin_centers - value))
            bin_indices.append(bin_idx)
        
        return bin_indices
    
    def _undiscretize_action(self, bin_indices: List[int]) -> np.ndarray:
        """Convert bin indices back to continuous values."""
        if not self.discretize_actions or self.bin_centers is None:
            return np.array(bin_indices)
        
        return np.array([self.bin_centers[idx] for idx in bin_indices])
    
    def _denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Denormalize action back to original scale."""
        if not self.normalize_actions or self.action_stats is None:
            return normalized_action
        
        q01 = self.action_stats["q01"]
        q99 = self.action_stats["q99"]
        mask = self.action_stats["mask"]
        
        # Denormalize from [-1, 1] range
        denormalized = 0.5 * (normalized_action + 1.0) * (q99 - q01) + q01
        
        # Apply mask
        result = np.where(mask, denormalized, normalized_action)
        
        return result
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        # Simplified conversion - in practice, you'd use proper Euler angle conversion
        # This is a placeholder that extracts meaningful rotation information
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def get(self, item: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """Get a single episode with pose-based actions."""
        episode = self.episodes[item]
        
        # Load video frames
        frames = self._extract_frames(episode['video_path'], self.max_frames)
        
        # Load pose data
        pose_data = self._load_pose_data(episode['pose_path'])
        
        # Create pose-based action conversation
        conversation = self._create_pose_action_conversation(episode['task_name'], pose_data['metadata'], pose_data)
        
        # Use egocentric frame as observation
        primary_image = frames[0] if frames else Image.new('RGB', (self.width or 320, self.height or 240))
        
        # Create annotation
        annotation = {
            'task_name': episode['task_name'],
            'episode_id': episode['episode_id'],
            'action_type': 'pose_based_with_fingertips_enhanced',
            'action_sequence_length': self.action_sequence_length,
            'action_dof': 21,  # 6 DOF hand + 15 DOF finger tips
            'action_processing': {
                'chunking': True,
                'temporal_aggregation': True,
                'normalization': self.normalize_actions,
                'discretization': self.discretize_actions,
                'n_action_bins': self.n_action_bins if self.discretize_actions else None
            },
            'hand_poses': {
                'left': pose_data.get('transforms', {}).get('leftHand', []),
                'right': pose_data.get('transforms', {}).get('rightHand', [])
            },
            'finger_tips': {
                'left': self._extract_finger_tip_positions(pose_data.get('transforms', {}), 'left', list(range(len(pose_data.get('transforms', {}).get('leftHand', []))))),
                'right': self._extract_finger_tip_positions(pose_data.get('transforms', {}), 'right', list(range(len(pose_data.get('transforms', {}).get('rightHand', [])))))
            },
            'action_stats': self.action_stats,
            'pose_data': pose_data
        }
        
        return {
            'style': self.style,
            'image': [primary_image],  # Egocentric observation
            'question': conversation['value'][0],
            'answers': conversation['value'][1],
            'annotation': annotation
        }


# Convenience classes for different EgoDex configurations
class EgoDexTrain(EgoDexDataset):
    """EgoDex training split dataset."""
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, split="train", **kwargs)


class EgoDexTest(EgoDexDataset):
    """EgoDex test split dataset."""
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, split="test", **kwargs)


class EgoDexAdditional(EgoDexDataset):
    """EgoDex additional data split dataset."""
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, split="additional", **kwargs)


# Pose-based action classes (treating hand poses as robot actions)
class EgoDexPoseActionsTrain(EgoDexPoseActions):
    """EgoDex training split with pose-based actions."""
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, split="train", **kwargs)


class EgoDexPoseActionsTest(EgoDexPoseActions):
    """EgoDex test split with pose-based actions."""
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, split="test", **kwargs)


class EgoDexPoseActionsAdditional(EgoDexPoseActions):
    """EgoDex additional data split with pose-based actions."""
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, split="additional", **kwargs)
