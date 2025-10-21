# EgoDex First Frame Extraction Script

This script extracts the first frame from each MP4 video in the EgoDex training data for visual inspection.

## Usage

### Basic Usage (All Tasks)
```bash
python scripts/extract_first_frames.py
```

### Process Specific Task
```bash
python scripts/extract_first_frames.py --task-filter "fry_egg"
```

### Process Multiple Tasks Matching Pattern
```bash
python scripts/extract_first_frames.py --task-filter "fry"
```

### Process Limited Number of Tasks (for testing)
```bash
python scripts/extract_first_frames.py --max-tasks 5
```

### Custom Data Directory
```bash
python scripts/extract_first_frames.py --data-dir /path/to/egodex/train
```

## Output

- First frames are saved as `{video_name}_first_frame.png` in the same directory as the MP4 files
- The script skips videos that already have extracted frames
- Progress bars show the extraction progress

## Examples

### Quick Test (2 tasks only)
```bash
python scripts/extract_first_frames.py --max-tasks 2
```

### Process All Cooking Tasks
```bash
python scripts/extract_first_frames.py --task-filter "cook"
```

### Process All Tasks (Full Dataset)
```bash
python scripts/extract_first_frames.py
```

## Notes

- The script uses OpenCV to extract frames
- PNG format is used for lossless image quality
- Existing first frame images are skipped to avoid re-processing
- The script handles errors gracefully and continues processing

