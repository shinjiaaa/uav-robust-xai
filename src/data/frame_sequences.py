"""Frame sequence handling for VisDrone video sequences."""

from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import defaultdict


def parse_visdrone_filename(filename: str) -> Dict[str, str]:
    """Parse VisDrone filename to extract sequence information.
    
    VisDrone filename format: {sequence_id}_{frame_number}_d_{object_id}.jpg
    Example: 0000001_02999_d_0000005.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        Dictionary with sequence_id, frame_number, object_id
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 3:
        sequence_id = parts[0]
        frame_number = parts[1]
        object_id = parts[-1] if len(parts) > 3 else None
        return {
            'sequence_id': sequence_id,
            'frame_number': int(frame_number) if frame_number.isdigit() else None,
            'object_id': object_id
        }
    return {'sequence_id': None, 'frame_number': None, 'object_id': None}


def group_images_by_sequence(image_files: List[Path]) -> Dict[str, List[Path]]:
    """Group images by sequence ID.
    
    Args:
        image_files: List of image file paths
        
    Returns:
        Dictionary mapping sequence_id to sorted list of frame paths
    """
    sequences = defaultdict(list)
    
    for img_file in image_files:
        info = parse_visdrone_filename(img_file.name)
        seq_id = info['sequence_id']
        if seq_id:
            sequences[seq_id].append(img_file)
    
    # Sort frames by frame number
    for seq_id in sequences:
        sequences[seq_id].sort(key=lambda p: parse_visdrone_filename(p.name).get('frame_number', 0) or 0)
    
    return dict(sequences)


def extract_continuous_clips(
    sequences: Dict[str, List[Path]],
    min_clip_length: int = 5,
    max_clip_length: int = 30
) -> List[Dict]:
    """Extract continuous frame clips from sequences.
    
    Args:
        sequences: Dictionary mapping sequence_id to frame paths
        min_clip_length: Minimum clip length
        max_clip_length: Maximum clip length
        
    Returns:
        List of clip dictionaries with sequence_id, start_frame, end_frame, frames
    """
    clips = []
    
    for seq_id, frames in sequences.items():
        if len(frames) < min_clip_length:
            continue
        
        # Extract clips of desired length
        for i in range(0, len(frames) - min_clip_length + 1, min_clip_length):
            clip_frames = frames[i:i + max_clip_length]
            if len(clip_frames) >= min_clip_length:
                clips.append({
                    'sequence_id': seq_id,
                    'clip_id': f"{seq_id}_clip_{i}",
                    'start_idx': i,
                    'end_idx': i + len(clip_frames) - 1,
                    'frames': clip_frames,
                    'length': len(clip_frames)
                })
    
    return clips


def find_tiny_object_clips(
    clips: List[Dict],
    annotations_dir: Path,
    tiny_config: Dict,
    sample_size: int = 500,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Find clips containing tiny objects.
    
    Args:
        clips: List of clip dictionaries
        annotations_dir: Directory containing annotation files
        tiny_config: Tiny object configuration (area_threshold, width_threshold, height_threshold)
        sample_size: Number of tiny objects to sample
        seed: Random seed
        
    Returns:
        List of clips with tiny object information
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    tiny_clips = []
    tiny_objects_found = []
    
    for clip in clips:
        clip_tiny_objects = []
        
        for frame_path in clip['frames']:
            # frame_path is Path object from clip['frames']
            frame_stem = frame_path.stem if isinstance(frame_path, Path) else Path(frame_path).stem
            ann_file = annotations_dir / f"{frame_stem}.txt"
            if not ann_file.exists():
                continue
            
            # Load image to get dimensions for bbox conversion
            try:
                from PIL import Image
                img = Image.open(frame_path)
                img_width, img_height = img.size
            except:
                continue
            
            # Load annotations
            with open(ann_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 8:
                        continue
                    
                    try:
                        bbox_left = float(parts[0])
                        bbox_top = float(parts[1])
                        bbox_width = float(parts[2])
                        bbox_height = float(parts[3])
                        object_category = int(parts[5])
                        
                        if object_category == 0:  # Skip ignored
                            continue
                        
                        area = bbox_width * bbox_height
                        
                        # Check if tiny object
                        is_tiny = (
                            area <= tiny_config['area_threshold'] and
                            bbox_width <= tiny_config['width_threshold'] and
                            bbox_height <= tiny_config['height_threshold']
                        )
                        
                        if is_tiny:
                            # frame_path is Path object from clip['frames']
                            frame_idx = clip['frames'].index(frame_path) if frame_path in clip['frames'] else 0
                            # Store as string (will be converted to relative path in script)
                            frame_path_str = str(frame_path)
                            clip_tiny_objects.append({
                                'frame_path': frame_path_str,
                                'frame_idx': frame_idx,
                                'bbox': (bbox_left, bbox_top, bbox_width, bbox_height),
                                'class_id': object_category,
                                'area': area,
                                'img_width': img_width,
                                'img_height': img_height
                            })
                            tiny_objects_found.append({
                                'clip_id': clip['clip_id'],
                                'frame_path': frame_path_str,
                                'bbox': (bbox_left, bbox_top, bbox_width, bbox_height),
                                'class_id': object_category,
                                'img_width': img_width,
                                'img_height': img_height
                            })
                    except (ValueError, IndexError):
                        continue
        
        if clip_tiny_objects:
            clip['tiny_objects'] = clip_tiny_objects
            tiny_clips.append(clip)
    
    # Sample tiny objects if needed
    if len(tiny_objects_found) > sample_size:
        tiny_objects_found = random.sample(tiny_objects_found, sample_size)
    
    return tiny_clips, tiny_objects_found
