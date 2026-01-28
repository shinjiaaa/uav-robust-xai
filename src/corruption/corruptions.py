"""Image corruption functions with severity control."""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image
import hashlib
import shutil
import yaml
from tqdm import tqdm


def apply_fog(image: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Apply fog corruption to image.
    
    Args:
        image: Input image (H, W, C) in [0, 255] range
        severity: Severity level 0-4 (or extreme values like 50)
        seed: Random seed for reproducibility
        
    Returns:
        Corrupted image
    """
    if severity == 0:
        return image.copy()
    
    # Fog parameters - INCREASED STRENGTH
    alpha_values = [0.0, 0.25, 0.50, 0.75, 0.90]  # Increased from [0.0, 0.15, 0.30, 0.45, 0.60]
    
    # Handle extreme severity (e.g., 50)
    if severity >= len(alpha_values):
        # Extreme severity: use maximum alpha (0.95)
        alpha = 0.95
    else:
        alpha = alpha_values[severity]
    
    # Create fog veil (white overlay)
    fog_veil = np.ones_like(image, dtype=np.float32) * 255.0
    
    # Apply slight blur to fog veil for realism
    fog_veil = cv2.GaussianBlur(fog_veil, (15, 15), 0)
    
    # Blend with original
    image_float = image.astype(np.float32)
    corrupted = (1 - alpha) * image_float + alpha * fog_veil
    
    # Reduce contrast slightly
    corrupted = corrupted * (1 - alpha * 0.1)
    
    return np.clip(corrupted, 0, 255).astype(np.uint8)


def apply_lowlight(image: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Apply low-light corruption to image.
    
    Args:
        image: Input image (H, W, C) in [0, 255] range
        severity: Severity level 0-4 (or extreme values like 50)
        seed: Random seed for reproducibility
        
    Returns:
        Corrupted image
    """
    if severity == 0:
        return image.copy()
    
    # Low-light parameters - INCREASED STRENGTH
    gamma_values = [1.0, 1.5, 2.0, 2.5, 3.0]  # Increased from [1.0, 1.2, 1.4, 1.6, 1.8]
    brightness_scale_values = [1.0, 0.75, 0.50, 0.35, 0.20]  # Decreased from [1.0, 0.90, 0.80, 0.70, 0.60]
    
    # Handle extreme severity (e.g., 50)
    if severity >= len(gamma_values):
        # Extreme severity: gamma = 5.0, brightness_scale = 0.1
        gamma = 5.0
        brightness_scale = 0.1
    else:
        gamma = gamma_values[severity]
        brightness_scale = brightness_scale_values[severity]
    
    # Normalize to [0, 1]
    image_norm = image.astype(np.float32) / 255.0
    
    # Apply gamma correction (darker)
    image_gamma = np.power(image_norm, gamma)
    
    # Apply brightness scaling
    image_dark = image_gamma * brightness_scale
    
    # Add slight noise
    np.random.seed(seed)
    noise = np.random.normal(0, 0.01 * severity, image_dark.shape).astype(np.float32)
    image_dark = image_dark + noise
    
    # Clip and convert back
    return np.clip(image_dark * 255.0, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, severity: int, seed: int = 42, image_id: str = "") -> np.ndarray:
    """Apply motion blur corruption to image.
    
    Args:
        image: Input image (H, W, C) in [0, 255] range
        severity: Severity level 0-4 (or extreme values like 50)
        seed: Random seed for reproducibility
        image_id: Image identifier for deterministic angle selection
        
    Returns:
        Corrupted image
    """
    if severity == 0:
        return image.copy()
    
    # Motion blur parameters - INCREASED STRENGTH
    kernel_lengths = [0, 5, 10, 15, 20]  # Increased from [0, 3, 6, 9, 12]
    
    # Handle extreme severity (e.g., 50)
    if severity >= len(kernel_lengths):
        # Extreme severity: kernel_length = 100
        kernel_length = 100
    else:
        kernel_length = kernel_lengths[severity]
    
    if kernel_length == 0:
        return image.copy()
    
    # Determine angle deterministically based on image_id and seed
    if image_id:
        # Hash image_id + seed to get deterministic angle
        hash_val = int(hashlib.md5(f"{image_id}_{seed}".encode()).hexdigest(), 16)
        angle = (hash_val % 360)  # 0-359 degrees
    else:
        angle = 0  # Default horizontal
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
    kernel[int((kernel_length - 1) / 2), :] = np.ones(kernel_length, dtype=np.float32)
    kernel = kernel / kernel_length
    
    # Rotate kernel
    M = cv2.getRotationMatrix2D((kernel_length / 2, kernel_length / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_length, kernel_length))
    
    # Apply blur to each channel
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
    else:
        blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred


def corrupt_image(
    image_path: Path,
    corruption_type: str,
    severity: int,
    output_path: Path,
    seed: int = 42
):
    """Apply corruption to an image and save.
    
    Args:
        image_path: Path to input image
        corruption_type: Type of corruption ('fog', 'lowlight', 'motion_blur')
        severity: Severity level 0-4
        output_path: Path to save corrupted image
        seed: Random seed
    """
    # Load image
    image = np.array(Image.open(image_path))
    
    # Get image ID for deterministic motion blur
    image_id = image_path.stem
    
    # Apply corruption
    if corruption_type == 'fog':
        corrupted = apply_fog(image, severity, seed)
    elif corruption_type == 'lowlight':
        corrupted = apply_lowlight(image, severity, seed)
    elif corruption_type == 'motion_blur':
        corrupted = apply_motion_blur(image, severity, seed, image_id)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrupted).save(output_path)


def generate_corruptions(
    config: Dict,
    splits: List[str],
    corruption_types: List[str] = None,
    severities: List[int] = None,
    force: bool = False
):
    """Generate corrupted images for all specified splits, corruptions, and severities.
    
    Args:
        config: Configuration dictionary
        splits: List of splits to process (e.g., ['val', 'test'])
        corruption_types: List of corruption types (default: from config)
        severities: List of severities (default: from config)
        force: Force regeneration even if exists
    """
    from src.utils.seed import set_seed
    
    if corruption_types is None:
        corruption_types = config['corruptions']['types']
    if severities is None:
        severities = config['corruptions']['severities']
    
    seed = config['seed']
    set_seed(seed)
    
    yolo_root = Path(config['dataset']['visdrone_yolo_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    for split in splits:
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist. Skipping {split}.")
            continue
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for corruption_type in corruption_types:
            for severity in severities:
                # Output directories
                corrupt_images_dir = corruptions_root / corruption_type / str(severity) / split / "images"
                corrupt_labels_dir = corruptions_root / corruption_type / str(severity) / split / "labels"
                
                corrupt_images_dir.mkdir(parents=True, exist_ok=True)
                corrupt_labels_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if already exists
                if not force and len(list(corrupt_images_dir.glob("*.jpg"))) > 0:
                    print(f"Skipping {corruption_type} severity {severity} {split} (already exists)")
                    continue
                
                print(f"Generating {corruption_type} severity {severity} for {split}...")
                
                for image_path in tqdm(image_files, desc=f"{corruption_type}-{severity}"):
                    # Copy label (unchanged)
                    label_path = labels_dir / (image_path.stem + ".txt")
                    if label_path.exists():
                        corrupt_label_path = corrupt_labels_dir / (image_path.stem + ".txt")
                        shutil.copy2(label_path, corrupt_label_path)
                    
                    # Generate corrupted image
                    corrupt_image_path = corrupt_images_dir / image_path.name
                    corrupt_image(image_path, corruption_type, severity, corrupt_image_path, seed)
                
                # Create YOLO dataset YAML for this corruption/severity
                yaml_path = corruptions_root / corruption_type / str(severity) / split / "data.yaml"
                # Ultralytics requires both 'train' and 'val' keys even if one is empty
                # Use the same split for both if only one split exists
                yolo_config = {
                    'path': str((corruptions_root / corruption_type / str(severity) / split).absolute()),
                    'train': f'images/{split}',  # Use same split for train if only val/test exists
                    'val': f'images/{split}',
                    'nc': 11,
                    'names': [
                        'pedestrian', 'people', 'bicycle', 'car', 'van',
                        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
                    ]
                }
                
                import yaml
                with open(yaml_path, 'w') as f:
                    yaml.dump(yolo_config, f, default_flow_style=False)
        
        # Also create "severity 0" which is just original images (symlink or copy)
        for corruption_type in corruption_types:
            severity_0_images = corruptions_root / corruption_type / "0" / split / "images"
            severity_0_labels = corruptions_root / corruption_type / "0" / split / "labels"
            
            if not severity_0_images.exists() or force:
                print(f"Creating severity 0 (original) for {corruption_type} {split}...")
                shutil.copytree(images_dir, severity_0_images, dirs_exist_ok=True)
                shutil.copytree(labels_dir, severity_0_labels, dirs_exist_ok=True)
                
                # Create YAML
                yaml_path = corruptions_root / corruption_type / "0" / split / "data.yaml"
                yolo_config = {
                    'path': str((corruptions_root / corruption_type / "0" / split).absolute()),
                    'train': f'images/{split}',  # Use same split for train
                    'val': f'images/{split}',
                    'nc': 11,
                    'names': [
                        'pedestrian', 'people', 'bicycle', 'car', 'van',
                        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
                    ]
                }
                
                import yaml
                with open(yaml_path, 'w') as f:
                    yaml.dump(yolo_config, f, default_flow_style=False)
