"""Bbox format conversion utilities."""

from typing import Tuple
from PIL import Image
from pathlib import Path


def visdrone_to_yolo_bbox(
    visdrone_bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """Convert VisDrone bbox format to YOLO format.
    
    VisDrone: (left, top, width, height) in pixels
    YOLO: (x_center, y_center, width, height) normalized [0, 1]
    
    Args:
        visdrone_bbox: (left, top, width, height) in pixels
        img_width: Image width
        img_height: Image height
        
    Returns:
        (x_center, y_center, width, height) normalized
    """
    left, top, width, height = visdrone_bbox
    
    # Convert to center coordinates
    x_center = left + width / 2
    y_center = top + height / 2
    
    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions.
    
    Args:
        image_path: Path to image file
        
    Returns:
        (width, height)
    """
    img = Image.open(image_path)
    return img.size
