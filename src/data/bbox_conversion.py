"""Bbox format conversion utilities."""

from typing import Tuple, Dict, Optional
from PIL import Image
from pathlib import Path
import numpy as np


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
    # Use context manager to ensure image is closed immediately
    with Image.open(image_path) as img:
        return img.size


def map_bbox_to_cam(
    bbox_xyxy: Tuple[float, float, float, float],
    img_shape: Tuple[int, int],
    cam_shape: Tuple[int, int],
    letterbox_meta: Optional[Dict] = None
) -> Tuple[float, float, float, float]:
    """Map bbox from original image coordinates to CAM coordinates.
    
    Args:
        bbox_xyxy: (x1, y1, x2, y2) in original image pixels
        img_shape: (img_width, img_height) original image dimensions
        cam_shape: (cam_height, cam_width) CAM map dimensions
        letterbox_meta: Optional dict with letterbox transformation info:
            {
                'scale': float,  # Resize scale factor
                'pad_top': int,  # Top padding in letterbox
                'pad_left': int,  # Left padding in letterbox
                'target_size': int  # Target size (e.g., 640)
            }
            If None, assumes CAM is same size as image (no letterbox)
        
    Returns:
        (x1, y1, x2, y2) in CAM coordinates (pixels)
    """
    img_width, img_height = img_shape
    cam_h, cam_w = cam_shape
    
    x1, y1, x2, y2 = bbox_xyxy
    
    if letterbox_meta is not None:
        # Letterbox transformation: scale + pad
        scale = letterbox_meta['scale']
        pad_left = letterbox_meta['pad_left']
        pad_top = letterbox_meta['pad_top']
        target_size = letterbox_meta['target_size']
        
        # Step 1: Scale bbox coordinates
        x1_scaled = x1 * scale
        y1_scaled = y1 * scale
        x2_scaled = x2 * scale
        y2_scaled = y2 * scale
        
        # Step 2: Add padding offset
        x1_cam = x1_scaled + pad_left
        y1_cam = y1_scaled + pad_top
        x2_cam = x2_scaled + pad_left
        y2_cam = y2_scaled + pad_top
        
        # Step 3: Scale to CAM dimensions (if CAM is not target_size)
        if cam_w != target_size or cam_h != target_size:
            scale_to_cam_w = cam_w / target_size
            scale_to_cam_h = cam_h / target_size
            x1_cam = x1_cam * scale_to_cam_w
            x2_cam = x2_cam * scale_to_cam_w
            y1_cam = y1_cam * scale_to_cam_h
            y2_cam = y2_cam * scale_to_cam_h
    else:
        # No letterbox: direct scaling
        scale_w = cam_w / img_width
        scale_h = cam_h / img_height
        x1_cam = x1 * scale_w
        y1_cam = y1 * scale_h
        x2_cam = x2 * scale_w
        y2_cam = y2 * scale_h
    
    # Clip to CAM bounds
    x1_cam = max(0, min(x1_cam, cam_w - 1))
    y1_cam = max(0, min(y1_cam, cam_h - 1))
    x2_cam = max(0, min(x2_cam, cam_w - 1))
    y2_cam = max(0, min(y2_cam, cam_h - 1))
    
    return (x1_cam, y1_cam, x2_cam, y2_cam)


def extract_letterbox_meta(
    img_shape: Tuple[int, int],
    target_size: int = 640
) -> Dict:
    """Extract letterbox transformation metadata.
    
    Args:
        img_shape: (img_width, img_height) original image dimensions
        target_size: Target size for letterbox (default: 640)
        
    Returns:
        Dict with letterbox metadata:
        {
            'scale': float,
            'pad_top': int,
            'pad_left': int,
            'target_size': int
        }
    """
    img_width, img_height = img_shape
    
    # Calculate scale to fit target_size while maintaining aspect ratio
    scale = min(target_size / img_width, target_size / img_height)
    new_w = int(img_width * scale)
    new_h = int(img_height * scale)
    
    # Letterbox padding (center alignment)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    
    return {
        'scale': scale,
        'pad_left': pad_left,
        'pad_top': pad_top,
        'target_size': target_size
    }
