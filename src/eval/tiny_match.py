"""Tiny object matching and analysis."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ultralytics.models import RTDETR
import torch
from PIL import Image


def load_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format label file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
    """
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (x_center, y_center, width, height) format (normalized).
    
    Args:
        box1: First box (x_center, y_center, width, height)
        box2: Second box (x_center, y_center, width, height)
        
    Returns:
        IoU value
    """
    # Convert to (x1, y1, x2, y2)
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def is_tiny_box(box: Tuple[float, float, float, float], img_width: int, img_height: int, config: Dict) -> bool:
    """Check if a box is considered tiny (small but detectable).
    
    Args:
        box: Box in (x_center, y_center, width, height) normalized format
        img_width: Image width in pixels
        img_height: Image height in pixels
        config: Configuration dictionary
        
    Returns:
        True if box is tiny (within size range for detection)
    """
    tiny_config = config['tiny_objects']
    width_px = box[2] * img_width
    height_px = box[3] * img_height
    area_px = width_px * height_px
    
    # Check if box is within detectable size range
    # Must be: area >= area_threshold AND (width >= width_threshold OR height >= height_threshold)
    # This ensures objects are large enough to be detected by COCO pretrained YOLO
    area_ok = area_px >= tiny_config['area_threshold']
    size_ok = width_px >= tiny_config['width_threshold'] or height_px >= tiny_config['height_threshold']
    
    # Also check upper bound to keep them "small" (not too large)
    # Use a reasonable upper bound (e.g., 100x100 = 10000 px²)
    max_area = tiny_config.get('max_area_threshold', 10000)
    area_not_too_large = area_px <= max_area
    
    return area_ok and size_ok and area_not_too_large


def sample_tiny_objects(config: Dict, split: str = 'val') -> List[Dict]:
    """Sample tiny GT boxes from dataset.
    
    Args:
        config: Configuration dictionary
        split: Dataset split to sample from
        
    Returns:
        List of sampled GT boxes with metadata
    """
    import random
    from src.utils.seed import set_seed
    
    seed = config['seed']
    set_seed(seed)
    
    yolo_root = Path(config['dataset']['visdrone_yolo_root'])
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split
    
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")
    
    # Collect all tiny boxes
    all_tiny_boxes = []
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    for image_path in image_files:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            continue
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Load GT boxes
        gt_boxes = load_yolo_label(label_path)
        
        for class_id, x_center, y_center, width, height in gt_boxes:
            box = (x_center, y_center, width, height)
            if is_tiny_box(box, img_width, img_height, config):
                all_tiny_boxes.append({
                    'image_id': image_path.stem,
                    'image_path': str(image_path),
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'img_width': img_width,
                    'img_height': img_height
                })
    
    # Sample
    sample_size = config['tiny_objects']['sample_size']
    if len(all_tiny_boxes) > sample_size:
        sampled = random.sample(all_tiny_boxes, sample_size)
    else:
        sampled = all_tiny_boxes
    
    return sampled


def match_prediction_to_gt(
    pred_boxes: List[Tuple[int, float, float, float, float, float]],  # (class, x, y, w, h, score)
    gt_box: Tuple[int, float, float, float, float],  # (class, x, y, w, h)
    iou_threshold: float = 0.5,
    same_class: bool = True
) -> Optional[Tuple[float, float]]:
    """Find best matching prediction for a GT box.
    
    Args:
        pred_boxes: List of predicted boxes
        gt_box: Ground truth box
        iou_threshold: Minimum IoU threshold for match
        same_class: Whether to require same class
        
    Returns:
        (score, iou) if match found, None otherwise
    """
    gt_class, gt_x, gt_y, gt_w, gt_h = gt_box
    gt_box_coords = (gt_x, gt_y, gt_w, gt_h)
    
    best_iou = 0.0
    best_score = 0.0
    found_match = False
    
    for pred_class, pred_x, pred_y, pred_w, pred_h, pred_score in pred_boxes:
        if same_class and pred_class != gt_class:
            continue
        
        pred_box_coords = (pred_x, pred_y, pred_w, pred_h)
        iou = compute_iou(gt_box_coords, pred_box_coords)
        
        if iou > best_iou:
            best_iou = iou
            best_score = pred_score
            found_match = True
    
    if found_match and best_iou >= iou_threshold:
        return (best_score, best_iou)
    
    return None


def run_inference_on_image(
    model_path: str,
    model_type: str,
    image_path: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Tuple[int, float, float, float, float, float]]:
    """Run inference on a single image.
    
    Args:
        model_path: Path to model
        model_type: 'yolo' or 'rtdetr'
        image_path: Path to image
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        device: Device to run on
        
    Returns:
        List of (class_id, x_center, y_center, width, height, score) in normalized coordinates
    """
    # Load model
    if model_type == 'yolo':
        model = YOLO(model_path)
    elif model_type == 'rtdetr':
        if RTDETR is None:
            print("Warning: RT-DETR not available, using YOLO instead")
            model = YOLO(model_path)
        else:
            model = RTDETR(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run inference
    results = model(str(image_path), conf=conf_thres, iou=iou_thres, device=device, verbose=False)
    
    # Extract boxes
    boxes = []
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                # Get normalized coordinates
                cls = int(box.cls[0])
                x_center = float(box.xywhn[0][0])
                y_center = float(box.xywhn[0][1])
                width = float(box.xywhn[0][2])
                height = float(box.xywhn[0][3])
                score = float(box.conf[0])
                boxes.append((cls, x_center, y_center, width, height, score))
    
    return boxes
