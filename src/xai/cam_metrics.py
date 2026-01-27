"""CAM distribution metrics for failure event analysis."""

import numpy as np
from typing import Tuple, Dict
from scipy import ndimage
from scipy.stats import entropy


def compute_energy_in_bbox(
    cam: np.ndarray,
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> float:
    """Compute energy (activation) inside GT bbox.
    
    Args:
        cam: CAM heatmap (H, W)
        bbox: (x_center, y_center, width, height) in normalized coordinates
        img_width: Image width
        img_height: Image height
        
    Returns:
        Ratio of energy inside bbox to total energy
    """
    # Convert normalized bbox to pixel coordinates
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    # Clip to image bounds
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    total_energy = np.sum(cam)
    if total_energy == 0:
        return 0.0
    
    bbox_energy = np.sum(cam[y1:y2+1, x1:x2+1])
    return bbox_energy / total_energy


def compute_activation_spread(
    cam: np.ndarray,
    threshold: float = 0.1
) -> float:
    """Compute spread of activations (spatial dispersion).
    
    Args:
        cam: CAM heatmap (H, W)
        threshold: Threshold for considering activation (relative to max)
        
    Returns:
        Spread metric (normalized)
    """
    if cam.max() == 0:
        return 0.0
    
    # Normalize
    cam_norm = cam / cam.max()
    
    # Find activated regions
    activated = cam_norm >= threshold
    
    if not np.any(activated):
        return 0.0
    
    # Compute center of mass
    y_coords, x_coords = np.where(activated)
    if len(y_coords) == 0:
        return 0.0
    
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    # Compute spread (distance from center)
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    spread = np.mean(distances)
    
    # Normalize by image diagonal
    max_distance = np.sqrt(cam.shape[0]**2 + cam.shape[1]**2)
    return spread / max_distance


def compute_cam_entropy(
    cam: np.ndarray
) -> float:
    """Compute entropy of CAM distribution.
    
    Higher entropy = more dispersed activations
    
    Args:
        cam: CAM heatmap (H, W)
        
    Returns:
        Entropy value
    """
    # Normalize to probability distribution
    cam_flat = cam.flatten()
    cam_flat = cam_flat + 1e-10  # Avoid log(0)
    cam_flat = cam_flat / cam_flat.sum()
    
    return entropy(cam_flat)


def compute_center_shift(
    cam1: np.ndarray,
    cam2: np.ndarray,
    threshold: float = 0.1
) -> float:
    """Compute shift in center of mass between two CAMs.
    
    Args:
        cam1: First CAM heatmap (H, W)
        cam2: Second CAM heatmap (H, W)
        threshold: Threshold for considering activation
        
    Returns:
        Normalized distance between centers
    """
    def get_center(cam):
        if cam.max() == 0:
            return None
        
        cam_norm = cam / cam.max()
        activated = cam_norm >= threshold
        
        if not np.any(activated):
            return None
        
        y_coords, x_coords = np.where(activated)
        return (np.mean(y_coords), np.mean(x_coords))
    
    center1 = get_center(cam1)
    center2 = get_center(cam2)
    
    if center1 is None or center2 is None:
        return 0.0
    
    # Compute distance
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # Normalize by image diagonal
    max_distance = np.sqrt(cam1.shape[0]**2 + cam1.shape[1]**2)
    return distance / max_distance


def compute_cam_metrics(
    cam: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    cam_shape: Tuple[int, int],
    letterbox_meta: Dict = None,
    baseline_cam: np.ndarray = None
) -> Dict[str, float]:
    """Compute all CAM distribution metrics (layer-invariant).
    
    Args:
        cam: Current CAM heatmap (H, W)
        bbox_xyxy: GT bbox (x1, y1, x2, y2) in original image pixels
        cam_shape: (cam_height, cam_width) CAM map dimensions
        letterbox_meta: Optional dict with letterbox transformation info
            If None, assumes CAM is same size as original image
        baseline_cam: Baseline CAM for comparison (optional)
        
    Returns:
        Dictionary of metrics
    """
    from src.data.bbox_conversion import map_bbox_to_cam
    
    metrics = {}
    
    # Map bbox to CAM coordinates
    cam_h, cam_w = cam_shape
    if letterbox_meta is not None:
        # Use letterbox metadata for accurate mapping
        img_shape = (letterbox_meta.get('original_width', cam_w), 
                     letterbox_meta.get('original_height', cam_h))
        bbox_cam = map_bbox_to_cam(bbox_xyxy, img_shape, cam_shape, letterbox_meta)
    else:
        # Direct scaling (no letterbox)
        img_w = letterbox_meta.get('original_width', cam_w) if letterbox_meta else cam_w
        img_h = letterbox_meta.get('original_height', cam_h) if letterbox_meta else cam_h
        scale_w = cam_w / img_w
        scale_h = cam_h / img_h
        x1, y1, x2, y2 = bbox_xyxy
        bbox_cam = (x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h)
    
    # Convert bbox_cam (x1, y1, x2, y2) to normalized center format for energy computation
    x1_cam, y1_cam, x2_cam, y2_cam = bbox_cam
    x_center_cam = (x1_cam + x2_cam) / 2
    y_center_cam = (y1_cam + y2_cam) / 2
    width_cam = x2_cam - x1_cam
    height_cam = y2_cam - y1_cam
    
    # Normalize to [0, 1] for energy computation
    bbox_norm = (x_center_cam / cam_w, y_center_cam / cam_h, 
                 width_cam / cam_w, height_cam / cam_h)
    
    # Energy in bbox
    metrics['energy_in_bbox'] = compute_energy_in_bbox(cam, bbox_norm, cam_w, cam_h)
    
    # Activation spread
    metrics['activation_spread'] = compute_activation_spread(cam)
    
    # Entropy
    metrics['entropy'] = compute_cam_entropy(cam)
    
    # Center shift (if baseline provided)
    if baseline_cam is not None:
        metrics['center_shift'] = compute_center_shift(baseline_cam, cam)
    else:
        metrics['center_shift'] = 0.0
    
    return metrics
