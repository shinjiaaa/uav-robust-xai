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
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    baseline_cam: np.ndarray = None
) -> Dict[str, float]:
    """Compute all CAM distribution metrics.
    
    Args:
        cam: Current CAM heatmap (H, W)
        bbox: GT bbox (x_center, y_center, width, height) normalized
        img_width: Image width
        img_height: Image height
        baseline_cam: Baseline CAM for comparison (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Energy in bbox
    metrics['energy_in_bbox'] = compute_energy_in_bbox(cam, bbox, img_width, img_height)
    
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
