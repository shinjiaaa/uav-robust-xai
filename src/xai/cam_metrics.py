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
    
    E_bbox = sum(CAM inside bbox) / sum(CAM)  [denominator = full CAM sum, not bbox sum]
    
    Design: CAM should be full-image; bbox is the measurement window. If CAM is bbox-crop-only,
    E_bbox is always 1.0 and distance/attention-leakage metrics are distorted.
    
    Args:
        cam: CAM heatmap (H, W)
        bbox: (x_center, y_center, width, height) in normalized coordinates
        img_width: Image width (CAM width when CAM is crop)
        img_height: Image height (CAM height when CAM is crop)
        
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
    
    total_energy = np.sum(cam)  # Full CAM sum (must be denominator for ratio)
    if total_energy == 0:
        return 0.0
    bbox_energy = np.sum(cam[y1:y2+1, x1:x2+1])
    return bbox_energy / total_energy  # Not bbox_energy/bbox_energy


def compute_ring_energy_ratio(
    cam: np.ndarray,
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    outer_scale: float = 1.25,
) -> float:
    """Ring Energy Ratio: E_ring_ratio = energy_bbox / (energy_bbox + energy_ring).
    
    ring = (outer_scale * bbox) \\ bbox (e.g. 1.25x expanded minus bbox).
    >0.5 → object-centric; ≈0.5 → similar to context; <0.5 → context-centric.
    Tiny-object friendly: compares bbox vs immediate surround instead of vs full image.
    
    Args:
        cam: CAM heatmap (H, W), full image
        bbox: (x_center, y_center, width, height) normalized
        img_width, img_height: CAM dimensions
        outer_scale: scale for outer region (default 1.25)
    
    Returns:
        Ratio in [0, 1], or 0.5 when denominator is 0 (neutral).
    """
    total = float(np.sum(cam))
    if total == 0:
        return 0.5
    energy_bbox = compute_energy_in_bbox(cam, bbox, img_width, img_height) * total
    energy_outer = compute_energy_in_bbox_expanded(cam, bbox, img_width, img_height, scale=outer_scale) * total
    energy_ring = energy_outer - energy_bbox  # ring = 1.25x \ bbox
    denom = energy_bbox + energy_ring  # = energy_outer
    if denom == 0:
        return 0.5
    return float(energy_bbox / denom)


def compute_energy_in_bbox_expanded(
    cam: np.ndarray,
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    scale: float = 1.1,
) -> float:
    """Compute energy inside bbox scaled by `scale` (center fixed, width/height scaled).
    Used for ROI sensitivity: 1.1x and 1.25x expanded bbox.
    """
    x_center, y_center, width, height = bbox
    w2 = min(0.5, width * scale / 2)
    h2 = min(0.5, height * scale / 2)
    x1 = int((x_center - w2) * img_width)
    y1 = int((y_center - h2) * img_height)
    x2 = int((x_center + w2) * img_width)
    y2 = int((y_center + h2) * img_height)
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    total_energy = np.sum(cam)  # Full CAM sum
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


def compute_bbox_center_activation_distance(
    cam: np.ndarray,
    bbox: Tuple[float, float, float, float],
    cam_w: int,
    cam_h: int,
    threshold: float = 0.1
) -> float:
    """Distance from bbox center to activation center of mass (normalized).
    bbox_dist: higher = CAM focus drifts from object.
    """
    if cam.max() == 0:
        return 0.0
    cam_norm = cam / cam.max()
    activated = cam_norm >= threshold
    if not np.any(activated):
        return 0.0
    y_coords, x_coords = np.where(activated)
    act_cy, act_cx = np.mean(y_coords), np.mean(x_coords)
    x_center, y_center, width, height = bbox
    # bbox is normalized [0,1]; convert to CAM pixel center
    bbox_cx = x_center * cam_w
    bbox_cy = y_center * cam_h
    dist = np.sqrt((act_cy - bbox_cy) ** 2 + (act_cx - bbox_cx) ** 2)
    max_d = np.sqrt(cam_h ** 2 + cam_w ** 2)
    return float(dist / max_d) if max_d > 0 else 0.0


def compute_peak_bbox_distance(
    cam: np.ndarray,
    bbox: Tuple[float, float, float, float],
    cam_w: int,
    cam_h: int
) -> float:
    """Distance from CAM peak (argmax) to bbox center (normalized).
    peak_dist: higher = peak activation is far from object.
    """
    flat_idx = np.argmax(cam)
    peak_y = flat_idx // cam.shape[1]
    peak_x = flat_idx % cam.shape[1]
    x_center, y_center, width, height = bbox
    bbox_cx = x_center * cam_w
    bbox_cy = y_center * cam_h
    dist = np.sqrt((peak_y - bbox_cy) ** 2 + (peak_x - bbox_cx) ** 2)
    max_d = np.sqrt(cam_h ** 2 + cam_w ** 2)
    return float(dist / max_d) if max_d > 0 else 0.0


def compute_ring_energy_ratio(
    cam: np.ndarray,
    bbox: Tuple[float, float, float, float],
    cam_w: int,
    cam_h: int,
    outer_scale: float = 1.25
) -> float:
    """Ratio of activation energy in ring (bbox to outer_scale*bbox) to total.
    E_ring_ratio: higher = more activation outside object bbox (dispersion).
    """
    x_center, y_center, width, height = bbox
    # pixel coords
    cx = x_center * cam_w
    cy = y_center * cam_h
    w = max(1, width * cam_w)
    h = max(1, height * cam_h)
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    x1 = max(0, min(x1, cam_w - 1))
    y1 = max(0, min(y1, cam_h - 1))
    x2 = max(0, min(x2, cam_w - 1))
    y2 = max(0, min(y2, cam_h - 1))
    # outer bbox (same center, outer_scale)
    ow = max(1, w * outer_scale)
    oh = max(1, h * outer_scale)
    ox1 = max(0, int(cx - ow / 2))
    oy1 = max(0, int(cy - oh / 2))
    ox2 = min(cam_w, int(cx + ow / 2) + 1)
    oy2 = min(cam_h, int(cy + oh / 2) + 1)
    total_energy = float(np.sum(cam))
    if total_energy == 0:
        return 0.0
    inner = np.sum(cam[y1:y2 + 1, x1:x2 + 1])
    outer_region = cam[oy1:oy2, ox1:ox2].copy()
    # mask out inner to get ring only
    ly, lx = outer_region.shape[0], outer_region.shape[1]
    iy1 = y1 - oy1
    iy2 = y2 - oy1 + 1
    ix1 = x1 - ox1
    ix2 = x2 - ox1 + 1
    if iy1 < 0:
        iy1 = 0
    if ix1 < 0:
        ix1 = 0
    if iy2 > ly:
        iy2 = ly
    if ix2 > lx:
        ix2 = lx
    outer_region[iy1:iy2, ix1:ix2] = 0
    ring_energy = float(np.sum(outer_region))
    return ring_energy / total_energy


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


def compute_activation_fragmentation(
    cam: np.ndarray,
    bbox_xyxy_cam: Tuple[float, float, float, float],
) -> float:
    """
    Fragmentation of activation inside bbox: entropy of the activation distribution
    within the bbox (normalized to sum=1). Higher = more dispersed/fragmented.

    Args:
        cam: CAM heatmap (H, W)
        bbox_xyxy_cam: (x1, y1, x2, y2) in CAM pixel coordinates

    Returns:
        Entropy value (non-negative)
    """
    x1, y1, x2, y2 = bbox_xyxy_cam
    x1, x2 = int(max(0, x1)), int(min(cam.shape[1], x2))
    y1, y2 = int(max(0, y1)), int(min(cam.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = cam[y1:y2, x1:x2]
    flat = crop.flatten() + 1e-10
    flat = flat / flat.sum()
    return float(entropy(flat))


def compute_bbox_center_activation_distance(
    cam: np.ndarray,
    bbox_xyxy_cam: Tuple[float, float, float, float],
    threshold: float = 0.1,
) -> float:
    """
    Distance between bbox center and activation center of mass (single CAM).
    Primary metric for gradual degradation: always continuous (no bimodal 0/1).

    Args:
        cam: CAM heatmap (H, W)
        bbox_xyxy_cam: (x1, y1, x2, y2) in CAM pixel coordinates
        threshold: relative threshold for activation (fraction of max)

    Returns:
        Normalized distance (0 = aligned, 1 = diagonal)
    """
    if cam.max() == 0:
        return 0.0
    cam_norm = cam / cam.max()
    activated = cam_norm >= threshold
    if not np.any(activated):
        return 0.0
    y_coords, x_coords = np.where(activated)
    act_cy = np.mean(y_coords)
    act_cx = np.mean(x_coords)
    x1, y1, x2, y2 = bbox_xyxy_cam
    bbox_cy = (y1 + y2) / 2.0
    bbox_cx = (x1 + x2) / 2.0
    dist = np.sqrt((act_cy - bbox_cy) ** 2 + (act_cx - bbox_cx) ** 2)
    max_dist = np.sqrt(cam.shape[0] ** 2 + cam.shape[1] ** 2)
    return float(dist / max_dist) if max_dist > 0 else 0.0


def compute_peak_bbox_distance(
    cam: np.ndarray,
    bbox_xyxy_cam: Tuple[float, float, float, float],
) -> float:
    """
    Distance between CAM peak (argmax) and bbox center. Continuous, stable for tiny objects.

    Args:
        cam: CAM heatmap (H, W)
        bbox_xyxy_cam: (x1, y1, x2, y2) in CAM pixel coordinates

    Returns:
        Normalized distance (0 = peak on bbox center, 1 = diagonal)
    """
    if cam.size == 0 or cam.max() == 0:
        return 0.0
    peak_flat = np.argmax(cam)
    peak_y = peak_flat // cam.shape[1]
    peak_x = peak_flat % cam.shape[1]
    x1, y1, x2, y2 = bbox_xyxy_cam
    bbox_cy = (y1 + y2) / 2.0
    bbox_cx = (x1 + x2) / 2.0
    dist = np.sqrt((peak_y - bbox_cy) ** 2 + (peak_x - bbox_cx) ** 2)
    max_dist = np.sqrt(cam.shape[0] ** 2 + cam.shape[1] ** 2)
    return float(dist / max_dist) if max_dist > 0 else 0.0


def compute_cam_metrics(
    cam: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    cam_shape: Tuple[int, int],
    letterbox_meta: Dict = None,
    baseline_cam: np.ndarray = None
) -> Dict[str, float]:
    """Compute all CAM distribution metrics (layer-invariant).
    
    Assumes full-image CAM; bbox is used only as a measurement window (energy in bbox,
    distance from bbox center/peak, etc.). Do not use bbox-cropped CAM here.
    
    Args:
        cam: Current CAM heatmap (H, W) — full image
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
    # Expanded bbox (ROI sensitivity: 보완 전략 3.2)
    metrics['energy_in_bbox_1_1x'] = compute_energy_in_bbox_expanded(cam, bbox_norm, cam_w, cam_h, scale=1.1)
    metrics['energy_in_bbox_1_25x'] = compute_energy_in_bbox_expanded(cam, bbox_norm, cam_w, cam_h, scale=1.25)
    # Ring Energy Ratio: bbox vs ring (1.25x \ bbox); >0.5 object-centric, <0.5 context-centric
    metrics['ring_energy_ratio'] = compute_ring_energy_ratio(cam, bbox_norm, cam_w, cam_h, outer_scale=1.25)
    # Full CAM sum (for 원인 분리: full vs ROI-only)
    metrics['full_cam_sum'] = float(np.sum(cam))
    
    # Activation spread
    metrics['activation_spread'] = compute_activation_spread(cam)

    metrics['entropy'] = compute_cam_entropy(cam)
    metrics['full_cam_entropy'] = metrics['entropy']
    
    # Center shift (if baseline provided)
    if baseline_cam is not None:
        metrics['center_shift'] = compute_center_shift(baseline_cam, cam)
    else:
        metrics['center_shift'] = 0.0

    # Tiny-object structural metrics
    bbox_cam_tuple = (x1_cam, y1_cam, x2_cam, y2_cam)
    try:
        metrics['activation_fragmentation'] = compute_activation_fragmentation(cam, bbox_cam_tuple)
    except Exception:
        metrics['activation_fragmentation'] = 0.0
    try:
        metrics['bbox_center_activation_distance'] = compute_bbox_center_activation_distance(cam, bbox_cam_tuple)
    except Exception:
        metrics['bbox_center_activation_distance'] = 0.0
    try:
        metrics['peak_bbox_distance'] = compute_peak_bbox_distance(cam, bbox_cam_tuple)
    except Exception:
        metrics['peak_bbox_distance'] = 0.0

    return metrics
