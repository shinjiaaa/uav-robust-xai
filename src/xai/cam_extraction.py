"""Multi-layer CAM extraction with quality gates."""

from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import cv2

from src.xai.gradcam_yolo import YOLOGradCAM
from src.xai.cam_qc import get_qc_status, check_cam_quality


def _crop_to_bbox(
    image: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    padding: int = 32,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop image to bbox with padding. Returns (crop, (x1,y1,x2,y2) used)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = max(padding, int(0.1 * min(bw, bh)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, w, h)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def extract_cam_multi_layer(
    gradcam_instances: Dict[str, YOLOGradCAM],
    image: np.ndarray,
    yolo_bbox: Tuple[float, float, float, float],
    class_id: int,
    layer_config: Dict,
    qc_config: Dict,
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
    orig_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Dict]:
    """Extract CAM from multiple layers (primary/secondary) simultaneously.
    
    Args:
        gradcam_instances: Dict mapping layer_role -> YOLOGradCAM instance
            e.g., {'primary': gradcam_primary, 'secondary': gradcam_secondary}
        image: Input image (H, W, 3) uint8
        yolo_bbox: (x_center, y_center, width, height) normalized
        class_id: Class ID for CAM generation
        layer_config: Dict with layer configurations:
            {
                'primary': {'name': str, 'role': str, 'required': bool},
                'secondary': {'name': str, 'role': str, 'required': bool}
            }
        qc_config: Dict with QC thresholds:
            {'finite_ratio_threshold', 'cam_sum_epsilon', 'cam_var_epsilon'}
        bbox_xyxy: Optional (x1, y1, x2, y2) in image pixels. If set with orig_shape,
            Grad-CAM is run on a crop of this region only (gradient only inside bbox).
        orig_shape: Optional (height, width) of the full image. Required when bbox_xyxy is set.
    
    Returns:
        Dict mapping layer_role -> result dict:
        {
            'primary': {
                'cam': np.ndarray or None,
                'cam_status': str,  # 'ok' or 'fail'
                'fail_reason': str or None,
                'letterbox_meta': Dict or None,
                'cam_shape': Tuple[int, int] or None
            },
            'secondary': {...} (same structure)
        }
    """
    results = {}
    # Bbox-ROI mode: run Grad-CAM on crop only (not recommended for gradual-curve analysis;
    # use full-image CAM so attention leakage and distance metrics are meaningful).
    use_bbox_roi = bbox_xyxy is not None and orig_shape is not None
    if use_bbox_roi:
        crop_img, _ = _crop_to_bbox(image, bbox_xyxy)
        roi_yolo_bbox = (0.5, 0.5, 1.0, 1.0)  # whole crop = object
    else:
        crop_img = image
        roi_yolo_bbox = yolo_bbox
    
    for layer_role, layer_info in layer_config.items():
        if layer_role not in gradcam_instances:
            # Layer not configured or instance not created
            results[layer_role] = {
                'cam': None,
                'cam_status': 'fail',
                'fail_reason': 'layer_not_configured',
                'letterbox_meta': None,
                'cam_shape': None,
                'exc_type': None,
                'exc_msg': None,
                'traceback_last_lines': None
            }
            continue
        
        gradcam = gradcam_instances[layer_role]
        extraction_error = None
        cam = None
        letterbox_meta = None
        exc_type = None
        exc_msg = None
        traceback_last_lines = None
        
        # Try to extract CAM (on crop in bbox-ROI mode, else full image)
        try:
            cam, letterbox_meta = gradcam.generate_cam(crop_img, roi_yolo_bbox, class_id)
            if use_bbox_roi and cam is not None:
                # Build full-image CAM: zeros except bbox region (resized crop CAM)
                orig_h, orig_w = orig_shape
                x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                cam_resized = cv2.resize(cam, (bw, bh), interpolation=cv2.INTER_LINEAR)
                cam_full = np.zeros((orig_h, orig_w), dtype=cam.dtype)
                cam_full[y1:y2, x1:x2] = cam_resized
                cam = cam_full
                letterbox_meta = None
        except RuntimeError as e:
            import traceback
            exc_type = type(e).__name__
            exc_msg = str(e)[:200]
            tb_lines = traceback.format_exc().split('\n')
            traceback_last_lines = '\n'.join([line for line in tb_lines[-4:] if line.strip()][-3:])
            
            error_msg = str(e).lower()
            if "activations not captured" in error_msg or "target_layer" in error_msg:
                extraction_error = "no_activation"
            elif "gradients not captured" in error_msg or "backward" in error_msg:
                extraction_error = "no_grad"
            elif "shape" in error_msg or "size" in error_msg:
                extraction_error = "shape_mismatch"
            elif "nan" in error_msg or "inf" in error_msg:
                extraction_error = "nan_cam"
            elif "memory" in error_msg or "out of memory" in error_msg:
                extraction_error = "memory_error"
            else:
                extraction_error = "runtime_error"
        except ValueError as e:
            import traceback
            exc_type = type(e).__name__
            exc_msg = str(e)[:200]
            tb_lines = traceback.format_exc().split('\n')
            traceback_last_lines = '\n'.join([line for line in tb_lines[-4:] if line.strip()][-3:])
            
            error_msg = str(e).lower()
            if "shape" in error_msg or "size" in error_msg:
                extraction_error = "shape_mismatch"
            else:
                extraction_error = "value_error"
        except Exception as e:
            # CRITICAL: Capture actual error information instead of "unknown_error"
            import traceback
            exc_type = type(e).__name__
            exc_msg = str(e)[:200]  # First 200 chars
            tb_lines = traceback.format_exc().split('\n')
            # Get last 3 non-empty lines of traceback
            traceback_last_lines = '\n'.join([line for line in tb_lines[-4:] if line.strip()][-3:])
            extraction_error = f"{exc_type}: {exc_msg[:100]}"  # Use actual error type
        
        # QC check (returns qc_stats with diagnostic info and quality label)
        # CRITICAL (RQ1): All CAMs are saved with quality labels (no hard filtering)
        cam_status, fail_reason, cam_quality, qc_stats = get_qc_status(cam, extraction_error)
        
        # Get CAM shape if available
        cam_shape = None
        if cam is not None:
            cam_shape = cam.shape
        
        # Store preprocessed input shape from letterbox_meta if available
        preprocessed_shape = None
        if letterbox_meta is not None:
            target_size = letterbox_meta.get('target_size', None)
            if target_size is not None:
                preprocessed_shape = (target_size, target_size)
        
        results[layer_role] = {
            'cam': cam,
            'cam_status': cam_status,  # "ok" for all quality levels (even flat/noisy)
            'fail_reason': fail_reason,  # Only for extraction errors (system errors)
            'cam_quality': cam_quality,  # RQ1: "high" | "flat" | "noisy" | "low_energy" | "empty" | "extraction_failed"
            'letterbox_meta': letterbox_meta,
            'cam_shape': cam_shape,
            'preprocessed_shape': preprocessed_shape,  # For diagnostic
            'exc_type': exc_type,
            'exc_msg': exc_msg,
            'traceback_last_lines': traceback_last_lines,
            # QC diagnostic stats
            'cam_min': qc_stats.get('cam_min'),
            'cam_max': qc_stats.get('cam_max'),
            'cam_sum': qc_stats.get('cam_sum'),
            'cam_var': qc_stats.get('cam_var'),
            'cam_std': qc_stats.get('cam_std'),
            'cam_dtype': qc_stats.get('cam_dtype'),
            'finite_ratio': qc_stats.get('finite_ratio')
        }
    
    return results
