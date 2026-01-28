"""Multi-layer CAM extraction with quality gates."""

from typing import Dict, Optional, Tuple, List
import numpy as np
import torch

from src.xai.gradcam_yolo import YOLOGradCAM
from src.xai.cam_qc import get_qc_status, check_cam_quality


def extract_cam_multi_layer(
    gradcam_instances: Dict[str, YOLOGradCAM],
    image: np.ndarray,
    yolo_bbox: Tuple[float, float, float, float],
    class_id: int,
    layer_config: Dict,
    qc_config: Dict
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
            {
                'finite_ratio_threshold': float,
                'cam_sum_epsilon': float,
                'cam_var_epsilon': float
            }
    
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
        
        # Try to extract CAM
        try:
            cam, letterbox_meta = gradcam.generate_cam(image, yolo_bbox, class_id)
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
