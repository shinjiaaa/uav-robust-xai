"""CAM Quality Gate (QC) functions for validating CAM maps."""

import numpy as np
from typing import Tuple, Optional, Dict


def check_cam_quality(
    cam: np.ndarray,
    finite_ratio_threshold: float = 0.99,
    cam_sum_epsilon: float = 1e-6,
    cam_var_epsilon: float = 1e-8
) -> Tuple[bool, Optional[str], Dict]:
    """Check CAM quality using QC rules.
    
    Args:
        cam: CAM heatmap (H, W)
        finite_ratio_threshold: Minimum ratio of finite values (default: 0.99)
        cam_sum_epsilon: Minimum total CAM energy (default: 1e-6)
        cam_var_epsilon: Minimum CAM variance (default: 1e-8)
        
    Returns:
        (is_valid, fail_reason, qc_stats)
        - is_valid: True if CAM passes all QC checks
        - fail_reason: None if valid, error code string if invalid
        - qc_stats: Dict with diagnostic info (cam_min, cam_max, cam_sum, cam_var, cam_shape, finite_ratio)
    """
    qc_stats = {
        'cam_min': None,
        'cam_max': None,
        'cam_sum': None,
        'cam_var': None,
        'cam_std': None,  # 추가: std for QC
        'cam_shape': None,
        'cam_dtype': None,
        'finite_ratio': None
    }
    
    if cam is None or cam.size == 0:
        qc_stats['cam_shape'] = None
        return False, "empty_cam", qc_stats
    
    # Store basic CAM info
    qc_stats['cam_shape'] = cam.shape
    qc_stats['cam_dtype'] = str(cam.dtype)
    
    # Check 1: Finite ratio (NaN/inf check)
    finite_mask = np.isfinite(cam)
    finite_ratio = np.sum(finite_mask) / cam.size
    qc_stats['finite_ratio'] = float(finite_ratio)
    
    if finite_ratio < finite_ratio_threshold:
        return False, "finite_ratio_fail", qc_stats
    
    # Use only finite values for remaining checks
    cam_finite = cam[finite_mask]
    
    if len(cam_finite) == 0:
        qc_stats['cam_sum'] = 0.0
        qc_stats['cam_var'] = 0.0
        qc_stats['cam_min'] = float(np.nanmin(cam)) if cam.size > 0 else None
        qc_stats['cam_max'] = float(np.nanmax(cam)) if cam.size > 0 else None
        return False, "no_finite_values", qc_stats
    
    # Store CAM statistics
    qc_stats['cam_min'] = float(np.min(cam_finite))
    qc_stats['cam_max'] = float(np.max(cam_finite))
    cam_std = float(np.std(cam_finite))
    qc_stats['cam_std'] = cam_std
    
    # Check 2: Total energy (cam_sum) - keep for diagnostic
    cam_sum = np.sum(cam_finite)
    qc_stats['cam_sum'] = float(cam_sum)
    
    # Check 3: QC 기준 변경 - sum 대신 max/std 기준 사용
    # cam_sum_fail은 CAM이 0에 가깝다는 뜻인데, tiny object + 특정 타겟 조건에서 매우 흔함
    # max/std 기준이 더 robust함
    cam_max = qc_stats['cam_max']
    cam_max_epsilon = 1e-4  # max가 너무 작으면 fail
    cam_std_epsilon = 1e-6  # std가 너무 작으면 fail (완전히 flat)
    
    if cam_max < cam_max_epsilon or cam_std < cam_std_epsilon:
        return False, "cam_too_flat", qc_stats
    
    # Check 4: Variance (not completely flat) - keep for additional check
    cam_var = np.var(cam_finite)
    qc_stats['cam_var'] = float(cam_var)
    
    if cam_var <= cam_var_epsilon:
        return False, "cam_var_fail", qc_stats
    
    return True, None, qc_stats


def get_qc_status(
    cam: Optional[np.ndarray],
    extraction_error: Optional[str] = None
) -> Tuple[str, Optional[str], Dict]:
    """Get CAM status and fail reason with diagnostic stats.
    
    Args:
        cam: CAM heatmap (H, W) or None if extraction failed
        extraction_error: Error code from CAM extraction (e.g., "shape_mismatch", "no_grad")
        
    Returns:
        (cam_status, fail_reason, qc_stats)
        - cam_status: "ok" or "fail"
        - fail_reason: None if ok, error code string if fail
        - qc_stats: Dict with diagnostic info (cam_min, cam_max, cam_sum, cam_var, cam_shape, finite_ratio)
    """
    qc_stats = {
        'cam_min': None,
        'cam_max': None,
        'cam_sum': None,
        'cam_var': None,
        'cam_std': None,  # 추가: std for QC
        'cam_shape': None,
        'cam_dtype': None,
        'finite_ratio': None
    }
    
    if extraction_error is not None:
        return "fail", extraction_error, qc_stats
    
    if cam is None:
        return "fail", "cam_extraction_failed", qc_stats
    
    is_valid, qc_fail_reason, qc_stats = check_cam_quality(cam)
    if is_valid:
        return "ok", None, qc_stats
    else:
        return "fail", qc_fail_reason, qc_stats
