"""CAM Quality Gate (QC) functions for validating CAM maps."""

import numpy as np
from typing import Tuple, Optional, Dict


def check_cam_quality(
    cam: np.ndarray,
    finite_ratio_threshold: float = 0.9,  # 완화: 0.99 → 0.9 (NaN 처리 후)
    cam_sum_epsilon: float = 1e-6,
    cam_var_epsilon: float = 1e-8
) -> Tuple[str, Optional[str], Dict]:
    """Check CAM quality using soft labels (RQ1 research design).
    
    CRITICAL CHANGE: Hard filter → Soft label
    - All CAMs are saved (not filtered out)
    - Quality labels: "high", "flat", "noisy", "low_energy"
    - This allows RQ1 analysis with both "all CAMs" and "high-quality only"
    
    Args:
        cam: CAM heatmap (H, W) - should already have NaN/Inf cleaned
        finite_ratio_threshold: Minimum ratio of finite values (default: 0.9, relaxed)
        cam_sum_epsilon: Minimum total CAM energy (for low_energy label)
        cam_var_epsilon: Minimum CAM variance (for flat label)
        
    Returns:
        (cam_quality, quality_flag, qc_stats)
        - cam_quality: "high" | "flat" | "noisy" | "low_energy" | "empty"
        - quality_flag: None if high, quality issue string if not (for backward compatibility)
        - qc_stats: Dict with diagnostic info (cam_min, cam_max, cam_sum, cam_var, cam_shape, finite_ratio)
    """
    qc_stats = {
        'cam_min': None,
        'cam_max': None,
        'cam_sum': None,
        'cam_var': None,
        'cam_std': None,
        'cam_shape': None,
        'cam_dtype': None,
        'finite_ratio': None
    }
    
    if cam is None or cam.size == 0:
        qc_stats['cam_shape'] = None
        return "empty", "empty_cam", qc_stats
    
    # Store basic CAM info
    qc_stats['cam_shape'] = cam.shape
    qc_stats['cam_dtype'] = str(cam.dtype)
    
    # Check 1: Finite ratio (NaN/inf check) - SOFT LABEL
    finite_mask = np.isfinite(cam)
    finite_ratio = np.sum(finite_mask) / cam.size
    qc_stats['finite_ratio'] = float(finite_ratio)
    
    is_noisy = finite_ratio < finite_ratio_threshold
    
    # Use only finite values for remaining checks
    cam_finite = cam[finite_mask] if np.any(finite_mask) else cam
    
    if len(cam_finite) == 0 or not np.any(finite_mask):
        qc_stats['cam_sum'] = 0.0
        qc_stats['cam_var'] = 0.0
        qc_stats['cam_min'] = float(np.nanmin(cam)) if cam.size > 0 else None
        qc_stats['cam_max'] = float(np.nanmax(cam)) if cam.size > 0 else None
        return "noisy", "no_finite_values", qc_stats
    
    # Store CAM statistics
    qc_stats['cam_min'] = float(np.min(cam_finite))
    qc_stats['cam_max'] = float(np.max(cam_finite))
    cam_std = float(np.std(cam_finite))
    qc_stats['cam_std'] = cam_std
    
    # Check 2: Total energy (cam_sum) - for low_energy label
    cam_sum = np.sum(cam_finite)
    qc_stats['cam_sum'] = float(cam_sum)
    
    # Check 3: Flatness (variance/std) - SOFT LABEL (not fail, just label)
    cam_var = np.var(cam_finite)
    qc_stats['cam_var'] = float(cam_var)
    
    cam_max = qc_stats['cam_max']
    cam_max_epsilon = 1e-4  # max threshold for flat detection
    cam_std_epsilon = 1e-6  # std threshold for flat detection
    
    is_flat = (cam_max < cam_max_epsilon) or (cam_std < cam_std_epsilon) or (cam_var <= cam_var_epsilon)
    
    # Check 4: Low energy - SOFT LABEL
    is_low_energy = cam_sum < cam_sum_epsilon
    
    # Determine quality label (priority: empty > noisy > flat > low_energy > high)
    if is_noisy:
        quality = "noisy"
        quality_flag = "finite_ratio_low"
    elif is_flat:
        quality = "flat"  # RQ1: "붕괴 상태 라벨" - 삭제하지 않고 기록
        quality_flag = "cam_flat"
    elif is_low_energy:
        quality = "low_energy"
        quality_flag = "cam_low_energy"
    else:
        quality = "high"
        quality_flag = None
    
    return quality, quality_flag, qc_stats


def get_qc_status(
    cam: Optional[np.ndarray],
    extraction_error: Optional[str] = None
) -> Tuple[str, Optional[str], str, Dict]:
    """Get CAM status with soft quality labels (RQ1 research design).
    
    CRITICAL CHANGE: All CAMs are saved with quality labels (no hard filtering)
    
    Args:
        cam: CAM heatmap (H, W) or None if extraction failed
        extraction_error: Error code from CAM extraction (e.g., "shape_mismatch", "no_grad")
        
    Returns:
        (cam_status, fail_reason, cam_quality, qc_stats)
        - cam_status: "ok" or "fail" (for backward compatibility, but all CAMs are saved)
        - fail_reason: None if ok, error code string if fail (extraction errors only)
        - cam_quality: "high" | "flat" | "noisy" | "low_energy" | "empty" | "extraction_failed"
        - qc_stats: Dict with diagnostic info (cam_min, cam_max, cam_sum, cam_var, cam_shape, finite_ratio)
    """
    qc_stats = {
        'cam_min': None,
        'cam_max': None,
        'cam_sum': None,
        'cam_var': None,
        'cam_std': None,
        'cam_shape': None,
        'cam_dtype': None,
        'finite_ratio': None
    }
    
    # System errors (extraction failed) - these are real failures
    if extraction_error is not None:
        # System errors: memory_error, shape_mismatch, no_activation, no_grad
        return "fail", extraction_error, "extraction_failed", qc_stats
    
    if cam is None:
        return "fail", "cam_extraction_failed", "extraction_failed", qc_stats
    
    # Quality check (soft labels - all CAMs are saved)
    cam_quality, quality_flag, qc_stats = check_cam_quality(cam)
    
    # cam_status: "ok" for all quality levels (even flat/noisy/low_energy)
    # Only "fail" for extraction errors (system errors)
    cam_status = "ok"
    fail_reason = None  # No fail_reason for quality issues (they're labels, not failures)
    
    return cam_status, fail_reason, cam_quality, qc_stats
