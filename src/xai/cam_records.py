"""CAM records storage with standardized schema."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import numpy as np


def create_cam_record(
    # Identifiers
    model: str,
    corruption: str,
    severity: int,
    image_id: str,
    object_id: Optional[str] = None,  # Can store object_uid for alignment
    frame_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    class_id: int = None,
    
    # Bbox (original image coordinates)
    bbox_x1: float = None,
    bbox_y1: float = None,
    bbox_x2: float = None,
    bbox_y2: float = None,
    
    # Layer info
    layer_role: str = None,  # 'primary' or 'secondary'
    layer_name: str = None,
    
    # CAM status (RQ1: soft labels - all CAMs saved)
    cam_status: str = None,  # 'ok' or 'fail' (for backward compatibility)
    fail_reason: Optional[str] = None,  # Only for extraction errors (system errors)
    cam_quality: Optional[str] = None,  # RQ1: 'high' | 'flat' | 'noisy' | 'low_energy' | 'empty' | 'extraction_failed'
    exc_type: Optional[str] = None,  # Exception type name (e.g., 'MemoryError', 'RuntimeError')
    exc_msg: Optional[str] = None,  # Exception message (first 200 chars)
    traceback_last_lines: Optional[str] = None,  # Last 1-3 lines of traceback
    # QC diagnostic stats (for cam_sum_fail debugging)
    cam_min: Optional[float] = None,
    cam_max: Optional[float] = None,
    cam_sum: Optional[float] = None,
    cam_var: Optional[float] = None,
    cam_std: Optional[float] = None,  # 추가: std for QC
    cam_dtype: Optional[str] = None,
    finite_ratio: Optional[float] = None,
    preprocessed_shape: Optional[Tuple[int, int]] = None,  # (H, W) after letterbox
    
    # CAM basics
    cam_h: Optional[int] = None,
    cam_w: Optional[int] = None,
    norm_method: str = "min_max",  # Normalization method
    
    # Metrics
    entropy: Optional[float] = None,
    activation_spread: Optional[float] = None,
    center_shift: Optional[float] = None,
    energy_in_bbox: Optional[float] = None,
    
    # Performance (from detection)
    detected: Optional[int] = None,  # 0 or 1
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    map_bucket: Optional[str] = None,
    
    # Additional metadata
    letterbox_meta: Optional[Dict] = None,
    failure_severity: Optional[int] = None,
    # Alignment analysis fields (for joining with risk_events)
    failure_event_id: Optional[str] = None,  # Links to risk_events.csv
    failure_type: Optional[str] = None,  # miss / score_drop / iou_drop
    # RQ1: CAM target selection (for miss cases)
    cam_target_class_id: Optional[int] = None,  # Class ID used for CAM generation (may differ from GT class_id)
    cam_target_type: Optional[str] = None  # "gt_class", "pred_class", "gt_class_miss"
) -> Dict:
    """Create a CAM record with standardized schema.
    
    Returns:
        Dict with all fields (None for missing values)
    """
    record = {
        # Identifiers
        'model': model,
        'corruption': corruption,
        'severity': severity,
        'image_id': image_id,
        'object_id': object_id,
        'frame_id': frame_id,
        'clip_id': clip_id,
        'class_id': class_id,
        
        # Bbox
        'bbox_x1': bbox_x1,
        'bbox_y1': bbox_y1,
        'bbox_x2': bbox_x2,
        'bbox_y2': bbox_y2,
        
        # Layer
        'layer_role': layer_role,
        'layer_name': layer_name,
        
        # Status (RQ1: soft labels)
        'cam_status': cam_status,
        'fail_reason': fail_reason,
        'cam_quality': cam_quality,  # RQ1: quality label for analysis
        'exc_type': exc_type,
        'exc_msg': exc_msg,
        'traceback_last_lines': traceback_last_lines,
        # QC diagnostic stats
        'cam_min': cam_min,
        'cam_max': cam_max,
        'cam_sum': cam_sum,
        'cam_var': cam_var,
        'cam_std': cam_std,  # 추가: std for QC
        'cam_dtype': cam_dtype,
        'finite_ratio': finite_ratio,
        'preprocessed_shape': f"{preprocessed_shape[0]}x{preprocessed_shape[1]}" if preprocessed_shape and isinstance(preprocessed_shape, tuple) and len(preprocessed_shape) == 2 else None,
        
        # CAM basics
        'cam_h': cam_h,
        'cam_w': cam_w,
        'norm_method': norm_method,
        
        # Metrics
        'entropy': entropy,
        'activation_spread': activation_spread,
        'center_shift': center_shift,
        'energy_in_bbox': energy_in_bbox,
        
        # Performance
        'detected': detected,
        'conf': conf,
        'iou': iou,
        'map_bucket': map_bucket,
        
        # Metadata
        'failure_severity': failure_severity,
        # Alignment analysis fields
        'failure_event_id': failure_event_id,
        'failure_type': failure_type,
        # RQ1: CAM target selection
        'cam_target_class_id': cam_target_class_id,
        'cam_target_type': cam_target_type
    }
    
    # Store letterbox_meta as JSON string if provided
    if letterbox_meta is not None:
        import json
        record['letterbox_meta'] = json.dumps(letterbox_meta)
    else:
        record['letterbox_meta'] = None
    
    return record


def save_cam_records(
    records: List[Dict],
    output_path: Path
):
    """Save CAM records to CSV file.
    
    Args:
        records: List of CAM record dicts
        output_path: Path to output CSV file
    """
    if len(records) == 0:
        print(f"[WARN] No CAM records to save to {output_path}")
        return
    
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(records)} CAM records to {output_path}")
