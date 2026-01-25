"""Dynamic refinement for Grad-CAM analysis: subdivide failure regions."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def detect_failure_region(
    cam_metrics_df: pd.DataFrame,
    threshold: float = 0.3,
    min_severity_gap: int = 1
) -> List[Dict]:
    """Detect "explosive" failure regions where CAM metrics change dramatically.
    
    Args:
        cam_metrics_df: DataFrame with CAM metrics across severities
        threshold: Threshold for detecting dramatic change in CAM metrics
        min_severity_gap: Minimum gap between severities to consider as failure region
        
    Returns:
        List of failure regions: [{'start_severity': s1, 'end_severity': s2, 'metric': '...'}, ...]
    """
    failure_regions = []
    
    # Group by model, corruption, image_id, class_id
    groups = cam_metrics_df.groupby(['model', 'corruption', 'image_id', 'class_id'])
    
    for (model, corruption, image_id, class_id), group in groups:
        group = group.sort_values('severity')
        
        if len(group) < 2:
            continue
        
        # Check each CAM metric
        cam_metrics = ['energy_in_bbox', 'activation_spread', 'entropy', 'center_shift']
        
        for metric in cam_metrics:
            if metric not in group.columns:
                continue
            
            values = group[metric].values
            severities = group['severity'].values
            
            # Find dramatic changes (explosive failure)
            for i in range(len(values) - 1):
                if severities[i+1] - severities[i] < min_severity_gap:
                    continue
                
                # Calculate relative change
                if values[i] != 0:
                    relative_change = abs((values[i+1] - values[i]) / values[i])
                else:
                    relative_change = abs(values[i+1])
                
                if relative_change >= threshold:
                    failure_regions.append({
                        'model': model,
                        'corruption': corruption,
                        'image_id': image_id,
                        'class_id': class_id,
                        'metric': metric,
                        'start_severity': int(severities[i]),
                        'end_severity': int(severities[i+1]),
                        'start_value': float(values[i]),
                        'end_value': float(values[i+1]),
                        'relative_change': float(relative_change)
                    })
    
    return failure_regions


def generate_subdivided_severities(
    start_severity: int,
    end_severity: int,
    num_steps: int = 10
) -> List[int]:
    """Generate subdivided severity levels between start and end.
    
    Args:
        start_severity: Starting severity
        end_severity: Ending severity
        num_steps: Number of subdivision steps
        
    Returns:
        List of subdivided severity levels (excluding start, including end)
    """
    if start_severity >= end_severity:
        return []
    
    # Generate evenly spaced severities between start and end
    step_size = (end_severity - start_severity) / num_steps
    subdivided = []
    
    for i in range(1, num_steps + 1):
        severity = start_severity + i * step_size
        subdivided.append(int(round(severity)))
    
    # Remove duplicates and ensure order
    subdivided = sorted(list(set(subdivided)))
    
    return subdivided
