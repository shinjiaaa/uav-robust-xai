"""Failure event detection and risk region identification."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def detect_failure_events(
    records_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Detect failure events (first miss, score/IoU drop) for each tiny object.
    
    Args:
        records_df: DataFrame with columns: model, corruption, severity, 
                   image_id, class_id, score, iou, miss
        config: Configuration dictionary
        
    Returns:
        DataFrame with failure event information
    """
    risk_config = config['risk_detection']
    
    failure_events = []
    
    # Group by model, corruption, image_id, class_id (tiny object)
    groups = records_df.groupby(['model', 'corruption', 'image_id', 'class_id'])
    
    for (model, corruption, image_id, class_id), group in groups:
        # Sort by severity
        group = group.sort_values(['severity'])
        
        # Baseline (severity 0)
        baseline = group[group['severity'] == 0]
        if len(baseline) == 0:
            continue
        
        baseline_matched = baseline[baseline['miss'] == 0]
        if len(baseline_matched) == 0:
            # Already failed at baseline
            continue
        
        baseline_score = baseline_matched['score'].mean()
        baseline_iou = baseline_matched['iou'].mean()
        
        # Find first failure event
        first_miss_severity = None
        first_miss_frame = None
        score_drop_severity = None
        score_drop_frame = None
        iou_drop_severity = None
        iou_drop_frame = None
        
        for severity in sorted(group['severity'].unique()):
            if severity == 0:
                continue
            
            severity_group = group[group['severity'] == severity]
            
            # Check for first miss
            if first_miss_severity is None:
                missed = severity_group[severity_group['miss'] == 1]
                if len(missed) > 0:
                    first_miss_severity = severity
                    first_miss_frame = 0  # Single image, no frame index
            
            # Check for score drop
            if score_drop_severity is None:
                matched = severity_group[severity_group['miss'] == 0]
                if len(matched) > 0:
                    avg_score = matched['score'].mean()
                    if baseline_score - avg_score >= risk_config['score_drop_threshold']:
                        score_drop_severity = severity
                        score_drop_frame = 0  # Single image, no frame index
            
            # Check for IoU drop
            if iou_drop_severity is None:
                matched = severity_group[severity_group['miss'] == 0]
                if len(matched) > 0:
                    avg_iou = matched['iou'].mean()
                    if baseline_iou - avg_iou >= risk_config['iou_drop_threshold']:
                        iou_drop_severity = severity
                        iou_drop_frame = 0  # Single image, no frame index
        
        # Determine failure event (earliest)
        failure_severity = None
        failure_frame = None
        failure_type = None
        
        events = []
        if first_miss_severity is not None:
            events.append(('miss', first_miss_severity, first_miss_frame))
        if score_drop_severity is not None:
            events.append(('score_drop', score_drop_severity, score_drop_frame))
        if iou_drop_severity is not None:
            events.append(('iou_drop', iou_drop_severity, iou_drop_frame))
        
        if events:
            # Find earliest event
            events.sort(key=lambda x: (x[1], x[2]))  # Sort by severity, then frame
            failure_type, failure_severity, failure_frame = events[0]
        
        if failure_severity is not None:
            failure_events.append({
                'model': model,
                'corruption': corruption,
                'image_id': image_id,
                'class_id': class_id,
                'failure_type': failure_type,
                'failure_severity': failure_severity,
                'failure_frame': failure_frame,
                'first_miss_severity': first_miss_severity,
                'first_miss_frame': first_miss_frame,
                'score_drop_severity': score_drop_severity,
                'score_drop_frame': score_drop_frame,
                'iou_drop_severity': iou_drop_severity,
                'iou_drop_frame': iou_drop_frame,
                'baseline_score': baseline_score,
                'baseline_iou': baseline_iou
            })
    
    return pd.DataFrame(failure_events)


def identify_risk_regions(
    metrics_df: pd.DataFrame,
    tiny_curves_df: pd.DataFrame,
    failure_events_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Identify risk regions based on performance degradation.
    
    Args:
        metrics_df: Dataset-wide metrics
        tiny_curves_df: Tiny object curves
        failure_events_df: Failure events
        config: Configuration dictionary
        
    Returns:
        DataFrame with risk region information
    """
    risk_config = config['risk_detection']
    risk_regions = []
    
    # Group by model and corruption
    for model in metrics_df['model'].unique():
        for corruption in metrics_df['corruption'].unique():
            model_corr_metrics = metrics_df[
                (metrics_df['model'] == model) &
                (metrics_df['corruption'] == corruption)
            ]
            
            if len(model_corr_metrics) == 0:
                continue
            
            # Find severity where mAP drops significantly
            baseline = model_corr_metrics[model_corr_metrics['severity'] == 0]
            if len(baseline) == 0:
                continue
            
            baseline_map = baseline['map50'].iloc[0]
            
            risk_severity_map = None
            for severity in sorted(model_corr_metrics['severity'].unique()):
                if severity == 0:
                    continue
                
                sev_metrics = model_corr_metrics[model_corr_metrics['severity'] == severity]
                if len(sev_metrics) == 0:
                    continue
                
                sev_map = sev_metrics['map50'].iloc[0]
                if baseline_map - sev_map >= risk_config['map_drop_threshold']:
                    risk_severity_map = severity
                    break
            
            # Find severity where miss rate exceeds threshold
            model_corr_tiny = tiny_curves_df[
                (tiny_curves_df['model'] == model) &
                (tiny_curves_df['corruption'] == corruption)
            ]
            
            risk_severity_miss = None
            if len(model_corr_tiny) > 0:
                for severity in sorted(model_corr_tiny['severity'].unique()):
                    if severity == 0:
                        continue
                    
                    sev_tiny = model_corr_tiny[model_corr_tiny['severity'] == severity]
                    if len(sev_tiny) == 0:
                        continue
                    
                    miss_rate = sev_tiny['miss_rate'].mean()
                    if miss_rate >= risk_config['miss_rate_threshold']:
                        risk_severity_miss = severity
                        break
            
            # Count failure events
            if len(failure_events_df) > 0 and 'model' in failure_events_df.columns:
                model_corr_failures = failure_events_df[
                    (failure_events_df['model'] == model) &
                    (failure_events_df['corruption'] == corruption)
                ]
                failure_count = len(model_corr_failures)
            else:
                failure_count = 0
            
            risk_regions.append({
                'model': model,
                'corruption': corruption,
                'risk_severity_map': risk_severity_map,
                'risk_severity_miss': risk_severity_miss,
                'risk_severity': min([s for s in [risk_severity_map, risk_severity_miss] if s is not None], default=None),
                'failure_count': failure_count,
                'baseline_map50': baseline_map
            })
    
    return pd.DataFrame(risk_regions)


def detect_instability(
    records_df: pd.DataFrame,
    window_size: int = 5,
    threshold: float = 0.3
) -> pd.DataFrame:
    """Detect instability regions (high variance in score/IoU).
    
    Note: For single images, instability is detected across severity levels, not frames.
    
    Args:
        records_df: DataFrame with detection records
        window_size: Window size for variance calculation (across severities)
        threshold: Variance threshold
        
    Returns:
        DataFrame with instability regions
    """
    instability_regions = []
    
    # Group by model, corruption, image_id, class_id
    groups = records_df.groupby(['model', 'corruption', 'image_id', 'class_id'])
    
    for (model, corruption, image_id, class_id), group in groups:
        group = group.sort_values('severity')
        
        if len(group) < window_size:
            continue
        
        # Calculate rolling variance across severities
        matched = group[group['miss'] == 0]
        if len(matched) < window_size:
            continue
        
        scores = matched['score'].values
        ious = matched['iou'].values
        severities = matched['severity'].values
        
        # Rolling variance across severities
        for i in range(len(scores) - window_size + 1):
            score_window = scores[i:i+window_size]
            iou_window = ious[i:i+window_size]
            
            score_var = np.var(score_window)
            iou_var = np.var(iou_window)
            
            if score_var >= threshold or iou_var >= threshold:
                instability_regions.append({
                    'model': model,
                    'corruption': corruption,
                    'image_id': image_id,
                    'class_id': class_id,
                    'start_severity': int(severities[i]),
                    'end_severity': int(severities[i+window_size-1]),
                    'score_variance': score_var,
                    'iou_variance': iou_var
                })
    
    return pd.DataFrame(instability_regions)
