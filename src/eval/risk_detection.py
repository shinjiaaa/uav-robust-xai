"""Risk region detection utilities."""

import pandas as pd
from typing import Dict, List, Optional, Tuple


def detect_risk_severity_map(
    metrics_df: pd.DataFrame,
    model: str,
    corruption: str,
    drop_threshold: float = 0.15
) -> Optional[int]:
    """Detect the severity where mAP@0.5 drops by threshold from severity 0.
    
    Args:
        metrics_df: DataFrame with metrics (must have model, corruption, severity, map50 columns)
        model: Model name
        corruption: Corruption type
        drop_threshold: Absolute drop threshold (default 0.15 = 15%)
        
    Returns:
        Severity level where risk starts, or None if not found
    """
    subset = metrics_df[
        (metrics_df['model'] == model) &
        (metrics_df['corruption'] == corruption)
    ].sort_values('severity')
    
    if len(subset) == 0:
        return None
    
    # Get baseline (severity 0)
    baseline = subset[subset['severity'] == 0]
    if len(baseline) == 0:
        return None
    
    baseline_map50 = baseline['map50'].iloc[0]
    
    # Find first severity where drop >= threshold
    for _, row in subset.iterrows():
        if row['severity'] == 0:
            continue
        
        drop = baseline_map50 - row['map50']
        if drop >= drop_threshold:
            return int(row['severity'])
    
    return None


def detect_risk_severity_miss(
    tiny_curves_df: pd.DataFrame,
    model: str,
    corruption: str,
    miss_threshold: float = 0.5
) -> Optional[int]:
    """Detect the severity where tiny miss rate reaches threshold.
    
    Args:
        tiny_curves_df: DataFrame with tiny curves (must have model, corruption, severity, miss_rate columns)
        model: Model name
        corruption: Corruption type
        miss_threshold: Miss rate threshold (default 0.5 = 50%)
        
    Returns:
        Severity level where risk starts, or None if not found
    """
    subset = tiny_curves_df[
        (tiny_curves_df['model'] == model) &
        (tiny_curves_df['corruption'] == corruption)
    ].sort_values('severity')
    
    if len(subset) == 0:
        return None
    
    # Find first severity where miss_rate >= threshold
    for _, row in subset.iterrows():
        if pd.notna(row['miss_rate']) and row['miss_rate'] >= miss_threshold:
            return int(row['severity'])
    
    return None


def compute_risk_regions(
    metrics_df: pd.DataFrame,
    tiny_curves_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Compute risk regions for all model/corruption combinations.
    
    Args:
        metrics_df: Dataset-wide metrics DataFrame
        tiny_curves_df: Tiny object curves DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with risk region information
    """
    risk_config = config['risk_detection']
    models = list(config['models'].keys())
    corruptions = config['corruptions']['types']
    
    risk_regions = []
    
    for model in models:
        for corruption in corruptions:
            risk_map = detect_risk_severity_map(
                metrics_df,
                model,
                corruption,
                drop_threshold=risk_config['map_drop_threshold']
            )
            
            risk_miss = detect_risk_severity_miss(
                tiny_curves_df,
                model,
                corruption,
                miss_threshold=risk_config['miss_rate_threshold']
            )
            
            risk_regions.append({
                'model': model,
                'corruption': corruption,
                'risk_severity_map': risk_map,
                'risk_severity_miss': risk_miss
            })
    
    return pd.DataFrame(risk_regions)
