"""
LLM-based report generation.

- Fixes:
  1) load_metrics() now loads all CSVs that summarize_metrics() expects
     (detection_records, failure_events, cam_metrics, risk_regions, instability, gradcam_error).
  2) Fixed multiple indentation / block-structure bugs that would crash at runtime:
     - performance_cam_alignment: CAM-metric loop was accidentally indented under dict literal
     - lead_lag_analysis: missing per-corruption dict init + same indentation issue
  3) Added defensive column handling and consistent key names.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
# import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os


# ============================================================
# CRITICAL: Failure Severity Definition (Single Source of Truth)
# ============================================================
# Failure severity is defined as: First miss occurrence severity (Option A)
# This is the severity at which the first miss (miss=1) occurs for a given image/object
# All tables (Table 3, Table 6, Table 7) must use this same definition
def get_failure_severity_for_corruption(
    fail_df: pd.DataFrame,
    corruption: str,
    det_df: Optional[pd.DataFrame] = None
) -> Optional[int]:
    """
    Get the failure severity for a corruption using a unified definition.

    Definition: First miss occurrence severity (minimum first_miss_severity from failure_events)
    If no failure events exist, use miss_rate threshold (>= 0.25) from detection records.

    Args:
        fail_df: DataFrame with failure_events
        corruption: Corruption type (fog, lowlight, motion_blur)
        det_df: Optional detection records DataFrame

    Returns:
        Failure severity (int) or None if no failure
    """
    # Method 1: Use failure_events.csv if available
    if fail_df is not None and len(fail_df) > 0 and 'corruption' in fail_df.columns:
        corr_fail = fail_df[fail_df['corruption'] == corruption].copy()
        if len(corr_fail) > 0:
            # Try first_miss_severity first (preferred)
            if 'first_miss_severity' in corr_fail.columns:
                first_miss_sevs = corr_fail['first_miss_severity'].dropna()
                if len(first_miss_sevs) > 0:
                    return int(first_miss_sevs.min())
            
            # Fallback to failure_severity if first_miss_severity is not available
            # This handles cases like lowlight where failure_type is score_drop/iou_drop
            if 'failure_severity' in corr_fail.columns:
                failure_sevs = corr_fail['failure_severity'].dropna()
                if len(failure_sevs) > 0:
                    return int(failure_sevs.min())

    # Method 2: Fallback to detection records (miss_rate >= 0.25)
    if det_df is not None and len(det_df) > 0 and 'corruption' in det_df.columns:
        corr_det = det_df[det_df['corruption'] == corruption].copy()
        if len(corr_det) > 0 and 'miss' in corr_det.columns and 'severity' in corr_det.columns:
            miss_by_sev = corr_det.groupby('severity')['miss'].mean()
            failure_severities = miss_by_sev[miss_by_sev >= 0.25].index.tolist()
            if failure_severities:
                return int(min(failure_severities))

    return None


def _read_csv_if_exists(path: Path) -> list:
    if path.exists():
        try:
            df = pd.read_csv(path)
            # Handle empty CSV files
            if len(df) == 0 or df.empty:
                return []
            return df.to_dict('records')
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return []
    return []


def load_metrics(config: Dict) -> Dict:
    """Load all computed metrics.

    NOTE: summarize_metrics() expects keys:
      - dataset
      - tiny_curves
      - tiny_records
      - gradcam              (legacy)
      - detection_records    (tiny per-object per-severity records)
      - failure_events       (failure_event.csv)
      - cam_metrics          (gradcam_metrics_timeseries.csv, cam_metrics.csv, or gradcam_metrics.csv)
      - risk_regions
      - instability
      - gradcam_errors       (gradcam_error.csv)

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with all metrics
    """
    results_root = Path(config['results']['root'])
    metrics: Dict[str, list] = {}

    # Dataset-wide metrics
    metrics_csv = results_root / "metrics_dataset.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        # Fill NaN with 0.0 (empty predictions -> 0 metrics, not missing)
        df = df.fillna(0.0)
        metrics['dataset'] = df.to_dict('records')
    else:
        metrics['dataset'] = []

    # Tiny object curves
    tiny_curves_csv = results_root / "tiny_curves.csv"
    metrics['tiny_curves'] = _read_csv_if_exists(tiny_curves_csv)

    # Tiny object records (legacy name you already used)
    tiny_records_csv = results_root / "tiny_records.csv"
    metrics['tiny_records'] = _read_csv_if_exists(tiny_records_csv)

    # Detection records (prefer detection_records.csv, fallback to tiny_records_timeseries.csv)
    det_csv = results_root / "detection_records.csv"
    ts_csv = results_root / "tiny_records_timeseries.csv"
    
    if det_csv.exists() and det_csv.stat().st_size > 0:
        metrics['detection_records'] = _read_csv_if_exists(det_csv)
        if not metrics['detection_records'] or len(metrics['detection_records']) == 0:
            print("[WARN] detection_records.csv exists but is empty")
            metrics['detection_records'] = []
        else:
            print(f"[INFO] Loaded {len(metrics['detection_records'])} records from detection_records.csv")
    elif ts_csv.exists() and ts_csv.stat().st_size > 0:
        print("[WARN] detection_records.csv not found, using tiny_records_timeseries.csv as fallback")
        metrics['detection_records'] = _read_csv_if_exists(ts_csv)
        if metrics['detection_records']:
            print(f"[INFO] Loaded {len(metrics['detection_records'])} records from tiny_records_timeseries.csv")
    else:
        print("[ERROR] Neither detection_records.csv nor tiny_records_timeseries.csv found.")
        print("[ERROR] Performance Metrics table will be skipped.")
        metrics['detection_records'] = []
    
    # tiny_records_timeseries.csv is kept separately for backward compatibility/CAM analysis
    if ts_csv.exists():
        metrics['tiny_records_timeseries'] = _read_csv_if_exists(ts_csv)
    else:
        metrics['tiny_records_timeseries'] = []

    # Grad-CAM metrics (legacy key)
    gradcam_csv = results_root / "gradcam_metrics.csv"
    metrics['gradcam'] = _read_csv_if_exists(gradcam_csv)

    # CAM metrics (the name summarize_metrics() uses)
    # CRITICAL: Prefer cam_records.csv (new format with cam_status) over legacy files
    # Legacy files (gradcam_metrics_timeseries.csv) may contain stale data from previous runs
    cam_records_csv = results_root / "cam_records.csv"
    if cam_records_csv.exists():
        metrics['cam_records'] = _read_csv_if_exists(cam_records_csv)
        print(f"[INFO] Loaded {len(metrics['cam_records'])} records from cam_records.csv")
    else:
        metrics['cam_records'] = []
    
    # Legacy CAM metrics (for backward compatibility, but NOT used for Table 7)
    cam_metrics_csv = results_root / "cam_metrics.csv"
    if cam_metrics_csv.exists():
        metrics['cam_metrics'] = _read_csv_if_exists(cam_metrics_csv)
    else:
        # Try timeseries version (actual file name) - but this is legacy, may be stale
        timeseries_cam_csv = results_root / "gradcam_metrics_timeseries.csv"
        if timeseries_cam_csv.exists():
            metrics['cam_metrics'] = _read_csv_if_exists(timeseries_cam_csv)
            print(f"[WARN] Using legacy gradcam_metrics_timeseries.csv (may contain stale data)")
        else:
            metrics['cam_metrics'] = metrics.get('gradcam', [])

    # Failure events
    failure_events_csv = results_root / "failure_event.csv"
    # also support plural naming if used elsewhere
    failure_events_csv_alt = results_root / "failure_events.csv"
    if failure_events_csv.exists():
        metrics['failure_events'] = _read_csv_if_exists(failure_events_csv)
    elif failure_events_csv_alt.exists():
        metrics['failure_events'] = _read_csv_if_exists(failure_events_csv_alt)
    else:
        metrics['failure_events'] = []

    # Risk regions / instability (optional)
    risk_regions_csv = results_root / "risk_regions.csv"
    metrics['risk_regions'] = _read_csv_if_exists(risk_regions_csv)

    # Instability regions (try both naming conventions)
    instability_csv = results_root / "instability.csv"
    instability_regions_csv = results_root / "instability_regions.csv"
    if instability_regions_csv.exists():
        metrics['instability'] = _read_csv_if_exists(instability_regions_csv)
    elif instability_csv.exists():
        metrics['instability'] = _read_csv_if_exists(instability_csv)
    else:
        metrics['instability'] = []

    # Grad-CAM errors (optional)
    gradcam_error_csv = results_root / "gradcam_error.csv"
    gradcam_errors_csv = results_root / "gradcam_errors.csv"  # plural form
    if gradcam_errors_csv.exists():
        metrics['gradcam_errors'] = _read_csv_if_exists(gradcam_errors_csv)
    elif gradcam_error_csv.exists():
        metrics['gradcam_errors'] = _read_csv_if_exists(gradcam_error_csv)
    else:
        metrics['gradcam_errors'] = []

    return metrics


def summarize_metrics(metrics: Dict) -> Dict:
    """Summarize metrics to reduce token usage.

    Args:
        metrics: Full metrics dictionary

    Returns:
        Summarized metrics dictionary
    """
    summarized: Dict = {}
    
    # CRITICAL: Load detection_records once at the beginning for data consistency checks
    detection_records = metrics.get('detection_records', [])
    detection_records_df = pd.DataFrame(detection_records) if detection_records else pd.DataFrame()
    
    # Get corruptions that actually have detection records (data consistency check)
    # This is used throughout summarize_metrics to ensure all tables use the same data source
    available_corruptions_from_det = set()
    if len(detection_records_df) > 0 and 'corruption' in detection_records_df.columns:
        available_corruptions_from_det = set(detection_records_df['corruption'].unique())

    # ----------------------------
    # Dataset metrics (Table 1)
    # CRITICAL: Only include corruptions that have actual detection records
    # ----------------------------
    dataset_metrics = metrics.get('dataset', [])
    
    if dataset_metrics:
        dataset_df = pd.DataFrame(dataset_metrics)

        dataset_by_corruption_severity = {}
        all_expected_corruptions = ['fog', 'lowlight', 'motion_blur']
        all_severities = [0, 1, 2, 3, 4]

        for corruption in all_expected_corruptions:
            # CRITICAL: Only process corruptions that have actual detection records
            # If dataset metrics exist but no detection records, mark as "no data"
            if corruption not in available_corruptions_from_det:
                # No detection records for this corruption - mark all severities as "no data"
                for severity in all_severities:
                    dataset_by_corruption_severity.setdefault(corruption, {})[severity] = {
                        'map50': None,
                        'map5095': None,
                        'precision': None,
                        'recall': None,
                        'n_frames_total': 0,
                        'n_frames_valid': 0,
                        'valid_ratio': 0.0,
                        'status': 'N/A (no detection records - data inconsistency)',
                        'evaluation_failed': False
                    }
                continue
            dataset_by_corruption_severity[corruption] = {}
            corr_df = dataset_df[dataset_df['corruption'] == corruption].copy() if 'corruption' in dataset_df.columns else pd.DataFrame()

            for severity in all_severities:
                sev_df = corr_df[corr_df['severity'] == severity].copy() if len(corr_df) > 0 else pd.DataFrame()

                if len(sev_df) > 0:
                    eval_failed = False
                    if 'eval_status' in sev_df.columns:
                        eval_failed = (sev_df['eval_status'] == 'error').any()
                    if 'pred_count' in sev_df.columns:
                        eval_failed = eval_failed or (sev_df['pred_count'] == 0).any()

                    if 'pred_count' in sev_df.columns and 'eval_status' in sev_df.columns:
                        valid_df = sev_df[(sev_df['pred_count'] > 0) & (sev_df['eval_status'] != 'error')]
                        # CRITICAL: n_frames_total should use pred_count (actual frames evaluated)
                        # not len(sev_df) which is just the number of CSV rows
                        # If metrics_dataset.csv has only 1 row per severity, pred_count indicates actual dataset size
                        n_frames_total = int(sev_df['pred_count'].iloc[0]) if len(sev_df) > 0 and pd.notna(sev_df['pred_count'].iloc[0]) else len(sev_df)
                        
                        # CRITICAL: n_frames_valid should use detection_records.csv, not metrics_dataset.csv rows
                        # metrics_dataset.csv has 1 row per severity, but detection_records.csv has actual frame-level data
                        if detection_records_df is not None and len(detection_records_df) > 0:
                            # Filter detection_records for this corruption/severity
                            det_sev_df = detection_records_df[
                                (detection_records_df['corruption'] == corruption) & 
                                (detection_records_df['severity'] == severity)
                            ]
                            if 'frame_id' in det_sev_df.columns:
                                n_frames_valid = det_sev_df['frame_id'].nunique()
                            elif 'image_id' in det_sev_df.columns:
                                n_frames_valid = det_sev_df['image_id'].nunique()
                            else:
                                n_frames_valid = len(det_sev_df)  # Fallback
                        else:
                            n_frames_valid = len(valid_df)  # Fallback: Number of valid CSV rows (usually 1 if subset evaluation)
                        
                        valid_ratio = float(n_frames_valid / n_frames_total) if n_frames_total > 0 else 0.0
                    else:
                        n_frames_total = len(sev_df)
                        # CRITICAL: n_frames_valid should use detection_records.csv, not metrics_dataset.csv rows
                        if detection_records_df is not None and len(detection_records_df) > 0:
                            # Filter detection_records for this corruption/severity
                            det_sev_df = detection_records_df[
                                (detection_records_df['corruption'] == corruption) & 
                                (detection_records_df['severity'] == severity)
                            ]
                            if 'frame_id' in det_sev_df.columns:
                                n_frames_valid = det_sev_df['frame_id'].nunique()
                            elif 'image_id' in det_sev_df.columns:
                                n_frames_valid = det_sev_df['image_id'].nunique()
                            else:
                                n_frames_valid = len(det_sev_df)  # Fallback
                        else:
                            n_frames_valid = n_frames_total if not eval_failed else 0
                        valid_ratio = 1.0 if not eval_failed else 0.0
                        valid_df = sev_df.copy() if n_frames_valid > 0 else pd.DataFrame()

                    # CRITICAL: If metrics_dataset.csv has only 1 row per severity, it's likely a subset/sample evaluation
                    # not a full dataset-wide evaluation. Mark this clearly.
                    # n_frames_total now uses pred_count (actual frames), so compare against that
                    if len(sev_df) == 1 and n_frames_total < 100:
                        # Likely subset evaluation - metrics_dataset.csv structure suggests this
                        # Still output the mAP values even if it's a subset
                        dataset_by_corruption_severity[corruption][severity] = {
                            'map50': float(valid_df['map50'].iloc[0]) if 'map50' in valid_df.columns and len(valid_df) > 0 else None,
                            'map5095': float(valid_df['map5095'].iloc[0]) if 'map5095' in valid_df.columns and len(valid_df) > 0 else None,
                            'precision': float(valid_df['precision'].iloc[0]) if 'precision' in valid_df.columns and len(valid_df) > 0 else None,
                            'recall': float(valid_df['recall'].iloc[0]) if 'recall' in valid_df.columns and len(valid_df) > 0 else None,
                            'n_frames_total': n_frames_total,
                            'n_frames_valid': n_frames_valid,
                            'valid_ratio': valid_ratio,
                            'status': f'Subset evaluation (n_frames_total={n_frames_total} from pred_count, CSV rows={len(sev_df)})',
                            'evaluation_failed': bool(eval_failed)
                        }
                    elif n_frames_total < 10:
                        dataset_by_corruption_severity[corruption][severity] = {
                            'map50': None,
                            'map5095': None,
                            'precision': None,
                            'recall': None,
                            'n_frames_total': n_frames_total,
                            'n_frames_valid': n_frames_valid,
                            'valid_ratio': valid_ratio,
                            'status': 'N/A (insufficient valid frames)',
                            'evaluation_failed': bool(eval_failed)
                        }
                    else:
                        dataset_by_corruption_severity[corruption][severity] = {
                            'map50': float(valid_df['map50'].mean()) if 'map50' in valid_df.columns else None,
                            'map5095': float(valid_df['map5095'].mean()) if 'map5095' in valid_df.columns else None,
                            'precision': float(valid_df['precision'].mean()) if 'precision' in valid_df.columns else None,
                            'recall': float(valid_df['recall'].mean()) if 'recall' in valid_df.columns else None,
                            'n_frames_total': n_frames_total,
                            'n_frames_valid': n_frames_valid,
                            'valid_ratio': valid_ratio,
                            'status': 'Computed from valid subset' if not eval_failed else 'Evaluation failed',
                            'evaluation_failed': bool(eval_failed)
                        }
                else:
                    dataset_by_corruption_severity[corruption][severity] = {
                        'map50': None,
                        'map5095': None,
                        'precision': None,
                        'recall': None,
                        'n_frames_total': 0,
                        'n_frames_valid': 0,
                        'valid_ratio': 0.0,
                        'status': 'N/A (no data)',
                        'evaluation_failed': False
                    }

        summarized['dataset_by_corruption_severity'] = dataset_by_corruption_severity
        summarized['dataset'] = dataset_df.to_dict('records') if len(dataset_df) > 0 else []
        summarized['dataset_evaluation_status'] = {
            'note': 'mAP computed from valid subset (pred_count > 0, eval_status != error)'
        }
    else:
        summarized['dataset'] = []
        summarized['dataset_evaluation_status'] = {}
        summarized['dataset_by_corruption_severity'] = {}

    # ----------------------------
    # Detection records summary (Table 2)
    # CRITICAL: detection_records.csv MUST exist for Performance Metrics table
    # NO FALLBACK: If missing, Table 2 will be skipped with explicit warning
    # ----------------------------
    # detection_records is already available from metrics.get('detection_records', []) at function start
    if detection_records and len(detection_records) > 0:
        df = pd.DataFrame(detection_records)

        # CRITICAL: Calculate actual object counts per corruption/severity
        # This is the actual number of tiny objects evaluated, not from metrics_dataset.csv
        n_objects_by_corruption_severity = {}
        if 'corruption' in df.columns and 'severity' in df.columns:
            for corruption in df['corruption'].unique():
                n_objects_by_corruption_severity[corruption] = {}
                corr_df = df[df['corruption'] == corruption]
                for severity in corr_df['severity'].unique():
                    sev_df = corr_df[corr_df['severity'] == severity]
                    n_objects_by_corruption_severity[corruption][severity] = len(sev_df)

        # CRITICAL: Calculate n_frames_valid using frame_id (or image_id as fallback)
        # This counts unique frames, not total records (objects × severities)
        if 'frame_id' in df.columns:
            n_frames_valid = df['frame_id'].nunique()
            frame_count_column = 'frame_id'
        elif 'image_id' in df.columns:
            n_frames_valid = df['image_id'].nunique()
            frame_count_column = 'image_id'
        else:
            # Fallback: if neither exists, use total records (legacy behavior)
            n_frames_valid = len(df)
            frame_count_column = 'total_records'
        
        overall_summary = {
            'total_records': len(detection_records),
            'n_objects_total': len(detection_records),  # Total detection records (objects × severities)
            'n_frames_valid': int(n_frames_valid),  # Unique frames (frame_id or image_id based)
            'frame_count_column': frame_count_column,  # Which column was used for counting
            'by_model': df.groupby('model').size().to_dict() if 'model' in df.columns else {},
            'by_corruption': df.groupby('corruption').size().to_dict() if 'corruption' in df.columns else {},
            'by_severity': df.groupby('severity').size().to_dict() if 'severity' in df.columns else {},
            'n_objects_by_corruption_severity': n_objects_by_corruption_severity,
            'miss_rate_by_severity': df.groupby('severity')['miss'].mean().to_dict() if 'miss' in df.columns else {},
            'avg_score_by_severity': df.groupby('severity')['score'].mean().to_dict() if 'score' in df.columns else {},
            'avg_iou_by_severity': df.groupby('severity')['iou'].mean().to_dict() if 'iou' in df.columns else {},
        }

        per_corruption = {}
        if 'corruption' in df.columns:
            for corruption in df['corruption'].unique():
                corr_df = df[df['corruption'] == corruption]
                per_corruption[corruption] = {
                    'miss_rate_by_severity': corr_df.groupby('severity')['miss'].mean().to_dict() if 'miss' in corr_df.columns else {},
                    'avg_score_by_severity': corr_df.groupby('severity')['score'].mean().to_dict() if 'score' in corr_df.columns else {},
                    'avg_iou_by_severity': corr_df.groupby('severity')['iou'].mean().to_dict() if 'iou' in corr_df.columns else {},
                }

        summarized['detection_summary'] = {**overall_summary, 'per_corruption': per_corruption}
        summarized['detection_summary']['has_data'] = True
    else:
        # CRITICAL: No detection_records - Table 2 (Performance Metrics) will be skipped
        # But check if data actually exists (may have been loaded from fallback file)
        if len(detection_records) == 0:
            summarized['detection_summary'] = {
                'has_data': False,
                'error': 'detection_records not found or empty',
                'message': 'Performance Metrics table (Table 2) cannot be generated without detection records. Please run scripts/03_detect_tiny_objects_timeseries.py to generate this file.'
            }
        else:
            # Data exists (may be from fallback file), but mark as available
            summarized['detection_summary'] = {
                'has_data': True,
                'note': 'Data loaded (may be from fallback file)'
            }

    # ----------------------------
    # Tiny curves pass-through
    # ----------------------------
    tiny_curves = metrics.get('tiny_curves', [])
    if len(tiny_curves) > 50:
        df = pd.DataFrame(tiny_curves)
        summarized['tiny_curves'] = {
            'total_records': len(tiny_curves),
            'summary': df.describe().to_dict() if len(df) > 0 else {}
        }
    else:
        summarized['tiny_curves'] = tiny_curves

    # ----------------------------
    # Failure events summary (Table 3)
    # CRITICAL: Only include corruptions that have actual detection records
    # ----------------------------
    failure_events = metrics.get('failure_events', [])
    det_df_for_failure = pd.DataFrame(detection_records) if detection_records else pd.DataFrame()

    if failure_events:
        df = pd.DataFrame(failure_events)
        fail_df_for_unified = df.copy()
        
        # Use the same available_corruptions_from_det from the beginning of summarize_metrics
        # (already computed from detection_records_df)
        
        failure_summary_by_corruption = {}
        for corruption in (df['corruption'].unique() if 'corruption' in df.columns else []):
            # CRITICAL: Only process corruptions that have actual detection records
            if corruption not in available_corruptions_from_det:
                # Skip corruptions without detection data (data inconsistency)
                continue
            
            corr_df = df[df['corruption'] == corruption].copy()
            unified_failure_sev = get_failure_severity_for_corruption(fail_df_for_unified, corruption, det_df_for_failure)

            # Count event types from failure_events.csv
            miss_events = len(corr_df[corr_df['failure_type'] == 'miss']) if 'failure_type' in corr_df.columns else 0
            score_drop_events = len(corr_df[corr_df['failure_type'] == 'score_drop']) if 'failure_type' in corr_df.columns else 0
            iou_drop_events = len(corr_df[corr_df['failure_type'] == 'iou_drop']) if 'failure_type' in corr_df.columns else 0

            failure_summary_by_corruption[corruption] = {
                'total_events': len(corr_df),
                'miss_events': int(miss_events),
                'score_drop_events': int(score_drop_events),
                'iou_drop_events': int(iou_drop_events),
                # Removed 'unified_failure_severity' - not shown in Failure Summary table
                'failure_severity_definition': 'First miss occurrence severity (minimum first_miss_severity)',
                'first_miss_severity': {
                    'min': float(corr_df['first_miss_severity'].min()) if 'first_miss_severity' in corr_df.columns and corr_df['first_miss_severity'].notna().any() else None,
                    'max': float(corr_df['first_miss_severity'].max()) if 'first_miss_severity' in corr_df.columns and corr_df['first_miss_severity'].notna().any() else None,
                    'mean': float(corr_df['first_miss_severity'].mean()) if 'first_miss_severity' in corr_df.columns and corr_df['first_miss_severity'].notna().any() else None,
                    'mode': float(corr_df['first_miss_severity'].mode().iloc[0]) if 'first_miss_severity' in corr_df.columns and corr_df['first_miss_severity'].notna().any() and len(corr_df['first_miss_severity'].mode()) > 0 else None,
                },
                'score_drop_severity': {
                    'min': float(corr_df['score_drop_severity'].min()) if 'score_drop_severity' in corr_df.columns and corr_df['score_drop_severity'].notna().any() else None,
                    'max': float(corr_df['score_drop_severity'].max()) if 'score_drop_severity' in corr_df.columns and corr_df['score_drop_severity'].notna().any() else None,
                    'mean': float(corr_df['score_drop_severity'].mean()) if 'score_drop_severity' in corr_df.columns and corr_df['score_drop_severity'].notna().any() else None,
                    'mode': float(corr_df['score_drop_severity'].mode().iloc[0]) if 'score_drop_severity' in corr_df.columns and corr_df['score_drop_severity'].notna().any() and len(corr_df['score_drop_severity'].mode()) > 0 else None,
                },
                'iou_drop_severity': {
                    'min': float(corr_df['iou_drop_severity'].min()) if 'iou_drop_severity' in corr_df.columns and corr_df['iou_drop_severity'].notna().any() else None,
                    'max': float(corr_df['iou_drop_severity'].max()) if 'iou_drop_severity' in corr_df.columns and corr_df['iou_drop_severity'].notna().any() else None,
                    'mean': float(corr_df['iou_drop_severity'].mean()) if 'iou_drop_severity' in corr_df.columns and corr_df['iou_drop_severity'].notna().any() else None,
                    'mode': float(corr_df['iou_drop_severity'].mode().iloc[0]) if 'iou_drop_severity' in corr_df.columns and corr_df['iou_drop_severity'].notna().any() and len(corr_df['iou_drop_severity'].mode()) > 0 else None,
                },
            }

        # event_type vs failure_type naming compatibility
        event_col = 'event_type' if 'event_type' in df.columns else ('failure_type' if 'failure_type' in df.columns else None)

        # CRITICAL: Filter failure events to only include corruptions with actual detection records
        # Use the same available_corruptions_from_det from the beginning of summarize_metrics
        if available_corruptions_from_det:
            df_filtered = df[df['corruption'].isin(available_corruptions_from_det)].copy() if 'corruption' in df.columns else df
        else:
            df_filtered = df.copy()

        summarized['failure_summary'] = {
            'total_events': len(df_filtered),  # Only count events from available corruptions
            'by_model': df_filtered.groupby('model').size().to_dict() if 'model' in df_filtered.columns else {},
            'by_corruption': df_filtered.groupby('corruption').size().to_dict() if 'corruption' in df_filtered.columns else {},
            'by_event_type': df_filtered.groupby(event_col).size().to_dict() if event_col and event_col in df_filtered.columns else {},
            'by_corruption_detail': failure_summary_by_corruption,
            'data_consistency_note': f'Only corruptions with detection records are included. Available: {sorted(available_corruptions_from_det)}' if available_corruptions_from_det else 'No detection records available for consistency check'
        }
    else:
        summarized['failure_summary'] = {}

    # ----------------------------
    # Risk regions (Table 9 support)
    # ----------------------------
    if len(detection_records) > 0:
        det_df_risk = pd.DataFrame(detection_records)
        risk_regions_dict = {}

        if 'corruption' in det_df_risk.columns and 'severity' in det_df_risk.columns and 'miss' in det_df_risk.columns:
            for corruption in det_df_risk['corruption'].unique():
                corr_det = det_df_risk[det_df_risk['corruption'] == corruption].copy()
                miss_by_sev = corr_det.groupby('severity')['miss'].mean()
                risk_severities = miss_by_sev[miss_by_sev >= 0.25].index.tolist()

                if risk_severities:
                    risk_sev = int(min(risk_severities))
                    risk_miss_rate = float(miss_by_sev[risk_sev])
                else:
                    risk_sev = None
                    risk_miss_rate = float(miss_by_sev.max()) if len(miss_by_sev) > 0 else None

                risk_regions_dict[corruption] = {
                    'corruption': corruption,
                    'risk_severity': risk_sev,
                    'miss_rate_at_risk': risk_miss_rate,
                    'mAP_drop_severity': None,
                    'notes': "miss_rate < 0.25 threshold" if risk_sev is None else f"miss_rate >= 0.25 at severity {risk_sev}"
                }

        summarized['risk_regions'] = list(risk_regions_dict.values())
    else:
        rr = metrics.get('risk_regions', [])
        summarized['risk_regions'] = rr[:20] if len(rr) > 20 else rr

    inst = metrics.get('instability', [])
    summarized['instability'] = inst[:20] if len(inst) > 20 else inst

    # ----------------------------
    # CAM metrics + pattern analyses
    # CRITICAL: Only include corruptions that have actual detection records
    # ----------------------------
    # CRITICAL: Use cam_records.csv (new format) instead of legacy cam_metrics
    cam_records = metrics.get('cam_records', [])
    cam_metrics = metrics.get('cam_metrics', [])  # Legacy fallback
    det_df = pd.DataFrame(detection_records) if detection_records else pd.DataFrame()
    fail_df = pd.DataFrame(failure_events) if failure_events else pd.DataFrame()

    # Prefer cam_records.csv if available (has cam_status and layer_role)
    if cam_records and len(cam_records) > 0:
        cam_records_df = pd.DataFrame(cam_records)
        if 'cam_status' in cam_records_df.columns:
            # Filter by cam_status='ok' AND layer_role='primary'
            if 'layer_role' in cam_records_df.columns:
                cam_df = cam_records_df[
                    (cam_records_df['cam_status'] == 'ok') & 
                    (cam_records_df['layer_role'] == 'primary')
                ].copy()
            else:
                cam_df = cam_records_df[cam_records_df['cam_status'] == 'ok'].copy()
        else:
            cam_df = cam_records_df.copy()
        has_cam_data = len(cam_df) > 0
        print(f"[INFO] Using cam_records.csv for CAM analysis ({len(cam_df)} OK primary records)")
    elif cam_metrics and len(cam_metrics) > 0:
        # Fallback to legacy format (may contain stale data)
        cam_df = pd.DataFrame(cam_metrics)
        has_cam_data = len(cam_df) > 0
        print(f"[WARN] Using legacy CAM metrics format (may contain stale data)")
    else:
        cam_df = pd.DataFrame()
        has_cam_data = False

    if has_cam_data and detection_records:

        cam_metrics_list = ['energy_in_bbox', 'activation_spread', 'entropy', 'center_shift']
        all_expected_corruptions = ['fog', 'lowlight', 'motion_blur']
        all_severities = [0, 1, 2, 3, 4]

        # ---- Table 4: CAM metrics by corruption x severity
        # CRITICAL: Only include corruptions that have actual detection records
        cam_metrics_by_corruption_severity: Dict[str, Dict[int, Dict]] = {}
        
        # CRITICAL: Use cam_records.csv (new format) instead of legacy gradcam_metrics_timeseries.csv
        # cam_records.csv has cam_status and layer_role fields for accurate filtering
        cam_records = metrics.get('cam_records', [])
        cam_records_df = pd.DataFrame(cam_records) if cam_records else pd.DataFrame()
        
        # If cam_records.csv exists, use it; otherwise fall back to legacy cam_metrics
        if len(cam_records_df) > 0 and 'cam_status' in cam_records_df.columns:
            # Use new format: filter by cam_status='ok' AND layer_role='primary'
            if 'layer_role' in cam_records_df.columns:
                cam_df = cam_records_df[
                    (cam_records_df['cam_status'] == 'ok') & 
                    (cam_records_df['layer_role'] == 'primary')
                ].copy()
            else:
                # Fallback: if layer_role doesn't exist, just filter by cam_status
                cam_df = cam_records_df[cam_records_df['cam_status'] == 'ok'].copy()
            print(f"[INFO] Using cam_records.csv for Table 7 ({len(cam_df)} OK primary records)")
        else:
            # Fallback to legacy format (may contain stale data)
            cam_df = pd.DataFrame(cam_metrics) if cam_metrics else pd.DataFrame()
            if len(cam_df) > 0:
                print(f"[WARN] Using legacy CAM metrics format (may contain stale data)")

        # Get corruptions that actually have detection records (data consistency check)
        # Use the same available_corruptions_from_det from dataset metrics section
        # (already computed at the beginning of summarize_metrics)

        # Get failure severity for each corruption (to include in CAM metrics table)
        failure_severity_by_corruption = {}
        for corruption in all_expected_corruptions:
            if corruption in available_corruptions_from_det:
                unified_failure_sev = get_failure_severity_for_corruption(fail_df, corruption, det_df)
                failure_severity_by_corruption[corruption] = unified_failure_sev
        
        for corruption in all_expected_corruptions:
            # CRITICAL: Only process corruptions that have actual detection records
            if corruption not in available_corruptions_from_det:
                # No detection records for this corruption - mark all severities as "no data"
                for sev in all_severities:
                    cam_metrics_by_corruption_severity.setdefault(corruption, {})[sev] = {
                        'energy_in_bbox_mean': None,
                        'activation_spread_mean': None,
                        'entropy_mean': None,
                        'center_shift_mean': None,
                        'n_cam_frames': 0,
                        'failure_severity': None,  # Add failure_severity
                        'note': 'N/A (no detection records - data inconsistency)'
                    }
                continue
            
            corr_cam = cam_df[cam_df['corruption'] == corruption].copy() if 'corruption' in cam_df.columns and len(cam_df) > 0 and corruption in cam_df['corruption'].unique() else pd.DataFrame()
            cam_metrics_by_corruption_severity[corruption] = {}
            
            # Get failure severity for this corruption
            unified_failure_sev = failure_severity_by_corruption.get(corruption)

            if len(corr_cam) > 0 and 'severity' in corr_cam.columns:
                for sev in all_severities:
                    sev_cam = corr_cam[corr_cam['severity'] == sev].copy()
                    if len(sev_cam) > 0:
                        # CRITICAL: Only count cam_status == "ok" as valid CAM frames
                        # (If using cam_records.csv, this is already filtered, but double-check for legacy data)
                        if 'cam_status' in sev_cam.columns:
                            sev_cam_ok = sev_cam[sev_cam['cam_status'] == 'ok'].copy()
                            n_cam_frames = len(sev_cam_ok)
                            # Use only OK records for metric computation
                            metric_df = sev_cam_ok
                        else:
                            # Fallback: if cam_status column doesn't exist, count all (legacy data)
                            n_cam_frames = len(sev_cam)
                            metric_df = sev_cam
                        
                        # Compute metrics only from OK records
                        cam_metrics_by_corruption_severity[corruption][sev] = {
                            'energy_in_bbox_mean': float(metric_df['energy_in_bbox'].mean()) if 'energy_in_bbox' in metric_df.columns and len(metric_df) > 0 else None,
                            'activation_spread_mean': float(metric_df['activation_spread'].mean()) if 'activation_spread' in metric_df.columns and len(metric_df) > 0 else None,
                            'entropy_mean': float(metric_df['entropy'].mean()) if 'entropy' in metric_df.columns and len(metric_df) > 0 else None,
                            'center_shift_mean': float(metric_df['center_shift'].mean()) if 'center_shift' in metric_df.columns and len(metric_df) > 0 else None,
                            'n_cam_frames': int(n_cam_frames),
                            'failure_severity': unified_failure_sev,  # Add failure_severity
                            'note': f'Computed from {n_cam_frames} CAM frames (cam_status=ok only)' if n_cam_frames > 0 else 'N/A (no valid CAM frames, all failed)'
                        }
                    else:
                        cam_metrics_by_corruption_severity[corruption][sev] = {
                            'energy_in_bbox_mean': None,
                            'activation_spread_mean': None,
                            'entropy_mean': None,
                            'center_shift_mean': None,
                            'n_cam_frames': 0,
                            'failure_severity': unified_failure_sev,  # Add failure_severity even when n_cam_frames=0
                            'note': 'N/A (n_cam_frames=0)'
                        }
            else:
                for sev in all_severities:
                    cam_metrics_by_corruption_severity[corruption][sev] = {
                        'energy_in_bbox_mean': None,
                        'activation_spread_mean': None,
                        'entropy_mean': None,
                        'center_shift_mean': None,
                        'n_cam_frames': 0,
                        'failure_severity': unified_failure_sev,  # Add failure_severity even when CAM data not available
                        'note': 'N/A (CAM data not available for this corruption, n_cam_frames=0)'
                    }

        summarized['cam_metrics_by_corruption_severity'] = cam_metrics_by_corruption_severity

        # ---- cam_pattern_summary (Table 5)
        # CRITICAL: Only include corruptions that have actual detection records
        cam_pattern_summary: Dict[str, Dict] = {}
        available_corruptions = cam_df['corruption'].unique().tolist() if 'corruption' in cam_df.columns else []

        for corruption in all_expected_corruptions:
            # CRITICAL: Only process corruptions that have actual detection records
            if corruption not in available_corruptions_from_det:
                cam_pattern_summary[corruption] = {'note': 'No detection records for this corruption (data inconsistency)', 'all_metrics_missing': True}
                continue
            
            if corruption not in available_corruptions:
                cam_pattern_summary[corruption] = {'note': 'CAM data not available for this corruption', 'all_metrics_missing': True}
                continue

            corr_cam = cam_df[cam_df['corruption'] == corruption].copy()
            if 'severity' not in corr_cam.columns:
                cam_pattern_summary[corruption] = {'note': 'CAM data missing severity column', 'all_metrics_missing': True}
                continue

            corr_cam_sorted = corr_cam.sort_values('severity')
            cam_pattern_summary[corruption] = {}

            for metric_name in cam_metrics_list:
                if metric_name not in corr_cam_sorted.columns:
                    continue

                sev_values = {}
                for sev in sorted(corr_cam_sorted['severity'].unique()):
                    sev_data = corr_cam_sorted[corr_cam_sorted['severity'] == sev].copy()
                    if len(sev_data) > 0:
                        mean_val = sev_data[metric_name].mean()
                        sev_values[int(sev)] = float(mean_val) if mean_val is not None else None

                if len(sev_values) < 2:
                    continue

                sev_seq = sorted(sev_values.keys())
                values_seq = [sev_values[s] for s in sev_seq]

                is_monotonic_increase = all(values_seq[i] <= values_seq[i + 1] for i in range(len(values_seq) - 1))
                is_monotonic_decrease = all(values_seq[i] >= values_seq[i + 1] for i in range(len(values_seq) - 1))
                is_monotonic = bool(is_monotonic_increase or is_monotonic_decrease)

                sev_0_val = sev_values.get(0, None)
                sev_4_val = sev_values.get(4, None)

                if sev_0_val is not None and sev_4_val is not None:
                    delta = sev_4_val - sev_0_val
                    normalized_delta = (delta / abs(sev_0_val)) if abs(sev_0_val) > 1e-10 else (delta if delta != 0 else 0.0)
                else:
                    delta = None
                    normalized_delta = None

                if delta is not None:
                    if abs(delta) < 1e-10:
                        direction = 'no_change'
                    elif delta > 0:
                        direction = 'increase'
                    else:
                        direction = 'decrease'
                else:
                    direction = 'unknown'

                cam_pattern_summary[corruption][metric_name] = {
                    'monotonic': is_monotonic,
                    'monotonic_direction': 'increase' if is_monotonic_increase else ('decrease' if is_monotonic_decrease else 'none'),
                    'delta': float(delta) if delta is not None else None,
                    'normalized_delta': float(normalized_delta) if normalized_delta is not None else None,
                    'direction': direction,
                    'values_by_severity': {k: float(v) for k, v in sev_values.items()}
                }

        summarized['cam_pattern_summary'] = cam_pattern_summary

        # ---- performance_cam_alignment (Table 6)
        # CRITICAL: Only include corruptions that have actual detection records
        performance_cam_alignment: Dict[str, Dict] = {}

        for corruption in all_expected_corruptions:
            # CRITICAL: Only process corruptions that have actual detection records
            if corruption not in available_corruptions_from_det:
                continue  # Skip corruptions without detection data
            
            corr_cam = cam_df[cam_df['corruption'] == corruption].copy() if 'corruption' in cam_df.columns else pd.DataFrame()
            corr_det = det_df[det_df['corruption'] == corruption].copy() if len(det_df) > 0 and 'corruption' in det_df.columns else pd.DataFrame()

            unified_failure_sev = get_failure_severity_for_corruption(fail_df, corruption, det_df)
            if unified_failure_sev is None:
                continue

            failure_sev = int(unified_failure_sev)
            if len(corr_cam) == 0 or 'severity' not in corr_cam.columns:
                continue

            cam_at_fail_sev = corr_cam[corr_cam['severity'] == failure_sev]
            if len(cam_at_fail_sev) == 0:
                continue

            fail_image_id = cam_at_fail_sev['image_id'].iloc[0] if 'image_id' in cam_at_fail_sev.columns else None
            fail_class_id = cam_at_fail_sev['class_id'].iloc[0] if 'class_id' in cam_at_fail_sev.columns else None

            if fail_image_id is None:
                continue

            if pd.notna(fail_class_id):
                fail_cam = corr_cam[(corr_cam.get('image_id') == fail_image_id) & (corr_cam.get('class_id') == fail_class_id)].copy()
            else:
                fail_cam = corr_cam[corr_cam.get('image_id') == fail_image_id].copy()

            cam_at_fail = fail_cam[fail_cam['severity'] == failure_sev]
            cam_at_prev = fail_cam[fail_cam['severity'] == (failure_sev - 1)] if failure_sev > 0 else pd.DataFrame()

            if len(cam_at_fail) == 0:
                continue

            performance_cam_alignment.setdefault(corruption, {})
            alignment_key = f"failure_at_{failure_sev}"

            entry = {
                'failure_severity': int(failure_sev),
                'failure_severity_definition': 'First miss occurrence severity (unified)',
                'image_id': str(fail_image_id),
                'cam_at_failure': {},
                'cam_at_prev': {},
                'cam_delta': {}
            }

            # CAM values at failure & prev (FIXED INDENTATION)
            for metric_name in cam_metrics_list:
                if metric_name in cam_at_fail.columns:
                    cam_fail_val = float(cam_at_fail[metric_name].mean())
                else:
                    cam_fail_val = None

                if len(cam_at_prev) > 0 and metric_name in cam_at_prev.columns:
                    cam_prev_val = float(cam_at_prev[metric_name].mean())
                else:
                    cam_prev_val = None

                entry['cam_at_failure'][metric_name] = cam_fail_val
                entry['cam_at_prev'][metric_name] = cam_prev_val
                entry['cam_delta'][metric_name] = (cam_fail_val - cam_prev_val) if (cam_fail_val is not None and cam_prev_val is not None) else None

            # Performance at failure severity K (must match Table 2 aggregation rules)
            perf_entry = {
                'failure_severity': int(failure_sev),
                'miss_rate_at_K': None,
                'avg_score_at_K': None,
                'avg_iou_at_K': None,
                'n_objects_total_at_K': 0,
                'n_matches_at_K': 0,
                'n_miss_at_K': 0,
                'note': 'Same aggregation as Table 2: avg_score/avg_iou from matched cases (miss=0) only'
            }

            if len(corr_det) > 0 and 'severity' in corr_det.columns:
                fail_perf_all = corr_det[corr_det['severity'] == failure_sev]
                if len(fail_perf_all) > 0 and 'miss' in fail_perf_all.columns:
                    n_objects_total_at_K = len(fail_perf_all)
                    n_miss_at_K = int(fail_perf_all['miss'].sum())
                    n_matches_at_K = n_objects_total_at_K - n_miss_at_K
                    miss_rate_at_K = float(n_miss_at_K / n_objects_total_at_K) if n_objects_total_at_K > 0 else 0.0

                    matched_df = fail_perf_all[fail_perf_all['miss'] == 0]
                    avg_score_at_K = float(matched_df['score'].mean()) if 'score' in matched_df.columns and len(matched_df) > 0 else None
                    avg_iou_at_K = float(matched_df['iou'].mean()) if 'iou' in matched_df.columns and len(matched_df) > 0 else None

                    perf_entry.update({
                        'miss_rate_at_K': miss_rate_at_K,
                        'avg_score_at_K': avg_score_at_K,
                        'avg_iou_at_K': avg_iou_at_K,
                        'n_objects_total_at_K': int(n_objects_total_at_K),
                        'n_matches_at_K': int(n_matches_at_K),
                        'n_miss_at_K': int(n_miss_at_K),
                    })

            entry['performance_at_failure'] = perf_entry
            performance_cam_alignment[corruption][alignment_key] = entry

        summarized['performance_cam_alignment'] = performance_cam_alignment

        # ---- lead_lag_analysis (Table 7) (FIXED dict init + indentation)
        # CRITICAL: Only include corruptions that have actual detection records
        lead_lag_analysis: Dict[str, Dict] = {}

        CAM_CHANGE_THRESHOLDS = {
            'center_shift': 0.01,
            'activation_spread': 0.02,
            'energy_in_bbox': 0.05,
            'entropy': 0.5
        }

        for corruption in all_expected_corruptions:
            # CRITICAL: Only process corruptions that have actual detection records
            if corruption not in available_corruptions_from_det:
                continue  # Skip corruptions without detection data
            
            corr_cam = cam_df[cam_df['corruption'] == corruption].copy() if 'corruption' in cam_df.columns else pd.DataFrame()
            if len(corr_cam) == 0 or 'severity' not in corr_cam.columns:
                continue

            unified_failure_sev = get_failure_severity_for_corruption(fail_df, corruption, det_df)
            if unified_failure_sev is None:
                continue

            failure_sev = int(unified_failure_sev)

            cam_at_fail_sev = corr_cam[corr_cam['severity'] == failure_sev]
            if len(cam_at_fail_sev) == 0:
                continue

            fail_image_id = cam_at_fail_sev['image_id'].iloc[0] if 'image_id' in cam_at_fail_sev.columns else None
            fail_class_id = cam_at_fail_sev['class_id'].iloc[0] if 'class_id' in cam_at_fail_sev.columns else None
            if fail_image_id is None:
                continue

            if pd.notna(fail_class_id):
                fail_cam = corr_cam[(corr_cam.get('image_id') == fail_image_id) & (corr_cam.get('class_id') == fail_class_id)].copy()
            else:
                fail_cam = corr_cam[corr_cam.get('image_id') == fail_image_id].copy()

            cam_at_prev = fail_cam[fail_cam['severity'] == (failure_sev - 1)] if failure_sev > 0 else pd.DataFrame()
            cam_at_fail = fail_cam[fail_cam['severity'] == failure_sev]

            if len(cam_at_prev) == 0 or len(cam_at_fail) == 0:
                continue

            lead_lag_analysis.setdefault(corruption, {})
            analysis_key = f"failure_at_{failure_sev}"

            entry = {
                'failure_severity': int(failure_sev),
                'failure_severity_definition': 'First miss occurrence severity (unified)',
                'cam_changes': {}
            }

            for metric_name in cam_metrics_list:
                if metric_name not in cam_at_prev.columns or metric_name not in cam_at_fail.columns:
                    entry['cam_changes'][metric_name] = {
                        'value_at_prev': None,
                        'value_at_failure': None,
                        'delta': None,
                        'threshold': float(CAM_CHANGE_THRESHOLDS.get(metric_name, 0.01)),
                        'is_significant_change': None,
                        'change_before_failure': None,
                        'change_at_failure': None,
                        'note': 'Metric not available in CAM data'
                    }
                    continue

                prev_val = float(cam_at_prev[metric_name].mean())
                fail_val = float(cam_at_fail[metric_name].mean())
                delta = float(fail_val - prev_val)

                threshold = float(CAM_CHANGE_THRESHOLDS.get(metric_name, 0.01))
                is_significant_change = bool(abs(delta) >= threshold)

                entry['cam_changes'][metric_name] = {
                    'value_at_prev': prev_val,
                    'value_at_failure': fail_val,
                    'delta': delta,
                    'threshold': threshold,
                    'is_significant_change': is_significant_change,
                    'change_before_failure': bool(abs(delta) < threshold),
                    'change_at_failure': bool(abs(delta) >= threshold),
                }

            lead_lag_analysis[corruption][analysis_key] = entry

        summarized['lead_lag_analysis'] = lead_lag_analysis

        # ---- cross_corruption_pattern (Table 8)
        # CRITICAL: Only include corruptions that have actual detection records
        cross_corruption_pattern: Dict[str, Dict] = {m: {} for m in cam_metrics_list}

        for metric_name in cam_metrics_list:
            for corruption in all_expected_corruptions:
                # CRITICAL: Only process corruptions that have actual detection records
                if corruption not in available_corruptions_from_det:
                    cross_corruption_pattern[metric_name][corruption] = "no_data"
                    continue
                
                corr_info = cam_pattern_summary.get(corruption, {})
                if corr_info.get('all_metrics_missing', False):
                    cross_corruption_pattern[metric_name][corruption] = "no_data"
                    continue
                if metric_name not in corr_info:
                    cross_corruption_pattern[metric_name][corruption] = "no_data"
                    continue

                delta = corr_info[metric_name].get('delta', None)
                if delta is None:
                    cross_corruption_pattern[metric_name][corruption] = "no_data"
                elif abs(delta) < 1e-10:
                    cross_corruption_pattern[metric_name][corruption] = "no_change"
                else:
                    if not corr_info[metric_name].get('monotonic', False):
                        cross_corruption_pattern[metric_name][corruption] = "non_monotonic_change"
                    else:
                        direction = corr_info[metric_name].get('monotonic_direction', 'none')
                        cross_corruption_pattern[metric_name][corruption] = "monotonic_increase" if direction == 'increase' else ("monotonic_decrease" if direction == 'decrease' else "non_monotonic_change")

        summarized['cross_corruption_pattern'] = cross_corruption_pattern

        # ---- final_summary (RQ1)
        rq1_evidence = {
            'corruptions_with_data': [],
            'corruptions_without_data': [],
            'monotonic_patterns': {},
            'significant_cam_deltas': {},
            'sufficient_evidence': False
        }

        for corruption in all_expected_corruptions:
            # CRITICAL: Only process corruptions that have actual detection records
            if corruption not in available_corruptions_from_det:
                rq1_evidence['corruptions_without_data'].append(corruption)
                continue
            
            corr_info = cam_pattern_summary.get(corruption, {})
            if corr_info.get('all_metrics_missing', False) or len(corr_info) == 0:
                rq1_evidence['corruptions_without_data'].append(corruption)
                continue

            rq1_evidence['corruptions_with_data'].append(corruption)
            rq1_evidence['monotonic_patterns'][corruption] = {}
            rq1_evidence['significant_cam_deltas'][corruption] = {}

            for metric_name in cam_metrics_list:
                if metric_name not in corr_info:
                    continue
                rq1_evidence['monotonic_patterns'][corruption][metric_name] = bool(corr_info[metric_name].get('monotonic', False))

                # use performance_cam_alignment deltas at failure
                if corruption in performance_cam_alignment:
                    for _, alignment_data in performance_cam_alignment[corruption].items():
                        delta = alignment_data.get('cam_delta', {}).get(metric_name, None)
                        if delta is None:
                            continue
                        thr = float(CAM_CHANGE_THRESHOLDS.get(metric_name, 0.01))
                        if abs(delta) >= thr:
                            rq1_evidence['significant_cam_deltas'][corruption][metric_name] = float(delta)
                            break

        corruptions_with_data = rq1_evidence['corruptions_with_data']
        if len(corruptions_with_data) >= 2:
            for metric_name in cam_metrics_list:
                monotonic_count = sum(1 for c in corruptions_with_data if rq1_evidence['monotonic_patterns'].get(c, {}).get(metric_name, False))
                significant_delta_count = sum(1 for c in corruptions_with_data if metric_name in rq1_evidence['significant_cam_deltas'].get(c, {}))
                if monotonic_count >= 2 or significant_delta_count >= 2:
                    rq1_evidence['sufficient_evidence'] = True
                    break

        final_summary = {
            'rq1_evidence': rq1_evidence,
            'rq1_answer': "Yes" if rq1_evidence['sufficient_evidence'] else ("Partial" if len(corruptions_with_data) >= 1 else "No"),
        }

        summarized['final_summary'] = final_summary

    else:
        # If no CAM metrics or no detection records, keep these empty
        summarized['cam_metrics_by_corruption_severity'] = {}
        summarized['cam_pattern_summary'] = {}
        summarized['performance_cam_alignment'] = {}
        summarized['lead_lag_analysis'] = {}
        summarized['cross_corruption_pattern'] = {}
        summarized['final_summary'] = {}

    # Keep gradcam error summary (optional)
    summarized['gradcam_errors'] = metrics.get('gradcam_errors', [])[:200]

    return summarized

import numpy as np

def normalize_for_json(obj):
    """
    Recursively convert dictionary keys to str
    and numpy types to native Python types.
    """
    if isinstance(obj, dict):
        return {
            str(k): normalize_for_json(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, (list, tuple)):
        return [normalize_for_json(v) for v in obj]

    elif isinstance(obj, np.generic):
        return obj.item()

    else:
        return obj


def generate_report_with_llm(config: Dict, metrics: Dict) -> str:
    """Generate report using LLM.

    Args:
        config: Configuration dictionary
        metrics: Dictionary with all metrics

    Returns:
        Generated report as markdown string
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")

    client = OpenAI(api_key=api_key)

    summarized_metrics = normalize_for_json(summarize_metrics(metrics))

    prompt = f"""You are a research assistant writing an experiment report. Generate a comprehensive markdown report based on the provided metrics data.

IMPORTANT RULES (CRITICAL - 논문 안전 수치만):
1. DO NOT invent or make up any numbers. Only use the values provided in the metrics data.
2. If a metric is missing, explicitly state "N/A" or "Data missing" (NOT "Not computed").
3. Be precise and factual. DO NOT add speculation, inference, prediction, or causality claims.
4. Use the exact numbers from the data, rounded appropriately (e.g., 2-3 decimal places for percentages).
5. CRITICAL: All tables must include count/sample size columns (n_objects_total, n_cam_frames, n_frames_valid, etc.).
6. CRITICAL: DO NOT use words like "predict", "precursor", "lead", "foreshadow" - only report observed changes.
7. CRITICAL: For CAM analysis, report ONLY observed values:
   - "At failure severity K, center_shift = X (delta from severity K-1 = Y)"
8. CRITICAL: Performance values in Table 6 must EXACTLY match Table 2 (same corruption, same severity).
9. CRITICAL: Report aggregation scope in notes (e.g., "avg_score/avg_iou from matched cases only").
10. CRITICAL: For mAP, report valid subset statistics (n_frames_valid, valid_ratio) and mark as "N/A (insufficient valid frames)" if n_frames_valid < 10.
11. CRITICAL: Table 4 (Failure Summary) must show corruption-specific event counts:
    - Each corruption (fog, lowlight, motion_blur) has DIFFERENT event counts
    - DO NOT sum across corruptions - report by_corruption_detail values for each corruption separately
    - miss_events, score_drop_events, iou_drop_events are per-corruption, NOT totals
    - DO NOT include "Unified Failure Severity" column in the Failure Summary table
    - Format: | Corruption | Total Events | Miss Events | Score Drop Events | IoU Drop Events |
12. CRITICAL: Table 2 (Detection Summary / Performance Metrics):
    - REQUIRES detection_records.csv to exist (created by scripts/03_detect_tiny_objects_timeseries.py)
    - Check detection_summary.has_data field in the metrics data
    - If detection_summary.has_data is False or missing:
      - DO NOT create Table 2 (Detection Summary / Performance Metrics)
      - DO NOT create any table with "Performance Metrics" or "Detection Summary" in the title
      - Instead, output ONLY this warning at the top: "⚠️ PERFORMANCE METRICS TABLE SKIPPED: detection_records.csv not found or empty. Please run scripts/03_detect_tiny_objects_timeseries.py to generate this file."
      - DO NOT create Table 6 or any other performance table if detection_summary.has_data is False
    - If detection_summary.has_data is True:
      - Create Table 2 with data from detection_records
      - n_objects_total comes from detection_records (sum of all detection records)
      - Report n_objects_by_corruption_severity if available
      - NOT from metrics_dataset.csv pred_count
    - IMPORTANT: Table 6 (if it exists) should be from dataset metrics, NOT from detection_records. 
      - Table 6 is separate from Table 2 and should only be created if dataset metrics exist
      - If detection_summary.has_data is False, you can still create Table 6 from dataset metrics, but DO NOT call it "Performance Metrics" or "Detection Summary"
13. CRITICAL: Grad-CAM Analysis Scope (Methods/Limitations):
    - CAM computation is performed only for failure events that satisfy:
      1) Detection records exist (tiny_records_timeseries.csv)
      2) Failure event detected (failure_events.csv)
      3) Successful CAM generation (no shape/device errors)
    - If a corruption has failure_events but 0 CAM records, explicitly state:
      "CAM generation failed for this corruption (see gradcam_errors.csv for details)"
    - Do NOT claim "no failure events" if failure_events exist but CAM records are 0
14. CRITICAL: Score/IoU Drop Definition (Methods):
    - Baseline: severity 0 for EACH corruption (corruption-specific baseline)
    - Threshold: score_drop_threshold and iou_drop_threshold (common across corruptions)
    - Detection: First severity where (baseline_score - current_score) >= threshold
    - This ensures fair comparison across different corruption types
15. CRITICAL: Failure Severity Range for CAM (Methods):
    - CAM is computed for severity 0 (baseline) through failure_severity (inclusive)
    - Each severity's CAM is computed INDEPENDENTLY (not cumulative)
    - CAM metrics compare each severity's CAM to baseline (severity 0)
    - This tracks CAM distribution changes as corruption severity increases up to failure point
16. CRITICAL: CAM Metrics Table MUST include failure_severity column:
    - The "CAM Metrics by Corruption Severity" table MUST have a "failure_severity" column
    - failure_severity is the failure severity for the ENTIRE corruption (same value for all severities of the same corruption)
    - failure_severity comes from cam_metrics_by_corruption_severity[corruption][severity]['failure_severity']
    - If n_cam_frames=0, still show failure_severity if it exists (do NOT mark as N/A)
    - Format: | Corruption | Severity | n_cam_frames | failure_severity | Energy in BBox Mean | Activation Spread Mean | Entropy Mean | Center Shift Mean |
    - Example: If lowlight has failure_severity=1, ALL lowlight severity rows (0-4) should show failure_severity=1
    - DO NOT omit the failure_severity column even if some values are None

Experiment Configuration:
- Seed: {config['seed']}
- Models: {', '.join(config['models'].keys())}
- Corruptions: {', '.join(config['corruptions']['types'])}
- Severities: {config['corruptions']['severities']}
- Tiny object threshold: {config['tiny_objects']['area_threshold']} pixels²

Metrics Data (Summarized, JSON format - compact):
{json.dumps(summarized_metrics, indent=None, separators=(',', ':'))}

Generate a markdown report with TABLES ONLY. All text descriptions should be commented out with <!-- -->.
"""
    response = client.chat.completions.create(
        model=config['llm_report']['model'],
        messages=[
            {"role": "system", "content": "You are a scientific report writer. Generate accurate, factual reports based only on provided data."},
            {"role": "user", "content": prompt}
        ],
        temperature=config['llm_report']['temperature'],
        max_tokens=config['llm_report']['max_tokens']
    )

    report = response.choices[0].message.content
    return report


def save_report(report: str, output_path: Path):
    """Save report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {output_path}")
