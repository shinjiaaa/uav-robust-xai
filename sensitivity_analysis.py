"""Sensitivity analysis for alignment calculation with different settings."""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_metrics_for_alignment():
    """Load metrics needed for alignment analysis."""
    results_root = Path('results')

    metrics = {}

    # Risk events (performance axis)
    risk_events_csv = results_root / "failure_events.csv"  # Assuming risk_events are in failure_events.csv
    if risk_events_csv.exists():
        df = pd.read_csv(risk_events_csv)
        # Create object_uid from image_id and class_id
        df['object_uid'] = df['image_id'].astype(str) + '_obj_' + df['class_id'].astype(str)
        # Add failure_event_id if not present
        if 'failure_event_id' not in df.columns:
            df['failure_event_id'] = df.index.astype(str)
        # Add start_severity (earliest)
        df['start_severity'] = df[['first_miss_severity', 'score_drop_severity', 'iou_drop_severity']].min(axis=1, skipna=True)
        metrics['risk_events'] = df.to_dict('records')
    else:
        metrics['risk_events'] = []

    # CAM records (cognition axis)
    cam_records_csv = results_root / "cam_records.csv"
    if cam_records_csv.exists():
        metrics['cam_records'] = pd.read_csv(cam_records_csv).to_dict('records')
    else:
        metrics['cam_records'] = []

    return metrics


def calculate_alignment_sensitivity(metrics, settings):
    """Calculate alignment with given settings."""
    risk_events = metrics.get('risk_events', [])
    cam_records = metrics.get('cam_records', [])

    if len(risk_events) == 0:
        return {}

    risk_events_df = pd.DataFrame(risk_events)
    cam_records_df = pd.DataFrame(cam_records) if len(cam_records) > 0 else pd.DataFrame()

    # For [C] and [D], process each failure_type separately
    if settings['perf_start_mode'] == 'separate':
        # Don't deduplicate, treat each failure_type as separate event
        pass  # Keep all rows
    else:
        # Deduplicate for earliest mode
        if 'corruption' in risk_events_df.columns and 'object_uid' in risk_events_df.columns and 'failure_type' in risk_events_df.columns:
            risk_events_df = risk_events_df.drop_duplicates(subset=['corruption', 'object_uid', 'failure_type'], keep='first').copy()

    # For [C] and [D], calculate perf_start based on mode
    if settings['perf_start_mode'] == 'separate':
        # Calculate earliest for each event
        def get_earliest_severity(row):
            severities = []
            if pd.notna(row.get('first_miss_severity')):
                severities.append(int(row['first_miss_severity']))
            if pd.notna(row.get('score_drop_severity')):
                severities.append(int(row['score_drop_severity']))
            if pd.notna(row.get('iou_drop_severity')):
                severities.append(int(row['iou_drop_severity']))
            return min(severities) if severities else None

        risk_events_df = risk_events_df.copy()
        risk_events_df['earliest_severity'] = risk_events_df.apply(get_earliest_severity, axis=1)
        # For separate mode, we'll process each failure_type separately later

    # CAM metrics for ensemble
    NEW_FOUR = ['bbox_center_activation_distance', 'peak_bbox_distance', 'activation_spread', 'ring_energy_ratio']
    OLD_FOUR = ['energy_in_bbox', 'activation_spread', 'entropy', 'center_shift']
    has_new = len(cam_records_df) > 0 and all(c in cam_records_df.columns for c in NEW_FOUR)
    has_old = len(cam_records_df) > 0 and all(c in cam_records_df.columns for c in OLD_FOUR)
    if has_new:
        CAM_METRICS_FOR_ENSEMBLE = NEW_FOUR
    elif has_old:
        CAM_METRICS_FOR_ENSEMBLE = OLD_FOUR
    else:
        CAM_METRICS_FOR_ENSEMBLE = ['activation_spread']

    REPRESENTATIVE_CAM_METRIC = 'activation_spread'
    DELTA_THRESHOLD = settings.get('delta_threshold', 0.2)  # default 0.2
    USE_CAM_ENSEMBLE = len(CAM_METRICS_FOR_ENSEMBLE) >= 2
    MIN_METRICS_CHANGED = settings['min_metrics_changed']

    alignment_analysis = {}

    def _cam_change_sev_delta(event_cam_primary, severity_order, baseline_sev=0):
        """Return (cam_change_sev, metric_used, change_type) using delta threshold."""
        baseline_cam = event_cam_primary[event_cam_primary['severity'] == baseline_sev]
        if len(baseline_cam) < 1:
            return None, None, None
        cam_change_sev = None
        metric_used = None
        change_type = None
        for sev in severity_order:
            if sev == baseline_sev:
                continue
            sev_cam = event_cam_primary[event_cam_primary['severity'] == sev]
            if len(sev_cam) == 0:
                continue
            n_changed = 0
            collapse_detected = False
            for m in CAM_METRICS_FOR_ENSEMBLE:
                if m not in baseline_cam.columns or m not in sev_cam.columns:
                    continue
                b_val = baseline_cam[m].dropna().mean()
                s_val = sev_cam[m].dropna().mean()
                if pd.notna(b_val) and pd.notna(s_val):
                    if s_val == 0 or abs(s_val) < 1e-3:
                        collapse_detected = True
                    delta = abs(s_val - b_val) / max(abs(b_val), 1e-6)
                    if delta >= DELTA_THRESHOLD:  # threshold
                        n_changed += 1
            if n_changed >= MIN_METRICS_CHANGED:
                cam_change_sev = sev
                metric_used = 'ensemble'
                if collapse_detected:
                    change_type = 'collapse'
                else:
                    change_type = 'gradual'
                break
        return (cam_change_sev, metric_used, change_type)

    # Process events
    for _, risk_event in risk_events_df.iterrows():
        event_id = risk_event.get('failure_event_id', '')
        corruption = risk_event.get('corruption', '')
        object_uid = risk_event.get('object_uid', '')
        failure_type = risk_event.get('failure_type', 'unknown')

        # Determine start_severity based on mode
        if settings['perf_start_mode'] == 'earliest':
            start_severity = int(risk_event.get('start_severity', -1))
        elif settings['perf_start_mode'] == 'separate':
            if failure_type == 'miss':
                start_severity = risk_event.get('first_miss_severity')
            elif failure_type == 'score_drop':
                start_severity = risk_event.get('score_drop_severity')
            elif failure_type == 'iou_drop':
                start_severity = risk_event.get('iou_drop_severity')
            else:
                start_severity = risk_event.get('earliest_severity')  # fallback
            if pd.isna(start_severity):
                continue
            start_severity = int(start_severity)
        else:
            start_severity = int(risk_event.get('start_severity', -1))

        if start_severity < 0:
            continue

        # Find CAM records
        event_cam = pd.DataFrame()
        if 'failure_event_id' in cam_records_df.columns:
            event_cam = cam_records_df[cam_records_df['failure_event_id'] == event_id].copy()
        if len(event_cam) == 0 and 'object_id' in cam_records_df.columns:
            event_cam = cam_records_df[cam_records_df['object_id'] == object_uid].copy()
        if len(event_cam) == 0:
            try:
                if '_cls' in object_uid:
                    image_id_from_uid = object_uid.split('_cls')[0].rsplit('_obj', 1)[0] if '_obj' in object_uid else object_uid.split('_cls')[0]
                    class_id_from_uid = int(object_uid.split('_cls')[1])
                else:
                    image_id_from_uid = object_uid
                    class_id_from_uid = None
                if 'image_id' in cam_records_df.columns and 'class_id' in cam_records_df.columns:
                    event_cam = cam_records_df[
                        (cam_records_df['image_id'] == image_id_from_uid) &
                        (cam_records_df['class_id'] == class_id_from_uid) &
                        (cam_records_df['corruption'] == corruption)
                    ].copy()
            except Exception:
                pass

        if len(event_cam) > 0:
            if 'severity' in event_cam.columns:
                event_cam = event_cam.copy()
                event_cam['severity'] = pd.to_numeric(event_cam['severity'], errors='coerce').fillna(-1).astype(int)
            event_cam = event_cam[
                (event_cam['corruption'] == corruption) &
                (event_cam['severity'] <= start_severity)
            ].copy()

        cam_change_severity = None
        cam_change_metric = None

        if len(event_cam) > 0:
            event_cam_primary = event_cam[event_cam.get('layer_role', 'primary') == 'primary'].copy()
            if len(event_cam_primary) > 0:
                severity_order = sorted(int(s) for s in event_cam_primary['severity'].dropna().unique() if pd.notna(s))
                cam_change_severity, cam_change_metric, change_type = _cam_change_sev_delta(
                    event_cam_primary, severity_order, baseline_sev=0
                )
                if cam_change_severity is not None and cam_change_severity > start_severity:
                    cam_change_severity = start_severity

        # Determine alignment
        if cam_change_severity is not None:
            lead_steps = int(start_severity) - int(cam_change_severity)
            if lead_steps > 0:
                alignment = 'lead'
            elif lead_steps == 0:
                alignment = 'coincident'
            else:
                alignment = 'lag'
        else:
            alignment = 'unavailable' if settings['separate_unavailable'] else None
            lead_steps = None

        if corruption not in alignment_analysis:
            alignment_analysis[corruption] = []

        alignment_analysis[corruption].append({
            'failure_event_id': event_id,
            'object_uid': object_uid,
            'corruption': corruption,
            'failure_type': failure_type,
            'performance_start_severity': start_severity,
            'cam_change_severity': cam_change_severity,
            'cam_change_metric': cam_change_metric,
            'change_type': change_type,
            'alignment': alignment,
            'lead_steps': lead_steps,
        })

    return alignment_analysis


def analyze_join_failures(risk_events_df, cam_records_df):
    """Analyze reasons for unmatched joins."""
    # Attempt merge
    merged = risk_events_df.merge(cam_records_df, left_on='object_uid', right_on='object_id', how='left', suffixes=('_risk', '_cam'))
    
    total = len(risk_events_df)
    unmatched_count = len(merged[merged['object_id'].isna()])
    matched_count = merged['object_id'].notna().sum()
    duplicate_matches = len(merged) - len(risk_events_df)
    
    print(f"Total: {total}, Matched: {matched_count}, Unmatched: {unmatched_count}")
    reasons = []
    
    for _, row in merged[merged['object_id'].isna()].iterrows():
        uid = row['object_uid']
        corruption = row['corruption']
        failure_type = row['failure_type']
        
        # Check if object_id exists in cam_records with different format
        if cam_records_df['object_id'].str.contains(uid.replace('_obj_', '_cls')).any():
            reason = 'format_difference (_obj_ vs _cls)'
        elif not cam_records_df[(cam_records_df['image_id'] == row['image_id']) & (cam_records_df['class_id'] == row['class_id'])].empty:
            reason = 'corruption_mismatch'
        elif not cam_records_df[cam_records_df['object_id'].str.startswith(row['image_id'])].empty:
            reason = 'severity_missing_in_cam'
        else:
            reason = 'no_matching_object_in_cam'
        
        reasons.append({
            'reason': reason,
            'object_uid': uid,
            'corruption': corruption,
            'failure_type': failure_type
        })
    
    if unmatched_count == 0:
        result = pd.DataFrame(columns=['reason', 'count', 'percentage', 'sample_object_uid'])
    else:
        # Count reasons
        reason_counts = pd.DataFrame(reasons).groupby('reason').size().reset_index(name='count')
        reason_counts['percentage'] = (reason_counts['count'] / unmatched_count * 100).round(1)
        
        # Samples
        samples = pd.DataFrame(reasons).groupby('reason').first().reset_index()[['reason', 'object_uid']]
        
        result = reason_counts.merge(samples, on='reason')
        result = result[['reason', 'count', 'percentage', 'object_uid']]
        result.columns = ['reason', 'count', 'percentage', 'sample_object_uid']
    
    return result, matched_count, unmatched_count, total


def check_aggregation_unit(alignment_analysis, risk_events_df):
    """Check if alignment aggregation is by event or row unit."""
    results = []
    for corruption, events in alignment_analysis.items():
        total_unique_events = len(set([e['object_uid'] + '_' + e['failure_type'] for e in events]))
        total_rows_used = len(events)
        duplication_ratio = total_rows_used / total_unique_events if total_unique_events > 0 else 0
        results.append({
            'corruption': corruption,
            'total_unique_events': total_unique_events,
            'total_rows_used': total_rows_used,
            'duplication_ratio': round(duplication_ratio, 2)
        })
    return pd.DataFrame(results)


def reaggregate_by_event_unit(alignment_analysis, cam_records_df, risk_events_df, method):
    """Reaggregate alignment by event unit with single cam_change_severity per event."""
    reaggregated = {}
    
    for corruption, events in alignment_analysis.items():
        reaggregated[corruption] = []
        for event in events:
            uid = event['object_uid']
            failure_type = event['failure_type']
            perf_start = event['performance_start_severity']
            
            # Get all CAM data for this event
            cam_data = cam_records_df[cam_records_df['object_id'] == uid].copy()
            if cam_data.empty:
                cam_change_sev = None
            else:
                # Filter by corruption and severity <= perf_start
                cam_data = cam_data[(cam_data['corruption'] == corruption) & (cam_data['severity'] <= perf_start)]
                if cam_data.empty:
                    cam_change_sev = None
                else:
                    # Aggregate cam_change_severity
                    if method == 'earliest':
                        # Earliest severity where change detected
                        severity_order = sorted(cam_data['severity'].unique())
                        cam_change_sev = None
                        for sev in severity_order:
                            if sev > 0:  # baseline is 0
                                # Simple check: if any metric changed (simplified)
                                cam_change_sev = sev
                                break
                    elif method == 'majority':
                        # Majority of changed metrics severity (simplified)
                        cam_change_sev = 1  # placeholder
                    elif method == 'median':
                        # Median severity of changes
                        cam_change_sev = 1  # placeholder
                    
            # Recalculate alignment
            if cam_change_sev is not None:
                lead_steps = perf_start - cam_change_sev
                if lead_steps > 0:
                    alignment = 'lead'
                elif lead_steps == 0:
                    alignment = 'coincident'
                else:
                    alignment = 'lag'
            else:
                alignment = 'unavailable'
            
            reaggregated[corruption].append({
                'object_uid': uid,
                'failure_type': failure_type,
                'performance_start_severity': perf_start,
                'cam_change_severity': cam_change_sev,
                'alignment': alignment
            })
    
    return reaggregated


def calculate_alignment_stats(alignment_data):
    """Calculate lead/coincident/lag/unavailable percentages."""
    total = sum(len(events) for events in alignment_data.values())
    lead = sum(len([e for e in events if e['alignment'] == 'lead']) for events in alignment_data.values())
    coincident = sum(len([e for e in events if e['alignment'] == 'coincident']) for events in alignment_data.values())
    lag = sum(len([e for e in events if e['alignment'] == 'lag']) for events in alignment_data.values())
    unavailable = sum(len([e for e in events if e['alignment'] == 'unavailable']) for events in alignment_data.values())
    
    return {
        'lead': lead / total * 100 if total > 0 else 0,
        'coincident': coincident / total * 100 if total > 0 else 0,
        'lag': lag / total * 100 if total > 0 else 0,
        'unavailable': unavailable / total * 100 if total > 0 else 0
    }


def get_severity_1_examples(sev1_analysis, limit=10):
    """Get 10 examples of severity 1 changes."""
    examples = sev1_analysis[:limit] if len(sev1_analysis) > limit else sev1_analysis
    return pd.DataFrame(examples)


def analyze_std_distribution(cam_records_df):
    """Analyze std distribution for baseline metrics."""
    CAM_METRICS = ['bbox_center_activation_distance', 'peak_bbox_distance', 'activation_spread', 'ring_energy_ratio']
    
    results = []
    for metric in CAM_METRICS:
        if metric in cam_records_df.columns:
            baseline_data = cam_records_df[cam_records_df['severity'] == 0][metric].dropna()
            if len(baseline_data) > 0:
                std_val = baseline_data.std()
                count = len(baseline_data)
                results.append({
                    'metric': metric,
                    'std': round(std_val, 6) if pd.notna(std_val) else 'NaN',
                    'count': count
                })
    
    return pd.DataFrame(results)


def analyze_low_std_cases(cam_records_df, threshold=1e-6):
    """Count cases where std < threshold."""
    CAM_METRICS = ['bbox_center_activation_distance', 'peak_bbox_distance', 'activation_spread', 'ring_energy_ratio']
    
    results = []
    for metric in CAM_METRICS:
        if metric in cam_records_df.columns:
            baseline_data = cam_records_df[cam_records_df['severity'] == 0][metric].dropna()
            if len(baseline_data) > 0:
                std_val = baseline_data.std()
                low_std_count = 1 if pd.notna(std_val) and std_val < threshold else 0
                results.append({
                    'metric': metric,
                    'std': round(std_val, 6) if pd.notna(std_val) else 'NaN',
                    'count': low_std_count
                })
    
    return pd.DataFrame(results)


def robustness_check(metrics, settings_list):
    """Check robustness with different z and min_metrics combinations."""
    results = {}
    
    for settings in settings_list:
        key = f"delta=0.2, min={settings['min_metrics_changed']}"
        alignment_analysis = calculate_alignment_sensitivity(metrics, settings)
        
        total_lead = sum(len([e for e in events if e['alignment'] == 'lead']) for events in alignment_analysis.values())
        total_coincident = sum(len([e for e in events if e['alignment'] == 'coincident']) for events in alignment_analysis.values())
        total_lag = sum(len([e for e in events if e['alignment'] == 'lag']) for events in alignment_analysis.values())
        total_unavailable = sum(len([e for e in events if e['alignment'] == 'unavailable']) for events in alignment_analysis.values())
        total = total_lead + total_coincident + total_lag + total_unavailable
        
        results[key] = {
            'lead': total_lead / total * 100 if total > 0 else 0,
            'coincident': total_coincident / total * 100 if total > 0 else 0,
            'lag': total_lag / total * 100 if total > 0 else 0,
            'unavailable': total_unavailable / total * 100 if total > 0 else 0
        }
    
    return results


def main():
    metrics = load_metrics_for_alignment()
    risk_events_df = pd.DataFrame(metrics['risk_events'])
    cam_records_df = pd.DataFrame(metrics['cam_records'])

    # CAM metrics 설정
    NEW_FOUR = ['bbox_center_activation_distance', 'peak_bbox_distance', 'activation_spread', 'ring_energy_ratio']
    OLD_FOUR = ['energy_in_bbox', 'activation_spread', 'entropy', 'center_shift']
    has_new = len(cam_records_df) > 0 and all(c in cam_records_df.columns for c in NEW_FOUR)
    has_old = len(cam_records_df) > 0 and all(c in cam_records_df.columns for c in OLD_FOUR)
    if has_new:
        CAM_METRICS_FOR_ENSEMBLE = NEW_FOUR
    elif has_old:
        CAM_METRICS_FOR_ENSEMBLE = OLD_FOUR
    else:
        CAM_METRICS_FOR_ENSEMBLE = ['activation_spread']

    # 1. Join 실패 원인 분석
    join_analysis, matched, unmatched, total = analyze_join_failures(risk_events_df, cam_records_df)
    print("1) Join 실패 원인 표")
    print(join_analysis.to_string(index=False))
    print()

    # Delta 분석
    print("Delta 계산 코드:")
    print("delta = abs(value - baseline) / max(abs(baseline), 1e-6)")
    print("change if delta >= threshold and min_metrics_changed satisfied")
    print()

    # Alignment 계산 (threshold 0.2)
    settings = {'min_metrics_changed': 1, 'separate_unavailable': True, 'perf_start_mode': 'separate', 'delta_threshold': 0.2}
    alignment_analysis = calculate_alignment_sensitivity(metrics, settings)

    # Change type별 alignment
    all_events = [e for events in alignment_analysis.values() for e in events]
    collapse_events = [e for e in all_events if e.get('change_type') == 'collapse']
    gradual_events = [e for e in all_events if e.get('change_type') == 'gradual']

    def calc_stats(events):
        lead = len([e for e in events if e['alignment'] == 'lead'])
        coincident = len([e for e in events if e['alignment'] == 'coincident'])
        lag = len([e for e in events if e['alignment'] == 'lag'])
        unavailable = len([e for e in events if e['alignment'] == 'unavailable'])
        total = lead + coincident + lag + unavailable
        return {
            'lead': lead / total * 100 if total > 0 else 0,
            'coincident': coincident / total * 100 if total > 0 else 0,
            'lag': lag / total * 100 if total > 0 else 0,
            'unavailable': unavailable / total * 100 if total > 0 else 0,
            'count': total
        }

    collapse_stats = calc_stats(collapse_events)
    gradual_stats = calc_stats(gradual_events)

    print("Change type별 alignment")
    print("change_type | lead% | coincident% | lag% | count")
    print("-" * 50)
    print(f"collapse    | {collapse_stats['lead']:.1f}  | {collapse_stats['coincident']:.1f}       | {collapse_stats['lag']:.1f}  | {collapse_stats['count']}")
    print(f"gradual     | {gradual_stats['lead']:.1f}  | {gradual_stats['coincident']:.1f}       | {gradual_stats['lag']:.1f}  | {gradual_stats['count']}")
    print()

    # Metric별 collapse 발생
    print("Metric별 collapse 발생")
    collapse_metric_counts = {}
    for e in collapse_events:
        object_uid = e.get('object_uid', '')
        corruption = e.get('corruption', '')
        cam_change_sev = e.get('cam_change_severity')
        if cam_change_sev is not None:
            event_cams = cam_records_df[
                (cam_records_df['object_id'] == object_uid) &
                (cam_records_df['corruption'] == corruption) &
                (cam_records_df['severity'] == cam_change_sev)
            ]
            for m in CAM_METRICS_FOR_ENSEMBLE:
                if m in event_cams.columns:
                    val = event_cams[m].dropna().mean()
                    if pd.notna(val) and (val == 0 or abs(val) < 1e-3):
                        if m not in collapse_metric_counts:
                            collapse_metric_counts[m] = 0
                        collapse_metric_counts[m] += 1
    total_collapse = sum(collapse_metric_counts.values())
    for m, count in collapse_metric_counts.items():
        pct = count / total_collapse * 100 if total_collapse > 0 else 0
        print(f"{m}: {count} ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()