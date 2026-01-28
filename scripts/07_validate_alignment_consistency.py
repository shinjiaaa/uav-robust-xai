"""Validate alignment analysis consistency between detection_records and risk_events.

This script performs 3 critical consistency checks:
(A) Same record sample validation: Compare is_score_drop calculation between 03 and 04
(B) Aggregation consistency check: Compare Table 2 drop rates vs risk_events counts
(C) Table X-detail validation: Verify perf_start_sev matches actual drop condition
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml


def validate_same_record_sample(detection_records_df: pd.DataFrame, risk_events_df: pd.DataFrame):
    """Test (A): Validate that same records have consistent is_score_drop calculation.
    
    Sample 20 random records and verify:
    - pred_score, base_pred_score values match
    - is_score_drop calculation matches between 03 and 04
    """
    print("=" * 60)
    print("Test (A): Same Record Sample Validation")
    print("=" * 60)
    
    # Filter records with matched=1 and base_pred_score > 0
    valid_records = detection_records_df[
        (detection_records_df['matched'] == 1) &
        (detection_records_df['base_pred_score'].notna()) &
        (detection_records_df['base_pred_score'] > 0)
    ].copy()
    
    if len(valid_records) == 0:
        print("[WARN] No valid records found for validation")
        return False
    
    # Sample 20 records (or all if less than 20)
    sample_size = min(20, len(valid_records))
    sample_records = valid_records.sample(n=sample_size, random_state=42)
    
    print(f"\nSampling {sample_size} records for validation...")
    
    # Constants from 04_detect_risk_events.py
    SCORE_DROP_RATIO = 0.5
    
    mismatches = []
    for idx, row in sample_records.iterrows():
        pred_score = row['pred_score']
        base_score = row['base_pred_score']
        is_score_drop_03 = row['is_score_drop']  # From 03
        
        # Recalculate using 04 logic
        is_score_drop_04_calc = 1 if pred_score <= base_score * SCORE_DROP_RATIO else 0
        
        if is_score_drop_03 != is_score_drop_04_calc:
            mismatches.append({
                'object_uid': row.get('object_uid', 'N/A'),
                'corruption': row.get('corruption', 'N/A'),
                'severity': row.get('severity', 'N/A'),
                'pred_score': pred_score,
                'base_pred_score': base_score,
                'is_score_drop_03': is_score_drop_03,
                'is_score_drop_04_calc': is_score_drop_04_calc,
                'threshold': base_score * SCORE_DROP_RATIO,
            })
    
    if len(mismatches) == 0:
        print(f"✅ PASS: All {sample_size} sampled records have consistent is_score_drop calculation")
        return True
    else:
        print(f"❌ FAIL: {len(mismatches)} mismatches found:")
        for m in mismatches[:5]:  # Show first 5
            print(f"  - {m['object_uid']} (corruption={m['corruption']}, sev={m['severity']}): "
                  f"pred={m['pred_score']:.4f}, base={m['base_pred_score']:.4f}, "
                  f"threshold={m['threshold']:.4f}, "
                  f"03={m['is_score_drop_03']}, 04_calc={m['is_score_drop_04_calc']}")
        return False


def validate_aggregation_consistency(detection_records_df: pd.DataFrame, risk_events_df: pd.DataFrame):
    """Test (B): Validate aggregation consistency between Table 2 and risk_events.
    
    Compare:
    - n_score_drop_records(sev=k): Number of records with is_score_drop=1 at severity k
    - n_score_drop_events(sev=k): Number of events where start_severity=k
    """
    print("\n" + "=" * 60)
    print("Test (B): Aggregation Consistency Check")
    print("=" * 60)
    
    # Calculate drop records by severity (Table 2 style)
    drop_records_by_sev = detection_records_df[
        detection_records_df['is_score_drop'] == 1
    ].groupby(['corruption', 'severity']).size().reset_index(name='n_score_drop_records')
    
    # Calculate drop events by start_severity (risk_events style)
    score_drop_events = risk_events_df[risk_events_df['failure_type'] == 'score_drop']
    drop_events_by_sev = score_drop_events.groupby(['corruption', 'start_severity']).size().reset_index(name='n_score_drop_events')
    drop_events_by_sev = drop_events_by_sev.rename(columns={'start_severity': 'severity'})
    
    # Merge for comparison
    comparison = drop_records_by_sev.merge(
        drop_events_by_sev,
        on=['corruption', 'severity'],
        how='outer'
    ).fillna(0)
    
    print("\nComparison: n_score_drop_records (Table 2) vs n_score_drop_events (risk_events)")
    print("-" * 60)
    print(f"{'Corruption':<15} {'Severity':<10} {'Drop Records':<15} {'Drop Events':<15} {'Status':<10}")
    print("-" * 60)
    
    all_consistent = True
    for _, row in comparison.iterrows():
        n_records = int(row['n_score_drop_records'])
        n_events = int(row['n_score_drop_events'])
        
        # Logic: n_records >= n_events (because one event can have multiple records)
        # But both should be > 0 or both = 0 (no extreme mismatch)
        if n_records == 0 and n_events > 0:
            status = "❌ FAIL"
            all_consistent = False
        elif n_records > 0 and n_events == 0:
            status = "❌ FAIL"
            all_consistent = False
        elif n_records > 0 and n_events > 0:
            status = "✅ OK"
        else:
            status = "✅ OK"
        
        print(f"{row['corruption']:<15} {int(row['severity']):<10} {n_records:<15} {n_events:<15} {status:<10}")
    
    if all_consistent:
        print("\n✅ PASS: Aggregation consistency check passed")
        print("   Note: n_records >= n_events is expected (one event can have multiple records)")
    else:
        print("\n❌ FAIL: Extreme mismatches found (0 vs >0)")
    
    return all_consistent


def validate_table_x_detail(detection_records_df: pd.DataFrame, risk_events_df: pd.DataFrame):
    """Test (C): Validate that Table X-detail perf_start_sev matches actual drop condition.
    
    For each score_drop event in risk_events:
    - Check that at perf_start_sev, pred_score <= base_pred_score * 0.5
    - Verify this is the FIRST severity where the condition is met
    """
    print("\n" + "=" * 60)
    print("Test (C): Table X-detail Validation")
    print("=" * 60)
    
    score_drop_events = risk_events_df[risk_events_df['failure_type'] == 'score_drop'].copy()
    
    if len(score_drop_events) == 0:
        print("[WARN] No score_drop events found for validation")
        return True
    
    print(f"\nValidating {len(score_drop_events)} score_drop events...")
    
    SCORE_DROP_RATIO = 0.5
    failures = []
    
    for _, event in score_drop_events.iterrows():
        object_uid = event['object_uid']
        corruption = event['corruption']
        perf_start_sev = event['start_severity']
        
        # Get all records for this object
        object_records = detection_records_df[
            (detection_records_df['object_uid'] == object_uid) &
            (detection_records_df['corruption'] == corruption)
        ].sort_values('severity')
        
        if len(object_records) == 0:
            failures.append({
                'object_uid': object_uid,
                'corruption': corruption,
                'perf_start_sev': perf_start_sev,
                'issue': 'No records found for this object'
            })
            continue
        
        # Get baseline
        baseline_record = object_records[object_records['severity'] == 0]
        if len(baseline_record) == 0:
            failures.append({
                'object_uid': object_uid,
                'corruption': corruption,
                'perf_start_sev': perf_start_sev,
                'issue': 'No baseline (severity 0) record found'
            })
            continue
        
        base_score = baseline_record.iloc[0]['base_pred_score']
        if base_score is None or base_score <= 0:
            failures.append({
                'object_uid': object_uid,
                'corruption': corruption,
                'perf_start_sev': perf_start_sev,
                'issue': f'Invalid baseline score: {base_score}'
            })
            continue
        
        # Check perf_start_sev record
        start_record = object_records[object_records['severity'] == perf_start_sev]
        if len(start_record) == 0:
            failures.append({
                'object_uid': object_uid,
                'corruption': corruption,
                'perf_start_sev': perf_start_sev,
                'issue': f'No record found at severity {perf_start_sev}'
            })
            continue
        
        start_pred_score = start_record.iloc[0]['pred_score']
        start_matched = start_record.iloc[0]['matched']
        threshold = base_score * SCORE_DROP_RATIO
        
        # Verify drop condition at perf_start_sev
        if start_matched != 1:
            failures.append({
                'object_uid': object_uid,
                'corruption': corruption,
                'perf_start_sev': perf_start_sev,
                'issue': f'Not matched at start_severity (matched={start_matched})'
            })
            continue
        
        if start_pred_score > threshold:
            failures.append({
                'object_uid': object_uid,
                'corruption': corruption,
                'perf_start_sev': perf_start_sev,
                'issue': f'Drop condition not met: pred_score={start_pred_score:.4f} > threshold={threshold:.4f}'
            })
            continue
        
        # Verify this is the FIRST severity where condition is met
        for sev in sorted(object_records['severity'].unique()):
            if sev >= perf_start_sev:
                break
            
            sev_record = object_records[object_records['severity'] == sev]
            if len(sev_record) == 0:
                continue
            
            sev_pred_score = sev_record.iloc[0]['pred_score']
            sev_matched = sev_record.iloc[0]['matched']
            
            if sev_matched == 1 and sev_pred_score <= threshold:
                failures.append({
                    'object_uid': object_uid,
                    'corruption': corruption,
                    'perf_start_sev': perf_start_sev,
                    'issue': f'Drop condition already met at earlier severity {sev} (pred_score={sev_pred_score:.4f} <= threshold={threshold:.4f})'
                })
                break
    
    if len(failures) == 0:
        print(f"✅ PASS: All {len(score_drop_events)} events validated successfully")
        return True
    else:
        print(f"❌ FAIL: {len(failures)} events failed validation:")
        for f in failures[:5]:  # Show first 5
            print(f"  - {f['object_uid']} (corruption={f['corruption']}, perf_start_sev={f['perf_start_sev']}): {f['issue']}")
        return False


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    results_dir = Path(config['results']['root'])
    
    print("=" * 60)
    print("Alignment Analysis Consistency Validation")
    print("=" * 60)
    print("\nThis script validates consistency between:")
    print("  - detection_records.csv (from 03_detect_tiny_objects_timeseries.py)")
    print("  - risk_events.csv (from 04_detect_risk_events.py)")
    print()
    
    # Load data
    detection_records_csv = results_dir / "detection_records.csv"
    risk_events_csv = results_dir / "risk_events.csv"
    
    if not detection_records_csv.exists():
        print(f"Error: {detection_records_csv} not found")
        print("Please run scripts/03_detect_tiny_objects_timeseries.py first")
        sys.exit(1)
    
    if not risk_events_csv.exists():
        print(f"Error: {risk_events_csv} not found")
        print("Please run scripts/04_detect_risk_events.py first")
        sys.exit(1)
    
    detection_records_df = pd.read_csv(detection_records_csv)
    risk_events_df = pd.read_csv(risk_events_csv)
    
    print(f"Loaded {len(detection_records_df)} detection records")
    print(f"Loaded {len(risk_events_df)} risk events")
    print()
    
    # Run tests
    test_a = validate_same_record_sample(detection_records_df, risk_events_df)
    test_b = validate_aggregation_consistency(detection_records_df, risk_events_df)
    test_c = validate_table_x_detail(detection_records_df, risk_events_df)
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Test (A): Same Record Sample Validation: {'✅ PASS' if test_a else '❌ FAIL'}")
    print(f"Test (B): Aggregation Consistency Check: {'✅ PASS' if test_b else '❌ FAIL'}")
    print(f"Test (C): Table X-detail Validation: {'✅ PASS' if test_c else '❌ FAIL'}")
    
    all_passed = test_a and test_b and test_c
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ Consistency validation complete. Alignment analysis is consistent.")
    else:
        print("\n❌ Consistency issues found. Please review the failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
