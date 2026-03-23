"""
Experiment A: 조기 경보성 비교 (Early Warning Comparison)

질문: Grad-CAM과 FastCAV 중 누가 더 빨리 위험 신호를 주는가?

설계:
- 같은 corruption × severity × object 이벤트
- 같은 성능 기준 (failure_events.csv의 start_severity)
- Grad-CAM: gradual/collapse change severity
- FastCAV: concept change severity (평균값 사용)

평가 지표:
- lead / coincident / lag / unavailable (각각의 비율)
- 평균 lead steps (average performance_start_sev - change_sev)
- corruption별 lead 비율
- collapse/gradual 분리성

출력:
- results/exp_A_alignment_comparison.csv
- results/exp_A_summary_table.csv
- 경고: lead/coincident/lag 비율 및 통계
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# FastCAV(11)와 동일 임계값으로 비교 (report.md 산출 시 0.15 사용)
GRADUAL_CHANGE_THRESHOLD = 0.15

def load_data():
    """Load Grad-CAM and FastCAV results."""
    results_dir = Path('results')
    
    # Load failure events (performance baseline)
    failure_events_path = results_dir / 'failure_events.csv'
    if not failure_events_path.exists():
        print(f"[ERROR] {failure_events_path} not found.")
        return None, None, None
    
    failure_df = pd.read_csv(failure_events_path)
    
    # Load Grad-CAM results (from sensitivity_analysis.py)
    # We need to reconstruct alignment analysis
    cam_records_path = results_dir / 'cam_records.csv'
    if not cam_records_path.exists():
        print(f"[ERROR] {cam_records_path} not found.")
        return None, None, None
    
    cam_df = pd.read_csv(cam_records_path)
    
    # Load FastCAV concept changes
    fastcav_path = results_dir / 'fastcav_concept_changes.csv'
    if not fastcav_path.exists():
        print(f"[ERROR] {fastcav_path} not found. Run 11_fastcav_concept_detection.py first.")
        return None, None, None
    
    fastcav_df = pd.read_csv(fastcav_path)
    
    return failure_df, cam_df, fastcav_df

def extract_gradcam_changes(cam_df):
    """Extract Grad-CAM change severity from cam_records."""
    # Group by (model, corruption, object_id) and identify change severity
    gradcam_changes = []
    
    if 'cam_status' in cam_df.columns:
        cam_df = cam_df[cam_df['cam_status'] == 'ok']
    if 'layer_role' in cam_df.columns:
        cam_df = cam_df[cam_df['layer_role'] == 'primary']
    
    cam_df = cam_df.copy()
    cam_df['severity'] = pd.to_numeric(cam_df['severity'], errors='coerce').fillna(-1).astype(int)
    
    group_cols = ['model', 'corruption', 'object_id']
    for col in group_cols:
        if col not in cam_df.columns:
            continue
    
    for (model, corr, obj), group in cam_df.groupby(group_cols, dropna=False):
        group = group[group['severity'] >= 0].sort_values('severity')
        sev_range = sorted(group['severity'].unique())
        
        if 0 not in sev_range or len(sev_range) < 2:
            continue
        
        # Detect collapse (activation_spread < 1e-3) and gradual (delta >= GRADUAL_CHANGE_THRESHOLD)
        baseline_spread = group[group['severity'] == 0]['activation_spread'].dropna().mean()
        
        if pd.isna(baseline_spread):
            continue
        
        collapse_change_sev = None
        gradual_change_sev = None
        
        for sev in sev_range:
            if sev == 0:
                continue
            
            sev_spread = group[group['severity'] == sev]['activation_spread'].dropna().mean()
            if pd.isna(sev_spread):
                continue
            
            # Collapse detection
            if collapse_change_sev is None and float(sev_spread) < 1e-3:
                collapse_change_sev = sev
            
            # Gradual detection (delta >= GRADUAL_CHANGE_THRESHOLD)
            if gradual_change_sev is None and baseline_spread != 0:
                delta = abs(sev_spread - baseline_spread) / max(abs(baseline_spread), 1e-6)
                if delta >= GRADUAL_CHANGE_THRESHOLD:
                    gradual_change_sev = sev
        
        # Report earliest change
        change_sev = None
        change_type = None
        if collapse_change_sev is not None and gradual_change_sev is not None:
            if collapse_change_sev <= gradual_change_sev:
                change_sev = collapse_change_sev
                change_type = 'collapse'
            else:
                change_sev = gradual_change_sev
                change_type = 'gradual'
        elif collapse_change_sev is not None:
            change_sev = collapse_change_sev
            change_type = 'collapse'
        elif gradual_change_sev is not None:
            change_sev = gradual_change_sev
            change_type = 'gradual'
        
        gradcam_changes.append({
            'model': model,
            'corruption': corr,
            'object_id': obj,
            'gradcam_change_severity': change_sev,
            'gradcam_change_type': change_type,
        })
    
    return pd.DataFrame(gradcam_changes)

def compute_fastcav_aggregate_change(fastcav_df):
    """Aggregate FastCAV concept changes per object."""
    # For each object, use the earliest concept change severity across all concepts
    fastcav_agg = []
    
    for (corr, obj), group in fastcav_df.groupby(['corruption', 'object_id']):
        changes = group[group['concept_change_severity'].notna()]
        
        if len(changes) == 0:
            fastcav_agg.append({
                'corruption': corr,
                'object_id': obj,
                'fastcav_concept_change_severity': None,
                'fastcav_change_type': None,
            })
        else:
            earliest_sev = changes['concept_change_severity'].min()
            change_types = changes[changes['concept_change_severity'] == earliest_sev]['concept_change_type'].unique()
            change_type = 'collapse' if 'collapse' in change_types else 'gradual'
            
            fastcav_agg.append({
                'corruption': corr,
                'object_id': obj,
                'fastcav_concept_change_severity': earliest_sev,
                'fastcav_change_type': change_type,
            })
    
    return pd.DataFrame(fastcav_agg)

def compute_alignment(failure_df, gradcam_df, fastcav_df):
    """Compute alignment (lead/coincident/lag) for both methods."""
    # Build object_uid from failure_events
    failure_df = failure_df.copy()
    if 'object_uid' not in failure_df.columns and 'image_id' in failure_df.columns and 'class_id' in failure_df.columns:
        failure_df['object_uid'] = failure_df['image_id'].astype(str) + '_obj_' + failure_df['class_id'].astype(str)
    
    # Merge Grad-CAM and FastCAV with failure events
    alignment_results = []
    
    for _, fevent in failure_df.iterrows():
        corruption = fevent.get('corruption', '')
        object_uid = fevent.get('object_uid', '')
        
        # Extract start_severity from failure event
        start_sev = None
        for col in ['start_severity', 'first_miss_severity', 'score_drop_severity', 'iou_drop_severity', 'failure_severity']:
            if col in fevent and pd.notna(fevent[col]):
                start_sev = int(fevent[col])
                break
        
        if start_sev is None or start_sev < 0:
            continue
        
        # Match with Grad-CAM
        gradcam_record = gradcam_df[
            (gradcam_df['corruption'] == corruption) & 
            (gradcam_df['object_id'] == object_uid)
        ]
        
        # Match with FastCAV
        fastcav_record = fastcav_df[
            (fastcav_df['corruption'] == corruption) & 
            (fastcav_df['object_id'] == object_uid)
        ]
        
        # Compute alignment for Grad-CAM
        gradcam_change_sev = None
        gradcam_change_type = None
        gradcam_alignment = 'unavailable'
        gradcam_lead = None
        
        if len(gradcam_record) > 0:
            gradcam_change_sev = gradcam_record.iloc[0]['gradcam_change_severity']
            gradcam_change_type = gradcam_record.iloc[0]['gradcam_change_type']
            
            if pd.notna(gradcam_change_sev):
                gradcam_change_sev = int(gradcam_change_sev)
                gradcam_lead = start_sev - gradcam_change_sev
                
                if gradcam_lead > 0:
                    gradcam_alignment = 'lead'
                elif gradcam_lead == 0:
                    gradcam_alignment = 'coincident'
                else:
                    gradcam_alignment = 'lag'
        
        # Compute alignment for FastCAV
        fastcav_change_sev = None
        fastcav_change_type = None
        fastcav_alignment = 'unavailable'
        fastcav_lead = None
        
        if len(fastcav_record) > 0:
            fastcav_change_sev = fastcav_record.iloc[0]['fastcav_concept_change_severity']
            fastcav_change_type = fastcav_record.iloc[0]['fastcav_change_type']
            
            if pd.notna(fastcav_change_sev):
                fastcav_change_sev = int(fastcav_change_sev)
                fastcav_lead = start_sev - fastcav_change_sev
                
                if fastcav_lead > 0:
                    fastcav_alignment = 'lead'
                elif fastcav_lead == 0:
                    fastcav_alignment = 'coincident'
                else:
                    fastcav_alignment = 'lag'
        
        alignment_results.append({
            'corruption': corruption,
            'object_uid': object_uid,
            'performance_start_severity': start_sev,
            'gradcam_change_severity': gradcam_change_sev,
            'gradcam_change_type': gradcam_change_type,
            'gradcam_alignment': gradcam_alignment,
            'gradcam_lead_steps': gradcam_lead,
            'fastcav_change_severity': fastcav_change_sev,
            'fastcav_change_type': fastcav_change_type,
            'fastcav_alignment': fastcav_alignment,
            'fastcav_lead_steps': fastcav_lead,
        })
    
    return pd.DataFrame(alignment_results)

def main():
    print("=" * 80)
    print("Experiment A: Early Warning Comparison (Grad-CAM vs FastCAV)")
    print("=" * 80)
    
    # Load data
    failure_df, cam_df, fastcav_df = load_data()
    if failure_df is None:
        sys.exit(1)
    
    print(f"Loaded {len(failure_df)} failure events")
    print(f"Loaded {len(cam_df)} CAM records")
    print(f"Loaded {len(fastcav_df)} FastCAV concept changes")
    
    # Extract Grad-CAM changes
    print("\nExtracting Grad-CAM changes...")
    gradcam_df = extract_gradcam_changes(cam_df)
    print(f"  Found {len(gradcam_df)} objects with Grad-CAM changes")
    
    # Aggregate FastCAV changes
    print("Aggregating FastCAV changes...")
    fastcav_agg = compute_fastcav_aggregate_change(fastcav_df)
    print(f"  Aggregated to {len(fastcav_agg)} objects")
    
    # Compute alignment
    print("Computing alignment...")
    alignment_df = compute_alignment(failure_df, gradcam_df, fastcav_agg)
    print(f"  Computed alignment for {len(alignment_df)} events")
    
    # Save detailed results
    results_dir = Path('results')
    alignment_path = results_dir / 'exp_A_alignment_comparison.csv'
    alignment_df.to_csv(alignment_path, index=False)
    print(f"\nSaved detailed alignment: {alignment_path}")
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("Summary: Alignment Comparison (Grad-CAM vs FastCAV)")
    print("=" * 80)
    
    for method in ['gradcam', 'fastcav']:
        align_col = f'{method}_alignment'
        lead_col = f'{method}_lead_steps'
        
        align_counts = alignment_df[align_col].value_counts()
        total = len(alignment_df)
        
        print(f"\n{method.upper()}:")
        for align_type in ['lead', 'coincident', 'lag', 'unavailable']:
            count = align_counts.get(align_type, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {align_type}: {count} ({pct:.1f}%)")
        
        # Mean lead steps
        lead_data = alignment_df[alignment_df[lead_col].notna()][lead_col]
        if len(lead_data) > 0:
            print(f"  Mean lead steps: {lead_data.mean():.2f}")
            print(f"  Std lead steps: {lead_data.std():.2f}")
    
    # Comparison by corruption
    print("\n" + "-" * 80)
    print("Per-Corruption Comparison")
    print("-" * 80)
    
    for corruption in sorted(alignment_df['corruption'].unique()):
        corr_df = alignment_df[alignment_df['corruption'] == corruption]
        print(f"\n{corruption}:")
        
        for method in ['gradcam', 'fastcav']:
            align_col = f'{method}_alignment'
            lead_col = f'{method}_lead_steps'
            
            align_counts = corr_df[align_col].value_counts()
            total = len(corr_df)
            
            lead_pct = align_counts.get('lead', 0) / total * 100 if total > 0 else 0
            avg_lead = corr_df[corr_df[lead_col].notna()][lead_col].mean()
            
            print(f"  {method}: lead={lead_pct:.1f}%, avg_lead_steps={avg_lead:.2f}")
    
    # Build summary table
    summary_data = []
    for method in ['gradcam', 'fastcav']:
        align_col = f'{method}_alignment'
        lead_col = f'{method}_lead_steps'
        
        align_counts = alignment_df[align_col].value_counts()
        total = len(alignment_df)
        
        for align_type in ['lead', 'coincident', 'lag', 'unavailable']:
            count = align_counts.get(align_type, 0)
            pct = count / total * 100 if total > 0 else 0
            
            summary_data.append({
                'method': method.upper(),
                'alignment': align_type,
                'count': count,
                'percentage': pct,
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = results_dir / 'exp_A_summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Next: Run exp_B_runtime_comparison.py to compare real-time performance")
    print("=" * 80)

if __name__ == "__main__":
    main()
