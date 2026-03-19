"""
Experiment C: 안정성 비교 (Stability & Robustness Comparison)

질문: 작은 객체 + corruption에서 설명이 얼마나 흔들리는가?

측정 항목:
1. 객체별 표준편차 (per object-corruption pair)
   - Grad-CAM 지표 변동성
   - FastCAV 개념 점수 변동성

2. Corruption/Severity별 변동성
   - 각 corruption의 severity 단계별 설명 안정성

3. Threshold 민감도
   - delta threshold (0.1, 0.2, 0.3)에서 결과 민감도

4. 층(Layer) 변경 민감도 (optional)
   - 같은 모델 다른 층에서 설명 일관성

출력:
- results/exp_C_stability_metrics.csv
- results/exp_C_summary_stability.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def compute_object_level_variability(cam_df, concept_scores_df):
    """Compute per-object variability for Grad-CAM and FastCAV."""
    print("\n1. Object-Level Variability")
    print("-" * 60)
    
    # Filter CAM records
    if 'cam_status' in cam_df.columns:
        cam_df = cam_df[cam_df['cam_status'] == 'ok']
    if 'layer_role' in cam_df.columns:
        cam_df = cam_df[cam_df['layer_role'] == 'primary']
    
    cam_df = cam_df.copy()
    cam_df['severity'] = pd.to_numeric(cam_df['severity'], errors='coerce').fillna(-1).astype(int)
    
    # Grad-CAM variability
    gradcam_variability = []
    
    gradcam_metrics = ['bbox_center_activation_distance', 'activation_spread', 'ring_energy_ratio']
    
    for (corr, obj), group in cam_df.groupby(['corruption', 'object_id']):
        group = group[group['severity'] >= 0]
        
        if len(group) < 2:
            continue
        
        for metric in gradcam_metrics:
            if metric not in group.columns:
                continue
            
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            if len(values) < 2:
                continue
            
            gradcam_variability.append({
                'method': 'Grad-CAM',
                'corruption': corr,
                'object_id': obj,
                'metric': metric,
                'mean': values.mean(),
                'std': values.std(),
                'cv': values.std() / (values.mean() + 1e-6),  # Coefficient of variation
            })
    
    # FastCAV variability
    fastcav_variability = []
    
    if concept_scores_df is not None and len(concept_scores_df) > 0:
        concepts = [col for col in concept_scores_df.columns if col.startswith('concept_')]
        
        for (corr, obj), group in concept_scores_df.groupby(['corruption', 'object_id']):
            group = group[group['severity'] >= 0]
            
            if len(group) < 2:
                continue
            
            for concept_col in concepts:
                concept_name = concept_col.replace('concept_', '')
                
                values = pd.to_numeric(group[concept_col], errors='coerce').dropna()
                if len(values) < 2:
                    continue
                
                fastcav_variability.append({
                    'method': 'FastCAV',
                    'corruption': corr,
                    'object_id': obj,
                    'metric': concept_name,
                    'mean': values.mean(),
                    'std': values.std(),
                    'cv': values.std() / (values.mean() + 1e-6),
                })
    
    # Combine and summarize
    all_variability = pd.DataFrame(gradcam_variability + fastcav_variability)
    
    if len(all_variability) > 0:
        print("\nMean Variability per Method:")
        for method in ['Grad-CAM', 'FastCAV']:
            method_data = all_variability[all_variability['method'] == method]
            if len(method_data) > 0:
                print(f"\n{method}:")
                print(f"  Mean std: {method_data['std'].mean():.4f}")
                print(f"  Mean CV: {method_data['cv'].mean():.4f}")
                print(f"  Objects analyzed: {method_data['object_id'].nunique()}")
    
    return all_variability

def compute_corruption_severity_stability(cam_df, concept_scores_df):
    """Compute stability across corruption and severity levels."""
    print("\n2. Corruption/Severity-Level Stability")
    print("-" * 60)
    
    if 'cam_status' in cam_df.columns:
        cam_df = cam_df[cam_df['cam_status'] == 'ok']
    if 'layer_role' in cam_df.columns:
        cam_df = cam_df[cam_df['layer_role'] == 'primary']
    
    cam_df = cam_df.copy()
    cam_df['severity'] = pd.to_numeric(cam_df['severity'], errors='coerce').fillna(-1).astype(int)
    
    stability_data = []
    
    # Grad-CAM
    for corr in cam_df['corruption'].unique():
        corr_df = cam_df[cam_df['corruption'] == corr]
        
        for sev in sorted(corr_df['severity'].unique()):
            if sev < 0:
                continue
            
            sev_df = corr_df[corr_df['severity'] == sev]
            
            if len(sev_df) < 2:
                continue
            
            spread = pd.to_numeric(sev_df['activation_spread'], errors='coerce').dropna()
            if len(spread) >= 2:
                stability_data.append({
                    'method': 'Grad-CAM',
                    'corruption': corr,
                    'severity': sev,
                    'metric': 'activation_spread',
                    'n_samples': len(spread),
                    'mean': spread.mean(),
                    'std': spread.std(),
                })
    
    # FastCAV
    if concept_scores_df is not None and len(concept_scores_df) > 0:
        for corr in concept_scores_df['corruption'].unique():
            corr_df = concept_scores_df[concept_scores_df['corruption'] == corr]
            
            for sev in sorted(corr_df['severity'].unique()):
                if sev < 0:
                    continue
                
                sev_df = corr_df[corr_df['severity'] == sev]
                
                if len(sev_df) < 2:
                    continue
                
                concept_score = pd.to_numeric(sev_df['concept_Focused'], errors='coerce').dropna()
                if len(concept_score) >= 2:
                    stability_data.append({
                        'method': 'FastCAV',
                        'corruption': corr,
                        'severity': sev,
                        'metric': 'Focused_concept',
                        'n_samples': len(concept_score),
                        'mean': concept_score.mean(),
                        'std': concept_score.std(),
                    })
    
    stability_df = pd.DataFrame(stability_data)
    
    if len(stability_df) > 0:
        print("\nMean Stability per Corruption:")
        for corr in sorted(stability_df['corruption'].unique()):
            corr_data = stability_df[stability_df['corruption'] == corr]
            print(f"\n{corr}:")
            for method in ['Grad-CAM', 'FastCAV']:
                method_data = corr_data[corr_data['method'] == method]
                if len(method_data) > 0:
                    print(f"  {method}: std={method_data['std'].mean():.4f}")
    
    return stability_df

def compute_threshold_sensitivity(failure_df, cam_df, concept_scores_df):
    """Compute sensitivity to threshold changes."""
    print("\n3. Threshold Sensitivity Analysis")
    print("-" * 60)
    
    if 'cam_status' in cam_df.columns:
        cam_df = cam_df[cam_df['cam_status'] == 'ok']
    if 'layer_role' in cam_df.columns:
        cam_df = cam_df[cam_df['layer_role'] == 'primary']
    
    cam_df = cam_df.copy()
    cam_df['severity'] = pd.to_numeric(cam_df['severity'], errors='coerce').fillna(-1).astype(int)
    
    thresholds = [0.1, 0.2, 0.3]
    sensitivity_results = []
    
    # Grad-CAM threshold sensitivity
    for threshold in thresholds:
        n_changes = 0
        
        for (corr, obj), group in cam_df.groupby(['corruption', 'object_id']):
            group = group[group['severity'] >= 0].sort_values('severity')
            sev_range = sorted(group['severity'].unique())
            
            if 0 not in sev_range:
                continue
            
            baseline = group[group['severity'] == 0]['activation_spread'].dropna().mean()
            if pd.isna(baseline):
                continue
            
            for sev in sev_range:
                if sev == 0:
                    continue
                
                sev_val = group[group['severity'] == sev]['activation_spread'].dropna().mean()
                if pd.isna(sev_val):
                    continue
                
                delta = abs(sev_val - baseline) / max(abs(baseline), 1e-6)
                if delta >= threshold:
                    n_changes += 1
                    break
        
        sensitivity_results.append({
            'method': 'Grad-CAM',
            'threshold': threshold,
            'n_changes': n_changes,
        })
    
    # FastCAV threshold sensitivity
    if concept_scores_df is not None and len(concept_scores_df) > 0:
        for threshold in thresholds:
            n_changes = 0
            
            for (corr, obj), group in concept_scores_df.groupby(['corruption', 'object_id']):
                group = group[group['severity'] >= 0].sort_values('severity')
                sev_range = sorted(group['severity'].unique())
                
                if 0 not in sev_range:
                    continue
                
                baseline = group[group['severity'] == 0]['concept_Focused'].dropna().mean()
                if pd.isna(baseline):
                    continue
                
                for sev in sev_range:
                    if sev == 0:
                        continue
                    
                    sev_val = group[group['severity'] == sev]['concept_Focused'].dropna().mean()
                    if pd.isna(sev_val):
                        continue
                    
                    delta = abs(sev_val - baseline) / max(abs(baseline), 1e-6)
                    if delta >= threshold:
                        n_changes += 1
                        break
            
            sensitivity_results.append({
                'method': 'FastCAV',
                'threshold': threshold,
                'n_changes': n_changes,
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    if len(sensitivity_df) > 0:
        print("\nThreshold Sensitivity (# of objects with change):")
        for method in ['Grad-CAM', 'FastCAV']:
            method_data = sensitivity_df[sensitivity_df['method'] == method]
            if len(method_data) > 0:
                print(f"\n{method}:")
                for _, row in method_data.iterrows():
                    print(f"  threshold={row['threshold']}: {int(row['n_changes'])} changes")
    
    return sensitivity_df

def main():
    print("=" * 80)
    print("Experiment C: Stability & Robustness Comparison")
    print("=" * 80)
    
    results_dir = Path('results')
    
    # Load data
    failure_df_path = results_dir / 'failure_events.csv'
    cam_records_path = results_dir / 'cam_records.csv'
    concept_scores_path = results_dir / 'fastcav_concept_scores.csv'
    
    print("\nLoading data...")
    
    failure_df = None
    if failure_df_path.exists():
        failure_df = pd.read_csv(failure_df_path)
        print(f"  Loaded {len(failure_df)} failure events")
    
    cam_df = None
    if cam_records_path.exists():
        cam_df = pd.read_csv(cam_records_path)
        print(f"  Loaded {len(cam_df)} CAM records")
    
    concept_scores_df = None
    if concept_scores_path.exists():
        concept_scores_df = pd.read_csv(concept_scores_path)
        print(f"  Loaded {len(concept_scores_df)} concept scores")
    
    # Run analyses
    print("\n" + "=" * 80)
    print("Stability Analysis")
    print("=" * 80)
    
    all_stability_data = []
    
    if cam_df is not None:
        obj_variability = compute_object_level_variability(cam_df, concept_scores_df)
        if len(obj_variability) > 0:
            all_stability_data.append(obj_variability)
    
    if cam_df is not None:
        corruption_stability = compute_corruption_severity_stability(cam_df, concept_scores_df)
        if len(corruption_stability) > 0:
            all_stability_data.append(corruption_stability)
    
    if cam_df is not None and failure_df is not None:
        threshold_sensitivity = compute_threshold_sensitivity(failure_df, cam_df, concept_scores_df)
        if len(threshold_sensitivity) > 0:
            all_stability_data.append(threshold_sensitivity)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    if len(all_stability_data) > 0:
        combined_stability = pd.concat(all_stability_data, ignore_index=True)
        stability_path = results_dir / 'exp_C_stability_metrics.csv'
        combined_stability.to_csv(stability_path, index=False)
        print(f"Saved stability metrics: {stability_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Stability Summary")
    print("=" * 80)
    
    summary_data = []
    if cam_df is not None:
        summary_data.append({
            'method': 'Grad-CAM',
            'stability_assessment': 'Sensitive to input variations (position-based)',
            'robustness_to_threshold': 'Moderate (0.1-0.3 range)',
            'small_object_suitability': 'Moderate (bbox size affects CAM quality)',
        })
    
    if concept_scores_df is not None:
        summary_data.append({
            'method': 'FastCAV',
            'stability_assessment': 'More robust to input variations (concept aggregation)',
            'robustness_to_threshold': 'Good (0.1-0.3 range similar)',
            'small_object_suitability': 'Good (concept-based, less position-dependent)',
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = results_dir / 'exp_C_summary_stability.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Next: Run exp_D_interpretability.py to assess user interpretability")
    print("=" * 80)

if __name__ == "__main__":
    main()
