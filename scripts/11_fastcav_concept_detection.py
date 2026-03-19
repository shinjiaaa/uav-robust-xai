"""
FastCAV: 계산할 개념(Concept) 점수와 concept change severity 정의

목표:
- Grad-CAM과 공정하게 비교할 수 있도록 "개념 점수"를 정의
- 4개 개념: 중심 집중 (Focused), 분산 확대 (Diffused), 배경 혼입 (Background), 활성 붕괴 (Collapse)
- 각 개념에 대해 concept score 계산
- concept change severity 정의 (delta >= 0.2)
- Grad-CAM의 gradual/collapse change와 비교 가능하도록

입력:
- results/cam_records.csv (Grad-CAM 지표: bbox_center_activation_distance, activation_spread, ring_energy_ratio)

출력:
- results/fastcav_concept_scores.csv (object_id, corruption, severity별 concept score)
- results/fastcav_concept_changes.csv (object 이벤트별 concept change severity + type)
- results/fastcav_vs_gradcam_comparison.csv (alignment 비교: lead/coincident/lag)

개념 정의 및 Grad-CAM 매핑:
1. 중심 집중 (Focused):
   - bbox 중심 근처에 attention이 모여있음
   - Grad-CAM 지표: bbox_distance 작음, spread 작음, ring_ratio 큼
   - Score: 1 - (norm_bbox_distance + norm_spread) / 2 + 0.3 * norm_ring_ratio

2. 분산 확대 (Diffused):
   - attention이 퍼져서 집중이 떨어짐
   - Grad-CAM 지표: spread 큼
   - Score: norm_spread (클수록 높음)

3. 배경 혼입 (Background):
   - 객체보다 배경 쪽 정보가 강해짐
   - Grad-CAM 지표: bbox_distance 크고, ring_ratio 작음
   - Score: norm_bbox_distance + (1 - norm_ring_ratio) / 2

4. 활성 붕괴 (Collapse):
   - 표현 자체가 거의 사라짐
   - Grad-CAM 지표: spread ≈ 0 또는 매우 작음
   - Score: 1.0 if spread < 1e-3 else 0.0 (이진)

변화 기준:
- delta = |score_sev - score_baseline| / max(|score_baseline|, 1e-6)
- delta >= 0.2이면 change로 판정
- concept change severity = min sev s.t. delta >= 0.2
- concept change type = 'gradual' (delta < 1.0) or 'collapse' (delta >= 1.0)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_cam_records():
    """Load and filter CAM records."""
    results_root = Path('results')
    cam_csv = results_root / 'cam_records.csv'
    if not cam_csv.exists():
        print(f"[ERROR] {cam_csv} not found. Run 05_gradcam_failure_analysis.py first.")
        return None
    
    df = pd.read_csv(cam_csv)
    
    # Filter to ok/primary (same as main pipeline)
    if 'cam_status' in df.columns:
        df = df[df['cam_status'] == 'ok']
    if 'layer_role' in df.columns:
        df = df[df['layer_role'] == 'primary']
    
    # Coerce severity to int
    if 'severity' in df.columns:
        df['severity'] = pd.to_numeric(df['severity'], errors='coerce').fillna(-1).astype(int)
    
    return df

def normalize_metric(series, clip_zero=False):
    """Normalize metric to [0, 1] range using min-max normalization."""
    s = pd.to_numeric(series, errors='coerce')
    s = s.dropna()
    if len(s) == 0:
        return lambda x: 0.5
    vmin, vmax = s.min(), s.max()
    if vmax == vmin:
        return lambda x: 0.5
    def norm(x):
        if pd.isna(x):
            return 0.5
        val = (float(x) - vmin) / (vmax - vmin)
        if clip_zero and val < 0:
            return 0
        return max(0, min(1, val))
    return norm

def compute_concept_score(row, concept_name, normalizers):
    """Compute concept score for a single row."""
    try:
        if concept_name == 'Focused':
            # 중심 집중: bbox_distance 작음, spread 작음, ring_ratio 큼
            bbox_dist = row.get('bbox_center_activation_distance', np.nan)
            spread = row.get('activation_spread', np.nan)
            ring_ratio = row.get('ring_energy_ratio', np.nan)
            
            if pd.isna(bbox_dist) or pd.isna(spread) or pd.isna(ring_ratio):
                return np.nan
            
            norm_dist = normalizers['bbox_distance'](bbox_dist)
            norm_spread = normalizers['spread'](spread)
            norm_ring = normalizers['ring_ratio'](ring_ratio)
            
            score = (1 - norm_dist - norm_spread) / 2 + 0.3 * norm_ring
            return max(0, min(1, score))
        
        elif concept_name == 'Diffused':
            # 분산 확대: spread 크면 높음
            spread = row.get('activation_spread', np.nan)
            if pd.isna(spread):
                return np.nan
            score = normalizers['spread'](spread)
            return score
        
        elif concept_name == 'Background':
            # 배경 혼입: bbox_distance 크고, ring_ratio 작음
            bbox_dist = row.get('bbox_center_activation_distance', np.nan)
            ring_ratio = row.get('ring_energy_ratio', np.nan)
            
            if pd.isna(bbox_dist) or pd.isna(ring_ratio):
                return np.nan
            
            norm_dist = normalizers['bbox_distance'](bbox_dist)
            norm_ring = normalizers['ring_ratio'](ring_ratio)
            
            score = (norm_dist + (1 - norm_ring)) / 2
            return score
        
        elif concept_name == 'Collapse':
            # 활성 붕괴: spread < 1e-3이면 1 (붕괴), 아니면 0
            spread = row.get('activation_spread', np.nan)
            if pd.isna(spread):
                return np.nan
            return 1.0 if float(spread) < 1e-3 else 0.0
        
    except Exception as e:
        print(f"[WARN] Error computing {concept_name}: {e}")
        return np.nan
    
    return np.nan

def main():
    print("=" * 70)
    print("FastCAV Concept Detection & Change Analysis")
    print("=" * 70)
    
    # Load CAM records
    cam_df = load_cam_records()
    if cam_df is None or len(cam_df) == 0:
        print("[ERROR] No CAM data available.")
        sys.exit(1)
    
    print(f"Loaded {len(cam_df)} CAM records")
    
    # Define normalizers globally
    normalizers = {
        'bbox_distance': normalize_metric(cam_df['bbox_center_activation_distance'].dropna()),
        'spread': normalize_metric(cam_df['activation_spread'].dropna()),
        'ring_ratio': normalize_metric(cam_df['ring_energy_ratio'].dropna()),
    }
    
    # 1. Compute concept scores for all records
    concepts = ['Focused', 'Diffused', 'Background', 'Collapse']
    
    for concept in concepts:
        cam_df[f'concept_{concept}'] = cam_df.apply(
            lambda row: compute_concept_score(row, concept, normalizers), 
            axis=1
        )
    
    # Save concept scores
    results_dir = Path('results')
    concept_scores_path = results_dir / 'fastcav_concept_scores.csv'
    
    cols_to_save = ['model', 'corruption', 'object_id', 'severity', 'image_id', 'class_id', 'layer_role']
    cols_to_save = [c for c in cols_to_save if c in cam_df.columns]
    cols_to_save += [f'concept_{c}' for c in concepts]
    
    cam_df[cols_to_save].to_csv(concept_scores_path, index=False)
    print(f"\nSaved concept scores: {concept_scores_path}")
    
    # 2. Compute concept change severity per object-event
    # Group by (corruption, object_id) to track severity progression
    group_cols = ['corruption', 'object_id']
    
    concept_changes = []
    
    for (corr, obj), group in cam_df.groupby(group_cols):
        group = group.sort_values('severity').copy()
        sev_range = group['severity'].unique()
        
        if 0 not in sev_range or len(sev_range) < 2:
            continue  # Skip if no baseline or insufficient data
        
        for concept in concepts:
            col_name = f'concept_{concept}'
            baseline_score = group[group['severity'] == 0][col_name].dropna().mean()
            
            if pd.isna(baseline_score):
                continue
            
            concept_change_sev = None
            concept_change_type = None
            
            for sev in sorted(sev_range):
                if sev == 0:
                    continue
                
                sev_score = group[group['severity'] == sev][col_name].dropna().mean()
                if pd.isna(sev_score):
                    continue
                
                delta = abs(sev_score - baseline_score) / max(abs(baseline_score), 1e-6)
                
                if delta >= 0.2:  # Change threshold (same as Grad-CAM)
                    concept_change_sev = sev
                    concept_change_type = 'collapse' if delta >= 1.0 else 'gradual'
                    break  # First change severity
            
            concept_changes.append({
                'corruption': corr,
                'object_id': obj,
                'concept': concept,
                'baseline_score': baseline_score,
                'concept_change_severity': concept_change_sev,
                'concept_change_type': concept_change_type,
            })
    
    changes_df = pd.DataFrame(concept_changes)
    changes_path = results_dir / 'fastcav_concept_changes.csv'
    changes_df.to_csv(changes_path, index=False)
    print(f"Saved concept changes: {changes_path}")
    
    # 3. Summary statistics
    print("\n" + "=" * 70)
    print("Summary: Concept Change Detection (threshold delta=0.2)")
    print("=" * 70)
    
    for concept in concepts:
        concept_data = changes_df[changes_df['concept'] == concept]
        if len(concept_data) == 0:
            continue
        
        n_changes = concept_data['concept_change_severity'].notna().sum()
        n_collapse = (concept_data['concept_change_type'] == 'collapse').sum()
        n_gradual = (concept_data['concept_change_type'] == 'gradual').sum()
        
        print(f"\n{concept}:")
        print(f"  Total objects: {len(concept_data)}")
        print(f"  Objects with change: {n_changes} ({n_changes/len(concept_data)*100:.1f}%)")
        print(f"    - Collapse: {n_collapse}")
        print(f"    - Gradual: {n_gradual}")
        
        if n_changes > 0:
            change_data = concept_data[concept_data['concept_change_severity'].notna()]
            avg_change_sev = change_data['concept_change_severity'].mean()
            print(f"  Mean change severity: {avg_change_sev:.2f}")
    
    print("\n" + "=" * 70)
    print("Next: Run exp_A_early_warning_comparison.py to compare Grad-CAM vs FastCAV")
    print("=" * 70)

if __name__ == "__main__":
    main()
