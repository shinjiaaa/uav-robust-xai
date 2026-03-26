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
- delta >= CHANGE_THRESHOLD (default 0.15) 이면 change로 판정
- concept change severity = min sev s.t. delta >= CHANGE_THRESHOLD
- concept change type = 'gradual' (delta < 1.0) or 'collapse' (delta >= 1.0)
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Grad-CAM과 동일 임계값으로 비교 (report.md 산출 시 0.15 사용)
CHANGE_THRESHOLD = 0.15
DEFAULT_VIS_CONF_THRESHOLD = 0.30
DEFAULT_TINY_AREA_RATIO_THRESHOLD = 0.01

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


def _safe_rel_path(root: Path, rel: str) -> Path:
    p = root / str(rel)
    if p.exists():
        return p
    alt = root / "datasets" / str(rel)
    return alt


def _bbox_area_ratio(row, root: Path, area_cache: dict, tiny_ratio_thr: float) -> tuple:
    """Return (bbox_area_ratio, tiny_rule_hit). Falls back to is_tiny when image area unknown."""
    gt_area = row.get('gt_area', np.nan)
    if pd.isna(gt_area):
        x1, y1 = row.get('gt_x1', np.nan), row.get('gt_y1', np.nan)
        x2, y2 = row.get('gt_x2', np.nan), row.get('gt_y2', np.nan)
        if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
            gt_area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        else:
            gt_area = np.nan

    img_rel = row.get('corrupted_image_path') if pd.notna(row.get('corrupted_image_path')) else row.get('image_path')
    image_area = np.nan
    if pd.notna(img_rel):
        key = str(img_rel)
        if key in area_cache:
            image_area = area_cache[key]
        else:
            p = _safe_rel_path(root, key)
            if p.exists():
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                    image_area = float(max(1, w * h))
                except Exception:
                    image_area = np.nan
            area_cache[key] = image_area

    if pd.notna(gt_area) and pd.notna(image_area) and image_area > 0:
        ratio = float(gt_area) / float(image_area)
        return ratio, bool(ratio <= tiny_ratio_thr)

    is_tiny = row.get('is_tiny', np.nan)
    if pd.notna(is_tiny):
        return np.nan, bool(int(is_tiny) == 1)
    return np.nan, False


def load_detection_records(tiny_ratio_thr: float, vis_conf_thr: float):
    """Load detection_records and derive tiny-object recognition concepts."""
    root = Path('.')
    p = root / 'results' / 'detection_records.csv'
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p)
    needed = ['corruption', 'severity', 'image_id', 'object_uid', 'matched']
    if not all(c in df.columns for c in needed):
        return None

    df = df.copy()
    df['severity'] = pd.to_numeric(df['severity'], errors='coerce').fillna(-1).astype(int)
    df['pred_score'] = pd.to_numeric(df.get('pred_score', np.nan), errors='coerce')
    df['matched'] = pd.to_numeric(df['matched'], errors='coerce').fillna(0).astype(int)
    area_cache = {}

    ratios = []
    tiny_hits = []
    for _, row in df.iterrows():
        ratio, hit = _bbox_area_ratio(row, root, area_cache, tiny_ratio_thr)
        ratios.append(ratio)
        tiny_hits.append(hit)
    df['bbox_area_ratio'] = ratios
    df['is_tiny_by_ratio'] = tiny_hits
    df = df[df['is_tiny_by_ratio'] == True].copy()

    # tiny_object_presence: 1 for tiny-object samples in this filtered table.
    df['concept_tiny_object_presence'] = 1.0

    # tiny_object_visibility: matched + confidence-aware recognition signal.
    # Conservative default: if miss then 0; if matched then normalized by confidence threshold.
    conf = df['pred_score'].fillna(0.0)
    vis = np.where(
        df['matched'] == 1,
        np.clip((conf - vis_conf_thr) / max(1e-6, 1.0 - vis_conf_thr), 0.0, 1.0),
        0.0,
    )
    df['concept_tiny_object_visibility'] = vis.astype(float)

    # tiny_object_separability (first pass proxy):
    # separability is high when matched and IoU is high with decent confidence.
    iou = pd.to_numeric(df.get('match_iou', np.nan), errors='coerce').fillna(0.0)
    sep = np.where(
        df['matched'] == 1,
        np.clip(0.5 * conf + 0.5 * iou, 0.0, 1.0),
        0.0,
    )
    df['concept_tiny_object_separability'] = sep.astype(float)
    df['miss_flag'] = (df['matched'] == 0).astype(int)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--change-threshold", type=float, default=CHANGE_THRESHOLD)
    parser.add_argument("--visibility-conf-threshold", type=float, default=DEFAULT_VIS_CONF_THRESHOLD)
    parser.add_argument("--tiny-area-ratio-threshold", type=float, default=DEFAULT_TINY_AREA_RATIO_THRESHOLD)
    args = parser.parse_args()

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
    
    # 1) 기존 환경 개념 점수 (cam_records 기반) - backward compatibility 유지
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
    
    # 2) 기존 환경 개념 변화 시점 (object-event)
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
                
                if delta >= float(args.change_threshold):  # configurable threshold
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
    
    # 3) 기존 요약
    print("\n" + "=" * 70)
    print(f"Summary: Concept Change Detection (threshold delta={float(args.change_threshold):.3f})")
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
    
    # 4) Tiny-object recognition-level concept pipeline (new)
    det_df = load_detection_records(
        tiny_ratio_thr=float(args.tiny_area_ratio_threshold),
        vis_conf_thr=float(args.visibility_conf_threshold),
    )
    if det_df is None or len(det_df) == 0:
        print("\n[WARN] detection_records.csv 기반 tiny-object concept 데이터가 없어 신규 파이프라인은 건너뜀.")
    else:
        tiny_cols = [
            'corruption', 'severity', 'image_id', 'object_uid',
            'concept_tiny_object_presence', 'concept_tiny_object_visibility', 'concept_tiny_object_separability',
            'matched', 'pred_score', 'miss_flag', 'bbox_area_ratio'
        ]
        tiny_scores_path = results_dir / 'fastcav_tiny_concept_scores.csv'
        det_df[tiny_cols].to_csv(tiny_scores_path, index=False)
        print(f"Saved tiny-object concept scores: {tiny_scores_path}")

        # severity summary by corruption
        gb = det_df.groupby(['corruption', 'severity'], dropna=False)
        tiny_summary = gb.agg(
            n_samples=('object_uid', 'count'),
            mean_tiny_object_presence=('concept_tiny_object_presence', 'mean'),
            mean_tiny_object_visibility=('concept_tiny_object_visibility', 'mean'),
            mean_tiny_object_separability=('concept_tiny_object_separability', 'mean'),
            mean_confidence=('pred_score', 'mean'),
            miss_rate=('miss_flag', 'mean'),
        ).reset_index()
        tiny_summary_path = results_dir / 'fastcav_tiny_severity_summary.csv'
        tiny_summary.to_csv(tiny_summary_path, index=False)
        print(f"Saved tiny severity summary: {tiny_summary_path}")

        # change onset by object for required primary concept: tiny_object_visibility
        changes = []
        for (corr, obj), g in det_df.groupby(['corruption', 'object_uid'], dropna=False):
            g = g.sort_values('severity')
            if 0 not in set(g['severity'].tolist()):
                continue
            b = g[g['severity'] == 0]['concept_tiny_object_visibility'].dropna().mean()
            if pd.isna(b):
                continue
            onset = None
            onset_type = None
            for sev in sorted(set(g['severity'].tolist())):
                if int(sev) == 0:
                    continue
                sv = g[g['severity'] == sev]['concept_tiny_object_visibility'].dropna().mean()
                if pd.isna(sv):
                    continue
                delta = abs(float(sv) - float(b)) / max(abs(float(b)), 1e-6)
                if delta >= float(args.change_threshold):
                    onset = int(sev)
                    onset_type = 'collapse' if delta >= 1.0 else 'gradual'
                    break
            changes.append({
                'corruption': corr,
                'object_id': obj,
                'image_id': g.iloc[0].get('image_id'),
                'class_id': g.iloc[0].get('gt_class_id') if 'gt_class_id' in g.columns else g.iloc[0].get('class_id'),
                'concept': 'tiny_object_visibility',
                'baseline_score': float(b),
                'concept_change_severity': onset,
                'concept_change_type': onset_type,
                'change_threshold': float(args.change_threshold),
            })
        tiny_changes_df = pd.DataFrame(changes)
        tiny_changes_path = results_dir / 'fastcav_tiny_concept_changes.csv'
        tiny_changes_df.to_csv(tiny_changes_path, index=False)
        print(f"Saved tiny concept changes: {tiny_changes_path}")

        # Bridge analysis: concept vs detection conf/miss correlation by corruption
        bridge_rows = []
        for corr, g in det_df.groupby('corruption', dropna=False):
            vis = pd.to_numeric(g['concept_tiny_object_visibility'], errors='coerce')
            conf = pd.to_numeric(g['pred_score'], errors='coerce')
            miss = pd.to_numeric(g['miss_flag'], errors='coerce')
            c1 = vis.corr(conf) if vis.notna().sum() > 1 and conf.notna().sum() > 1 else np.nan
            c2 = vis.corr(miss) if vis.notna().sum() > 1 and miss.notna().sum() > 1 else np.nan
            bridge_rows.append({
                'corruption': corr,
                'corr_visibility_confidence': c1,
                'corr_visibility_miss_flag': c2,
                'n_samples': int(len(g)),
            })
        bridge_df = pd.DataFrame(bridge_rows)
        bridge_path = results_dir / 'fastcav_tiny_bridge_analysis.csv'
        bridge_df.to_csv(bridge_path, index=False)
        print(f"Saved tiny bridge analysis: {bridge_path}")

        # Early-warning alignment: tiny visibility onset vs performance start severity
        fpath = results_dir / 'failure_events.csv'
        if fpath.exists() and fpath.stat().st_size > 0:
            fe = pd.read_csv(fpath)
            fe = fe.copy()
            if 'object_uid' not in fe.columns and 'image_id' in fe.columns and 'class_id' in fe.columns:
                fe['object_uid'] = fe['image_id'].astype(str) + '_obj_' + fe['class_id'].astype(str)

            def _start_sev(r):
                for col in ['start_severity', 'first_miss_severity', 'score_drop_severity', 'iou_drop_severity', 'failure_severity']:
                    v = r.get(col)
                    if pd.notna(v):
                        return int(v)
                return None

            rows = []
            for _, r in fe.iterrows():
                corr = r.get('corruption')
                oid = r.get('object_uid')
                img_id = r.get('image_id')
                cls_id = r.get('class_id')
                start = _start_sev(r)
                if start is None:
                    continue
                sub = tiny_changes_df[(tiny_changes_df['corruption'] == corr) & (tiny_changes_df['object_id'] == oid)]
                if len(sub) == 0 and pd.notna(img_id) and pd.notna(cls_id):
                    sub = tiny_changes_df[
                        (tiny_changes_df['corruption'] == corr)
                        & (tiny_changes_df['image_id'].astype(str) == str(img_id))
                        & (pd.to_numeric(tiny_changes_df['class_id'], errors='coerce') == float(cls_id))
                    ]
                if len(sub) == 0 or pd.isna(sub.iloc[0]['concept_change_severity']):
                    rows.append({'corruption': corr, 'object_id': oid, 'performance_start_severity': start,
                                 'concept_change_severity': np.nan, 'alignment': 'unavailable', 'lead_steps': np.nan})
                    continue
                ch = int(sub.iloc[0]['concept_change_severity'])
                lead = int(start) - ch
                align = 'lead' if lead > 0 else ('coincident' if lead == 0 else 'lag')
                rows.append({'corruption': corr, 'object_id': oid, 'performance_start_severity': start,
                             'concept_change_severity': ch, 'alignment': align, 'lead_steps': lead})
            ew_df = pd.DataFrame(rows)
            ew_path = results_dir / 'fastcav_tiny_early_warning_summary.csv'
            ew_df.to_csv(ew_path, index=False)
            print(f"Saved tiny early-warning rows: {ew_path}")

            # corruption-wise summary
            summary_rows = []
            for corr, g in ew_df.groupby('corruption', dropna=False):
                total = len(g)
                if total == 0:
                    continue
                lead_pct = 100.0 * float((g['alignment'] == 'lead').sum()) / total
                coinc_pct = 100.0 * float((g['alignment'] == 'coincident').sum()) / total
                lag_pct = 100.0 * float((g['alignment'] == 'lag').sum()) / total
                unavail_pct = 100.0 * float((g['alignment'] == 'unavailable').sum()) / total
                ls = pd.to_numeric(g.loc[g['alignment'] == 'lead', 'lead_steps'], errors='coerce').dropna()
                mean_lead = float(ls.mean()) if len(ls) else np.nan
                # mean onset over available rows
                av = pd.to_numeric(g['concept_change_severity'], errors='coerce').dropna()
                mean_onset = float(av.mean()) if len(av) else np.nan
                summary_rows.append({
                    'corruption': corr,
                    'mean_visibility_onset_severity': mean_onset,
                    'lead_pct': lead_pct,
                    'coincident_pct': coinc_pct,
                    'lag_pct': lag_pct,
                    'unavailable_pct': unavail_pct,
                    'mean_lead_steps': mean_lead,
                    'n_total': int(total),
                })
            ew_corr_df = pd.DataFrame(summary_rows)
            ew_corr_path = results_dir / 'fastcav_tiny_corruption_summary.csv'
            ew_corr_df.to_csv(ew_corr_path, index=False)
            print(f"Saved tiny early-warning corruption summary: {ew_corr_path}")

    print("\n" + "=" * 70)
    print("Next: Run exp_A_early_warning_comparison.py and exp_A_threshold_trend_validation.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
