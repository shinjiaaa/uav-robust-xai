"""DASC 실험 산출물 생성 스크립트.

DASC 실험 설계에 따른 산출물:
1. 이미지 변조 단계별 모델 성능 (IoU curve, mAP)
2. 모델 성능 저하 단계 vs Grad-CAM 패턴 붕괴 단계 비교
3. 각 노이즈 유형별 Grad-CAM 전조 패턴 일관성
4. HTML 프로토타입용 JSON 요약
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml
from src.utils.plot import plot_metric_curves


def load_dasc_data(config: dict) -> dict:
    """Load all data required for DASC deliverables."""
    results_root = Path(config['results']['root'])
    data = {}
    
    # Detection records (prefer detection_records.csv)
    for name, fname in [
        ('detection_records', 'detection_records.csv'),
        ('tiny_records', 'tiny_records_timeseries.csv'),
    ]:
        p = results_root / fname
        if p.exists() and p.stat().st_size > 0:
            data['detection'] = pd.read_csv(p)
            break
    if 'detection' not in data:
        data['detection'] = pd.DataFrame()
    
    # CAM records
    cam_path = results_root / "cam_records.csv"
    if cam_path.exists() and cam_path.stat().st_size > 0:
        data['cam'] = pd.read_csv(cam_path)
    else:
        data['cam'] = pd.DataFrame()
    
    # Metrics dataset (mAP)
    metrics_path = results_root / "metrics_dataset.csv"
    if metrics_path.exists():
        data['metrics'] = pd.read_csv(metrics_path)
    else:
        data['metrics'] = pd.DataFrame()
    
    # Risk events (model degradation stage)
    risk_path = results_root / "risk_events.csv"
    if risk_path.exists() and risk_path.stat().st_size > 0:
        data['risk_events'] = pd.read_csv(risk_path)
    else:
        fallback = results_root / "failure_events.csv"
        if fallback.exists():
            data['risk_events'] = pd.read_csv(fallback)
        else:
            data['risk_events'] = pd.DataFrame()
    
    return data


def compute_iou_curve(det_df: pd.DataFrame, corruptions: list, severities: list) -> pd.DataFrame:
    """Compute mean IoU per severity per corruption."""
    if det_df.empty:
        return pd.DataFrame()
    
    iou_col = 'match_iou' if 'match_iou' in det_df.columns else 'iou'
    if iou_col not in det_df.columns:
        return pd.DataFrame()
    
    model_col = 'model_id' if 'model_id' in det_df.columns else 'model'
    if model_col not in det_df.columns:
        det_df = det_df.copy()
        det_df['model_id'] = 'yolo_generic'
        model_col = 'model_id'
    
    rows = []
    for corruption in corruptions:
        sub = det_df[(det_df['corruption'] == corruption) & (det_df['matched'] == 1)]
        if sub.empty:
            continue
        for severity in severities:
            sev_sub = sub[sub['severity'] == severity]
            mean_iou = sev_sub[iou_col].mean() if len(sev_sub) > 0 else np.nan
            rows.append({
                'corruption': corruption,
                'severity': severity,
                'mean_iou': float(mean_iou) if not np.isnan(mean_iou) else 0.0,
                'n_matched': len(sev_sub)
            })
    return pd.DataFrame(rows)


def compute_miss_rate_curve(det_df: pd.DataFrame, corruptions: list, severities: list) -> pd.DataFrame:
    """Compute miss rate per severity per corruption."""
    if det_df.empty:
        return pd.DataFrame()
    
    miss_col = 'miss' if 'miss' in det_df.columns else 'is_miss'
    if miss_col not in det_df.columns:
        if 'matched' in det_df.columns:
            det_df = det_df.copy()
            det_df['miss'] = (det_df['matched'] == 0).astype(int)
            miss_col = 'miss'
        else:
            return pd.DataFrame()
    
    rows = []
    for corruption in corruptions:
        sub = det_df[det_df['corruption'] == corruption]
        if sub.empty:
            continue
        for severity in severities:
            sev_sub = sub[sub['severity'] == severity]
            miss_rate = sev_sub[miss_col].mean() if len(sev_sub) > 0 else np.nan
            rows.append({
                'corruption': corruption,
                'severity': severity,
                'miss_rate': float(miss_rate) if not np.isnan(miss_rate) else 0.0,
                'n_total': len(sev_sub)
            })
    return pd.DataFrame(rows)


def detect_model_degradation_stage(miss_curve: pd.DataFrame, threshold: float = 0.25) -> dict:
    """첫 번째로 miss_rate >= threshold인 severity 반환 (모델 성능 저하 단계)."""
    result = {}
    for corruption in miss_curve['corruption'].unique():
        sub = miss_curve[miss_curve['corruption'] == corruption].sort_values('severity')
        degraded = sub[sub['miss_rate'] >= threshold]
        if len(degraded) > 0:
            result[corruption] = int(degraded.iloc[0]['severity'])
        else:
            result[corruption] = None
    return result


def detect_gradcam_breakdown_stage(
    cam_df: pd.DataFrame,
    corruptions: list,
    severities: list,
    metric: str = 'energy_in_bbox',
    threshold: float = 0.3
) -> dict:
    """Grad-CAM 패턴 붕괴 단계: baseline(sev0) 대비 metric이 threshold 이상 감소한 첫 severity."""
    if cam_df.empty or metric not in cam_df.columns:
        return {c: None for c in corruptions}
    
    ok_cam = cam_df[cam_df['cam_status'] == 'ok'] if 'cam_status' in cam_df.columns else cam_df
    if ok_cam.empty:
        return {c: None for c in corruptions}
    
    result = {}
    for corruption in corruptions:
        sub = ok_cam[ok_cam['corruption'] == corruption]
        if sub.empty:
            result[corruption] = None
            continue
        base_row = sub[sub['severity'] == 0]
        if len(base_row) == 0:
            result[corruption] = None
            continue
        base_val = base_row[metric].mean()
        if base_val <= 0 or np.isnan(base_val):
            result[corruption] = None
            continue
        for sev in sorted(severities):
            if sev == 0:
                continue
            sev_sub = sub[sub['severity'] == sev]
            if len(sev_sub) == 0:
                continue
            mean_val = sev_sub[metric].mean()
            if np.isnan(mean_val):
                continue
            drop_ratio = 1.0 - (mean_val / base_val)
            if drop_ratio >= threshold:
                result[corruption] = sev
                break
        else:
            result[corruption] = None
    return result


def check_consistency(model_stages: dict, gradcam_stages: dict) -> dict:
    """노이즈 유형별 Grad-CAM이 모델보다 먼저 붕괴했는지 일관성 확인."""
    consistency = {}
    for corruption in model_stages:
        m_sev = model_stages[corruption]
        g_sev = gradcam_stages.get(corruption)
        if m_sev is None and g_sev is None:
            consistency[corruption] = {'predicts_degradation': None, 'lead': None}
        elif g_sev is None:
            consistency[corruption] = {'predicts_degradation': False, 'lead': None}
        elif m_sev is None:
            consistency[corruption] = {'predicts_degradation': True, 'lead': g_sev}
        else:
            predicts = g_sev < m_sev  # Grad-CAM 붕괴가 모델 저하보다 먼저
            lead = m_sev - g_sev if predicts else 0
            consistency[corruption] = {'predicts_degradation': predicts, 'lead': int(lead)}
    return consistency


def main():
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    corruptions = config['corruptions']['types']
    severities = config['corruptions']['severities']
    results_root = Path(config['results']['root'])
    dasc_cfg = config.get('dasc', {})
    model_th = dasc_cfg.get('model_degradation_threshold', 0.25)
    cam_th = dasc_cfg.get('gradcam_breakdown_threshold', 0.3)
    
    print("=" * 60)
    print("DASC Deliverables Generation")
    print("=" * 60)
    print()
    
    # Load data
    print("1. Loading data...")
    data = load_dasc_data(config)
    det_df = data['detection']
    cam_df = data['cam']
    
    if det_df.empty:
        print("   [WARN] No detection records. Run pipeline steps 01-03 first.")
    
    # IoU curve
    print("\n2. Computing IoU curve...")
    iou_curve = compute_iou_curve(det_df, corruptions, severities)
    if not iou_curve.empty:
        print(f"   IoU curve: {len(iou_curve)} records")
    
    # Miss rate curve
    print("\n3. Computing miss rate curve...")
    miss_curve = compute_miss_rate_curve(det_df, corruptions, severities)
    
    # Model degradation stage
    print("\n4. Detecting model degradation stage...")
    model_stages = detect_model_degradation_stage(miss_curve, threshold=model_th)
    for c, s in model_stages.items():
        print(f"   {c}: severity {s}")
    
    # Grad-CAM breakdown stage
    print("\n5. Detecting Grad-CAM pattern breakdown stage...")
    gradcam_stages = detect_gradcam_breakdown_stage(
        cam_df, corruptions, severities,
        metric='energy_in_bbox', threshold=cam_th
    )
    for c, s in gradcam_stages.items():
        print(f"   {c}: severity {s}")
    
    # Consistency
    print("\n6. Checking consistency across noise types...")
    consistency = check_consistency(model_stages, gradcam_stages)
    for c, v in consistency.items():
        pred = v['predicts_degradation']
        lead = v['lead']
        if pred is None:
            print(f"   {c}: N/A")
        elif pred:
            print(f"   {c}: Grad-CAM predicts degradation (lead: {lead} severity)")
        else:
            print(f"   {c}: Grad-CAM does NOT predict before model degradation")
    
    # mAP from metrics
    map_by_corruption = {}
    if not data['metrics'].empty and 'corruption' in data['metrics'].columns:
        for c in corruptions:
            sub = data['metrics'][data['metrics']['corruption'] == c]
            if len(sub) > 0:
                map50 = sub['mAP50'].mean() if 'mAP50' in sub.columns else None
                map5095 = sub['mAP50-95'].mean() if 'mAP50-95' in sub.columns else None
                map_by_corruption[c] = {'mAP50': float(map50) if map50 is not None else None,
                                        'mAP50-95': float(map5095) if map5095 is not None else None}
    else:
        map_by_corruption = {c: {'mAP50': None, 'mAP50-95': None} for c in corruptions}
    
    # Save curves as CSV
    curves_dir = Path(config['results'].get('dasc_curves_dir', 'results/dasc_curves'))
    curves_dir.mkdir(parents=True, exist_ok=True)
    if not iou_curve.empty:
        iou_curve.to_csv(curves_dir / "iou_curve.csv", index=False)
        print(f"\n   Saved {curves_dir / 'iou_curve.csv'}")
    if not miss_curve.empty:
        miss_curve.to_csv(curves_dir / "miss_rate_curve.csv", index=False)
        print(f"   Saved {curves_dir / 'miss_rate_curve.csv'}")
    
    # Plot curves
    if not iou_curve.empty and 'mean_iou' in iou_curve.columns:
        # Need model column for plot_metric_curves - add dummy if missing
        plot_df = iou_curve.copy()
        if 'model' not in plot_df.columns:
            plot_df['model'] = 'yolo_generic'
        plot_metric_curves(
            plot_df, 'mean_iou',
            curves_dir / "iou_curve.png",
            title="IoU Curve by Corruption (DASC)",
            ylabel="Mean IoU"
        )
        print(f"   Saved {curves_dir / 'iou_curve.png'}")
    
    if not miss_curve.empty and 'miss_rate' in miss_curve.columns:
        plot_df = miss_curve.copy()
        if 'model' not in plot_df.columns:
            plot_df['model'] = 'yolo_generic'
        plot_metric_curves(
            plot_df, 'miss_rate',
            curves_dir / "miss_rate_curve.png",
            title="Miss Rate Curve by Corruption (DASC)",
            ylabel="Miss Rate"
        )
        print(f"   Saved {curves_dir / 'miss_rate_curve.png'}")
    
    # Heatmap sample paths for prototype (first sample per corruption/severity)
    results_root = Path(config['results']['root'])
    heatmap_dir = Path(config['results'].get('heatmap_samples_dir', 'results/heatmap_samples'))
    heatmap_samples = []
    if heatmap_dir.exists():
        for c in corruptions:
            for s in severities:
                p = heatmap_dir / "yolo_generic" / c / f"L{s}"
                if p.exists():
                    for f in sorted(p.glob("*.png"))[:2]:
                        try:
                            rel = str(f.relative_to(Path.cwd()))
                        except ValueError:
                            rel = str(f)
                        heatmap_samples.append({
                            'corruption': c,
                            'severity': int(s),
                            'path': rel.replace('\\', '/')
                        })

    # DASC summary JSON for HTML prototype
    summary = {
        'experiment': 'DASC',
        'description': '객체 탐지 모델 성능 저하 전 Grad-CAM 전조 붕괴 예측 평가',
        'corruptions': corruptions,
        'severities': severities,
        'iou_curve': iou_curve.to_dict('records') if not iou_curve.empty else [],
        'miss_rate_curve': miss_curve.to_dict('records') if not miss_curve.empty else [],
        'model_degradation_stage': model_stages,
        'gradcam_breakdown_stage': gradcam_stages,
        'consistency': consistency,
        'map_by_corruption': map_by_corruption,
        'heatmap_samples': heatmap_samples,
        'thresholds': {'model_degradation': model_th, 'gradcam_breakdown': cam_th}
    }
    
    summary_path = Path(config['results'].get('dasc_summary_json', 'results/dasc_summary.json'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n7. Saved DASC summary to {summary_path}")
    print("\n[OK] DASC deliverables complete!")


if __name__ == "__main__":
    main()
