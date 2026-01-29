"""Detect risk events from detection_records.csv and generate risk_events.csv.

This script implements the 3-stage automation pipeline:
1. detection_records.csv → risk_events.csv (위험구간 시작점/유형 자동 검출)
2. risk_events 기준으로 CAM 계산 구간 자동 선택
3. 두 축을 severity/time으로 join해서 alignment analysis

Risk events are defined using 3 curves:
- Miss-rate curve: mean(is_miss | severity=s)
- Score curve: mean(pred_score | matched=1, severity=s)
- IoU curve: mean(match_iou | matched=1, severity=s)

Risk region start is determined by priority:
1. miss occurrence (strongest signal)
2. score_drop (if no miss)
3. iou_drop (if no miss and no score_drop)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml
from src.utils.seed import set_seed


# Constants (overridden by config when provided)
IOU_MISS_TH = 0.5  # IoU threshold for miss detection
SCORE_DROP_RATIO_DEFAULT = 0.5  # Default: 50% of baseline. Use 0.9 for more events (RQ1).
IOU_DROP_ABSOLUTE = 0.2  # IoU drop threshold (absolute)
# RQ1 severity window for CAM: ensure high-severity coverage (fix Table 7 sev2~4 N/A)
# Option A: [0, 1, 2, 3, 4] always. Option B (compromise): [0 .. min(4, perf_start_sev+3)]
SEV_WINDOW_FULL = True  # If True: cam_sev_from=0, cam_sev_to=4 always. If False: use POST below.
SEV_WINDOW_POST = 3   # severities after start when not full (min 4 levels: 0..min(4, s_star+3))
MAX_SEVERITY = 4      # dataset max (e.g. VisDrone 0..4)


def make_risk_events(detection_df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Generate risk_events.csv from detection_records.csv.
    
    Args:
        detection_df: DataFrame with detection_records.csv content
        
    Returns:
        DataFrame with risk events (one row per risk event)
    """
    # RQ1: Configurable score_drop_ratio (r=0.9 → more events, r=0.5 → stronger drop only)
    risk_cfg = (config or {}).get('risk_detection', {})
    score_drop_ratio = float(risk_cfg.get('score_drop_ratio', SCORE_DROP_RATIO_DEFAULT))
    
    # Key columns for grouping (object 시계열 단위)
    key_cols = ["run_id", "model_id", "corruption", "clip_id", "frame_idx", "object_uid"]
    
    # Check if required columns exist
    required_cols = key_cols + ["severity", "matched", "match_iou", "pred_score"]
    missing_cols = [col for col in required_cols if col not in detection_df.columns]
    if missing_cols:
        # Try legacy column names
        if "model" in detection_df.columns and "model_id" not in detection_df.columns:
            detection_df["model_id"] = detection_df["model"]
        if "image_id" in detection_df.columns and "object_uid" not in detection_df.columns:
            # Generate object_uid from image_id if missing
            if "class_id" in detection_df.columns:
                detection_df["object_uid"] = detection_df["image_id"].astype(str) + "_obj_" + detection_df["class_id"].astype(str)
            else:
                detection_df["object_uid"] = detection_df["image_id"].astype(str) + "_obj_0"
        if "run_id" not in detection_df.columns:
            # Generate run_id if missing (use first timestamp or default)
            detection_df["run_id"] = "default_run"
        if "clip_id" not in detection_df.columns:
            detection_df["clip_id"] = ""
        if "frame_idx" not in detection_df.columns:
            detection_df["frame_idx"] = 0
    
    df = detection_df.copy()
    
    # 1. Miss 정의
    if "is_miss" not in df.columns:
        if "miss" in df.columns:
            df["is_miss"] = (df["miss"] == 1) | (df.get("match_iou", df.get("iou", 0)).fillna(0.0) < IOU_MISS_TH)
        else:
            df["is_miss"] = (df["matched"] == 0) | (df.get("match_iou", df.get("iou", 0)).fillna(0.0) < IOU_MISS_TH)
    
    # 2. base(=severity 0) 기준값 조인
    base_cols = key_cols + ["pred_score", "match_iou"]
    available_base_cols = [col for col in base_cols if col in df.columns]
    
    if "match_iou" not in df.columns and "iou" in df.columns:
        df["match_iou"] = df["iou"]
    if "pred_score" not in df.columns and "score" in df.columns:
        df["pred_score"] = df["score"]
    
    base = df[df["severity"] == 0][available_base_cols].rename(
        columns={"pred_score": "base_pred_score", "match_iou": "base_match_iou"}
    )
    
    if len(base) > 0:
        df = df.merge(base, on=[col for col in key_cols if col in base.columns], how="left")
    else:
        df["base_pred_score"] = None
        df["base_match_iou"] = None
    
    # 3. Drop 규칙 (간단 버전: 비율/절대 혼합)
    base_score = df["base_pred_score"].fillna(0.0).clip(lower=1e-6)
    base_iou = df["base_match_iou"].fillna(0.0)
    
    if "is_score_drop" not in df.columns:
        # score(sev) / score0 < r  →  pred_score <= base_score * score_drop_ratio
        df["is_score_drop"] = (df["matched"] == 1) & (df["pred_score"] <= base_score * score_drop_ratio)
    
    if "is_iou_drop" not in df.columns:
        df["is_iou_drop"] = (df["matched"] == 1) & (df["match_iou"] <= (base_iou - IOU_DROP_ABSOLUTE))
    
    # 4. severity 순으로 정렬
    df = df.sort_values(key_cols + ["severity"])
    
    # 5. 이벤트 생성
    events = []
    for k, g in df.groupby(key_cols, sort=False):
        # 각 유형별 최초 발생 severity
        def first_sev(mask_col):
            if mask_col not in g.columns:
                return None
            idx = g.index[g[mask_col].fillna(False)].tolist()
            return int(g.loc[idx[0], "severity"]) if idx else None
        
        s_miss = first_sev("is_miss")
        s_score = first_sev("is_score_drop")
        s_iou = first_sev("is_iou_drop")
        
        # 위험 이벤트가 없는 경우 skip
        if s_miss is None and s_score is None and s_iou is None:
            continue
        
        # 우선순위로 event type/시작점 결정
        if s_miss is not None:
            s_star, ftype = s_miss, "miss"
        elif s_score is not None:
            s_star, ftype = s_score, "score_drop"
        else:
            s_star, ftype = s_iou, "iou_drop"
        
        # CAM severity window: full [0..4] or [0..min(4, s_star+3)] for high-sev coverage (Table 7)
        if SEV_WINDOW_FULL:
            cam_s_from = 0
            cam_s_to = MAX_SEVERITY
        else:
            cam_s_from = 0
            cam_s_to = min(MAX_SEVERITY, s_star + SEV_WINDOW_POST)
        
        run_id, model_id, corr, clip_id, frame_idx, obj = k
        event_id = f"{run_id}|{model_id}|{corr}|{clip_id}|{frame_idx}|{obj}|S{s_star}|{ftype}"
        
        events.append({
            "run_id": run_id,
            "model_id": model_id,
            "corruption": corr,
            "clip_id": clip_id if pd.notna(clip_id) else "",
            "frame_idx": int(frame_idx) if pd.notna(frame_idx) else 0,
            "object_uid": obj,
            "failure_type": ftype,
            "start_severity": int(s_star),
            "cam_sev_from": int(cam_s_from),
            "cam_sev_to": int(cam_s_to),
            "failure_event_id": event_id,
        })
    
    return pd.DataFrame(events)


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Risk Event Detection (3-Stage Automation Pipeline: Stage 1)")
    print("=" * 60)
    print()
    print("This script generates risk_events.csv from detection_records.csv")
    print("Risk events define:")
    print("  - Miss-rate curve: mean(is_miss | severity=s)")
    print("  - Score curve: mean(pred_score | matched=1, severity=s)")
    print("  - IoU curve: mean(match_iou | matched=1, severity=s)")
    print()
    
    results_dir = Path(config['results']['root'])
    
    # Process each model separately
    # RQ1: Risk events for Grad-CAM alignment; YOLO only (2 models)
    all_models = list(config['models'].keys())
    models = [m for m in all_models if config['models'][m].get('type') == 'yolo']
    if not models:
        models = all_models
    print(f"\nProcessing {len(models)} models: {', '.join(models)}")
    
    all_risk_events = []  # For combined file
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Load per-model detection records
        model_detection_records_csv = results_dir / f"detection_records_{model_name}.csv"
        model_legacy_csv = results_dir / f"tiny_records_timeseries_{model_name}.csv"
        
        # Fallback to combined file if per-model file doesn't exist
        if not model_detection_records_csv.exists() and not model_legacy_csv.exists():
            combined_detection_records_csv = results_dir / "detection_records.csv"
            combined_legacy_csv = results_dir / "tiny_records_timeseries.csv"
            
            if combined_detection_records_csv.exists() and combined_detection_records_csv.stat().st_size > 0:
                print(f"[INFO] Per-model file not found, using combined file and filtering by model_id={model_name}")
                records_df = pd.read_csv(combined_detection_records_csv)
                records_df = records_df[records_df['model_id'] == model_name].copy()
            elif combined_legacy_csv.exists() and combined_legacy_csv.stat().st_size > 0:
                print(f"[INFO] Per-model file not found, using combined legacy file and filtering by model_id={model_name}")
                records_df = pd.read_csv(combined_legacy_csv)
                records_df = records_df[records_df['model_id'] == model_name].copy()
            else:
                print(f"[WARN] No detection records found for model {model_name}, skipping")
                continue
        elif model_detection_records_csv.exists() and model_detection_records_csv.stat().st_size > 0:
            records_df = pd.read_csv(model_detection_records_csv)
            print(f"Loaded {len(records_df)} records from detection_records_{model_name}.csv")
        elif model_legacy_csv.exists() and model_legacy_csv.stat().st_size > 0:
            records_df = pd.read_csv(model_legacy_csv)
            print(f"Loaded {len(records_df)} records from tiny_records_timeseries_{model_name}.csv (legacy)")
        else:
            print(f"[WARN] Detection records file exists but is empty for model {model_name}, skipping")
            continue
        
        if len(records_df) == 0:
            print(f"[WARN] No records found for model {model_name}, skipping")
            continue
        
        # Generate risk events (pass config for score_drop_ratio, etc.)
        print(f"\nDetecting risk events for {model_name}...")
        risk_events_df = make_risk_events(records_df, config=config)
        print(f"Found {len(risk_events_df)} risk events")
        
        if len(risk_events_df) > 0:
            # Print summary by failure type
            print(f"\nRisk events by failure type ({model_name}):")
            for ftype, count in risk_events_df['failure_type'].value_counts().items():
                print(f"  {ftype}: {count} events")
            
            print(f"\nRisk events by corruption ({model_name}):")
            for corr, count in risk_events_df['corruption'].value_counts().items():
                print(f"  {corr}: {count} events")
            
            # Save per-model risk events
            model_risk_events_csv = results_dir / f"risk_events_{model_name}.csv"
            risk_events_df.to_csv(model_risk_events_csv, index=False)
            print(f"\nSaved {len(risk_events_df)} risk events to {model_risk_events_csv}")
            
            # Collect for combined file
            all_risk_events.append(risk_events_df)
            
            # Print CAM computation scope
            print(f"\nCAM computation scope ({model_name}):")
            print(f"  Total events: {len(risk_events_df)}")
            print(f"  CAM severity range: {risk_events_df['cam_sev_from'].min()} to {risk_events_df['cam_sev_to'].max()}")
        else:
            print(f"\n[WARN] No risk events detected for {model_name}. This may indicate:")
            print("  - All objects detected successfully across all severities")
            print("  - Or detection_records may be incomplete")
    
    # Save combined risk events file (all models) for backward compatibility
    if len(all_risk_events) > 0:
        combined_risk_events_df = pd.concat(all_risk_events, ignore_index=True)
        combined_risk_events_csv = results_dir / "risk_events.csv"
        combined_risk_events_df.to_csv(combined_risk_events_csv, index=False)
        print(f"\nSaved {len(combined_risk_events_df)} total risk events (all models) to {combined_risk_events_csv}")
    
    print("\n[OK] Risk event detection complete for all models!")
    print("\nNext steps:")
    print("  1. Run scripts/05_gradcam_failure_analysis.py (will use risk_events_{model}.csv to compute CAM only for risk regions)")
    print("  2. Run scripts/06_llm_report.py (will generate alignment analysis from risk_events + cam_records)")


if __name__ == "__main__":
    main()
