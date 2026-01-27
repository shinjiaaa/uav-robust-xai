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


# Constants
IOU_MISS_TH = 0.5  # IoU threshold for miss detection
SCORE_DROP_RATIO = 0.5  # Score drop threshold (50% of baseline)
IOU_DROP_ABSOLUTE = 0.2  # IoU drop threshold (absolute)
W_PRE = 2  # CAM을 몇 severity 전까지 볼지 (직전 2단계 + 시작단계)


def make_risk_events(detection_df: pd.DataFrame) -> pd.DataFrame:
    """Generate risk_events.csv from detection_records.csv.
    
    Args:
        detection_df: DataFrame with detection_records.csv content
        
    Returns:
        DataFrame with risk events (one row per risk event)
    """
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
        df["is_score_drop"] = (df["matched"] == 1) & (df["pred_score"] <= base_score * SCORE_DROP_RATIO)
    
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
        
        # CAM 계산 구간 (직전 W_PRE 단계 + 시작단계)
        cam_s_from = max(0, s_star - W_PRE)
        cam_s_to = s_star
        
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
    
    # Load detection records (prefer detection_records.csv, fallback to tiny_records_timeseries.csv)
    detection_records_csv = results_dir / "detection_records.csv"
    legacy_csv = results_dir / "tiny_records_timeseries.csv"
    
    if detection_records_csv.exists() and detection_records_csv.stat().st_size > 0:
        records_df = pd.read_csv(detection_records_csv)
        print(f"Loaded {len(records_df)} records from detection_records.csv")
    elif legacy_csv.exists() and legacy_csv.stat().st_size > 0:
        records_df = pd.read_csv(legacy_csv)
        print(f"Loaded {len(records_df)} records from tiny_records_timeseries.csv (legacy)")
    else:
        print("Error: Neither detection_records.csv nor tiny_records_timeseries.csv found.")
        print("Please run scripts/03_detect_tiny_objects_timeseries.py first")
        sys.exit(1)
    
    # Generate risk events
    print("\nDetecting risk events...")
    risk_events_df = make_risk_events(records_df)
    print(f"Found {len(risk_events_df)} risk events")
    
    if len(risk_events_df) > 0:
        # Print summary by failure type
        print("\nRisk events by failure type:")
        for ftype, count in risk_events_df['failure_type'].value_counts().items():
            print(f"  {ftype}: {count} events")
        
        print("\nRisk events by corruption:")
        for corr, count in risk_events_df['corruption'].value_counts().items():
            print(f"  {corr}: {count} events")
        
        # Save risk events
        risk_events_csv = results_dir / "risk_events.csv"
        risk_events_df.to_csv(risk_events_csv, index=False)
        print(f"\nSaved {len(risk_events_df)} risk events to {risk_events_csv}")
        
        # Print CAM computation scope
        print("\nCAM computation scope (from risk_events):")
        print(f"  Total events: {len(risk_events_df)}")
        print(f"  CAM severity range: {risk_events_df['cam_sev_from'].min()} to {risk_events_df['cam_sev_to'].max()}")
        print(f"  This reduces CAM computation from all frames×all severities to risk regions only")
    else:
        print("\n[WARN] No risk events detected. This may indicate:")
        print("  - All objects detected successfully across all severities")
        print("  - Or detection_records.csv may be incomplete")
    
    print("\n[OK] Risk event detection complete!")
    print("\nNext steps:")
    print("  1. Run scripts/05_gradcam_failure_analysis.py (will use risk_events.csv to compute CAM only for risk regions)")
    print("  2. Run scripts/06_llm_report.py (will generate alignment analysis from risk_events + cam_records)")


if __name__ == "__main__":
    main()
