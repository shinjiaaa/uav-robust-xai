"""Detect failure events and identify risk regions."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.failure_detection import detect_failure_events, identify_risk_regions, detect_instability
from src.eval.metrics import evaluate_all_models
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Failure Event Detection and Risk Region Identification")
    print("=" * 60)
    print()
    
    results_dir = Path(config['results']['root'])
    
    # Load detection records
    records_csv = results_dir / "tiny_records_timeseries.csv"
    if not records_csv.exists():
        print("Error: Detection records not found. Run scripts/03_detect_tiny_objects_timeseries.py first")
        sys.exit(1)
    
    records_df = pd.read_csv(records_csv)
    print(f"Loaded {len(records_df)} detection records")
    
    # Detect failure events
    print("\n1. Detecting failure events...")
    failure_events_df = detect_failure_events(records_df, config)
    print(f"   Found {len(failure_events_df)} failure events")
    
    failure_events_csv = results_dir / "failure_events.csv"
    failure_events_df.to_csv(failure_events_csv, index=False)
    print(f"   Saved to {failure_events_csv}")
    
    # Detect instability (across severities for single images)
    print("\n2. Detecting instability regions...")
    instability_df = detect_instability(
        records_df,
        window_size=3,  # Reduced for single images (severity-based)
        threshold=config['risk_detection']['instability_threshold']
    )
    print(f"   Found {len(instability_df)} instability regions")
    
    if len(instability_df) > 0:
        instability_csv = results_dir / "instability_regions.csv"
        instability_df.to_csv(instability_csv, index=False)
        print(f"   Saved to {instability_csv}")
    
    # Compute dataset-wide metrics (if not exists)
    metrics_csv = Path(config['results']['metrics_csv'])
    if not metrics_csv.exists():
        print("\n3. Computing dataset-wide metrics...")
        evaluate_all_models(
            config,
            models=['yolo_generic'],  # Only Generic YOLO
            splits=config['evaluation']['splits'],
            corruption_types=config['corruptions']['types'],
            severities=config['corruptions']['severities'],
            output_csv=metrics_csv
        )
    else:
        print("\n3. Loading existing dataset-wide metrics...")
    
    metrics_df = pd.read_csv(metrics_csv)
    
    # Compute tiny curves
    print("\n4. Computing tiny object curves...")
    tiny_curves = []
    for model in records_df['model'].unique():
        for corruption in records_df['corruption'].unique():
            for severity in records_df['severity'].unique():
                subset = records_df[
                    (records_df['model'] == model) &
                    (records_df['corruption'] == corruption) &
                    (records_df['severity'] == severity)
                ]
                
                if len(subset) == 0:
                    continue
                
                miss_rate = subset['miss'].mean()
                matched = subset[subset['miss'] == 0]
                
                if len(matched) > 0:
                    avg_score = matched['score'].mean()
                    avg_iou = matched['iou'].mean()
                else:
                    avg_score = None
                    avg_iou = None
                
                # Compare to baseline
                baseline = records_df[
                    (records_df['model'] == model) &
                    (records_df['corruption'] == corruption) &
                    (records_df['severity'] == 0)
                ]
                baseline_matched = baseline[baseline['miss'] == 0]
                
                if len(baseline_matched) > 0 and len(matched) > 0:
                    score_drop = baseline_matched['score'].mean() - avg_score if avg_score is not None else None
                    iou_drop = baseline_matched['iou'].mean() - avg_iou if avg_iou is not None else None
                else:
                    score_drop = None
                    iou_drop = None
                
                tiny_curves.append({
                    'model': model,
                    'corruption': corruption,
                    'severity': severity,
                    'miss_rate': miss_rate,
                    'avg_score': avg_score,
                    'avg_iou': avg_iou,
                    'score_drop': score_drop,
                    'iou_drop': iou_drop
                })
    
    tiny_curves_df = pd.DataFrame(tiny_curves)
    tiny_curves_csv = results_dir / "tiny_curves.csv"
    tiny_curves_df.to_csv(tiny_curves_csv, index=False)
    print(f"   Saved to {tiny_curves_csv}")
    
    # Identify risk regions
    print("\n5. Identifying risk regions...")
    risk_regions_df = identify_risk_regions(metrics_df, tiny_curves_df, failure_events_df, config)
    print(f"   Identified {len(risk_regions_df)} risk regions")
    
    risk_regions_csv = results_dir / "risk_regions.csv"
    risk_regions_df.to_csv(risk_regions_csv, index=False)
    print(f"   Saved to {risk_regions_csv}")
    
    print("\n[OK] Failure event detection complete!")


if __name__ == "__main__":
    main()
