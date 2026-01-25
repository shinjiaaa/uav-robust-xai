"""Generate final report using LLM with failure-event based analysis."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report.llm_report import generate_report_with_llm, save_report
from src.utils.io import load_yaml
import json


def load_metrics(config):
    """Load all computed metrics for new experiment design."""
    results_root = Path(config['results']['root'])
    
    metrics = {}
    
    # Dataset-wide metrics
    metrics_csv = results_root / "metrics_dataset.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        # Fill NaN with 0.0 (empty predictions -> 0 metrics, not missing)
        df = df.fillna(0.0)
        metrics['dataset'] = df.to_dict('records')
    else:
        metrics['dataset'] = []
    
    # Detection records
    timeseries_csv = results_root / "tiny_records_timeseries.csv"
    if timeseries_csv.exists():
        metrics['detection_records'] = pd.read_csv(timeseries_csv).to_dict('records')
    else:
        metrics['detection_records'] = []
    
    # Tiny object curves
    tiny_curves_csv = results_root / "tiny_curves.csv"
    if tiny_curves_csv.exists():
        metrics['tiny_curves'] = pd.read_csv(tiny_curves_csv).to_dict('records')
    else:
        metrics['tiny_curves'] = []
    
    # Failure events
    failure_events_csv = results_root / "failure_events.csv"
    if failure_events_csv.exists():
        metrics['failure_events'] = pd.read_csv(failure_events_csv).to_dict('records')
    else:
        metrics['failure_events'] = []
    
    # Risk regions
    risk_regions_csv = results_root / "risk_regions.csv"
    if risk_regions_csv.exists():
        metrics['risk_regions'] = pd.read_csv(risk_regions_csv).to_dict('records')
    else:
        metrics['risk_regions'] = []
    
    # Instability regions
    instability_csv = results_root / "instability_regions.csv"
    if instability_csv.exists():
        metrics['instability'] = pd.read_csv(instability_csv).to_dict('records')
    else:
        metrics['instability'] = []
    
    # CAM metrics (time-series)
    cam_metrics_csv = results_root / "gradcam_metrics_timeseries.csv"
    if cam_metrics_csv.exists():
        metrics['cam_metrics'] = pd.read_csv(cam_metrics_csv).to_dict('records')
    else:
        metrics['cam_metrics'] = []
    
    return metrics


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    
    print("=" * 60)
    print("Generating LLM Report (Failure-Event Based)")
    print("=" * 60)
    print()
    
    # Load metrics
    print("1. Loading metrics...")
    try:
        metrics = load_metrics(config)
        print(f"  Dataset metrics: {len(metrics['dataset'])} records")
        print(f"  Detection records: {len(metrics['detection_records'])} records")
        print(f"  Tiny curves: {len(metrics['tiny_curves'])} records")
        print(f"  Failure events: {len(metrics['failure_events'])} events")
        print(f"  Risk regions: {len(metrics['risk_regions'])} regions")
        print(f"  CAM metrics: {len(metrics['cam_metrics'])} records")
    except Exception as e:
        print(f"  Error loading metrics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate report
    print("\n2. Generating report with LLM...")
    try:
        # Update prompt for new experiment design
        report = generate_report_with_llm(config, metrics)
        print("  [OK] Report generated")
    except Exception as e:
        print(f"  [ERROR] Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save report
    print("\n3. Saving report...")
    report_path = Path(config['results']['report_md'])
    save_report(report, report_path)
    
    print("\n[OK] Report generation complete!")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
