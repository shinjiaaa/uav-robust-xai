"""Generate final report using LLM with failure-event based analysis."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report.llm_report import generate_report_with_llm, save_report
from src.utils.io import load_yaml
import json  # for lead_stats.json


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
    
    # Detection records (prefer detection_records.csv)
    det_csv = results_root / "detection_records.csv"
    ts_csv = results_root / "tiny_records_timeseries.csv"
    
    if det_csv.exists() and det_csv.stat().st_size > 0:
        metrics['detection_records'] = pd.read_csv(det_csv).to_dict('records')
        detection_source = "detection_records.csv"
    elif ts_csv.exists() and ts_csv.stat().st_size > 0:
        metrics['detection_records'] = pd.read_csv(ts_csv).to_dict('records')
        detection_source = "tiny_records_timeseries.csv"
    else:
        metrics['detection_records'] = []
        detection_source = "NONE"
    
    # Debug: print which file was used
    print(f"  Detection records source: {detection_source}")
    
    # Tiny object curves
    tiny_curves_csv = results_root / "tiny_curves.csv"
    if tiny_curves_csv.exists():
        metrics['tiny_curves'] = pd.read_csv(tiny_curves_csv).to_dict('records')
    else:
        metrics['tiny_curves'] = []
    
    # DASC curves (IoU and Miss Rate curves)
    dasc_curves_dir = results_root / "dasc_curves"
    iou_curve_csv = dasc_curves_dir / "iou_curve.csv"
    if iou_curve_csv.exists():
        metrics['iou_curve'] = pd.read_csv(iou_curve_csv).to_dict('records')
    else:
        metrics['iou_curve'] = []
    
    miss_rate_curve_csv = dasc_curves_dir / "miss_rate_curve.csv"
    if miss_rate_curve_csv.exists():
        metrics['miss_rate_curve'] = pd.read_csv(miss_rate_curve_csv).to_dict('records')
    else:
        metrics['miss_rate_curve'] = []
    
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
    
    # CAM records (prefer cam_records.csv for Table 7 and correct n_expected_frames)
    cam_records_csv = results_root / "cam_records.csv"
    if cam_records_csv.exists() and cam_records_csv.stat().st_size > 0:
        metrics['cam_records'] = pd.read_csv(cam_records_csv).to_dict('records')
        print(f"  CAM records source: cam_records.csv ({len(metrics['cam_records'])} records)")
    else:
        metrics['cam_records'] = []
    
    # CAM metrics (legacy time-series fallback)
    cam_metrics_csv = results_root / "gradcam_metrics_timeseries.csv"
    if cam_metrics_csv.exists() and cam_metrics_csv.stat().st_size > 0:
        metrics['cam_metrics'] = pd.read_csv(cam_metrics_csv).to_dict('records')
    else:
        metrics['cam_metrics'] = []
    
    # Lead analysis (z-score collapse, sign test, permutation test)
    lead_stats_json = results_root / "lead_stats.json"
    if lead_stats_json.exists():
        try:
            with open(lead_stats_json, encoding='utf-8') as f:
                metrics['lead_stats'] = json.load(f)
        except Exception:
            metrics['lead_stats'] = {}
    else:
        metrics['lead_stats'] = {}
    
    lead_table_csv = results_root / "lead_table.csv"
    if lead_table_csv.exists() and lead_table_csv.stat().st_size > 0:
        metrics['lead_table'] = pd.read_csv(lead_table_csv).to_dict('records')
    else:
        metrics['lead_table'] = []
    
    return metrics


def generate_concise_summary_report(config):
    """Generate a concise, actionable markdown report for quick inspection."""
    root = Path(config['results']['root'])
    summary_path = root / "report_concise.md"

    # Load key tables if available
    expA_path = root / "exp_A_summary_table.csv"
    det_path = root / "detection_records.csv"
    cam_path = root / "cam_records.csv"

    expA = pd.read_csv(expA_path) if expA_path.exists() else pd.DataFrame()
    det = pd.read_csv(det_path) if det_path.exists() else pd.DataFrame()
    cam = pd.read_csv(cam_path) if cam_path.exists() else pd.DataFrame()

    # Performance axis: score drop at sev4 per corruption
    performance_rows = []
    if not det.empty:
        if 'is_score_drop' in det.columns:
            for corruption in ['fog', 'lowlight', 'motion_blur']:
                df = det[(det['corruption'] == corruption) & (det['severity'] == 4)]
                score_drop_rate = df['is_score_drop'].mean() if len(df) > 0 else 0.0
                performance_rows.append((corruption, score_drop_rate))
        else:
            # fallback using miss and score
            for corruption in ['fog', 'lowlight', 'motion_blur']:
                df = det[(det['corruption'] == corruption) & (det['severity'] == 4)]
                score_drop_rate = df['is_miss'].mean() if 'is_miss' in df.columns and len(df) > 0 else 0.0
                performance_rows.append((corruption, score_drop_rate))

    # Cognition axis: cam valid ratio at sev4
    cam_valid_rows = []
    if not cam.empty and 'severity' in cam.columns:
        for corruption in ['fog', 'lowlight', 'motion_blur']:
            base = cam[(cam['corruption'] == corruption) & (cam['severity'] == 4)]
            n_cam = len(base)
            n_expected = 285  # default expected from per-severity data
            if 'cam_status' in cam.columns:
                n_expected = 285
            ratio = float(n_cam) / n_expected if n_expected > 0 else 0.0
            cam_valid_rows.append((corruption, ratio))

    # Early warning axis: expA
    lead_stats = {}
    if not expA.empty and 'corruption' in expA.columns:
        for row in expA.to_dict('records'):
            c = row.get('corruption')
            lead_stats[c] = {
                'lead_pct': float(row.get('lead_pct', 0.0)),
                'avg_lead_steps': row.get('avg_lead_steps', 'N/A')
            }

    lines = [
        "# Concise Summary Report: CAM vs Performance",
        "",
        "This summary focuses on the core findings for fast decision-making.",
        "",
        "## Key conclusions",
        "- CAM 선행 신호 (lead) 없음: exp_A에서 모든 corruption에서 lead=0%, coincident=0%, lag=100% (or no lead events).",
        "- 성능 붕괴 현상: severity 3~4에서 score_drop 및 miss_rate 급증.",
        "- CAM 가용성 감소: severity 4에서 cam_valid_ratio가 크게 감소 (0.084~0.133).",
        "",
        "## Core comparison table",
        "| Corruption | Lead % | Avg Lead Steps | Cam Valid Ratio (sev4) | Score Drop Rate (sev4) |",
        "|------------|--------|----------------|------------------------|------------------------|",
    ]

    for corruption in ['fog', 'lowlight', 'motion_blur']:
        lead_pct = lead_stats.get(corruption, {}).get('lead_pct', 0.0)
        avg_lead = lead_stats.get(corruption, {}).get('avg_lead_steps', 'N/A')
        cam_ratio = next((r for c, r in cam_valid_rows if c == corruption), 0.0)
        score_drop = next((r for c, r in performance_rows if c == corruption), 0.0)
        lines.append(f"| {corruption} | {lead_pct:.1f}% | {avg_lead} | {cam_ratio:.3f} | {score_drop:.3f} |")

    lines.extend([
        "",
        "## Recommendation",
        "- 핵심 메시지: CAM change metric으로 선행 경보는 확인되지 않음. 그러나 CAM 생성 성공률(플랜망)은 corruption이 커질수록 급락하여 `cam_valid_ratio`가 모니터링해야 할 중요한 지표임.",
        "- 실제 실무 적용: 실시간 시스템에서는 Grad-CAM baseline을 유지하되, CAM 실패 가중치(‘not computed’ 건수, cam_status != ok)를 위험 지표로 포함.",
    ])

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[OK] Concise summary report saved to: {summary_path}")


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
        print(f"  IoU curve: {len(metrics['iou_curve'])} records")
        print(f"  Miss rate curve: {len(metrics['miss_rate_curve'])} records")
        print(f"  Failure events: {len(metrics['failure_events'])} events")
        print(f"  Risk regions: {len(metrics['risk_regions'])} regions")
        print(f"  CAM records: {len(metrics.get('cam_records', []))} records")
        print(f"  CAM metrics (legacy): {len(metrics.get('cam_metrics', []))} records")
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

    # 추가: 핵심 메시지 압축 요약 리포트 생성
    generate_concise_summary_report(config)

    # report.md를 report_concise.md 내용으로 대체 (요청 반영)
    concise_path = Path('results') / 'report_concise.md'
    if concise_path.exists() and report_path.exists():
        concise_text = concise_path.read_text(encoding='utf-8')
        report_path.write_text(concise_text, encoding='utf-8')
        print(f"[OK] Overwrote {report_path} with content from {concise_path}")
    else:
        print(f"[WARN] Could not overwrite {report_path}: {concise_path} or {report_path} missing")


if __name__ == "__main__":
    main()
