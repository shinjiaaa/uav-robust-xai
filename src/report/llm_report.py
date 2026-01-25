"""LLM-based report generation."""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os


def load_metrics(config: Dict) -> Dict:
    """Load all computed metrics.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with all metrics
    """
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
    
    # Tiny object curves
    tiny_curves_csv = results_root / "tiny_curves.csv"
    if tiny_curves_csv.exists():
        metrics['tiny_curves'] = pd.read_csv(tiny_curves_csv).to_dict('records')
    else:
        metrics['tiny_curves'] = []
    
    # Tiny object records
    tiny_records_csv = results_root / "tiny_records.csv"
    if tiny_records_csv.exists():
        metrics['tiny_records'] = pd.read_csv(tiny_records_csv).to_dict('records')
    else:
        metrics['tiny_records'] = []
    
    # Grad-CAM metrics
    gradcam_csv = results_root / "gradcam_metrics.csv"
    if gradcam_csv.exists():
        metrics['gradcam'] = pd.read_csv(gradcam_csv).to_dict('records')
    else:
        metrics['gradcam'] = []
    
    return metrics


def summarize_metrics(metrics: Dict) -> Dict:
    """Summarize metrics to reduce token usage.
    
    Args:
        metrics: Full metrics dictionary
        
    Returns:
        Summarized metrics dictionary
    """
    summarized = {}
    
    # Dataset metrics: keep as is (small)
    summarized['dataset'] = metrics.get('dataset', [])
    
    # Detection records: aggregate statistics (per corruption and overall)
    detection_records = metrics.get('detection_records', [])
    if detection_records:
        df = pd.DataFrame(detection_records)
        
        # Overall statistics
        overall_summary = {
            'total_records': len(detection_records),
            'by_model': df.groupby('model').size().to_dict() if 'model' in df.columns else {},
            'by_corruption': df.groupby('corruption').size().to_dict() if 'corruption' in df.columns else {},
            'by_severity': df.groupby('severity').size().to_dict() if 'severity' in df.columns else {},
            'miss_rate_by_severity': df.groupby('severity')['miss'].mean().to_dict() if 'miss' in df.columns else {},
            'avg_score_by_severity': df.groupby('severity')['score'].mean().to_dict() if 'score' in df.columns else {},
            'avg_iou_by_severity': df.groupby('severity')['iou'].mean().to_dict() if 'iou' in df.columns else {},
        }
        
        # Per-corruption statistics (critical for clarity)
        per_corruption = {}
        if 'corruption' in df.columns:
            for corruption in df['corruption'].unique():
                corr_df = df[df['corruption'] == corruption]
                per_corruption[corruption] = {
                    'miss_rate_by_severity': corr_df.groupby('severity')['miss'].mean().to_dict() if 'miss' in corr_df.columns else {},
                    'avg_score_by_severity': corr_df.groupby('severity')['score'].mean().to_dict() if 'score' in corr_df.columns else {},
                    'avg_iou_by_severity': corr_df.groupby('severity')['iou'].mean().to_dict() if 'iou' in corr_df.columns else {},
                }
        
        summarized['detection_summary'] = {
            **overall_summary,
            'per_corruption': per_corruption  # Add per-corruption breakdown
        }
    else:
        summarized['detection_summary'] = {}
    
    # Tiny curves: keep as is (small)
    summarized['tiny_curves'] = metrics.get('tiny_curves', [])
    
    # Failure events: aggregate statistics
    failure_events = metrics.get('failure_events', [])
    if failure_events:
        df = pd.DataFrame(failure_events)
        summarized['failure_summary'] = {
            'total_events': len(failure_events),
            'by_model': df.groupby('model').size().to_dict() if 'model' in df.columns else {},
            'by_corruption': df.groupby('corruption').size().to_dict() if 'corruption' in df.columns else {},
            'by_event_type': df.groupby('event_type').size().to_dict() if 'event_type' in df.columns else {},
            'avg_failure_severity': df['failure_severity'].mean() if 'failure_severity' in df.columns else None,
            'failure_severity_distribution': df['failure_severity'].value_counts().to_dict() if 'failure_severity' in df.columns else {},
        }
    else:
        summarized['failure_summary'] = {}
    
    # Risk regions: keep as is (small)
    summarized['risk_regions'] = metrics.get('risk_regions', [])
    summarized['instability'] = metrics.get('instability', [])
    
    # CAM metrics: aggregate statistics (with lead-lag analysis)
    cam_metrics = metrics.get('cam_metrics', [])
    if cam_metrics:
        df = pd.DataFrame(cam_metrics)
        
        # Overall CAM statistics
        cam_summary = {
            'total_records': len(cam_metrics),
            'by_model': df.groupby('model').size().to_dict() if 'model' in df.columns else {},
            'by_corruption': df.groupby('corruption').size().to_dict() if 'corruption' in df.columns else {},
            'by_severity': df.groupby('severity').size().to_dict() if 'severity' in df.columns else {},
            'avg_energy_in_bbox': df.groupby('severity')['energy_in_bbox'].mean().to_dict() if 'energy_in_bbox' in df.columns else {},
            'avg_activation_spread': df.groupby('severity')['activation_spread'].mean().to_dict() if 'activation_spread' in df.columns else {},
            'avg_entropy': df.groupby('severity')['entropy'].mean().to_dict() if 'entropy' in df.columns else {},
            'avg_center_shift': df.groupby('severity')['center_shift'].mean().to_dict() if 'center_shift' in df.columns else {},
        }
        
        # Lead-lag analysis: CAM changes before failure
        # Group by failure_severity and check CAM at severity K-1, K-2
        if 'failure_severity' in df.columns:
            lead_lag_analysis = {}
            for failure_sev in df['failure_severity'].unique():
                failure_df = df[df['failure_severity'] == failure_sev]
                if failure_sev > 0:
                    # Check CAM at severity K-1 (before failure)
                    prev_sev = failure_sev - 1
                    prev_cam = failure_df[failure_df['severity'] == prev_sev]
                    if len(prev_cam) > 0:
                        lead_lag_analysis[f'failure_at_{int(failure_sev)}'] = {
                            'cam_at_severity_k_minus_1': {
                                'avg_center_shift': prev_cam['center_shift'].mean() if 'center_shift' in prev_cam.columns else None,
                                'avg_activation_spread': prev_cam['activation_spread'].mean() if 'activation_spread' in prev_cam.columns else None,
                            }
                        }
            cam_summary['lead_lag_analysis'] = lead_lag_analysis
        
        summarized['cam_summary'] = cam_summary
        # Sample a few detailed records for context (max 10)
        summarized['cam_samples'] = cam_metrics[:10] if len(cam_metrics) > 10 else cam_metrics
    else:
        summarized['cam_summary'] = {}
        summarized['cam_samples'] = []
    
    return summarized


def generate_report_with_llm(config: Dict, metrics: Dict) -> str:
    """Generate report using LLM.
    
    Args:
        config: Configuration dictionary
        metrics: Dictionary with all metrics
        
    Returns:
        Generated report as markdown string
    """
    # Load API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")
    
    client = OpenAI(api_key=api_key)
    
    # Summarize metrics to reduce token usage
    summarized_metrics = summarize_metrics(metrics)
    
    # Prepare prompt
    prompt = f"""You are a research assistant writing an experiment report. Generate a comprehensive markdown report based on the provided metrics data.

IMPORTANT RULES:
1. DO NOT invent or make up any numbers. Only use the values provided in the metrics data.
2. If a metric is missing, explicitly state "Metric not available" or "Data missing".
3. Be precise and factual. Do not add speculation beyond what the data shows.
4. Use the exact numbers from the data, rounded appropriately (e.g., 2-3 decimal places for percentages).
5. CRITICAL: For "tiny miss rate >= X" statements, check the actual miss rate values in the data tables first.
   - If the maximum miss rate in the data is 0.28 or 0.29, DO NOT claim ">= 50%"
   - Use realistic thresholds: 0.25 (25%) or 0.30 (30%) based on actual observed values
   - Report the exact severity where the threshold is reached using actual data values

Experiment Configuration:
- Seed: {config['seed']}
- Models: {', '.join(config['models'].keys())}
- Corruptions: {', '.join(config['corruptions']['types'])}
- Severities: {config['corruptions']['severities']}
- Tiny object threshold: {config['tiny_objects']['area_threshold']} pixels²

Metrics Data (Summarized, JSON format):
{json.dumps(summarized_metrics, indent=2)}

Generate a markdown report with the following structure:

# Robustness Evaluation Report: VisDrone Object Detection

## 1. Experiment Setup
- Dataset: VisDrone
- Models evaluated: [list models]
- Corruptions: [list corruptions]
- Severity levels: 0-4
- Random seed: {config['seed']}
- Evaluation scope: 
  - Dataset-wide mAP: [specify if computed on full dataset or tiny object subset]
  - Tiny object analysis: Based on sampled {config['experiment']['sample_size']} tiny objects (one per image)
  - If mAP@0.5 baseline is low (e.g., 0.049), clarify whether this is:
    * Full VisDrone validation set evaluation, OR
    * Tiny object subset evaluation, OR
    * Class mismatch issue (COCO pretrained model on VisDrone classes)

## 2. Main Findings

### 2.1 Dataset-wide Performance
For each corruption type, summarize:
- Which model performs best at severity 0 (baseline)
- At what severity level does performance drop significantly (risk region)
- Compare models across severities
- IMPORTANT: If mAP values are missing for severity 1-4, explicitly state "Dataset-wide mAP not computed for severity 1-4" and note that risk region conclusions are based on tiny object analysis only

### 2.2 Tiny Object Analysis (Single Images)
- Miss rate trends across severities (specify if values are per-corruption or overall average)
- Score and IoU drop patterns for ALL corruption types (fog, lowlight, motion_blur) - not just one example
- Instability regions (high variance across severities)
- Which corruption affects tiny objects most
- CRITICAL: When reporting miss rates, clearly specify:
  - "Overall average miss rate" vs "Per-corruption miss rate"
  - Example: "Overall miss rate at severity 4: 0.16" vs "Fog miss rate at severity 4: 0.16" vs "Motion blur miss rate at severity 4: 0.29"

### 2.3 Failure Event Analysis
- First miss events: severity where detection first fails
- Score/IoU drop events: significant degradation points
- Failure event distribution across corruptions and models

### 2.4 Grad-CAM Distribution Analysis (Failure-Event Based)
For failure events, analyze CAM distribution metrics across severities up to failure:
- Energy in bbox: activation concentration in GT region
- Activation spread: spatial dispersion changes
- Entropy: distribution entropy evolution
- Center shift: movement of activation center
- CRITICAL: Analyze lead-lag relationship - do CAM changes occur BEFORE failure (K steps before miss)?
  - If CAM metrics change at severity K-1 or K-2 before failure at severity K, this suggests CAM as a "precursor signal"
  - If CAM changes only at the same severity as failure, state "CAM changes co-occur with failure but do not precede it"
- Identify consistent CAM change patterns that co-occur with performance degradation

## 3. Evidence Tables
Include key metrics in tabular format:
- mAP@0.5 and mAP@0.5:0.95 per model/corruption/severity
  - If values are missing for severity 1-4, mark as "Not computed" (not "Data missing")
  - Note: "mAP values shown are for [full dataset / tiny subset]"
- Tiny object miss rates per corruption type (fog, lowlight, motion_blur) and overall average
- Score and IoU drop patterns for ALL corruption types (not just fog)
- Failure event counts and types
- CAM distribution metrics (before failure events)
  - Include lead-lag analysis: CAM changes at severity K-1 vs failure at severity K

## 4. Risk Region Detection
For each model and corruption:
- Severity where mAP drops by >=15% from baseline
  - If dataset-wide mAP is missing for severity 1-4, state: "Dataset-wide mAP not available. Risk region conclusions based on tiny object analysis only."
- Severity where tiny miss rate >= 0.25 (25%) or >= 0.3 (30%) - USE ACTUAL VALUES FROM DATA, NOT HARDCODED 50%
  - Specify per-corruption: "lowlight: severity 4 (0.28)", "motion_blur: severity 4 (0.29)"
- Frame regions with high instability
- Summary of vulnerability patterns

CRITICAL: When reporting "tiny miss rate >= X", you MUST:
1. Check the actual miss rate values in the provided data
2. Use realistic thresholds (0.25 or 0.3) based on the actual maximum values observed
3. NEVER claim ">= 50%" if the actual maximum miss rate in the data is lower (e.g., 0.28, 0.29)
4. Report the actual severity where the threshold is reached, using the exact values from the data tables
5. Specify whether the threshold is reached for specific corruptions or overall average

## 5. CAM Change Pattern Generalization
For each corruption type and model:
- Do CAM distribution metrics show consistent change patterns BEFORE failure (lead-lag analysis)?
  - Check if CAM changes at severity K-1 occur before failure at severity K
  - If CAM changes precede failure, this supports "CAM as precursor signal" claim
  - If CAM changes only co-occur with failure, state "CAM changes coincide with failure but do not precede it"
- Are these patterns generalizable across different corruption types?
- Are these patterns generalizable across different models (Generic/FT/RT-DETR)?
- Identify CAM-based risk signals that appear before performance degradation

## 5. Reproducibility
- Random seed used: {config['seed']}
- Configuration hash: [if available]
- Run commands: [list the script execution commands]

Remember: Only use numbers from the provided data. If data is missing, state it clearly."""

    # Call LLM
    response = client.chat.completions.create(
        model=config['llm_report']['model'],
        messages=[
            {"role": "system", "content": "You are a scientific report writer. Generate accurate, factual reports based only on provided data."},
            {"role": "user", "content": prompt}
        ],
        temperature=config['llm_report']['temperature'],
        max_tokens=config['llm_report']['max_tokens']
    )
    
    report = response.choices[0].message.content
    
    return report


def save_report(report: str, output_path: Path):
    """Save report to file.
    
    Args:
        report: Report content
        output_path: Path to save report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {output_path}")
