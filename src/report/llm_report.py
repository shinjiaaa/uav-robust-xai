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
        metrics['dataset'] = pd.read_csv(metrics_csv).to_dict('records')
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
    
    # Prepare prompt
    prompt = f"""You are a research assistant writing an experiment report. Generate a comprehensive markdown report based on the provided metrics data.

IMPORTANT RULES:
1. DO NOT invent or make up any numbers. Only use the values provided in the metrics data.
2. If a metric is missing, explicitly state "Metric not available" or "Data missing".
3. Be precise and factual. Do not add speculation beyond what the data shows.
4. Use the exact numbers from the data, rounded appropriately (e.g., 2-3 decimal places for percentages).

Experiment Configuration:
- Seed: {config['seed']}
- Models: {', '.join(config['models'].keys())}
- Corruptions: {', '.join(config['corruptions']['types'])}
- Severities: {config['corruptions']['severities']}
- Tiny object threshold: {config['tiny_objects']['area_threshold']} pixels²

Metrics Data (JSON format):
{json.dumps(metrics, indent=2)}

Generate a markdown report with the following structure:

# Robustness Evaluation Report: VisDrone Object Detection

## 1. Experiment Setup
- Dataset: VisDrone
- Models evaluated: [list models]
- Corruptions: [list corruptions]
- Severity levels: 0-4
- Random seed: {config['seed']}

## 2. Main Findings

### 2.1 Dataset-wide Performance
For each corruption type, summarize:
- Which model performs best at severity 0 (baseline)
- At what severity level does performance drop significantly (risk region)
- Compare models across severities

### 2.2 Tiny Object Analysis (Continuous Frame Sequences)
- Miss rate trends across severities and frames
- Score and IoU drop patterns (time-series)
- Instability regions (high variance frames)
- Which corruption affects tiny objects most

### 2.3 Failure Event Analysis
- First miss events: severity/frame where detection first fails
- Score/IoU drop events: significant degradation points
- Failure event distribution across corruptions and models

### 2.4 Grad-CAM Distribution Analysis (Failure-Event Based)
For failure events, analyze CAM distribution metrics in the W-frame window before failure:
- Energy in bbox: activation concentration in GT region
- Activation spread: spatial dispersion changes
- Entropy: distribution entropy evolution
- Center shift: movement of activation center
- Identify consistent CAM change patterns that co-occur with performance degradation

## 3. Evidence Tables
Include key metrics in tabular format:
- mAP@0.5 and mAP@0.5:0.95 per model/corruption/severity
- Tiny object miss rates (time-series aggregated)
- Failure event counts and types
- CAM distribution metrics (before failure events)

## 4. Risk Region Detection
For each model and corruption:
- Severity where mAP drops by >=15% from baseline
- Severity where tiny miss rate >= 50%
- Frame regions with high instability
- Summary of vulnerability patterns

## 5. CAM Change Pattern Generalization
For each corruption type and model:
- Do CAM distribution metrics show consistent change patterns before failure?
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
