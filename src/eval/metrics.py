"""Evaluation metrics computation."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from ultralytics import YOLO
import torch

# RT-DETR might not be available in all ultralytics versions
try:
    from ultralytics import RTDETR
except ImportError:
    # Fallback: use YOLO for RT-DETR if not available
    RTDETR = None


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (x_center, y_center, width, height) format (normalized).
    
    Args:
        box1: First box (x_center, y_center, width, height)
        box2: Second box (x_center, y_center, width, height)
        
    Returns:
        IoU value
    """
    # Convert to (x1, y1, x2, y2)
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_model_dataset_wide(
    model_path: str,
    model_type: str,
    dataset_yaml: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Evaluate model on dataset and return metrics.
    
    Args:
        model_path: Path to model checkpoint or model name
        model_type: 'yolo' or 'rtdetr'
        dataset_yaml: Path to dataset YAML file
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        device: Device to run on
        
    Returns:
        Dictionary with metrics: map50, map5095, precision, recall
    """
    # Load model
    if model_type == 'yolo':
        model = YOLO(model_path)
    elif model_type == 'rtdetr':
        if RTDETR is None:
            # Fallback to YOLO if RT-DETR not available
            print("Warning: RT-DETR not available, using YOLO instead")
            model = YOLO(model_path)
        else:
            model = RTDETR(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run validation
    # CRITICAL: Set workers=0 on Windows to avoid multiprocessing issues
    import platform
    workers = 0 if platform.system() == 'Windows' else 4
    
    try:
        results = model.val(
            data=str(dataset_yaml),
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=False,
            workers=workers  # Fix Windows multiprocessing issue
        )
    except Exception as e:
        # If validation fails, return error status
        print(f"ERROR in model.val(): {e}")
        import traceback
        traceback.print_exc()
        return {
            'map50': 0.0,
            'map5095': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'pred_count': 0,
            'eval_status': 'error'
        }
    
    # Extract metrics with NaN handling (empty predictions -> 0.0)
    def safe_float(value, default=0.0):
        """Convert to float, handling NaN and None."""
        if value is None:
            return default
        try:
            val = float(value)
            # Check for NaN
            if val != val:  # NaN check (NaN != NaN)
                return default
            return val
        except (ValueError, TypeError):
            return default
    
    # Extract metrics with robust error handling
    try:
        # Check if results.box exists and has metrics
        if not hasattr(results, 'box'):
            print("WARNING: results.box does not exist")
            return {
                'map50': 0.0,
                'map5095': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'pred_count': 0,
                'eval_status': 'error'
            }
        
        metrics = {
            'map50': safe_float(results.box.map50 if hasattr(results.box, 'map50') else None),
            'map5095': safe_float(results.box.map if hasattr(results.box, 'map') else None),
            'precision': safe_float(results.box.mp if hasattr(results.box, 'mp') else None),
            'recall': safe_float(results.box.mr if hasattr(results.box, 'mr') else None),
        }
        
        # Additional info for debugging
        metrics['pred_count'] = getattr(results.box, 'nc', 0)  # Number of classes (proxy for predictions)
        metrics['eval_status'] = 'ok'
        
        # Debug: Print actual values if they seem wrong
        if metrics['map50'] == 0.0 and metrics['map5095'] == 0.0:
            print(f"WARNING: mAP is 0.0. Checking results.box attributes...")
            print(f"  results.box attributes: {dir(results.box)}")
            if hasattr(results.box, 'map50'):
                print(f"  results.box.map50 (raw): {results.box.map50}")
            if hasattr(results.box, 'map'):
                print(f"  results.box.map (raw): {results.box.map}")
        
        return metrics
    except Exception as e:
        print(f"ERROR extracting metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'map50': 0.0,
            'map5095': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'pred_count': 0,
            'eval_status': 'error'
        }


def evaluate_all_models(
    config: Dict,
    models: List[str],
    splits: List[str],
    corruption_types: List[str],
    severities: List[int],
    output_csv: Path
):
    """Evaluate all models on all corruption/severity combinations.
    
    Args:
        config: Configuration dictionary
        models: List of model names to evaluate
        splits: List of splits to evaluate on
        corruption_types: List of corruption types
        severities: List of severities
        output_csv: Path to save results CSV
    """
    from datetime import datetime
    import uuid
    
    results = []
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    inference_config = config['inference']
    conf_thres = inference_config['conf_thres']
    iou_thres = inference_config['iou_thres']
    
    for model_name in models:
        model_config = config['models'][model_name]
        model_type = model_config['type']
        
        # Get model path
        if model_config['fine_tuned'] and Path(model_config['checkpoint']).exists():
            model_path = model_config['checkpoint']
        else:
            model_path = model_config['pretrained']
        
        for split in splits:
            for corruption in corruption_types:
                for severity in severities:
                    # Dataset path
                    if severity == 0:
                        # Use original dataset
                        dataset_yaml = Path(config['dataset']['visdrone_yolo_root']) / "visdrone.yaml"
                        # But we need to point to the right split
                        # For now, use corruption path structure
                        dataset_yaml = Path(config['dataset']['corruptions_root']) / corruption / str(severity) / split / "data.yaml"
                    else:
                        dataset_yaml = Path(config['dataset']['corruptions_root']) / corruption / str(severity) / split / "data.yaml"
                    
                    if not dataset_yaml.exists():
                        print(f"Warning: {dataset_yaml} does not exist. Skipping.")
                        continue
                    
                    print(f"Evaluating {model_name} on {corruption} severity {severity} {split}...")
                    
                    try:
                        metrics = evaluate_model_dataset_wide(
                            model_path,
                            model_type,
                            dataset_yaml,
                            conf_thres,
                            iou_thres
                        )
                        
                        results.append({
                            'model': model_name,
                            'split': split,
                            'corruption': corruption,
                            'severity': severity,
                            'map50': metrics['map50'],
                            'map5095': metrics['map5095'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'pred_count': metrics.get('pred_count', 0),
                            'eval_status': metrics.get('eval_status', 'ok'),
                            'run_id': run_id,
                            'timestamp': timestamp
                        })
                    except Exception as e:
                        print(f"Error evaluating {model_name} on {corruption} severity {severity}: {e}")
                        # Record error case with 0 metrics
                        results.append({
                            'model': model_name,
                            'split': split,
                            'corruption': corruption,
                            'severity': severity,
                            'map50': 0.0,
                            'map5095': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'pred_count': 0,
                            'eval_status': 'error',
                            'run_id': run_id,
                            'timestamp': timestamp
                        })
                        continue
    
    # Save to CSV with NaN handling
    df = pd.DataFrame(results)
    # Fill NaN values with 0.0 (empty predictions -> 0 metrics)
    df = df.fillna(0.0)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
    
    # Print summary of empty predictions
    if 'eval_status' in df.columns:
        error_count = (df['eval_status'] == 'error').sum()
        if error_count > 0:
            print(f"Warning: {error_count} evaluations had errors (recorded as 0.0 metrics)")
