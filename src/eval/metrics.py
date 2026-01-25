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
    results = model.val(
        data=str(dataset_yaml),
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        verbose=False
    )
    
    # Extract metrics
    metrics = {
        'map50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        'map5095': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
        'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
    }
    
    return metrics


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
                            'run_id': run_id,
                            'timestamp': timestamp
                        })
                    except Exception as e:
                        print(f"Error evaluating {model_name} on {corruption} severity {severity}: {e}")
                        continue
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
