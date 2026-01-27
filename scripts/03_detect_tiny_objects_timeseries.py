"""Detect tiny objects and create time-series logs."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.tiny_match import run_inference_on_image, match_prediction_to_gt, load_yolo_label
from src.data.bbox_conversion import visdrone_to_yolo_bbox, get_image_dimensions
from src.utils.io import load_yaml, load_json
from src.utils.seed import set_seed
from tqdm import tqdm


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Tiny Object Detection")
    print("=" * 60)
    print()
    
    # Load tiny objects
    results_dir = Path(config['results']['root'])
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    
    if not tiny_objects_file.exists():
        print("Error: Tiny objects not found. Run scripts/01_sample_tiny_objects.py first.")
        sys.exit(1)
    
    tiny_objects = load_json(tiny_objects_file)
    
    print(f"Loaded {len(tiny_objects)} tiny objects")
    
    # Process each model (only yolo_generic for pilot)
    models = ['yolo_generic']  # Only Generic YOLO
    corruptions = config['corruptions']['types']
    severities = config['corruptions']['severities']
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    records = []
    
    for model_name in models:
        model_config = config['models'][model_name]
        model_type = model_config['type']
        
        if model_config['fine_tuned'] and Path(model_config['checkpoint']).exists():
            model_path = model_config['checkpoint']
        else:
            model_path = model_config['pretrained']
        
        # CRITICAL: Check if model file is accessible before processing
        if not Path(model_path).exists():
            print(f"[ERROR] Model file not found: {model_path}")
            continue
        
        # Try to check file access (detect if locked)
        try:
            with open(model_path, 'rb') as f:
                f.read(1)  # Try to read first byte
        except (OSError, IOError, PermissionError) as e:
            print(f"[ERROR] Cannot access model file (may be locked by another process): {model_path}")
            print(f"[ERROR] Error: {e}")
            print(f"[ERROR] Please close other processes using this model file and try again")
            continue
        
        print(f"\nProcessing model: {model_name}")
        
        for corruption in corruptions:
            corruption_record_count = 0
            corruption_missing_image_count = 0
            
            for severity in severities:
                print(f"  {corruption} severity {severity}...")
                
                # CRITICAL: Process in very small batches to reduce memory pressure
                batch_size = 5  # Reduced to 5 images per batch for memory-constrained systems
                for batch_start in range(0, len(tiny_objects), batch_size):
                    batch_end = min(batch_start + batch_size, len(tiny_objects))
                    batch_objects = tiny_objects[batch_start:batch_end]
                    
                    for tiny_obj in batch_objects:
                        image_id = tiny_obj['image_id']
                        frame_rel_path = tiny_obj['frame_path']
                        
                        # Get image path
                        if severity == 0:
                            image_path = visdrone_root / frame_rel_path
                        else:
                            # New structure: corruptions_root / corruption / severity / images
                            image_path = corruptions_root / corruption / str(severity) / "images" / Path(frame_rel_path).name
                        
                        if not image_path.exists():
                            corruption_missing_image_count += 1
                            continue
                        
                        # Get annotation
                        ann_path = visdrone_root / "VisDrone2019-DET-val" / "annotations" / f"{Path(frame_rel_path).stem}.txt"
                        if not ann_path.exists():
                            continue
                        
                        # Run inference
                        try:
                            pred_boxes = run_inference_on_image(
                                model_path,
                                model_type,
                                image_path,
                                conf_thres=config['inference']['conf_thres'],
                                iou_thres=config['inference']['iou_thres']
                            )
                        except MemoryError as e:
                            print(f"  [ERROR] Memory error for {image_path.name}: {e}")
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        except Exception as e:
                            print(f"  [WARN] Inference error for {image_path.name}: {e}")
                            continue
                        
                        # Match to GT
                        # Convert VisDrone bbox to YOLO format
                        img_width, img_height = get_image_dimensions(image_path)
                        visdrone_bbox = tiny_obj['bbox']  # (left, top, width, height) in pixels
                        yolo_bbox = visdrone_to_yolo_bbox(visdrone_bbox, img_width, img_height)
                        
                        # GT box format: (class_id, x_center, y_center, width, height) normalized
                        class_id = tiny_obj['class_id']
                        gt_box = (class_id, yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3])
                        
                        match_result = match_prediction_to_gt(
                            pred_boxes,
                            gt_box,
                            iou_threshold=config['evaluation']['tiny_match_iou_threshold'],
                            same_class=config['evaluation']['tiny_match_same_class']
                        )
                        
                        # Extract match results and bbox information
                        if match_result:
                            score, iou, pred_bbox_norm = match_result  # pred_bbox_norm: (x_center, y_center, width, height) normalized
                            miss = 0
                            
                            # Convert normalized pred_bbox to pixel coordinates (x1, y1, x2, y2)
                            pred_x_center, pred_y_center, pred_w, pred_h = pred_bbox_norm
                            pred_x1 = (pred_x_center - pred_w / 2) * img_width
                            pred_y1 = (pred_y_center - pred_h / 2) * img_height
                            pred_x2 = (pred_x_center + pred_w / 2) * img_width
                            pred_y2 = (pred_y_center + pred_h / 2) * img_height
                        else:
                            score, iou = None, None
                            miss = 1
                            pred_x1, pred_y1, pred_x2, pred_y2 = None, None, None, None
                        
                        # GT bbox in pixel coordinates (x1, y1, x2, y2)
                        gt_left, gt_top, gt_width, gt_height = visdrone_bbox
                        gt_x1 = gt_left
                        gt_y1 = gt_top
                        gt_x2 = gt_left + gt_width
                        gt_y2 = gt_top + gt_height
                        
                        # CRITICAL: Include frame-level identifiers and bbox information for proper frame counting
                        records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'severity': severity,
                            'image_id': image_id,
                            'frame_path': str(frame_rel_path),
                            'frame_id': Path(frame_rel_path).stem,  # Frame key (filename without extension)
                            'clip_id': tiny_obj.get('clip_id', ''),  # Clip ID if available
                            'object_id': tiny_obj.get('object_id', ''),  # Object ID if available
                            'class_id': class_id,
                            # GT bbox (pixel coordinates)
                            'gt_bbox_x1': gt_x1,
                            'gt_bbox_y1': gt_y1,
                            'gt_bbox_x2': gt_x2,
                            'gt_bbox_y2': gt_y2,
                            # Pred bbox (pixel coordinates, None if miss)
                            'pred_bbox_x1': pred_x1,
                            'pred_bbox_y1': pred_y1,
                            'pred_bbox_x2': pred_x2,
                            'pred_bbox_y2': pred_y2,
                            # Performance metrics
                            'conf': score,  # Confidence score (same as score)
                            'score': score,  # Alias for backward compatibility
                            'iou': iou,
                            'miss': miss
                        })
                        corruption_record_count += 1
                    
                    # CRITICAL: Aggressive memory cleanup after each batch
                    gc.collect()
                    gc.collect()  # Second pass for stubborn references
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            # Log corruption summary
            print(f"  [{corruption}] Total records: {corruption_record_count}, Missing images: {corruption_missing_image_count}")
            
            # CRITICAL: Aggressive memory cleanup after each corruption
            from src.eval.tiny_match import clear_model_cache
            clear_model_cache()  # Clear model cache
            gc.collect()
            gc.collect()  # Second pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # Second pass
    
    # CRITICAL: Final cleanup before saving
    from src.eval.tiny_match import clear_model_cache
    clear_model_cache()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Save records
    print("\nSaving detection records...")
    records_df = pd.DataFrame(records)
    
    # CRITICAL: Save as both tiny_records_timeseries.csv (for backward compatibility)
    # and detection_records.csv (for explicit report usage)
    timeseries_csv = results_dir / "tiny_records_timeseries.csv"
    detection_records_csv = results_dir / "detection_records.csv"
    
    records_df.to_csv(timeseries_csv, index=False)
    records_df.to_csv(detection_records_csv, index=False)
    
    print(f"  Saved to {timeseries_csv}")
    print(f"  Saved to {detection_records_csv}")
    print(f"  Total records: {len(records)}")
    
    # Final cleanup
    del records_df
    del records
    gc.collect()
    
    print("\n[OK] Detection complete!")


if __name__ == "__main__":
    main()
