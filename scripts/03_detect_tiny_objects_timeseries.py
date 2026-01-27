"""Detect tiny objects and create time-series logs."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.tiny_match import run_inference_on_image, match_prediction_to_gt, load_yolo_label, compute_iou
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
    
    # VisDrone class names mapping (10 classes)
    VISDRONE_CLASS_NAMES = {
        0: 'ignored regions',
        1: 'pedestrian',
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle',
        9: 'bus',
        10: 'motor',
        11: 'others'
    }
    
    # COCO class names (for YOLO predictions) - full 80 classes
    COCO_CLASS_NAMES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
        15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    
    # Generate run_id from timestamp
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Get dataset info
    dataset = "VisDrone2019-DET"
    split = config['evaluation']['splits'][0] if 'evaluation' in config and 'splits' in config['evaluation'] else "val"
    model_family = "YOLO"  # Default, can be extended
    
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
                        # CRITICAL: Get all predictions for debug info (best_iou_any_pred, best_score_any_pred)
                        n_preds_total = len(pred_boxes)
                        best_iou_any_pred = 0.0
                        best_score_any_pred = 0.0
                        best_pred_class_any = None
                        
                        # Find best match regardless of class (for debug)
                        gt_box_coords = (yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3])
                        for pred_class, pred_x, pred_y, pred_w, pred_h, pred_score in pred_boxes:
                            pred_box_coords = (pred_x, pred_y, pred_w, pred_h)
                            iou_any = compute_iou(gt_box_coords, pred_box_coords)
                            if iou_any > best_iou_any_pred:
                                best_iou_any_pred = iou_any
                                best_score_any_pred = pred_score
                                best_pred_class_any = pred_class
                        
                        if match_result:
                            score, iou, pred_bbox_norm = match_result  # pred_bbox_norm: (x_center, y_center, width, height) normalized
                            matched = 1
                            miss = 0
                            
                            # Extract pred_class_id from matched prediction
                            # Find the matched prediction's class
                            pred_class_id = None
                            for pred_class, pred_x, pred_y, pred_w, pred_h, pred_score in pred_boxes:
                                pred_box_coords = (pred_x, pred_y, pred_w, pred_h)
                                if abs(pred_score - score) < 1e-6:  # Same score (matched prediction)
                                    iou_check = compute_iou(gt_box_coords, pred_box_coords)
                                    if abs(iou_check - iou) < 1e-6:  # Same IoU (matched prediction)
                                        pred_class_id = pred_class
                                        break
                            
                            # Convert normalized pred_bbox to pixel coordinates (x1, y1, x2, y2)
                            pred_x_center, pred_y_center, pred_w, pred_h = pred_bbox_norm
                            pred_x1 = (pred_x_center - pred_w / 2) * img_width
                            pred_y1 = (pred_y_center - pred_h / 2) * img_height
                            pred_x2 = (pred_x_center + pred_w / 2) * img_width
                            pred_y2 = (pred_y_center + pred_h / 2) * img_height
                        else:
                            score, iou = None, None
                            matched = 0
                            miss = 1
                            pred_class_id = None
                            pred_x1, pred_y1, pred_x2, pred_y2 = None, None, None, None
                        
                        # GT bbox in pixel coordinates (x1, y1, x2, y2)
                        gt_left, gt_top, gt_width, gt_height = visdrone_bbox
                        gt_x1 = gt_left
                        gt_y1 = gt_top
                        gt_x2 = gt_left + gt_width
                        gt_y2 = gt_top + gt_height
                        gt_area = gt_width * gt_height
                        
                        # Generate object_uid: unique identifier for this tiny object across all frames/severities
                        # Format: {image_id}_obj{object_index}_{class_id}
                        object_index = tiny_obj.get('object_index', tiny_obj.get('object_id', ''))
                        if not object_index:
                            # Fallback: use hash of bbox position
                            object_index = hash((gt_x1, gt_y1, gt_x2, gt_y2)) % 10000
                        object_uid = f"{image_id}_obj{object_index}_cls{class_id}"
                        
                        # Frame index (within clip) - for now, assume single image (frame_idx=0)
                        # TODO: If processing video clips, extract frame_idx from clip_id
                        frame_idx = tiny_obj.get('frame_idx', 0)
                        
                        # Get class names
                        gt_class_name = VISDRONE_CLASS_NAMES.get(class_id, f'class_{class_id}')
                        pred_class_name = COCO_CLASS_NAMES.get(pred_class_id, f'class_{pred_class_id}') if pred_class_id is not None else None
                        
                        # Determine if tiny (based on area threshold)
                        is_tiny = 1 if gt_area <= config['tiny_objects']['area_threshold'] else 0
                        
                        # Get iou_threshold from config
                        iou_threshold = config['evaluation']['tiny_match_iou_threshold']
                        
                        # Pred rank (1 if matched, None if miss)
                        pred_rank = 1 if matched == 1 else None
                        
                        # CRITICAL: Standard schema for RQ1 automatic report generation
                        record = {
                            # A. 실험 키 (재현성/조인키) - 표준 스키마
                            'run_id': run_id,
                            'model_id': model_name,
                            'model_family': model_family,
                            'dataset': dataset,
                            'split': split,
                            'corruption': corruption,
                            'severity': severity,
                            'clip_id': tiny_obj.get('clip_id', ''),
                            'frame_idx': frame_idx,
                            'image_id': image_id,
                            'image_path': str(frame_rel_path),  # Original image path
                            'corrupted_image_path': str(image_path),  # 실제 사용한 변조 이미지 경로
                            
                            # B. 타이니 객체 추적 (100개 "각각"을 보장)
                            'object_uid': object_uid,
                            'gt_class_id': class_id,
                            'gt_class_name': gt_class_name,
                            'gt_x1': gt_x1,  # 표준 스키마: gt_x1 (not gt_bbox_x1)
                            'gt_y1': gt_y1,
                            'gt_x2': gt_x2,
                            'gt_y2': gt_y2,
                            'gt_area': gt_area,
                            'is_tiny': is_tiny,
                            
                            # 매칭 설정
                            'match_policy': 'max_iou_same_class' if config['evaluation']['tiny_match_same_class'] else 'max_iou_any_class',
                            'iou_threshold': iou_threshold,
                            
                            # C. 매칭 & 성능 원시값 (모든 curve의 원천)
                            'matched': matched,
                            'is_miss': 1 if matched == 0 or (iou is not None and iou < iou_threshold) else 0,
                            'pred_rank': pred_rank,
                            'pred_class_id': pred_class_id,
                            'pred_class_name': pred_class_name,
                            'pred_score': score if score is not None else 0.0,
                            'match_iou': iou if iou is not None else 0.0,
                            'pred_x1': pred_x1 if pred_x1 is not None else None,  # 표준 스키마: pred_x1 (not pred_bbox_x1)
                            'pred_y1': pred_y1 if pred_y1 is not None else None,
                            'pred_x2': pred_x2 if pred_x2 is not None else None,
                            'pred_y2': pred_y2 if pred_y2 is not None else None,
                            
                            # E. 디버그 정보
                            'n_preds_total': n_preds_total,
                            'best_iou_any_pred': best_iou_any_pred,
                            'best_score_any_pred': best_score_any_pred,
                            
                            # F. drop curve를 "바로" 만들기 위한 기준값 (나중에 추가됨)
                            'base_pred_score': None,  # Will be filled in post-processing
                            'base_match_iou': None,
                            'delta_score': None,
                            'delta_iou': None,
                            
                            # D. failure 이벤트 기반 요약 (나중에 추가됨)
                            'is_score_drop': None,  # Will be filled in post-processing
                            'is_iou_drop': None,
                            'failure_type': None,
                            'failure_event_id': None,
                            'notes': None
                        }
                        
                        # Legacy aliases for backward compatibility (keep for now)
                        record['model'] = model_name
                        record['class_id'] = class_id
                        record['object_id'] = tiny_obj.get('object_id', '')
                        record['frame_path'] = str(frame_rel_path)
                        record['frame_id'] = Path(frame_rel_path).stem
                        record['gt_bbox_x1'] = gt_x1
                        record['gt_bbox_y1'] = gt_y1
                        record['gt_bbox_x2'] = gt_x2
                        record['gt_bbox_y2'] = gt_y2
                        record['pred_bbox_x1'] = pred_x1
                        record['pred_bbox_y1'] = pred_y1
                        record['pred_bbox_x2'] = pred_x2
                        record['pred_bbox_y2'] = pred_y2
                        record['conf'] = score
                        record['score'] = score
                        record['iou'] = iou
                        record['miss'] = miss
                        record['best_pred_class_any'] = best_pred_class_any
                        
                        records.append(record)
                        corruption_record_count += 1
                    
                    # CRITICAL: Aggressive memory cleanup after each batch
                    gc.collect()
                    gc.collect()  # Second pass for stubborn references
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            # CRITICAL: Post-process records to add baseline comparisons and failure flags
            # This requires severity 0 baseline values for each object_uid (within this corruption)
            # Filter records for current corruption only
            corruption_records = [r for r in records if r.get('corruption') == corruption and r.get('model') == model_name]
            if len(corruption_records) > 0:
                # Create baseline lookup (severity 0 values per object_uid for this corruption)
                baseline_lookup = {}
                for rec in corruption_records:
                    if rec['severity'] == 0:
                        object_uid = rec['object_uid']
                        baseline_lookup[object_uid] = {
                            'base_pred_score': rec['pred_score'],
                            'base_match_iou': rec['match_iou'],
                            'base_matched': rec['matched']
                        }
                
                # Add baseline comparisons and failure flags to all records of this corruption
                for rec in corruption_records:
                    object_uid = rec['object_uid']
                    baseline = baseline_lookup.get(object_uid, {})
                    
                    # F. 기준값 비교 (표준 스키마)
                    rec['base_pred_score'] = baseline.get('base_pred_score', None)
                    rec['base_match_iou'] = baseline.get('base_match_iou', None)
                    
                    if rec['base_pred_score'] is not None:
                        rec['delta_score'] = rec['pred_score'] - rec['base_pred_score']
                    else:
                        rec['delta_score'] = None
                    
                    if rec['base_match_iou'] is not None:
                        rec['delta_iou'] = rec['match_iou'] - rec['base_match_iou']
                    else:
                        rec['delta_iou'] = None
                    
                    # D. 실패 플래그 (표준 스키마)
                    # is_miss는 이미 record 생성 시 계산됨, 여기서는 업데이트만
                    iou_thresh = rec.get('iou_threshold', 0.5)
                    rec['is_miss'] = 1 if rec['matched'] == 0 or rec['match_iou'] < iou_thresh else 0
                    
                    # IoU drop: baseline IoU 대비 0.2 이상 감소
                    if rec['base_match_iou'] is not None and rec['base_match_iou'] > 0:
                        iou_drop_threshold = 0.2
                        rec['is_iou_drop'] = 1 if rec['delta_iou'] is not None and rec['delta_iou'] < -iou_drop_threshold else 0
                    else:
                        rec['is_iou_drop'] = 0
                    
                    # Score drop: baseline score 대비 0.3 이상 감소
                    if rec['base_pred_score'] is not None and rec['base_pred_score'] > 0:
                        score_drop_threshold = 0.3
                        rec['is_score_drop'] = 1 if rec['delta_score'] is not None and rec['delta_score'] < -score_drop_threshold else 0
                    else:
                        rec['is_score_drop'] = 0
                    
                    # Failure type
                    if rec['is_miss'] == 1:
                        rec['failure_type'] = 'miss'
                    elif rec['is_iou_drop'] == 1:
                        rec['failure_type'] = 'iou_drop'
                    elif rec['is_score_drop'] == 1:
                        rec['failure_type'] = 'score_drop'
                    else:
                        rec['failure_type'] = 'none'
                    
                    # Failure event ID (will be assigned later in failure detection script)
                    rec['failure_event_id'] = None
            
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
    
    # CRITICAL: Reorder columns to match standard schema (RQ1 automatic report generation)
    standard_columns = [
        'run_id', 'model_id', 'model_family', 'dataset', 'split', 'corruption', 'severity',
        'clip_id', 'frame_idx', 'image_id', 'image_path', 'corrupted_image_path',
        'object_uid', 'gt_class_id', 'gt_class_name', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'gt_area', 'is_tiny',
        'match_policy', 'iou_threshold', 'matched', 'is_miss', 'pred_rank',
        'pred_class_id', 'pred_class_name', 'pred_score', 'match_iou',
        'pred_x1', 'pred_y1', 'pred_x2', 'pred_y2',
        'n_preds_total', 'best_iou_any_pred', 'best_score_any_pred',
        'base_pred_score', 'base_match_iou', 'delta_score', 'delta_iou',
        'is_score_drop', 'is_iou_drop', 'failure_type', 'failure_event_id', 'notes'
    ]
    
    # Add any extra columns that exist but aren't in standard schema (legacy columns)
    extra_columns = [col for col in records_df.columns if col not in standard_columns]
    all_columns = standard_columns + extra_columns
    
    # Reorder DataFrame
    available_columns = [col for col in all_columns if col in records_df.columns]
    records_df = records_df[available_columns]
    
    # CRITICAL: Save as both tiny_records_timeseries.csv (for backward compatibility)
    # and detection_records.csv (for explicit report usage with standard schema)
    timeseries_csv = results_dir / "tiny_records_timeseries.csv"
    detection_records_csv = results_dir / "detection_records.csv"
    
    records_df.to_csv(timeseries_csv, index=False)
    records_df.to_csv(detection_records_csv, index=False)
    
    print(f"  Saved {len(records_df)} records to {detection_records_csv}")
    print(f"  Standard schema columns: {len([c for c in standard_columns if c in records_df.columns])}/{len(standard_columns)}")
    
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
