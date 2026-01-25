"""Detect tiny objects and create time-series logs."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

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
    print("Tiny Object Detection (Time-series)")
    print("=" * 60)
    print()
    
    # Load clips and tiny objects
    results_dir = Path(config['results']['root'])
    clips_file = results_dir / "frame_clips.json"
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    
    if not clips_file.exists() or not tiny_objects_file.exists():
        print("Error: Frame clips or tiny objects not found. Run previous scripts first.")
        sys.exit(1)
    
    clips = load_json(clips_file)
    tiny_objects = load_json(tiny_objects_file)
    
    print(f"Loaded {len(clips)} clips, {len(tiny_objects)} tiny objects")
    
    # Process each model
    models = list(config['models'].keys())
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
        
        print(f"\nProcessing model: {model_name}")
        
        for corruption in corruptions:
            for severity in severities:
                print(f"  {corruption} severity {severity}...")
                
                for tiny_obj in tqdm(tiny_objects, desc=f"    {model_name}-{corruption}-{severity}", leave=False):
                    clip_id = tiny_obj['clip_id']
                    frame_rel_path = tiny_obj['frame_path']
                    
                    # Find frame index in clip
                    clip = next((c for c in clips if c['clip_id'] == clip_id), None)
                    if not clip:
                        continue
                    
                    frame_idx = next((i for i, f in enumerate(clip['frames']) if f == frame_rel_path), None)
                    if frame_idx is None:
                        continue
                    
                    # Get image path
                    if severity == 0:
                        image_path = visdrone_root / frame_rel_path
                    else:
                        image_path = corruptions_root / "sequences" / clip_id / corruption / str(severity) / "images" / Path(frame_rel_path).name
                    
                    if not image_path.exists():
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
                    except Exception as e:
                        continue
                    
                    # Match to GT
                    # Convert VisDrone bbox to YOLO format
                    img_width, img_height = get_image_dimensions(image_path)
                    visdrone_bbox = tiny_obj['bbox']  # (left, top, width, height) in pixels
                    gt_box = visdrone_to_yolo_bbox(visdrone_bbox, img_width, img_height)
                    
                    match_result = match_prediction_to_gt(
                        pred_boxes,
                        gt_box,
                        iou_threshold=config['evaluation']['tiny_match_iou_threshold'],
                        same_class=config['evaluation']['tiny_match_same_class']
                    )
                    
                    if match_result:
                        score, iou = match_result
                        miss = 0
                    else:
                        score, iou = None, None
                        miss = 1
                    
                    records.append({
                        'model': model_name,
                        'corruption': corruption,
                        'severity': severity,
                        'clip_id': clip_id,
                        'frame_idx': frame_idx,
                        'image_id': Path(frame_rel_path).stem,
                        'class_id': tiny_obj['class_id'],
                        'score': score,
                        'iou': iou,
                        'miss': miss
                    })
    
    # Save records
    print("\nSaving time-series records...")
    records_df = pd.DataFrame(records)
    records_csv = results_dir / "tiny_records_timeseries.csv"
    records_df.to_csv(records_csv, index=False)
    print(f"  Saved to {records_csv}")
    print(f"  Total records: {len(records)}")
    
    print("\n[OK] Time-series detection complete!")


if __name__ == "__main__":
    main()
