"""Grad-CAM analysis for failure events."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai.gradcam_yolo import YOLOGradCAM
from src.xai.cam_metrics import compute_cam_metrics
from src.data.bbox_conversion import visdrone_to_yolo_bbox, get_image_dimensions
from src.utils.io import load_yaml, load_json
from src.utils.seed import set_seed
from ultralytics import YOLO
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
    print("Grad-CAM Failure Event Analysis")
    print("=" * 60)
    print()
    
    if not config['gradcam']['enabled']:
        print("Grad-CAM analysis is disabled in config.")
        sys.exit(0)
    
    results_dir = Path(config['results']['root'])
    
    # Load failure events
    failure_events_csv = results_dir / "failure_events.csv"
    if not failure_events_csv.exists():
        print("Error: Failure events not found. Run scripts/04_detect_failure_events.py first")
        sys.exit(1)
    
    failure_events_df = pd.read_csv(failure_events_csv)
    print(f"Loaded {len(failure_events_df)} failure events")
    
    # Load clips and tiny objects
    clips_file = results_dir / "frame_clips.json"
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    clips = load_json(clips_file)
    tiny_objects = load_json(tiny_objects_file)
    
    # Create tiny object lookup
    tiny_obj_lookup = {}
    for obj in tiny_objects:
        key = (obj['clip_id'], obj['frame_path'], obj['class_id'])
        tiny_obj_lookup[key] = obj
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    window_size = config['risk_detection']['window_size']
    gradcam_config = config['gradcam']
    
    cam_metrics_records = []
    
    # Process failure events (limit to max_samples)
    max_samples = gradcam_config.get('max_samples', 20)
    sample_events = failure_events_df.head(max_samples)
    
    print(f"\nProcessing {len(sample_events)} failure events...")
    
    for _, event in tqdm(sample_events.iterrows(), total=len(sample_events), desc="Processing events"):
        model_name = event['model']
        corruption = event['corruption']
        image_id = event['image_id']
        class_id = event['class_id']
        failure_severity = int(event['failure_severity'])
        failure_frame = int(event['failure_frame'])
        
        # Skip RT-DETR (Grad-CAM is YOLO-specific)
        model_config = config['models'][model_name]
        if model_config['type'] != 'yolo':
            continue
        
        # Get model path
        if model_config['fine_tuned'] and Path(model_config['checkpoint']).exists():
            model_path = model_config['checkpoint']
        else:
            model_path = model_config['pretrained']
        
        # Load model and setup Grad-CAM
        model = YOLO(model_path)
        gradcam = YOLOGradCAM(model, target_layer_name=gradcam_config['target_layer'])
        
        # Find clip and frame
        clip = next((c for c in clips if any(Path(f).stem == image_id for f in c['frames'])), None)
        if not clip:
            continue
        
        # Get frame path
        frame_path = next((f for f in clip['frames'] if Path(f).stem == image_id), None)
        if not frame_path:
            continue
        
        # Get tiny object bbox
        key = (event['clip_id'], frame_path, class_id)
        tiny_obj = tiny_obj_lookup.get(key)
        if not tiny_obj:
            continue
        
        # Analyze window before failure
        frame_idx = next((i for i, f in enumerate(clip['frames']) if f == frame_path), None)
        if frame_idx is None:
            continue
        
        start_frame = max(0, frame_idx - window_size)
        analysis_frames = clip['frames'][start_frame:frame_idx + 1]
        
        baseline_cam = None
        
        for frame_rel_path in analysis_frames:
            frame_idx_curr = clip['frames'].index(frame_rel_path)
            
            # Process each severity up to failure severity
            for severity in range(0, failure_severity + 1):
                # Get image path
                if severity == 0:
                    image_path = visdrone_root / frame_rel_path
                else:
                    image_path = corruptions_root / "sequences" / event['clip_id'] / corruption / str(severity) / "images" / Path(frame_rel_path).name
                
                if not image_path.exists():
                    continue
                
                # Load image
                image = np.array(Image.open(image_path))
                img_height, img_width = image.shape[:2]
                
                # Generate CAM
                try:
                    # Convert VisDrone bbox to YOLO format
                    visdrone_bbox = tiny_obj['bbox']  # (left, top, width, height) in pixels
                    yolo_bbox = visdrone_to_yolo_bbox(visdrone_bbox, img_width, img_height)
                    
                    cam = gradcam.generate_cam(
                        image,
                        yolo_bbox,
                        class_id
                    )
                except Exception as e:
                    continue
                
                # Store baseline (severity 0, first frame)
                if severity == 0 and frame_idx_curr == start_frame:
                    baseline_cam = cam
                
                # Compute metrics
                if baseline_cam is not None:
                    metrics = compute_cam_metrics(
                        cam,
                        yolo_bbox,
                        img_width,
                        img_height,
                        baseline_cam=baseline_cam
                    )
                    
                    cam_metrics_records.append({
                        'model': model_name,
                        'corruption': corruption,
                        'severity': severity,
                        'clip_id': event['clip_id'],
                        'frame_idx': frame_idx_curr,
                        'image_id': Path(frame_rel_path).stem,
                        'class_id': class_id,
                        'failure_severity': failure_severity,
                        'failure_frame': failure_frame,
                        'distance_to_failure': failure_frame - frame_idx_curr,
                        **metrics
                    })
    
    # Save CAM metrics
    if len(cam_metrics_records) > 0:
        print("\nSaving CAM metrics...")
        cam_metrics_df = pd.DataFrame(cam_metrics_records)
        cam_metrics_csv = results_dir / "gradcam_metrics_timeseries.csv"
        cam_metrics_df.to_csv(cam_metrics_csv, index=False)
        print(f"  Saved to {cam_metrics_csv}")
        print(f"  Total CAM metrics: {len(cam_metrics_records)}")
    else:
        print("\nNo CAM metrics computed")
    
    print("\n[OK] Grad-CAM failure analysis complete!")


if __name__ == "__main__":
    main()
