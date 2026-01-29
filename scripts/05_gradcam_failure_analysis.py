"""Grad-CAM analysis for failure events."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import gc

# Set PIL safety limits to prevent decompression bombs
Image.MAX_IMAGE_PIXELS = 40_000_000  # 40MP limit (adjust if needed)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai.gradcam_yolo import YOLOGradCAM
from src.xai.cam_metrics import compute_cam_metrics
from src.xai.cam_qc import get_qc_status
from src.xai.cam_extraction import extract_cam_multi_layer
from src.xai.cam_records import create_cam_record, save_cam_records
from src.xai.dynamic_refinement import detect_failure_region, generate_subdivided_severities
from src.corruption.corruptions import corrupt_image
from src.data.bbox_conversion import visdrone_to_yolo_bbox, get_image_dimensions, extract_letterbox_meta
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
    
    # CRITICAL: Load detection_records.csv to get matched/prediction information for CAM target selection
    # This is required for RQ1: CAM generation even when matched=0 (miss)
    detection_records_csv = results_dir / "detection_records.csv"
    detection_records_df = None
    if detection_records_csv.exists() and detection_records_csv.stat().st_size > 0:
        try:
            detection_records_df = pd.read_csv(detection_records_csv)
            print(f"[INFO] Loaded {len(detection_records_df)} detection records for CAM target selection")
        except Exception as e:
            print(f"[WARN] Failed to load detection_records.csv: {e}")
            print(f"[WARN] CAM will use GT class_id as target (may not work for miss cases)")
    
    # CRITICAL: Load risk_events.csv (new format with CAM computation scope)
    # If risk_events.csv exists, use it; otherwise fallback to failure_events.csv (legacy)
    risk_events_csv = results_dir / "risk_events.csv"
    failure_events_csv = results_dir / "failure_events.csv"
    
    use_risk_events = False
    if risk_events_csv.exists() and risk_events_csv.stat().st_size > 0:
        try:
            risk_events_df = pd.read_csv(risk_events_csv)
            if len(risk_events_df) > 0:
                failure_events_df = risk_events_df  # Use risk_events as failure_events
                use_risk_events = True
                print(f"[INFO] Loaded {len(risk_events_df)} risk events from risk_events.csv")
                print(f"[INFO] CAM will be computed only for risk regions (cam_sev_from to cam_sev_to)")
        except Exception as e:
            print(f"[WARN] Failed to load risk_events.csv: {e}")
            use_risk_events = False
    
    if not use_risk_events:
        # Fallback to legacy failure_events.csv
        if not failure_events_csv.exists():
            print("Error: Neither risk_events.csv nor failure_events.csv found.")
            print("Please run scripts/04_detect_risk_events.py (or scripts/04_detect_failure_events.py) first")
            sys.exit(1)
        
        try:
            failure_events_df = pd.read_csv(failure_events_csv)
            if len(failure_events_df) == 0:
                print("Warning: No failure events found. Skipping Grad-CAM analysis.")
                sys.exit(0)
        except pd.errors.EmptyDataError:
            print("Warning: Failure events file is empty. Skipping Grad-CAM analysis.")
            sys.exit(0)
        
        print(f"[INFO] Loaded {len(failure_events_df)} failure events from failure_events.csv (legacy)")
        print(f"[WARN] Using legacy format - CAM will be computed for all severities 0 to failure_severity")
    
    print(f"Total events to process: {len(failure_events_df)}")
    
    # Load tiny objects
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    tiny_objects = load_json(tiny_objects_file)
    
    # object_uid-based tiny_obj join: primary by object_uid, fallback by (image_id, class_id)
    # Avoids overwriting when multiple tiny objects share same (image_id, class_id) -> fewer "missing tiny_obj" / CAM 0
    tiny_obj_by_uid = {}
    tiny_obj_by_ic = {}  # (image_id, class_id) -> list of objs (first used as fallback for legacy events)
    for obj in tiny_objects:
        uid = obj.get('object_uid')
        if uid:
            tiny_obj_by_uid[uid] = obj
        image_id = obj.get('image_id', '')
        class_id = obj.get('class_id', obj.get('gt_class_id'))
        if image_id != '' and class_id is not None:
            key_ic = (image_id, int(class_id))
            if key_ic not in tiny_obj_by_ic:
                tiny_obj_by_ic[key_ic] = []
            tiny_obj_by_ic[key_ic].append(obj)
    # Legacy single-key lookup (last writer wins) - only used if both uid and ic fallback fail
    tiny_obj_lookup = {}
    for obj in tiny_objects:
        key = (obj.get('image_id', ''), obj.get('class_id', obj.get('gt_class_id')))
        tiny_obj_lookup[key] = obj
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    gradcam_config = config['gradcam']
    layer_config = gradcam_config.get('layers', {})
    qc_config = gradcam_config.get('quality_gate', {})
    
    # New schema: cam_records (replaces cam_metrics_records)
    cam_records = []
    error_count = 0  # Initialize error counter
    error_records = []  # Store error details for analysis
    device_checked = False  # Flag to print device info once
    
    # RQ1: Process all events when max_samples is null/unset; no debug subset
    max_samples = gradcam_config.get('max_samples')
    cap_events = (max_samples is not None and max_samples > 0)
    
    if use_risk_events:
        # Process all risk events (they are already scoped to risk regions)
        sample_events = failure_events_df.copy()
        if cap_events:
            sample_events = sample_events.head(max_samples)
            print(f"\n[INFO] Using risk_events.csv - capped to {len(sample_events)} events (max_samples={max_samples})")
        else:
            print(f"\n[INFO] Using risk_events.csv - processing all {len(sample_events)} risk events")
        print("[INFO] CAM will be computed only for risk regions (reduces computation cost)")
    else:
        # Legacy: process all or cap by max_samples
        if cap_events:
            sample_events = (failure_events_df.groupby("corruption", group_keys=False)
                             .apply(lambda g: g.head(max(max_samples // 3, 1))).reset_index(drop=True))
            print(f"\n[INFO] Using legacy failure_events.csv - capped to {len(sample_events)} events (max_samples={max_samples})")
        else:
            sample_events = failure_events_df.copy()
            print(f"\n[INFO] Using legacy failure_events.csv - processing all {len(sample_events)} events")
                 
    print("\n[DBG] sample_events corruption distribution:")
    print(sample_events["corruption"].value_counts())
    print(f"\nProcessing {len(sample_events)} events...")
    
    # Track CAM generation success/failure by corruption
    corruption_cam_stats = {corr: {'success': 0, 'failed': 0} for corr in sample_events['corruption'].unique()}
    
    # CRITICAL: Load model ONCE outside the loop to avoid memory accumulation
    model_name = 'yolo_generic'  # Only process yolo_generic
    model_config = config['models'][model_name]
    if model_config['type'] != 'yolo':
        print(f"[ERROR] Model {model_name} is not YOLO type")
        sys.exit(1)
    
    # Get model path
    if model_config['fine_tuned'] and Path(model_config['checkpoint']).exists():
        model_path = model_config['checkpoint']
    else:
        model_path = model_config['pretrained']
    
    print(f"\nLoading model: {model_path}")
    
    # CRITICAL: Check file access with retry logic
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            # Check if model file is accessible
            if not Path(model_path).exists():
                print(f"[ERROR] Model file not found: {model_path}")
                sys.exit(1)
            
            # Try to open file to check if it's locked
            try:
                with open(model_path, 'rb') as f:
                    f.read(1)  # Try to read first byte
            except (OSError, IOError, PermissionError) as e:
                if attempt < max_retries - 1:
                    import time
                    print(f"[WARN] Model file locked (attempt {attempt+1}/{max_retries}), waiting {retry_delay}s...")
                    print(f"[WARN] If this persists, run: python scripts/kill_python_processes.py")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print(f"[ERROR] Cannot access model file (may be locked by another process): {model_path}")
                    print(f"[ERROR] Error: {e}")
                    print(f"[ERROR] Please run: python scripts/kill_python_processes.py")
                    sys.exit(1)
            
            # Load model with explicit task specification
            yolo = YOLO(model_path, task='detect')
            torch_model = yolo.model
            break  # Success
            
        except Exception as e:
            error_msg = str(e).lower()
            if attempt < max_retries - 1:
                import time
                print(f"[WARN] Model loading failed (attempt {attempt+1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                if "cannot access" in error_msg or "being used" in error_msg or "locked" in error_msg or "permission" in error_msg:
                    print(f"[ERROR] Model file access denied: {model_path}")
                    print(f"[ERROR] Another process may be using this model file")
                    print(f"[ERROR] Please run: python scripts/kill_python_processes.py")
                    sys.exit(1)
                raise
    
    # CRITICAL: Force model to GPU if available to reduce CPU RAM usage
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch_model = torch_model.to(device)
        print(f"  Model moved to GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"  Model on CPU (GPU not available)")
        print(f"  [WARN] CPU mode will use more RAM - ensure sufficient memory")
    
    # Stage B: Initialize Grad-CAM for multiple layers (primary/secondary)
    device_str = str(device) if 'device' in locals() else ("cuda:0" if torch.cuda.is_available() else "cpu")
    gradcam_instances = {}
    
    for layer_role, layer_info in layer_config.items():
        layer_name = layer_info['name']
        required = layer_info.get('required', True)
        
        try:
            gradcam = YOLOGradCAM(torch_model, target_layer_name=layer_name, device=device_str)
            actual_device = next(gradcam.model.parameters()).device
            gradcam_instances[layer_role] = gradcam
            print(f"  [{layer_role.upper()}] CAM model device: {actual_device}")
            print(f"  [{layer_role.upper()}] Using target layer: {layer_name}")
        except Exception as e:
            if required:
                print(f"  [ERROR] Failed to initialize required {layer_role} layer {layer_name}: {e}")
                sys.exit(1)
            else:
                print(f"  [WARN] Failed to initialize optional {layer_role} layer {layer_name}: {e}")
                print(f"  [WARN] Continuing without {layer_role} layer")
    
    if len(gradcam_instances) == 0:
        print("[ERROR] Failed to initialize any Grad-CAM layers")
        sys.exit(1)
    
    for _, event in tqdm(sample_events.iterrows(), total=len(sample_events), desc="Processing events"):
        # Only process yolo_generic (already filtered by model loading)
        event_model = event.get('model', event.get('model_id', ''))
        if event_model != model_name:
            continue
        
        # Ensure class_id/image_id/object_uid exist so post-loop or later use does not raise UnboundLocalError
        class_id = event.get('class_id', 0)
        image_id = event.get('image_id', '')
        object_uid = event.get('object_uid', '')
        corruption = event['corruption']
        
        # CRITICAL: Use risk_events format if available (has object_uid, cam_sev_from, cam_sev_to)
        if use_risk_events and 'object_uid' in event and 'cam_sev_from' in event and 'cam_sev_to' in event:
            object_uid = event['object_uid']
            cam_sev_from = int(event['cam_sev_from'])
            cam_sev_to = int(event['cam_sev_to'])
            start_severity = int(event['start_severity'])
            failure_type = event.get('failure_type', 'unknown')
            failure_event_id = event.get('failure_event_id', '')
            
            # Parse image_id, class_id from object_uid for fallback lookup
            # object_uid format from 01/03: {image_id}_obj_{class_id}_{index} (e.g. 0000213_05745_d_0000247_obj_4_0)
            try:
                if '_cls' in object_uid:
                    parts = object_uid.split('_cls')
                    image_id = parts[0].rsplit('_obj', 1)[0] if '_obj' in parts[0] else parts[0]
                    class_id = int(parts[1])
                elif '_obj' in object_uid:
                    parts = object_uid.split('_obj', 1)
                    image_id = parts[0]
                    rest = parts[1] if len(parts) > 1 else ''
                    class_id = int(rest.split('_')[0]) if rest and rest.split('_')[0].isdigit() else 0
                else:
                    image_id = object_uid
                    class_id = 0
            except Exception as e:
                image_id = event.get('image_id', object_uid)
                class_id = event.get('class_id', 0)
                print(f"[WARN] Failed to parse object_uid {object_uid}: {e}, using image_id={image_id}, class_id={class_id}")
            
            # Lookup: object_uid first (tiny_obj_by_uid), then (image_id, class_id) first match (tiny_obj_by_ic)
            tiny_obj = tiny_obj_by_uid.get(object_uid) if object_uid else None
            if tiny_obj is None and image_id is not None and class_id is not None:
                objs_ic = tiny_obj_by_ic.get((image_id, int(class_id)), [])
                tiny_obj = objs_ic[0] if objs_ic else None
            if tiny_obj is None:
                tiny_obj = tiny_obj_lookup.get((image_id, class_id))
            if not tiny_obj:
                print(f"[DBG] missing tiny_obj for {corruption} {image_id} class={class_id} (object_uid={object_uid})")
                continue
            
            frame_rel_path = tiny_obj['frame_path']
            failure_severity = start_severity  # Use start_severity as failure_severity
            
            # CRITICAL: Process only CAM computation scope (cam_sev_from to cam_sev_to)
            severity_range = range(cam_sev_from, cam_sev_to + 1)
        else:
            # Legacy format: use failure_events.csv structure
            image_id = event['image_id']
            class_id = event['class_id']
            failure_severity = int(event['failure_severity'])
            failure_event_id = event.get('failure_event_id', '')
            failure_type = 'unknown'
            object_uid = event.get('object_uid', '') or f"{image_id}_obj_{class_id}"
            
            # Lookup: object_uid first, then (image_id, class_id) first from list
            tiny_obj = tiny_obj_by_uid.get(object_uid) if object_uid else None
            if tiny_obj is None and image_id is not None and class_id is not None:
                objs_ic = tiny_obj_by_ic.get((image_id, int(class_id)), [])
                tiny_obj = objs_ic[0] if objs_ic else None
            if tiny_obj is None:
                tiny_obj = tiny_obj_lookup.get((image_id, class_id))
            if not tiny_obj:
                print(f"[DBG] missing tiny_obj for {corruption} {image_id} class={class_id}")
                continue
            
            frame_rel_path = tiny_obj['frame_path']
            
            # Legacy: Process all severities 0 to failure_severity
            severity_range = range(0, failure_severity + 1)
        
        # Store baseline CAMs per layer (primary/secondary)
        baseline_cams = {}  # {layer_role: cam}
        
        # CRITICAL (RQ1): Initialize CAM target variables (will be set per severity)
        cam_target_class_id = class_id  # Default: GT class_id
        cam_target_type = "gt_class"  # Track target type for debugging
        
        for severity in severity_range:
            # Get image path
            if severity == 0:
                image_path = visdrone_root / frame_rel_path
            else:
                # New structure: corruptions_root / corruption / severity / images
                image_path = corruptions_root / corruption / str(severity) / "images" / Path(frame_rel_path).name
            
            if not image_path.exists():
                # B-2: Log skipped attempt so n_expected = n_ok + n_failed + n_skipped
                rec_skip = create_cam_record(
                    model=model_name, corruption=corruption, severity=severity,
                    image_id=image_id, class_id=class_id, object_id=object_uid,
                    bbox_x1=tiny_obj['bbox'][0], bbox_y1=tiny_obj['bbox'][1],
                    bbox_x2=tiny_obj['bbox'][0] + tiny_obj['bbox'][2], bbox_y2=tiny_obj['bbox'][1] + tiny_obj['bbox'][3],
                    layer_role='primary', layer_name=layer_config.get('primary', {}).get('name', 'unknown'),
                    cam_status='skipped', fail_reason='image_missing', exc_type=None, exc_msg=None,
                    failure_severity=failure_severity
                )
                if use_risk_events and 'failure_event_id' in event:
                    rec_skip['failure_event_id'] = event['failure_event_id']
                    rec_skip['failure_type'] = failure_type
                cam_records.append(rec_skip)
                continue
            
            # Load image with memory-efficient approach
            try:
                # Use context manager for safe PIL image loading
                with Image.open(image_path) as pil_image:
                    # Check image size before processing
                    w, h = pil_image.size
                    pixel_count = w * h
                    
                    # CRITICAL: Very aggressive size limit for memory-constrained systems
                    MAX_PIXELS = 10_000_000  # Reduced to 10MP for low-memory systems
                    if pixel_count > MAX_PIXELS:
                        print(f"  [WARN] Image too large ({w}x{h}={pixel_count} pixels): {image_path}")
                        error_records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'image_id': image_id,
                            'class_id': class_id,
                            'severity': severity,
                            'failure_severity': failure_severity,
                            'error_type': 'ImageTooLarge',
                            'error_message': f'Image size {w}x{h} exceeds {MAX_PIXELS//1_000_000}MP limit',
                            'target_layer': layer_config.get('primary', {}).get('name', 'unknown'),
                            'exc_type': 'ImageTooLarge',
                            'exc_msg': f'Image size {w}x{h} exceeds {MAX_PIXELS//1_000_000}MP limit'
                        })
                        rec_skip = create_cam_record(
                            model=model_name, corruption=corruption, severity=severity,
                            image_id=image_id, class_id=class_id, object_id=object_uid,
                            bbox_x1=tiny_obj['bbox'][0], bbox_y1=tiny_obj['bbox'][1],
                            bbox_x2=tiny_obj['bbox'][0] + tiny_obj['bbox'][2], bbox_y2=tiny_obj['bbox'][1] + tiny_obj['bbox'][3],
                            layer_role='primary', layer_name=layer_config.get('primary', {}).get('name', 'unknown'),
                            cam_status='skipped', fail_reason='image_too_large',
                            exc_type='ImageTooLarge', exc_msg=f'Image size {w}x{h} exceeds {MAX_PIXELS//1_000_000}MP limit',
                            failure_severity=failure_severity
                        )
                        if use_risk_events and 'failure_event_id' in event:
                            rec_skip['failure_event_id'] = event['failure_event_id']
                            rec_skip['failure_type'] = failure_type
                        cam_records.append(rec_skip)
                        continue
                    
                    # CRITICAL: Very aggressive resizing for memory-constrained systems
                    MAX_SIDE = 800  # Reduced to 800px to minimize memory usage
                    scale = MAX_SIDE / max(w, h) if max(w, h) > MAX_SIDE else 1.0
                    if scale < 1.0:
                        new_w, new_h = int(w * scale), int(h * scale)
                        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        w, h = new_w, new_h
                    
                    # Convert to RGB if needed
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Use np.asarray instead of np.array to avoid unnecessary copy
                    image = np.asarray(pil_image, dtype=np.uint8)
                    img_height, img_width = image.shape[:2]
                
                # PIL image is automatically closed by context manager
                # Explicit cleanup
                del pil_image
                # Force immediate garbage collection
                gc.collect()
                
            except MemoryError as e:
                print(f"  [ERROR] CPU Memory error loading image {image_path}: {e}")
                error_records.append({
                    'model': model_name,
                    'corruption': corruption,
                    'image_id': image_id,
                    'class_id': class_id,
                    'severity': severity,
                    'failure_severity': failure_severity,
                    'error_type': 'CPU_MemoryError',
                    'error_message': str(e)[:200],
                    'target_layer': layer_config.get('primary', {}).get('name', 'unknown'),
                    'exc_type': type(e).__name__,
                    'exc_msg': str(e)[:200]
                })
                rec_fail = create_cam_record(
                    model=model_name, corruption=corruption, severity=severity,
                    image_id=image_id, class_id=class_id, object_id=object_uid,
                    bbox_x1=tiny_obj['bbox'][0], bbox_y1=tiny_obj['bbox'][1],
                    bbox_x2=tiny_obj['bbox'][0] + tiny_obj['bbox'][2], bbox_y2=tiny_obj['bbox'][1] + tiny_obj['bbox'][3],
                    layer_role='primary', layer_name=layer_config.get('primary', {}).get('name', 'unknown'),
                    cam_status='fail', fail_reason='oom', exc_type=type(e).__name__, exc_msg=str(e)[:200],
                    failure_severity=failure_severity
                )
                if use_risk_events and 'failure_event_id' in event:
                    rec_fail['failure_event_id'] = event['failure_event_id']
                    rec_fail['failure_type'] = failure_type
                cam_records.append(rec_fail)
                gc.collect()
                continue
            except (OSError, IOError, ValueError) as e:
                # Handle corrupted images, decompression bombs, etc.
                print(f"  [ERROR] Image loading error {image_path}: {e}")
                error_records.append({
                    'model': model_name,
                    'corruption': corruption,
                    'image_id': image_id,
                    'class_id': class_id,
                    'severity': severity,
                    'failure_severity': failure_severity,
                    'error_type': type(e).__name__,
                    'error_message': str(e)[:200],
                    'target_layer': layer_config.get('primary', {}).get('name', 'unknown'),
                    'exc_type': type(e).__name__,
                    'exc_msg': str(e)[:200]
                })
                rec_fail = create_cam_record(
                    model=model_name, corruption=corruption, severity=severity,
                    image_id=image_id, class_id=class_id, object_id=object_uid,
                    bbox_x1=tiny_obj['bbox'][0], bbox_y1=tiny_obj['bbox'][1],
                    bbox_x2=tiny_obj['bbox'][0] + tiny_obj['bbox'][2], bbox_y2=tiny_obj['bbox'][1] + tiny_obj['bbox'][3],
                    layer_role='primary', layer_name=layer_config.get('primary', {}).get('name', 'unknown'),
                    cam_status='fail', fail_reason='image_load_failed', exc_type=type(e).__name__, exc_msg=str(e)[:200],
                    failure_severity=failure_severity
                )
                if use_risk_events and 'failure_event_id' in event:
                    rec_fail['failure_event_id'] = event['failure_event_id']
                    rec_fail['failure_type'] = failure_type
                cam_records.append(rec_fail)
                gc.collect()
                continue
            
            # DEBUG: Check image shape consistency across severities
            if severity == 0:
                baseline_shape = (img_height, img_width)
                print(f"[DBG] Baseline shape (severity 0): {baseline_shape} for {image_id}")
            else:
                current_shape = (img_height, img_width)
                if current_shape != baseline_shape:
                    print(f"[DBG] SHAPE MISMATCH: {image_id} severity {severity}: expected {baseline_shape}, got {current_shape}")
            
            # Stage B: Extract CAM from multiple layers (primary/secondary)
            visdrone_bbox = tiny_obj['bbox']  # (left, top, width, height) in pixels
            yolo_bbox = visdrone_to_yolo_bbox(visdrone_bbox, img_width, img_height)
            
            # CRITICAL (RQ1): Determine CAM target class_id based on matched status (per severity)
            # For RQ1, we need CAM even when matched=0 (miss) to analyze failure regions
            cam_target_class_id = class_id  # Default: GT class_id
            cam_target_type = "gt_class"  # Track target type for debugging
            
            if detection_records_df is not None:
                # Try to find matched prediction for this object+severity
                object_records = detection_records_df[
                    (detection_records_df.get('object_uid', '') == object_uid) &
                    (detection_records_df.get('corruption', '') == corruption) &
                    (detection_records_df.get('severity', -1) == severity)
                ]
                
                if len(object_records) > 0:
                    record = object_records.iloc[0]
                    matched = record.get('matched', 0)
                    
                    if matched == 1:
                        # Matched: use prediction's class_id (more meaningful for CAM)
                        pred_class_id = record.get('pred_class_id')
                        if pred_class_id is not None and not pd.isna(pred_class_id):
                            cam_target_class_id = int(pred_class_id)
                            cam_target_type = "pred_class"
                    else:
                        # Miss: use GT class_id (RQ1 requirement: CAM even for miss)
                        cam_target_class_id = class_id
                        cam_target_type = "gt_class_miss"
                        # Note: This allows CAM generation for miss cases, critical for RQ1
            
            # Extract CAMs from all configured layers
            cam_results = extract_cam_multi_layer(
                gradcam_instances,
                image,
                yolo_bbox,
                cam_target_class_id,  # Use determined target class_id
                layer_config,
                qc_config
            )
            
            # CRITICAL: Force garbage collection after CAM generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process each layer's CAM result
            for layer_role, layer_result in cam_results.items():
                cam = layer_result['cam']
                cam_status = layer_result['cam_status']
                fail_reason = layer_result['fail_reason']
                letterbox_meta = layer_result['letterbox_meta']
                cam_shape = layer_result['cam_shape']
                
                # Extract QC diagnostic stats for debugging
                cam_min = layer_result.get('cam_min')
                cam_max = layer_result.get('cam_max')
                cam_sum = layer_result.get('cam_sum')
                cam_var = layer_result.get('cam_var')
                cam_std = layer_result.get('cam_std')  # 추가: std for QC
                cam_dtype = layer_result.get('cam_dtype')
                finite_ratio = layer_result.get('finite_ratio')
                preprocessed_shape = layer_result.get('preprocessed_shape')
                
                # Stage C: Quality Gate (RQ1: soft labels - all CAMs saved)
                # CRITICAL (RQ1): All CAMs are saved with quality labels (no hard filtering)
                # Only extraction errors (system errors) are marked as 'fail'
                # Quality issues (flat/noisy/low_energy) are labels, not failures
                cam_quality = layer_result.get('cam_quality', 'unknown')
                
                # Only skip if extraction failed (system error) - quality issues are still saved
                extraction_errors = ['no_activation', 'no_grad', 'shape_mismatch', 'memory_error', 'cam_extraction_failed', 'layer_not_configured']
                is_extraction_error = (cam_status == 'fail' and fail_reason in extraction_errors)
                
                if is_extraction_error:
                    # Extract error information from layer_result
                    exc_type = layer_result.get('exc_type')
                    exc_msg = layer_result.get('exc_msg')
                    traceback_last_lines = layer_result.get('traceback_last_lines')
                    
                    # Parse preprocessed_shape if it's a string
                    preprocessed_shape_tuple = None
                    if preprocessed_shape and isinstance(preprocessed_shape, str):
                        try:
                            h, w = map(int, preprocessed_shape.split('x'))
                            preprocessed_shape_tuple = (h, w)
                        except:
                            pass
                    elif preprocessed_shape:
                        preprocessed_shape_tuple = preprocessed_shape
                    
                    # Create record for extraction failures (system errors)
                    # CRITICAL: Include object_uid and failure_event_id for alignment analysis
                    # RQ1: Include CAM target information and quality label
                    record = create_cam_record(
                        model=model_name,
                        corruption=corruption,
                        severity=severity,
                        image_id=image_id,
                        class_id=class_id,
                        object_id=object_uid if 'object_uid' in locals() else None,  # Store object_uid
                        cam_target_class_id=cam_target_class_id if 'cam_target_class_id' in locals() else class_id,  # RQ1
                        cam_target_type=cam_target_type if 'cam_target_type' in locals() else "gt_class",  # RQ1
                        bbox_x1=visdrone_bbox[0],
                        bbox_y1=visdrone_bbox[1],
                        bbox_x2=visdrone_bbox[0] + visdrone_bbox[2],
                        bbox_y2=visdrone_bbox[1] + visdrone_bbox[3],
                        layer_role=layer_role,
                        layer_name=layer_config[layer_role]['name'],
                        cam_status='fail',  # System error (extraction failed)
                        fail_reason=fail_reason,
                        cam_quality=cam_quality if 'cam_quality' in locals() else 'extraction_failed',  # RQ1: quality label
                        exc_type=exc_type,
                        exc_msg=exc_msg,
                        traceback_last_lines=traceback_last_lines,
                        # QC diagnostic stats
                        cam_min=cam_min,
                        cam_max=cam_max,
                        cam_sum=cam_sum,
                        cam_var=cam_var,
                        cam_std=cam_std,  # 추가: std for QC
                        cam_dtype=cam_dtype,
                        finite_ratio=finite_ratio,
                        preprocessed_shape=preprocessed_shape_tuple,
                        failure_severity=failure_severity
                    )
                    
                    # Add failure_event_id if available (for alignment analysis)
                    if use_risk_events and 'failure_event_id' in event:
                        record['failure_event_id'] = event['failure_event_id']
                        record['failure_type'] = failure_type
                    cam_records.append(record)
                    if layer_role == 'primary':
                        # RQ1: Only count extraction errors as "failed", not quality issues
                        corruption_cam_stats[corruption]['failed'] += 1
                    continue  # Skip metrics computation for extraction errors
                
                # Store baseline (severity 0) per layer
                if severity == 0:
                    baseline_cams[layer_role] = cam.copy() if cam is not None else None
                
                # Stage D: Compute metrics (layer-invariant)
                # RQ1: Compute metrics for all CAMs (including flat/noisy/low_energy)
                # Only skip if extraction failed (system error)
                if cam is not None and baseline_cams.get(layer_role) is not None:
                    # Convert visdrone bbox to xyxy format
                    bbox_xyxy = (
                        visdrone_bbox[0],
                        visdrone_bbox[1],
                        visdrone_bbox[0] + visdrone_bbox[2],
                        visdrone_bbox[1] + visdrone_bbox[3]
                    )
                    
                    # Add original image dimensions to letterbox_meta
                    if letterbox_meta is not None:
                        letterbox_meta['original_width'] = img_width
                        letterbox_meta['original_height'] = img_height
                    
                    metrics = compute_cam_metrics(
                        cam,
                        bbox_xyxy,
                        cam_shape,
                        letterbox_meta=letterbox_meta,
                        baseline_cam=baseline_cams[layer_role]
                    )
                    
                    # Extract QC diagnostic stats (even for OK records, for consistency)
                    cam_min = layer_result.get('cam_min')
                    cam_max = layer_result.get('cam_max')
                    cam_sum = layer_result.get('cam_sum')
                    cam_var = layer_result.get('cam_var')
                    cam_std = layer_result.get('cam_std')  # 추가: std for QC
                    cam_dtype = layer_result.get('cam_dtype')
                    finite_ratio = layer_result.get('finite_ratio')
                    preprocessed_shape = layer_result.get('preprocessed_shape')
                    
                    # Parse preprocessed_shape if it's a string
                    preprocessed_shape_tuple = None
                    if preprocessed_shape and isinstance(preprocessed_shape, str):
                        try:
                            h, w = map(int, preprocessed_shape.split('x'))
                            preprocessed_shape_tuple = (h, w)
                        except:
                            pass
                    elif preprocessed_shape:
                        preprocessed_shape_tuple = preprocessed_shape
                    
                    # Create record with new schema (RQ1: all CAMs saved with quality labels)
                    # CRITICAL: Include object_uid and failure_event_id for alignment analysis
                    # RQ1: Include CAM target information and quality label
                    cam_quality = layer_result.get('cam_quality', 'high')  # RQ1: quality label
                    record = create_cam_record(
                        model=model_name,
                        corruption=corruption,
                        severity=severity,
                        image_id=image_id,
                        class_id=class_id,
                        object_id=object_uid if 'object_uid' in locals() else None,  # Store object_uid
                        cam_target_class_id=cam_target_class_id,  # RQ1: track target class used for CAM
                        cam_target_type=cam_target_type,  # RQ1: track target type (gt_class/pred_class/gt_class_miss)
                        bbox_x1=visdrone_bbox[0],
                        bbox_y1=visdrone_bbox[1],
                        bbox_x2=visdrone_bbox[0] + visdrone_bbox[2],
                        bbox_y2=visdrone_bbox[1] + visdrone_bbox[3],
                        layer_role=layer_role,
                        layer_name=layer_config[layer_role]['name'],
                        cam_status='ok',  # RQ1: All quality levels are 'ok' (even flat/noisy/low_energy)
                        fail_reason=None,  # RQ1: No fail_reason for quality issues (they're labels)
                        cam_quality=cam_quality,  # RQ1: 'high' | 'flat' | 'noisy' | 'low_energy'
                        cam_h=cam_shape[0] if cam_shape else None,
                        cam_w=cam_shape[1] if cam_shape else None,
                        entropy=float(metrics['entropy']),
                        activation_spread=float(metrics['activation_spread']),
                        center_shift=float(metrics['center_shift']),
                        energy_in_bbox=float(metrics['energy_in_bbox']),
                        letterbox_meta=letterbox_meta,
                        # QC diagnostic stats (for consistency and debugging)
                        cam_min=cam_min,
                        cam_max=cam_max,
                        cam_sum=cam_sum,
                        cam_var=cam_var,
                        cam_std=cam_std,  # 추가: std for QC
                        cam_dtype=cam_dtype,
                        finite_ratio=finite_ratio,
                        preprocessed_shape=preprocessed_shape_tuple,
                        failure_severity=failure_severity
                    )
                    
                    # Add failure_event_id if available (for alignment analysis)
                    if use_risk_events and 'failure_event_id' in event:
                        record['failure_event_id'] = event['failure_event_id']
                        record['failure_type'] = failure_type
                    cam_records.append(record)
                    if layer_role == 'primary':
                        corruption_cam_stats[corruption]['success'] += 1
                
                # Clean up CAM for this layer
                if cam is not None:
                    del cam
            
            # CRITICAL: Aggressive cleanup after processing all layers
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Force Python garbage collection
            gc.collect()
            # Force PyTorch to free unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for CUDA operations to complete
            else:
                # On CPU, explicitly clear PyTorch cache
                torch.cuda.empty_cache() if hasattr(torch.cuda, 'empty_cache') else None
        
        # CRITICAL: Clean up after each failure event (all severities processed)
        baseline_cams.clear()
        # Aggressive cleanup after each event
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            # On CPU, try to free any cached allocations
            import sys
            if sys.platform == 'win32':
                # Windows: force memory release
                gc.collect()
                gc.collect()  # Second pass for stubborn references
    
    # CRITICAL: Clean up model and gradcam after all events
    for layer_role, gradcam_instance in gradcam_instances.items():
        if gradcam_instance is not None:
            del gradcam_instance
    gradcam_instances.clear()
    del torch_model
    del yolo
    # Multiple passes of garbage collection for thorough cleanup
    gc.collect()
    gc.collect()
    gc.collect()  # Third pass for stubborn references
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Second pass
        print("\n[INFO] CUDA cache cleared after all events")
    else:
        print("\n[INFO] CPU memory cleaned up after all events")
    
    # Print CAM generation statistics by corruption
    print("\n[DBG] CAM generation statistics by corruption:")
    for corr, stats in corruption_cam_stats.items():
        total = stats['success'] + stats['failed']
        if total > 0:
            print(f"  {corr}: {stats['success']} success, {stats['failed']} failed (total attempts: {total})")
    
    # Stage E: Save CAM records with new schema
    if len(cam_records) > 0:
        print("\nSaving CAM records...")
        cam_records_csv = results_dir / "cam_records.csv"
        save_cam_records(cam_records, cam_records_csv)
        
        # B-2: CAM fail/skip aggregate table (severity x corruption: n_expected, n_ok, n_failed, n_skipped, top_fail_reason)
        cam_df = pd.DataFrame(cam_records)
        if 'cam_status' in cam_df.columns and 'fail_reason' in cam_df.columns:
            # One row per (model, corruption, severity) - primary layer only to avoid double count
            cam_primary = cam_df[cam_df['layer_role'] == 'primary'] if 'layer_role' in cam_df.columns else cam_df
            if len(cam_primary) == 0 and 'layer_role' in cam_df.columns:
                cam_primary = cam_df
            grp = cam_primary.groupby(['model', 'corruption', 'severity'], dropna=False)
            n_expected = grp.size()
            n_ok = cam_primary.groupby(['model', 'corruption', 'severity'], dropna=False)['cam_status'].apply(lambda s: (s == 'ok').sum())
            n_failed = cam_primary.groupby(['model', 'corruption', 'severity'], dropna=False)['cam_status'].apply(lambda s: (s == 'fail').sum())
            n_skipped = cam_primary.groupby(['model', 'corruption', 'severity'], dropna=False)['cam_status'].apply(lambda s: (s == 'skipped').sum())
            def _top_fail_reason(s):
                s = s.dropna()
                if len(s) == 0:
                    return None
                vc = s.value_counts()
                return vc.index[0] if len(vc) > 0 else None
            failed_skipped = cam_primary[cam_primary['cam_status'].isin(['fail', 'skipped'])]
            top_fail = failed_skipped.groupby(['model', 'corruption', 'severity'], dropna=False)['fail_reason'].agg(_top_fail_reason) if len(failed_skipped) > 0 else pd.Series(dtype=object)
            top_fail = top_fail.reindex(n_expected.index)  # align with all groups
            summary_df = pd.DataFrame({
                'n_expected': n_expected,
                'n_ok': n_ok,
                'n_failed': n_failed,
                'n_skipped': n_skipped,
                'top_fail_reason': top_fail
            }).reset_index()
            summary_csv = results_dir / "cam_fail_summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"  Saved CAM fail summary to {summary_csv}")
        
        # Also save legacy format for backward compatibility
        # CRITICAL: Use primary ok → secondary ok → any ok fallback strategy
        # 1) Collect all OK records
        ok_recs = [r for r in cam_records if r.get('cam_status') == 'ok']
        
        # 2) Priority-based layer selection (primary → secondary → any)
        prefer_roles = ['primary', 'secondary']
        picked = []
        for role in prefer_roles:
            picked = [r for r in ok_recs if r.get('layer_role') == role]
            if len(picked) > 0:
                break
        
        # 3) Fallback: if no primary/secondary, use any OK record
        if len(picked) == 0:
            picked = ok_recs
        
        # 4) Build legacy records from picked records
        legacy_records = []
        for rec in picked:
            legacy_records.append({
                'model': rec['model'],
                'corruption': rec['corruption'],
                'severity': rec['severity'],
                'image_id': rec['image_id'],
                'class_id': rec['class_id'],
                'failure_severity': rec['failure_severity'],
                'energy_in_bbox': rec['energy_in_bbox'],
                'activation_spread': rec['activation_spread'],
                'entropy': rec['entropy'],
                'center_shift': rec['center_shift'],
            })
        
        # Initialize cam_metrics_df for dynamic refinement (even if empty)
        cam_metrics_df = None
        cam_metrics_csv = results_dir / "gradcam_metrics_timeseries.csv"
        
        if len(legacy_records) > 0:
            cam_metrics_df = pd.DataFrame(legacy_records)
            cam_metrics_df.to_csv(cam_metrics_csv, index=False)
            layer_used = picked[0].get('layer_role', 'unknown') if len(picked) > 0 else 'unknown'
            print(f"  Saved legacy format to {cam_metrics_csv} ({len(legacy_records)} {layer_used} layer records)")
        else:
            # CRITICAL: legacy_records가 0일 때도 "빈 CSV로 덮어쓰기" 해야 함
            # 이렇게 해야 예전 실행 결과가 남아있지 않고, 리포트가 현재 상태(0/N/A)를 반영함
            cam_metrics_df = pd.DataFrame(columns=[
                'model', 'corruption', 'severity', 'image_id', 'class_id',
                'failure_severity', 'energy_in_bbox', 'activation_spread',
                'entropy', 'center_shift'
            ])
            cam_metrics_df.to_csv(cam_metrics_csv, index=False)  # << 이 줄이 핵심 (덮어쓰기)
            print(f"  [WARN] No legacy records to save; wrote empty {cam_metrics_csv}")
        
        if error_count > 0:
            print(f"  [WARN] Skipped {error_count} CAM generations due to errors")
    else:
        print("\nNo CAM records computed")
        if error_count > 0:
            print(f"  [ERROR] All {error_count} CAM generations failed")
        # CRITICAL: Initialize empty DataFrame and overwrite CSV to clear old data
        cam_metrics_df = pd.DataFrame(columns=[
            'model', 'corruption', 'severity', 'image_id', 'class_id',
            'failure_severity', 'energy_in_bbox', 'activation_spread',
            'entropy', 'center_shift'
        ])
        cam_metrics_csv = results_dir / "gradcam_metrics_timeseries.csv"
        cam_metrics_df.to_csv(cam_metrics_csv, index=False)  # Overwrite with empty CSV
        print(f"  [WARN] No CAM records computed; wrote empty {cam_metrics_csv}")
    
    # Save error log for bias analysis
    if len(error_records) > 0:
        print("\nSaving CAM error log...")
        error_df = pd.DataFrame(error_records)
        error_csv = results_dir / "gradcam_errors.csv"
        error_df.to_csv(error_csv, index=False)
        print(f"  Saved to {error_csv}")
        print(f"  Total errors: {len(error_records)}")
        
        # Print error statistics
        print("\n  Error statistics:")
        if 'corruption' in error_df.columns:
            print(f"    By corruption:")
            for corr, count in error_df['corruption'].value_counts().items():
                print(f"      {corr}: {count} errors")
        if 'severity' in error_df.columns:
            print(f"    By severity:")
            for sev, count in error_df['severity'].value_counts().items():
                print(f"      {sev}: {count} errors")
        if 'error_type' in error_df.columns:
            print(f"    By error type:")
            for err_type, count in error_df['error_type'].value_counts().items():
                print(f"      {err_type}: {count} errors")
    
    # Dynamic refinement: detect failure regions and subdivide
    refinement_config = gradcam_config.get('dynamic_refinement', {})
    if refinement_config.get('enabled', False):
        print("\n" + "=" * 60)
        print("Dynamic Refinement: Detecting and Subdividing Failure Regions")
        print("=" * 60)
        
        # CRITICAL: If cam_metrics_df is empty, try to load from disk CSV
        if (cam_metrics_df is None or len(cam_metrics_df) == 0) and cam_metrics_csv.exists():
            try:
                cam_metrics_df = pd.read_csv(cam_metrics_csv)
                print(f"  [INFO] Loaded CAM metrics from disk: {cam_metrics_csv} ({len(cam_metrics_df)} records)")
            except Exception as e:
                print(f"  [WARN] Failed to load CAM metrics from {cam_metrics_csv}: {e}")
                cam_metrics_df = None
        
        # Check if cam_metrics_df exists and has data
        if cam_metrics_df is None or len(cam_metrics_df) == 0:
            print("  [WARN] No CAM metrics available for dynamic refinement. Skipping.")
            failure_regions = []
        else:
            # Detect failure regions
            failure_regions = detect_failure_region(
                cam_metrics_df,
                threshold=refinement_config.get('failure_detection_threshold', 0.3)
            )
        
        if len(failure_regions) > 0:
            print(f"\nDetected {len(failure_regions)} failure regions")
            
            # Subdivide and re-analyze
            subdivision_steps = refinement_config.get('subdivision_steps', 10)
            refined_records = []
            
            print(f"\nSubdividing failure regions into {subdivision_steps} steps...")
            
            for region in tqdm(failure_regions[:20], desc="Refining regions"):  # Limit to 20 regions
                start_sev = region['start_severity']
                end_sev = region['end_severity']
                
                # Generate subdivided severities
                subdivided_severities = generate_subdivided_severities(
                    start_sev,
                    end_sev,
                    num_steps=subdivision_steps
                )
                
                if len(subdivided_severities) == 0:
                    continue
                
                # Re-analyze with subdivided severities
                model_name = region['model']
                
                # Only process yolo_generic
                if model_name != 'yolo_generic':
                    continue
                
                corruption = region['corruption']
                image_id = region['image_id']
                class_id = region['class_id']
                
                # Get model
                model_config = config['models'][model_name]
                if model_config['type'] != 'yolo':
                    continue
                
                if model_config['fine_tuned'] and Path(model_config['checkpoint']).exists():
                    model_path = model_config['checkpoint']
                else:
                    model_path = model_config['pretrained']
                
                # CRITICAL: Reuse the same model/gradcam from main loop if possible
                # For now, create new instance but clean up properly
                yolo_refined = YOLO(model_path)
                torch_model_refined = yolo_refined.model
                
                # Use primary layer for refinement
                primary_layer_name = layer_config.get('primary', {}).get('name', 'model.18.cv2.conv')
                gradcam_refined = None
                try:
                    gradcam_refined = YOLOGradCAM(torch_model_refined, target_layer_name=primary_layer_name)
                except Exception as e:
                    print(f"  [ERROR] Failed to initialize Grad-CAM for refinement: {e}")
                    del yolo_refined
                    del torch_model_refined
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue  # Skip this region if layer fails
                
                if gradcam_refined is None:
                    del yolo_refined
                    del torch_model_refined
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Get tiny object (same join as main loop: object_uid -> (image_id, class_id) first)
                object_uid_ref = region.get('object_uid', '') or (f"{image_id}_obj_{class_id}" if image_id else '')
                tiny_obj = tiny_obj_by_uid.get(object_uid_ref) if object_uid_ref else None
                if tiny_obj is None and image_id is not None and class_id is not None:
                    objs_ic = tiny_obj_by_ic.get((image_id, int(class_id)), [])
                    tiny_obj = objs_ic[0] if objs_ic else None
                if tiny_obj is None:
                    tiny_obj = tiny_obj_lookup.get((image_id, class_id))
                if not tiny_obj:
                    print(f"[DBG] missing tiny_obj for {corruption} {image_id} class={class_id}")
                    continue
                
                frame_rel_path = tiny_obj['frame_path']
                
                # Get baseline CAM (severity 0)
                baseline_image_path = visdrone_root / frame_rel_path
                if not baseline_image_path.exists():
                    continue
                
                # Load baseline image with memory-efficient approach
                try:
                    with Image.open(baseline_image_path) as pil_image:
                        w, h = pil_image.size
                        pixel_count = w * h
                        
                        if pixel_count > 40_000_000:
                            print(f"  [WARN] Baseline image too large ({w}x{h}): {baseline_image_path}")
                            continue
                        
                        # Resize if too large
                        MAX_SIDE = 1280
                        scale = MAX_SIDE / max(w, h) if max(w, h) > MAX_SIDE else 1.0
                        if scale < 1.0:
                            new_w, new_h = int(w * scale), int(h * scale)
                            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            w, h = new_w, new_h
                        
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        baseline_image = np.asarray(pil_image, dtype=np.uint8)
                        img_height, img_width = baseline_image.shape[:2]
                        baseline_shape_refined = (img_height, img_width)
                        print(f"[DBG] Refined baseline shape (severity 0): {baseline_shape_refined} for {image_id}")
                except MemoryError as e:
                    print(f"  [ERROR] CPU Memory error loading baseline image: {e}")
                    gc.collect()
                    continue
                except (OSError, IOError, ValueError) as e:
                    print(f"  [ERROR] Image loading error: {e}")
                    gc.collect()
                    continue
                visdrone_bbox = tiny_obj['bbox']
                yolo_bbox = visdrone_to_yolo_bbox(visdrone_bbox, img_width, img_height)
                
                try:
                    # CRITICAL: generate_cam returns (cam, meta_dict) tuple, unpack it
                    baseline_cam_result = gradcam_refined.generate_cam(baseline_image, yolo_bbox, class_id)
                    if isinstance(baseline_cam_result, tuple):
                        baseline_cam = baseline_cam_result[0].copy()  # Extract cam array and copy
                    else:
                        baseline_cam = baseline_cam_result.copy()  # Fallback: already array
                except Exception as e:
                    print(f"  [WARN] Failed to generate baseline CAM for refined analysis: {e}")
                    del baseline_image
                    del yolo_refined
                    del torch_model_refined
                    del gradcam_refined
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Analyze subdivided severities
                for sub_severity in subdivided_severities:
                    # Generate corrupted image if not exists
                    if sub_severity == 0:
                        image_path = baseline_image_path
                    else:
                        corrupt_dir = corruptions_root / corruption / str(sub_severity) / "images"
                        corrupt_dir.mkdir(parents=True, exist_ok=True)
                        image_path = corrupt_dir / Path(frame_rel_path).name
                        
                        if not image_path.exists():
                            # Generate corruption on-the-fly
                            corrupt_image(
                                baseline_image_path,
                                corruption,
                                sub_severity,
                                image_path,
                                seed=config['seed']
                            )
                    
                    if not image_path.exists():
                        continue
                    
                    # Load and analyze
                    # Load image with memory-efficient approach
                    try:
                        with Image.open(image_path) as pil_image:
                            w, h = pil_image.size
                            pixel_count = w * h
                            
                            if pixel_count > 40_000_000:
                                print(f"  [WARN] Image too large ({w}x{h}): {image_path}")
                                continue
                            
                            # Resize if too large
                            MAX_SIDE = 1280
                            scale = MAX_SIDE / max(w, h) if max(w, h) > MAX_SIDE else 1.0
                            if scale < 1.0:
                                new_w, new_h = int(w * scale), int(h * scale)
                                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                w, h = new_w, new_h
                            
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            image = np.asarray(pil_image, dtype=np.uint8)
                            img_height, img_width = image.shape[:2]
                    except MemoryError as e:
                        print(f"  [ERROR] CPU Memory error loading image: {e}")
                        gc.collect()
                        continue
                    except (OSError, IOError, ValueError) as e:
                        print(f"  [ERROR] Image loading error: {e}")
                        gc.collect()
                        continue
                    current_shape_refined = image.shape[:2]
                    if current_shape_refined != baseline_shape_refined:
                        print(f"[DBG] REFINED SHAPE MISMATCH: {image_id} sub_severity {sub_severity}: expected {baseline_shape_refined}, got {current_shape_refined}")
                    
                    # Try CAM generation (single layer for refinement)
                    cam = None
                    try:
                        cam, letterbox_meta_refined = gradcam_refined.generate_cam(image, yolo_bbox, class_id)
                    except torch.cuda.OutOfMemoryError as e:
                        error_msg = str(e)
                        error_records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'image_id': image_id,
                            'class_id': class_id,
                            'severity': sub_severity,
                            'failure_severity': end_sev,
                            'error_type': 'CUDA_OOM',
                            'error_message': error_msg[:200],
                            'refined': True,
                            'original_start_sev': start_sev,
                            'original_end_sev': end_sev,
                            'target_layer': primary_layer_name
                        })
                        del image
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    except MemoryError as e:
                        error_msg = str(e)
                        error_records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'image_id': image_id,
                            'class_id': class_id,
                            'severity': sub_severity,
                            'failure_severity': end_sev,
                            'error_type': 'CPU_MemoryError',
                            'error_message': error_msg[:200],
                            'refined': True,
                            'original_start_sev': start_sev,
                            'original_end_sev': end_sev,
                            'target_layer': primary_layer_name
                        })
                        del image
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    except (RuntimeError, ValueError) as e:
                        error_msg = str(e)
                        error_records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'image_id': image_id,
                            'class_id': class_id,
                            'severity': sub_severity,
                            'failure_severity': end_sev,
                            'error_type': type(e).__name__,
                            'error_message': error_msg[:200],
                            'refined': True,
                            'original_start_sev': start_sev,
                            'original_end_sev': end_sev,
                            'target_layer': primary_layer_name
                        })
                        del image
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue  # Exit retry loop
                    
                    if cam is None:
                        # Clean up before continuing
                        del image
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue  # Skip this severity
                    
                    # Compute metrics if CAM succeeded
                    metrics = None
                    try:
                        # CRITICAL: Ensure baseline_cam is numpy array (not tuple)
                        baseline_cam_for_metrics = baseline_cam
                        if isinstance(baseline_cam, tuple):
                            baseline_cam_for_metrics = baseline_cam[0]
                        metrics = compute_cam_metrics(
                            cam,
                            yolo_bbox,
                            img_width,
                            img_height,
                            baseline_cam=baseline_cam_for_metrics.copy()  # Copy to avoid reference sharing
                        )
                        
                        # CRITICAL: Create new dict with explicit values (not references)
                        record = {
                            'model': model_name,
                            'corruption': corruption,
                            'severity': sub_severity,
                            'image_id': image_id,
                            'class_id': class_id,
                            'failure_severity': end_sev,
                            'refined': True,  # Mark as refined
                            'original_start_sev': start_sev,
                            'original_end_sev': end_sev,
                            'energy_in_bbox': float(metrics['energy_in_bbox']),  # Explicit conversion
                            'activation_spread': float(metrics['activation_spread']),
                            'entropy': float(metrics['entropy']),
                            'center_shift': float(metrics['center_shift']),
                        }
                        refined_records.append(record)
                    except Exception as e:
                        # Metrics computation error
                        error_records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'image_id': image_id,
                            'class_id': class_id,
                            'severity': sub_severity,
                            'failure_severity': end_sev,
                            'error_type': type(e).__name__,
                            'error_message': str(e)[:200],
                            'refined': True,
                            'original_start_sev': start_sev,
                            'original_end_sev': end_sev,
                            'target_layer': 'metrics_computation'
                        })
                    finally:
                        # CRITICAL: Clean up memory after each refined severity
                        del image
                        if cam is not None:
                            del cam
                        if metrics is not None:
                            del metrics
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # CRITICAL: Clean up after refined analysis
            if 'baseline_image' in locals():
                del baseline_image
            if 'baseline_cam' in locals():
                del baseline_cam
            if 'gradcam_refined' in locals():
                del gradcam_refined
            if 'torch_model_refined' in locals():
                del torch_model_refined
            if 'yolo_refined' in locals():
                del yolo_refined
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Append refined records
            if len(refined_records) > 0:
                print(f"\nGenerated {len(refined_records)} refined CAM metrics")
                refined_df = pd.DataFrame(refined_records)
                
                # Append to existing metrics (ensure cam_metrics_df exists)
                if cam_metrics_df is None or len(cam_metrics_df) == 0:
                    cam_metrics_df = refined_df
                else:
                    cam_metrics_df = pd.concat([cam_metrics_df, refined_df], ignore_index=True)
                
                cam_metrics_df.to_csv(cam_metrics_csv, index=False)
                print(f"  Updated {cam_metrics_csv}")
                print(f"  Total CAM metrics: {len(cam_metrics_df)}")
        
        # Update error log after refinement
        if len(error_records) > 0:
            error_df = pd.DataFrame(error_records)
            error_csv = results_dir / "gradcam_errors.csv"
            error_df.to_csv(error_csv, index=False)
            print(f"\n  Updated error log: {error_csv} ({len(error_records)} total errors)")
    
    print("\n[OK] Grad-CAM failure analysis complete!")


if __name__ == "__main__":
    main()
