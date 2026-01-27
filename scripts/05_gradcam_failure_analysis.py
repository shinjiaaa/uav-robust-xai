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
from src.xai.dynamic_refinement import detect_failure_region, generate_subdivided_severities
from src.corruption.corruptions import corrupt_image
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
    
    # Check if file is empty
    try:
        failure_events_df = pd.read_csv(failure_events_csv)
        if len(failure_events_df) == 0:
            print("Warning: No failure events found. Skipping Grad-CAM analysis.")
            sys.exit(0)
    except pd.errors.EmptyDataError:
        print("Warning: Failure events file is empty. Skipping Grad-CAM analysis.")
        sys.exit(0)
    
    print(f"Loaded {len(failure_events_df)} failure events")
    
    # Load tiny objects
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    tiny_objects = load_json(tiny_objects_file)
    
    # Create tiny object lookup
    tiny_obj_lookup = {}
    for obj in tiny_objects:
        key = (obj['image_id'], obj['class_id'])
        tiny_obj_lookup[key] = obj
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    gradcam_config = config['gradcam']
    
    cam_metrics_records = []
    error_count = 0  # Initialize error counter
    error_records = []  # Store error details for analysis
    device_checked = False  # Flag to print device info once
    
    # Process failure events (limit to max_samples)
    max_samples = gradcam_config.get('max_samples', 20)
    
    sample_events = (failure_events_df
                 .groupby("corruption", group_keys=False)
                 .head(max_samples//3))
                 
    print("\n[DBG] sample_events corruption distribution:")
    print(sample_events["corruption"].value_counts())
    print(f"\nProcessing {len(sample_events)} failure events...")
    
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
    yolo = YOLO(model_path)
    torch_model = yolo.model
    
    # CRITICAL: Force model to GPU if available to reduce CPU RAM usage
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch_model = torch_model.to(device)
        print(f"  Model moved to GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"  Model on CPU (GPU not available)")
        print(f"  [WARN] CPU mode will use more RAM - ensure sufficient memory")
    
    # Fallback layer candidates (most stable first)
    target_layer_candidates = [
        gradcam_config['target_layer'],  # Primary: model.18.cv2.conv
        "model.18.cv2.conv",  # Explicit backup
        "model.16.conv",  # Alternative backbone layer
        "model.19.conv",  # Another backbone option
    ]
    
    # Initialize Grad-CAM once with explicit device
    gradcam = None
    device_str = str(device) if 'device' in locals() else ("cuda:0" if torch.cuda.is_available() else "cpu")
    for layer_name in target_layer_candidates:
        try:
            gradcam = YOLOGradCAM(torch_model, target_layer_name=layer_name, device=device_str)
            actual_device = next(gradcam.model.parameters()).device
            print(f"  CAM model device: {actual_device}")
            print(f"  Using target layer: {layer_name}")
            break
        except Exception as e:
            if layer_name == target_layer_candidates[-1]:  # Last candidate failed
                print(f"  [ERROR] Failed to initialize Grad-CAM with all fallback layers: {e}")
                sys.exit(1)
    
    if gradcam is None:
        print("[ERROR] Failed to initialize Grad-CAM")
        sys.exit(1)
    
    for _, event in tqdm(sample_events.iterrows(), total=len(sample_events), desc="Processing events"):
        # Only process yolo_generic (already filtered by model loading)
        if event['model'] != model_name:
            continue
        
        corruption = event['corruption']
        image_id = event['image_id']
        class_id = event['class_id']
        failure_severity = int(event['failure_severity'])
        
        # Get tiny object bbox
        key = (image_id, class_id)
        tiny_obj = tiny_obj_lookup.get(key)
        if not tiny_obj:
            print(f"[DBG] missing tiny_obj for {corruption} {image_id} class={class_id}")
            continue
        
        frame_rel_path = tiny_obj['frame_path']
        
        # Process each severity up to failure severity
        baseline_cam = None
        
        for severity in range(0, failure_severity + 1):
            # Get image path
            if severity == 0:
                image_path = visdrone_root / frame_rel_path
            else:
                # New structure: corruptions_root / corruption / severity / images
                image_path = corruptions_root / corruption / str(severity) / "images" / Path(frame_rel_path).name
            
            if not image_path.exists():
                continue
            
            # Load image with memory-efficient approach
            try:
                # Use context manager for safe PIL image loading
                with Image.open(image_path) as pil_image:
                    # Check image size before processing
                    w, h = pil_image.size
                    pixel_count = w * h
                    
                    # Skip if image is too large (safety check)
                    if pixel_count > 40_000_000:  # 40MP limit
                        print(f"  [WARN] Image too large ({w}x{h}={pixel_count} pixels): {image_path}")
                        error_records.append({
                            'model': model_name,
                            'corruption': corruption,
                            'image_id': image_id,
                            'class_id': class_id,
                            'severity': severity,
                            'failure_severity': failure_severity,
                            'error_type': 'ImageTooLarge',
                            'error_message': f'Image size {w}x{h} exceeds 40MP limit',
                            'target_layer': gradcam_config['target_layer']
                        })
                        continue
                    
                    # Resize if too large (max side 1280px to reduce memory)
                    MAX_SIDE = 1280
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
                    'target_layer': gradcam_config['target_layer']
                })
                # Clean up and skip
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
                    'target_layer': gradcam_config['target_layer']
                })
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
            
            # Generate CAM with fallback layer retry
            cam = None
            visdrone_bbox = tiny_obj['bbox']  # (left, top, width, height) in pixels
            yolo_bbox = visdrone_to_yolo_bbox(visdrone_bbox, img_width, img_height)

            # Try to generate CAM (gradcam is already initialized)
            try:
                cam = gradcam.generate_cam(
                    image,
                    yolo_bbox,
                    class_id
                )
                # CRITICAL: Force garbage collection after CAM generation to free graph memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                # CUDA OOM
                print(f"[DBG] CUDA OOM: {str(e)[:120]}")
                error_msg = str(e)
                error_count += 1
                if error_count <= 5:
                    print(f"  [WARN] CUDA OOM for {image_id} severity {severity}: {e}")
                
                error_records.append({
                    'model': model_name,
                    'corruption': corruption,
                    'image_id': image_id,
                    'class_id': class_id,
                    'severity': severity,
                    'failure_severity': failure_severity,
                    'error_type': 'CUDA_OOM',
                    'error_message': error_msg[:200],
                    'target_layer': gradcam_config['target_layer']
                })
                # Clean up and continue
                del image
                gc.collect()
                torch.cuda.empty_cache()
                continue
            except MemoryError as e:
                # CPU RAM OOM
                print(f"[DBG] CPU MemoryError: {str(e)[:120]}")
                error_msg = str(e)
                error_count += 1
                if error_count <= 5:
                    print(f"  [WARN] CPU MemoryError for {image_id} severity {severity}: {e}")
                
                error_records.append({
                    'model': model_name,
                    'corruption': corruption,
                    'image_id': image_id,
                    'class_id': class_id,
                    'severity': severity,
                    'failure_severity': failure_severity,
                    'error_type': 'CPU_MemoryError',
                    'error_message': error_msg[:200],
                    'target_layer': gradcam_config['target_layer']
                })
                # Clean up and continue
                del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except (RuntimeError, ValueError) as e:
                # Other runtime errors
                print(f"[DBG] RuntimeError: {str(e)[:120]}")
                error_msg = str(e)
                error_count += 1
                if error_count <= 5:
                    print(f"  [WARN] Failed to generate CAM for {image_id} severity {severity}: {e}")
                
                error_records.append({
                    'model': model_name,
                    'corruption': corruption,
                    'image_id': image_id,
                    'class_id': class_id,
                    'severity': severity,
                    'failure_severity': failure_severity,
                    'error_type': type(e).__name__,
                    'error_message': error_msg[:200],
                    'target_layer': gradcam_config['target_layer']
                })
                # Clean up and continue
                del image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if cam is None:
                # Clean up memory before continuing
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue  # Skip this severity if all layers failed
            
            # Store baseline (severity 0) - use copy to avoid reference sharing
            if severity == 0:
                baseline_cam = cam.copy()  # CRITICAL: Copy to avoid reference sharing across corruptions
                # Force garbage collection after baseline creation
                gc.collect()
            
            # Compute metrics
            metrics = None
            if baseline_cam is not None:
                # CRITICAL: Create new metrics dict for each record to avoid reference sharing
                # Use baseline_cam directly (already a copy from severity 0)
                # Only copy if we need to modify it, but compute_cam_metrics doesn't modify
                metrics = compute_cam_metrics(
                    cam,
                    yolo_bbox,
                    img_width,
                    img_height,
                    baseline_cam=baseline_cam  # Use directly, compute_cam_metrics doesn't modify it
                )
                
                # CRITICAL: Create new dict with explicit values (not references)
                record = {
                    'model': model_name,
                    'corruption': corruption,
                    'severity': severity,
                    'image_id': image_id,
                    'class_id': class_id,
                    'failure_severity': failure_severity,
                    'energy_in_bbox': float(metrics['energy_in_bbox']),  # Explicit conversion
                    'activation_spread': float(metrics['activation_spread']),
                    'entropy': float(metrics['entropy']),
                    'center_shift': float(metrics['center_shift']),
                }
                cam_metrics_records.append(record)
                corruption_cam_stats[corruption]['success'] += 1
            
            if cam is None:
                corruption_cam_stats[corruption]['failed'] += 1
            
            # CRITICAL: Clean up memory after each severity
            # Delete in order to ensure proper cleanup
            del image
            if cam is not None:
                del cam
            if metrics is not None:
                del metrics
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
        if baseline_cam is not None:
            del baseline_cam
            baseline_cam = None
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
    del gradcam
    del torch_model
    del yolo
    # Multiple passes of garbage collection for thorough cleanup
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("\n[INFO] CUDA cache cleared after all events")
    else:
        print("\n[INFO] CPU memory cleaned up after all events")
    
    # Print CAM generation statistics by corruption
    print("\n[DBG] CAM generation statistics by corruption:")
    for corr, stats in corruption_cam_stats.items():
        total = stats['success'] + stats['failed']
        if total > 0:
            print(f"  {corr}: {stats['success']} success, {stats['failed']} failed (total attempts: {total})")
    
    # Save initial CAM metrics
    if len(cam_metrics_records) > 0:
        print("\nSaving initial CAM metrics...")
        cam_metrics_df = pd.DataFrame(cam_metrics_records)
        cam_metrics_csv = results_dir / "gradcam_metrics_timeseries.csv"
        cam_metrics_df.to_csv(cam_metrics_csv, index=False)
        print(f"  Saved to {cam_metrics_csv}")
        print(f"  Total CAM metrics: {len(cam_metrics_records)}")
        if error_count > 0:
            print(f"  [WARN] Skipped {error_count} CAM generations due to errors")
    else:
        print("\nNo CAM metrics computed")
        if error_count > 0:
            print(f"  [ERROR] All {error_count} CAM generations failed")
        return
    
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
        
        # Detect failure regions
        failure_regions = detect_failure_region(
            cam_metrics_df,
            threshold=refinement_config.get('failure_detection_threshold', 0.3)
        )
        
        print(f"\nDetected {len(failure_regions)} failure regions")
        
        if len(failure_regions) > 0:
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
                
                # Use fallback layers for refinement too
                gradcam_refined = None
                for layer_name in target_layer_candidates:
                    try:
                        gradcam_refined = YOLOGradCAM(torch_model_refined, target_layer_name=layer_name)
                        break
                    except Exception:
                        if layer_name == target_layer_candidates[-1]:
                            del yolo_refined
                            del torch_model_refined
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue  # Skip this region if all layers fail
                
                if gradcam_refined is None:
                    del yolo_refined
                    del torch_model_refined
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Get tiny object
                key = (image_id, class_id)
                tiny_obj = tiny_obj_lookup.get(key)
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
                    baseline_cam = gradcam_refined.generate_cam(baseline_image, yolo_bbox, class_id)
                    baseline_cam = baseline_cam.copy()  # CRITICAL: Copy to avoid reference sharing
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
                    
                    # Try CAM generation with fallback layers
                    cam = None
                    for layer_name in target_layer_candidates:
                        try:
                            # Use the refined gradcam instance
                            cam = gradcam_refined.generate_cam(image, yolo_bbox, class_id)
                            break  # Success
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
                                'target_layer': layer_name
                            })
                            del image
                            gc.collect()
                            torch.cuda.empty_cache()
                            break
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
                                'target_layer': layer_name
                            })
                            del image
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            break
                        except (RuntimeError, ValueError) as e:
                            error_msg = str(e)
                            # Check if it's a shape mismatch error
                            if "Sizes of tensors must match" in error_msg or "Expected size" in error_msg:
                                if layer_name != target_layer_candidates[-1]:
                                    continue  # Try next candidate
                            # If last candidate or non-shape error, record and skip
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
                                'target_layer': layer_name
                            })
                            del image
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            break  # Exit retry loop
                    
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
                        metrics = compute_cam_metrics(
                            cam,
                            yolo_bbox,
                            img_width,
                            img_height,
                            baseline_cam=baseline_cam.copy()  # Copy to avoid reference sharing
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
            del baseline_image
            del baseline_cam
            del gradcam_refined
            del torch_model_refined
            del yolo_refined
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Append refined records
            if len(refined_records) > 0:
                print(f"\nGenerated {len(refined_records)} refined CAM metrics")
                refined_df = pd.DataFrame(refined_records)
                
                # Append to existing metrics
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
