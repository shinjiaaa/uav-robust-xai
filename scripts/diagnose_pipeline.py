"""Diagnose pipeline issues: why lowlight/motion_blur have no detection records."""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml, load_json


def main():
    """Diagnose pipeline issues."""
    config_path = Path("configs/experiment.yaml")
    config = load_yaml(config_path)
    
    results_dir = Path(config['results']['root'])
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    print("=" * 80)
    print("PIPELINE DIAGNOSIS: Why lowlight/motion_blur have no detection records?")
    print("=" * 80)
    print()
    
    # Step 1: Check tiny objects
    print("1. TINY OBJECTS SAMPLING")
    print("-" * 80)
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    if tiny_objects_file.exists():
        tiny_objects = load_json(tiny_objects_file)
        print(f"   [OK] Found {len(tiny_objects)} tiny objects")
        unique_images = set(obj['image_id'] for obj in tiny_objects)
        print(f"   [OK] Unique images: {len(unique_images)}")
    else:
        print("   [ERROR] tiny_objects_samples.json not found")
        return
    print()
    
    # Step 2: Check corrupted images
    print("2. CORRUPTED IMAGE GENERATION")
    print("-" * 80)
    corruptions = config['corruptions']['types']
    severities = config['corruptions']['severities']
    
    corruption_stats = {}
    for corruption in corruptions:
        corruption_stats[corruption] = {}
        for severity in severities:
            if severity == 0:
                continue
            corrupt_dir = corruptions_root / corruption / str(severity) / "images"
            if corrupt_dir.exists():
                image_files = list(corrupt_dir.glob("*.jpg"))
                corruption_stats[corruption][severity] = len(image_files)
            else:
                corruption_stats[corruption][severity] = 0
    
    for corruption in corruptions:
        total = sum(corruption_stats[corruption].values())
        print(f"   {corruption}:")
        for severity in severities:
            if severity == 0:
                continue
            count = corruption_stats[corruption].get(severity, 0)
            print(f"     Severity {severity}: {count} images")
        print(f"     Total (severity 1-4): {total} images")
        print()
    
    # Step 3: Check detection records
    print("3. DETECTION RECORDS")
    print("-" * 80)
    records_csv = results_dir / "tiny_records_timeseries.csv"
    if records_csv.exists():
        records_df = pd.read_csv(records_csv)
        print(f"   [OK] Found {len(records_df)} detection records")
        
        if len(records_df) > 0:
            print("\n   By corruption:")
            by_corr = records_df.groupby('corruption').size()
            for corruption in corruptions:
                count = by_corr.get(corruption, 0)
                status = "[OK]" if count > 0 else "[MISSING]"
                print(f"     {status} {corruption}: {count} records")
            
            print("\n   By corruption × severity:")
            by_corr_sev = records_df.groupby(['corruption', 'severity']).size()
            for corruption in corruptions:
                print(f"     {corruption}:")
                for severity in severities:
                    count = by_corr_sev.get((corruption, severity), 0)
                    status = "[OK]" if count > 0 else "[MISSING]"
                    print(f"       {status} Severity {severity}: {count} records")
            
            # Check for missing images
            print("\n   Missing image analysis:")
            for corruption in corruptions:
                corr_df = records_df[records_df['corruption'] == corruption]
                if len(corr_df) == 0:
                    print(f"     {corruption}: NO RECORDS")
                    # Check if images exist
                    missing_count = 0
                    for severity in severities:
                        if severity == 0:
                            continue
                        corrupt_dir = corruptions_root / corruption / str(severity) / "images"
                        if not corrupt_dir.exists():
                            missing_count += 1
                    if missing_count > 0:
                        print(f"       → {missing_count} severity directories missing")
                    else:
                        # Check if image files match tiny objects
                        sample_obj = tiny_objects[0]
                        image_name = Path(sample_obj['frame_path']).name
                        sample_sev_dir = corruptions_root / corruption / "1" / "images"
                        if sample_sev_dir.exists():
                            sample_image = sample_sev_dir / image_name
                            if not sample_image.exists():
                                print(f"       → Sample image {image_name} not found in {sample_sev_dir}")
                            else:
                                print(f"       → Images exist but no detection records (inference may have failed)")
        else:
            print("   [ERROR] No detection records found")
    else:
        print("   [ERROR] tiny_records_timeseries.csv not found")
    print()
    
    # Step 4: Check failure events
    print("4. FAILURE EVENTS")
    print("-" * 80)
    failure_events_csv = results_dir / "failure_events.csv"
    if failure_events_csv.exists():
        failure_df = pd.read_csv(failure_events_csv)
        print(f"   [OK] Found {len(failure_df)} failure events")
        
        if len(failure_df) > 0:
            print("\n   By corruption:")
            by_corr = failure_df.groupby('corruption').size()
            for corruption in corruptions:
                count = by_corr.get(corruption, 0)
                status = "[OK]" if count > 0 else "[MISSING]"
                print(f"     {status} {corruption}: {count} events")
    else:
        print("   [ERROR] failure_events.csv not found")
    print()
    
    # Step 5: Check Grad-CAM
    print("5. GRAD-CAM METRICS")
    print("-" * 80)
    gradcam_csv = results_dir / "gradcam_metrics_timeseries.csv"
    if gradcam_csv.exists():
        cam_df = pd.read_csv(gradcam_csv)
        print(f"   [OK] Found {len(cam_df)} CAM metric records")
        
        if len(cam_df) > 0:
            print("\n   By corruption:")
            by_corr = cam_df.groupby('corruption').size()
            for corruption in corruptions:
                count = by_corr.get(corruption, 0)
                status = "[OK]" if count > 0 else "[MISSING]"
                print(f"     {status} {corruption}: {count} CAM records")
    else:
        print("   [ERROR] gradcam_metrics_timeseries.csv not found")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if records_csv.exists():
        records_df = pd.read_csv(records_csv)
        for corruption in corruptions:
            count = len(records_df[records_df['corruption'] == corruption])
            if count == 0:
                print(f"[WARNING] {corruption}: NO DETECTION RECORDS")
                print(f"  → Check: Are corrupted images generated?")
                print(f"  → Check: Does 03_detect_tiny_objects_timeseries.py process this corruption?")
                print(f"  → Check: Are image paths correct in the script?")
    print()


if __name__ == "__main__":
    main()
