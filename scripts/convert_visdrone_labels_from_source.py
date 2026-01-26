"""Convert VisDrone source labels to YOLO format for corrupted datasets.

This script reads original VisDrone annotations and converts them to YOLO format
for the corrupted dataset structure.
"""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml


def convert_visdrone_annotation_to_yolo(
    visdrone_annotation_path: Path,
    image_path: Path,
    output_label_path: Path
):
    """Convert VisDrone annotation to YOLO format.
    
    VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    YOLO format: <class> <x_center> <y_center> <width> <height> (normalized, space-separated)
    
    Args:
        visdrone_annotation_path: Path to VisDrone annotation file
        image_path: Path to corresponding image file
        output_label_path: Path to save YOLO format label
    """
    # Get image dimensions
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return False
    
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Read VisDrone annotation
    if not visdrone_annotation_path.exists():
        print(f"Warning: Annotation not found: {visdrone_annotation_path}")
        return False
    
    lines = visdrone_annotation_path.read_text().strip().split('\n')
    
    yolo_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse VisDrone format: x,y,w,h,score,class,occlusion,truncation
        parts = line.split(',')
        if len(parts) < 6:
            continue
        
        try:
            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_width = float(parts[2])
            bbox_height = float(parts[3])
            score = float(parts[4])
            class_id = int(parts[5])
            
            # Skip if bbox is invalid
            if bbox_width <= 0 or bbox_height <= 0:
                continue
            
            # Skip if score is 0 (ignored regions in VisDrone)
            if score == 0:
                continue
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (bbox_left + bbox_width / 2) / img_width
            y_center = (bbox_top + bbox_height / 2) / img_height
            width = bbox_width / img_width
            height = bbox_height / img_height
            
            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            # Format: class x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
            
        except (ValueError, IndexError) as e:
            continue
    
    # Write YOLO format label
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text('\n'.join(yolo_lines) + '\n' if yolo_lines else '')
    return True


def main():
    """Convert VisDrone source annotations to YOLO format for specified corruptions."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    # Convert motion_blur severity 0 and 1
    corruption = "motion_blur"
    severities = [0, 1]
    split = "val"
    
    print("=" * 80)
    print("Converting VisDrone Source Annotations to YOLO Format")
    print("=" * 80)
    print(f"Corruption: {corruption}")
    print(f"Severities: {severities}")
    print(f"Split: {split}")
    print()
    
    # Source annotations directory
    source_annotations_dir = visdrone_root / "VisDrone2019-DET-val" / "annotations"
    
    if not source_annotations_dir.exists():
        print(f"Error: Source annotations directory not found: {source_annotations_dir}")
        sys.exit(1)
    
    total_converted = 0
    total_skipped = 0
    
    for severity in severities:
        labels_dir = corruptions_root / corruption / str(severity) / split / "labels"
        images_dir = corruptions_root / corruption / str(severity) / split / "images"
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist. Skipping severity {severity}.")
            continue
        
        print(f"Processing {corruption} severity {severity} {split}...")
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg"))
        print(f"  Found {len(image_files)} images")
        
        for image_path in image_files:
            # Find corresponding source annotation
            annotation_name = image_path.stem + ".txt"
            source_annotation_path = source_annotations_dir / annotation_name
            
            if not source_annotation_path.exists():
                print(f"  Warning: Source annotation not found for {image_path.name}")
                total_skipped += 1
                continue
            
            # Output label path
            output_label_path = labels_dir / annotation_name
            
            # Convert
            if convert_visdrone_annotation_to_yolo(
                source_annotation_path,
                image_path,
                output_label_path
            ):
                total_converted += 1
            else:
                total_skipped += 1
        
        print(f"  Converted: {total_converted}, Skipped: {total_skipped}")
    
    print()
    print("=" * 80)
    print(f"Conversion complete!")
    print(f"  Total converted: {total_converted}")
    print(f"  Total skipped: {total_skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
