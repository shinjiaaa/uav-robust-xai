"""Convert VisDrone label format to YOLO format.

VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
YOLO format: <class> <x_center> <y_center> <width> <height> (normalized, space-separated)
"""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml


def convert_visdrone_to_yolo_label(
    visdrone_label_path: Path,
    image_path: Path,
    output_label_path: Path
):
    """Convert a single VisDrone label file to YOLO format.
    
    Args:
        visdrone_label_path: Path to VisDrone format label file
        image_path: Path to corresponding image file (for dimensions)
        output_label_path: Path to save YOLO format label
    """
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Read VisDrone label
    if not visdrone_label_path.exists():
        print(f"Warning: {visdrone_label_path} does not exist")
        return
    
    lines = visdrone_label_path.read_text().strip().split('\n')
    
    yolo_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if already in YOLO format (space-separated, 5 values)
        if ' ' in line and ',' not in line:
            parts = line.split()
            if len(parts) == 5:
                # Already YOLO format: class x_center y_center width height
                try:
                    # Validate: first should be int (class), rest should be floats [0,1]
                    class_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    if all(0.0 <= c <= 1.0 for c in coords):
                        yolo_lines.append(line)
                        continue
                except ValueError:
                    pass
        
        # Parse VisDrone format: x,y,w,h,score,class,occlusion,truncation
        parts = line.split(',')
        if len(parts) < 6:
            # Invalid format
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
            print(f"Warning: Failed to parse line in {visdrone_label_path}: {line[:50]}... Error: {e}")
            continue
    
    # Write YOLO format label
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text('\n'.join(yolo_lines) + '\n' if yolo_lines else '')


def main():
    """Convert all VisDrone labels in lowlight/0 and lowlight/1 to YOLO format."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    # Convert lowlight severity 0 and 1
    corruption = "lowlight"
    severities = [0, 1]
    splits = ["val"]  # Add "train" if needed
    
    print("=" * 80)
    print("Converting VisDrone Labels to YOLO Format")
    print("=" * 80)
    print(f"Corruption: {corruption}")
    print(f"Severities: {severities}")
    print()
    
    total_converted = 0
    total_skipped = 0
    
    for severity in severities:
        for split in splits:
            labels_dir = corruptions_root / corruption / str(severity) / split / "labels"
            images_dir = corruptions_root / corruption / str(severity) / split / "images"
            
            if not labels_dir.exists():
                print(f"Warning: {labels_dir} does not exist. Skipping.")
                continue
            
            if not images_dir.exists():
                print(f"Warning: {images_dir} does not exist. Skipping.")
                continue
            
            print(f"Processing {corruption} severity {severity} {split}...")
            
            label_files = list(labels_dir.glob("*.txt"))
            print(f"  Found {len(label_files)} label files")
            
            for label_file in label_files:
                # Find corresponding image
                image_name = label_file.stem + ".jpg"
                image_path = images_dir / image_name
                
                if not image_path.exists():
                    # Try other extensions
                    for ext in [".png", ".PNG", ".JPG", ".jpeg", ".JPEG"]:
                        alt_image_path = images_dir / (label_file.stem + ext)
                        if alt_image_path.exists():
                            image_path = alt_image_path
                            break
                    else:
                        print(f"  Warning: Image not found for {label_file.name}")
                        total_skipped += 1
                        continue
                
                # Convert label
                convert_visdrone_to_yolo_label(
                    label_file,
                    image_path,
                    label_file  # Overwrite original
                )
                total_converted += 1
            
            print(f"  Converted {total_converted} labels, skipped {total_skipped}")
    
    print()
    print("=" * 80)
    print(f"Conversion complete!")
    print(f"  Total converted: {total_converted}")
    print(f"  Total skipped: {total_skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
