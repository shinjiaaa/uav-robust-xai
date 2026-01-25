"""Sample tiny objects from single images (no frame clips)."""

import sys
from pathlib import Path
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml, save_json
from src.utils.seed import set_seed
from PIL import Image


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Sampling Tiny Objects from Single Images")
    print("=" * 60)
    print()
    
    # Load images
    visdrone_root = Path(config['dataset']['visdrone_root'])
    split = config['evaluation']['splits'][0]  # Use first split
    
    if (visdrone_root / f"VisDrone2019-DET-{split}").exists():
        split_dir = visdrone_root / f"VisDrone2019-DET-{split}"
    else:
        print(f"Error: Split directory not found")
        sys.exit(1)
    
    images_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    annotations_dir = split_dir / "annotations" if (split_dir / "annotations").exists() else split_dir
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    # Find all tiny objects
    print("\n1. Finding tiny objects...")
    tiny_config = config['tiny_objects']
    experiment_config = config.get('experiment', {})
    one_per_image = experiment_config.get('one_per_image', False)
    sample_size = experiment_config.get('sample_size', tiny_config['sample_size'])
    
    all_tiny_objects = []
    image_tiny_objects = {}  # Track tiny objects per image
    
    for image_path in image_files:
        image_stem = image_path.stem
        ann_file = annotations_dir / f"{image_stem}.txt"
        
        if not ann_file.exists():
            continue
        
        # Load image to get dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            continue
        
        # Load annotations
        image_tiny_list = []
        
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 8:
                    continue
                
                try:
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    object_category = int(parts[5])
                    
                    if object_category == 0:  # Skip ignored
                        continue
                    
                    area = bbox_width * bbox_height
                    
                    # Check if tiny object (detectable small object)
                    # Must be: area >= area_threshold AND (width >= width_threshold OR height >= height_threshold)
                    area_ok = area >= tiny_config['area_threshold']
                    size_ok = bbox_width >= tiny_config['width_threshold'] or bbox_height >= tiny_config['height_threshold']
                    max_area = tiny_config.get('max_area_threshold', 20000)
                    area_not_too_large = area <= max_area
                    is_tiny = area_ok and size_ok and area_not_too_large
                    
                    if is_tiny:
                        # Convert to relative path
                        try:
                            frame_rel_path = str(image_path.relative_to(visdrone_root))
                        except ValueError:
                            frame_rel_path = str(image_path)
                        
                        tiny_obj = {
                            'image_id': image_stem,
                            'frame_path': frame_rel_path,
                            'bbox': (bbox_left, bbox_top, bbox_width, bbox_height),
                            'class_id': object_category,
                            'area': area,
                            'img_width': img_width,
                            'img_height': img_height
                        }
                        
                        image_tiny_list.append(tiny_obj)
                except (ValueError, IndexError):
                    continue
        
        # If one_per_image, select only one tiny object per image
        if one_per_image and len(image_tiny_list) > 0:
            image_key = str(image_path)
            if image_key not in image_tiny_objects:
                selected = random.choice(image_tiny_list)
                image_tiny_objects[image_key] = selected
                all_tiny_objects.append(selected)
        else:
            all_tiny_objects.extend(image_tiny_list)
    
    print(f"   Found {len(all_tiny_objects)} tiny objects")
    
    # Sample if needed
    if len(all_tiny_objects) > sample_size:
        random.seed(config['seed'])
        all_tiny_objects = random.sample(all_tiny_objects, sample_size)
    
    print(f"   Sampled {len(all_tiny_objects)} tiny objects")
    
    # Save results
    print("\n2. Saving results...")
    results_dir = Path(config['results']['root'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    save_json(all_tiny_objects, tiny_objects_file)
    print(f"   Saved to {tiny_objects_file}")
    
    print("\n[OK] Tiny object sampling complete!")


if __name__ == "__main__":
    main()
