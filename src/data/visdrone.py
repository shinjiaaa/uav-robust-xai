"""VisDrone dataset download and preprocessing."""

import os
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import shutil


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Chunk size for streaming
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {e}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check if file is valid ZIP
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid ZIP file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def download_visdrone(config: Dict, force: bool = False):
    """Download VisDrone dataset using Ultralytics download method.
    
    Args:
        config: Configuration dictionary
        force: Force re-download even if exists
    """
    visdrone_root = Path(config['dataset']['visdrone_root'])
    visdrone_root.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if not force and (visdrone_root / "VisDrone2019-DET-val").exists():
        print(f"Dataset already exists at {visdrone_root}")
        return
    
    # Use Ultralytics download method (as per official docs)
    try:
        from ultralytics.utils.downloads import download
        from ultralytics.utils import ASSETS_URL
        
        print("Downloading VisDrone dataset using Ultralytics method...")
        print(f"Download directory: {visdrone_root}")
        
        # URLs from Ultralytics ASSETS
        urls = [
            f"{ASSETS_URL}/VisDrone2019-DET-train.zip",
            f"{ASSETS_URL}/VisDrone2019-DET-val.zip",
            f"{ASSETS_URL}/VisDrone2019-DET-test-dev.zip",
        ]
        
        # Download using Ultralytics download function
        download(urls, dir=visdrone_root, threads=4)
        
        print("Download complete!")
        
    except Exception as e:
        print(f"Error downloading via Ultralytics: {e}")
        print("Falling back to manual download instructions...")
        print("Please download manually from: https://github.com/VisDrone/VisDrone-Dataset")
        raise


def convert_visdrone_to_yolo(config: Dict):
    """Convert VisDrone annotations to YOLO format.
    
    VisDrone annotation format (per line):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    YOLO format (per line):
    <class_id> <x_center> <y_center> <width> <height>
    (normalized coordinates)
    
    Args:
        config: Configuration dictionary
    """
    visdrone_root = Path(config['dataset']['visdrone_root'])
    yolo_root = Path(config['dataset']['visdrone_yolo_root'])
    
    # VisDrone class mapping (10 classes)
    # 0: ignored regions, 1: pedestrian, 2: people, 3: bicycle, 4: car,
    # 5: van, 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor, 11: others
    # We map to YOLO: ignore 0, map 1-11 to 0-10
    class_mapping = {i: i - 1 for i in range(1, 12)}  # 1->0, 2->1, ..., 11->10
    
    splits = ['train', 'val']
    
    for split in splits:
        visdrone_split_dir = visdrone_root / f"VisDrone2019-DET-{split}"
        if not visdrone_split_dir.exists():
            print(f"Warning: {visdrone_split_dir} does not exist. Skipping {split}.")
            continue
        
        images_dir = visdrone_split_dir / "images"
        annotations_dir = visdrone_split_dir / "annotations"
        
        yolo_images_dir = yolo_root / "images" / split
        yolo_labels_dir = yolo_root / "labels" / split
        
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting {split} split...")
        
        annotation_files = list(annotations_dir.glob("*.txt"))
        for ann_file in tqdm(annotation_files, desc=f"Processing {split}"):
            image_name = ann_file.stem + ".jpg"
            image_path = images_dir / image_name
            
            if not image_path.exists():
                # Try .png
                image_path = images_dir / (ann_file.stem + ".png")
                if not image_path.exists():
                    continue
            
            # Copy image
            shutil.copy2(image_path, yolo_images_dir / image_name)
            
            # Convert annotations
            yolo_label_path = yolo_labels_dir / (ann_file.stem + ".txt")
            with open(ann_file, 'r') as f_in, open(yolo_label_path, 'w') as f_out:
                # Read image dimensions (we'll need to get this from the image)
                from PIL import Image
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 6:
                        continue
                    
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    object_category = int(parts[5])
                    
                    # Skip ignored regions (category 0)
                    if object_category == 0:
                        continue
                    
                    # Skip if category not in mapping
                    if object_category not in class_mapping:
                        continue
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (bbox_left + bbox_width / 2) / img_width
                    y_center = (bbox_top + bbox_height / 2) / img_height
                    width = bbox_width / img_width
                    height = bbox_height / img_height
                    
                    # Clamp to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Skip if bbox is too small
                    if width < 0.001 or height < 0.001:
                        continue
                    
                    yolo_class = class_mapping[object_category]
                    f_out.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Create YOLO dataset YAML
    yaml_path = yolo_root / "visdrone.yaml"
    yolo_config = {
        'path': str(yolo_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 11,  # number of classes
        'names': [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
        ]
    }
    
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    
    print(f"YOLO dataset created at {yolo_root}")
    print(f"Dataset config saved to {yaml_path}")
