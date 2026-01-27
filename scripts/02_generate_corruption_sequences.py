"""Generate corruptions for single images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corruption.corruptions import corrupt_image
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
    print("Generating Corruptions for Single Images")
    print("=" * 60)
    print()
    
    # Load tiny objects
    results_dir = Path(config['results']['root'])
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    if not tiny_objects_file.exists():
        print(f"Error: Tiny objects not found. Run scripts/01_sample_tiny_objects.py first")
        sys.exit(1)
    
    tiny_objects = load_json(tiny_objects_file)
    print(f"Loaded {len(tiny_objects)} tiny objects")
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    corruptions = config['corruptions']['types']
    severities = config['corruptions']['severities']
    
    print(f"\nGenerating corruptions for {len(corruptions)} types, {len(severities)} severities...")
    
    # Get unique images
    unique_images = {}
    for tiny_obj in tiny_objects:
        image_id = tiny_obj['image_id']
        frame_path = tiny_obj['frame_path']
        if image_id not in unique_images:
            unique_images[image_id] = frame_path
    
    print(f"Processing {len(unique_images)} unique images")
    
    # Process each image
    for image_id, frame_rel_path in tqdm(unique_images.items(), desc="Processing images"):
        frame_path = visdrone_root / frame_rel_path
        if not frame_path.exists():
            continue
        
        for corruption_type in corruptions:
            for severity in severities:
                if severity == 0:
                    # Skip severity 0 (use original)
                    continue
                
                # Create output directory: corruptions_root / corruption / severity / images
                corrupt_dir = corruptions_root / corruption_type / str(severity) / "images"
                corrupt_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate corrupted image
                output_path = corrupt_dir / frame_path.name
                
                # Skip if already exists
                if output_path.exists():
                    continue
                
                corrupt_image(
                    frame_path,
                    corruption_type,
                    severity,
                    output_path,
                    seed=config['seed']
                )
    
    print("\n[OK] Corruption generation complete!")


if __name__ == "__main__":
    main()
